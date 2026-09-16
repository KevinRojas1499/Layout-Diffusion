"""
Masking diffusion over purely continuous data, unmasking only (no insertion/
deletion -- the active/padding set is fixed by the data itself, decided
externally rather than predicted by the model).

This is a reduced special case of multimodal_interpolant.py's joint
discrete+continuous interpolant: there, geometry (continuous) and category
(discrete) are separate channels, category drives masking via a dedicated
mask_token, and unmasking is a per-vocab-class Poisson race (see
discrete_interpolant.py for the same idea in a cleaner, fully-discrete form).
Here there is only one continuous channel -- geometry and category are
concatenated by the caller into one vector before this ever sees them -- so:

  - Masking is content-agnostic: a masked position is just zeroed, on the
    whole vector, exactly like multimodal_interpolant.py already does for its
    continuous channel alone.
  - There is no vocab to race over, so "which value does this become" is not
    a discrete choice -- it's a Gaussian centered at the model's own clean-data
    prediction (mean = prediction, variance from the same schedule used
    everywhere else in this codebase), evaluated at every active position
    (not just revealed ones), since the reveal step below consults exactly
    this prediction for masked positions too.
  - The reveal *timing* needs no learned rate at all: for masking diffusion,
    the marginal hazard for any still-masked position is the closed-form
    dalpha(t)/(1-alpha(t)) -- a property of the noise schedule, not of the
    data. Tau-leaping with this rate decides how many positions reveal each
    step; the model only has to supply the value to reveal them with.

Deliberately deferred (see conversation history, not implemented here):
  - Insertion/deletion (variable active-set size). Active positions are fixed
    for the whole trajectory, same as multimodal_interpolant.py's
    fixed_length=True mode.
  - The model predicting "this position is actually padding" -- the active
    set is supplied externally (e.g. sampled from a length distribution),
    not chosen by revealing to a pad value, since there's no discrete pad
    token in a continuous channel.

The "masked" signal reaches the model via models/transformer.py's existing
discrete cat_tokens/d_encoder pathway, repurposed: cat_tokens is fed 0/1
(unmasked/masked) instead of a category id, with vocab_size=2. No model code
changes needed. label_logits/clean_data_unmasking come along for the ride and
are simply ignored -- only clean_data and (once insertion is added) the
already-existing insertion_rate are used.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from multimodal_interpolant import MultimodalModelPrediction


@dataclass
class ContinuousMaskingResult:
    xt: Tensor          # [B, L, D] noisy/masked data fed to the model
    is_masked: Tensor   # [B, L] bool -- True where still hidden
    active: Tensor      # [B, L] bool -- True where a real (non-padding) position
    x1: Tensor          # [B, L, D] clean target
    t: Tensor           # [B]


class ContinuousMaskingInterpolant:

    def __init__(self, euclidean_dim: int):
        self.euclidean_dim = euclidean_dim

    def alpha(self, t: Tensor) -> Tensor:
        return t

    def dalpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)

    def sample_interpolant(self, t: Tensor, x1: Tensor, active: Tensor) -> ContinuousMaskingResult:
        # Each active position gets its own independent uniform unmasking
        # time; before it, the position is hidden. No insertion means no
        # deletion_time coupling (c.f. multimodal_interpolant.py's
        # get_masking_and_deletion_time) -- every active position is
        # available to be masked from t=0.
        unmasking_time = torch.rand(x1.shape[:2], device=x1.device)
        is_masked = active & (t.unsqueeze(-1) < unmasking_time)

        alpha_t = self.alpha(t).view(-1, 1, 1)
        xt = alpha_t * x1 + (1 - alpha_t) * torch.randn_like(x1)
        xt = torch.where((active & ~is_masked).unsqueeze(-1), xt, torch.zeros_like(xt))

        return ContinuousMaskingResult(xt=xt, is_masked=is_masked, active=active, x1=x1, t=t)

    def compute_loss(self, model, batch) -> dict:
        x1 = batch['x']
        active = batch['mask']
        t = self.sample_time(x1.shape[0], x1.device)
        sample = self.sample_interpolant(t, x1, active)

        prediction: MultimodalModelPrediction = model(
            euclidean_tokens=sample.xt,
            cat_tokens=sample.is_masked.long(),
            pos_mask=active, symbols_mask=active,
            pos_time=t, symbols_time=t,
        )

        # One denoising loss, over every active position -- including still-
        # masked ones, since sampling's reveal step (below) consults this same
        # prediction there. No split between "revealed" and "masked" targets
        # the way multimodal_interpolant.py needs for its discrete channel.
        loss = (sample.x1 - prediction.clean_data).pow(2).sum(dim=-1)[active]
        loss = loss.mean() / self.euclidean_dim

        zero = torch.zeros((), device=x1.device, dtype=loss.dtype)
        return {
            'dsm_loss': loss,
            'discrete_unmasking_loss': zero,
            'euclidean_unmasking_loss': zero,
            'insertion_loss': zero,
        }

    @torch.no_grad()
    def sampling(self, model, num_steps: int, active: Tensor) -> Tensor:
        """active: [B, L] bool, fixed for the whole trajectory (see module docstring)."""
        B, L = active.shape
        device = active.device
        xt = torch.zeros((B, L, self.euclidean_dim), device=device)
        is_masked = active.clone()

        ts = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        dt = 1.0 / num_steps
        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(B)
            is_last_step = i == num_steps - 1

            prediction: MultimodalModelPrediction = model(
                euclidean_tokens=xt,
                cat_tokens=is_masked.long(),
                pos_mask=active, symbols_mask=active,
                pos_time=t, symbols_time=t,
            )

            # Denoise the already-revealed positions.
            revealed = active & ~is_masked
            drift = (prediction.clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1)).clamp(min=1e-5)
            xt = torch.where(revealed.unsqueeze(-1), xt + drift * dt, xt)

            # Decide which masked positions reveal this step. The rate is the
            # closed-form dalpha/(1-alpha) for every masked position alike --
            # nothing learned here, see module docstring.
            masked = active & is_masked
            if is_last_step:
                reveal = masked  # nothing may still be masked once t reaches 1
            else:
                rate = (self.dalpha(t) / (1 - self.alpha(t)).clamp(min=1e-5)).view(-1, 1)
                num_events = torch.distributions.Poisson(rate * dt).sample()
                reveal = masked & (num_events > 0)

            alpha_t = self.alpha(t).view(-1, 1, 1)
            new_value = alpha_t * prediction.clean_data + (1 - alpha_t) * torch.randn_like(xt)
            xt = torch.where(reveal.unsqueeze(-1), new_value, xt)
            is_masked = is_masked & ~reveal

        return xt
