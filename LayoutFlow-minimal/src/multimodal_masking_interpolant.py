from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class MultimodalMaskingResult:
    xt: Tensor          # (B, L, geom_dim) continuous geometry input, zeroed where masked
    yt: Tensor          # (B, L) discrete category input, mask_token id where masked
    is_masked: Tensor   # (B, L) bool, shared between both modalities -- an element is
                         # revealed or not as one atomic unit
    active: Tensor
    x1: Tensor
    y1: Tensor
    t: Tensor


class MultimodalMaskingInterpolant:
    '''
    Same unmasking-only, no-insertion masking process as ContinuousMaskingInterpolant,
    but geometry and category are no longer folded into one continuous (analog-bit)
    channel. Geometry keeps the continuous noise-interpolation treatment (mask ->
    zero, reveal -> denoise). Category gets a genuine discrete treatment: masked ->
    a real [MASK] embedding id (not a blanked continuous value), reveal -> the model
    predicts a categorical distribution over the vocab, trained with cross-entropy.
    This mirrors the old (pre-LayoutFlow-minimal) multimodal_interpolant.py, whose
    own internal ablation found discrete category handling beats analog-bit
    regression by a wide margin (FID 5.7-9.9 vs 20.6-414 on RICO).
    '''

    def __init__(self, geom_dim: int, num_cat: int):
        self.geom_dim = geom_dim
        self.num_cat = num_cat
        self.mask_token = num_cat   # one reserved id past the real category ids

    def alpha(self, t):
        return t

    def dalpha(self, t):
        return torch.ones_like(t)

    def sample_time(self, batch_size, device):
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)

    def sample_interpolant(self, t, x1, y1, active):
        unmasking_time = torch.rand(x1.shape[:2], device=x1.device)
        is_masked = active & (t.unsqueeze(-1) < unmasking_time)

        alpha_t = self.alpha(t).view(-1, 1, 1)
        xt = alpha_t * x1 + (1 - alpha_t) * torch.randn_like(x1)
        xt = torch.where((active & ~is_masked).unsqueeze(-1), xt, torch.zeros_like(xt))

        yt = torch.where(is_masked, torch.full_like(y1, self.mask_token), y1)
        yt = torch.where(active, yt, torch.zeros_like(y1))  # pad slots: arbitrary, never used in any loss/context that matters

        return MultimodalMaskingResult(xt=xt, yt=yt, is_masked=is_masked, active=active, x1=x1, y1=y1, t=t)

    def compute_loss(self, model, batch) -> dict:
        x1 = batch['x']; y1 = batch['y']; active = batch['mask']
        t = self.sample_time(x1.shape[0], x1.device)
        sample = self.sample_interpolant(t, x1, y1, active)
        geom_pred, cat_logits = model(sample.xt, sample.yt, sample.is_masked, t)

        revealed = active & ~sample.is_masked
        geom_err = (sample.x1 - geom_pred).pow(2).sum(dim=-1)[revealed]
        geom_loss = geom_err.mean() / self.geom_dim if geom_err.numel() > 0 else geom_pred.sum() * 0

        masked = active & sample.is_masked
        if masked.any():
            cat_loss = F.cross_entropy(cat_logits[masked], sample.y1[masked].long(), reduction='mean')
        else:
            cat_loss = cat_logits.sum() * 0

        return {'geom_loss': geom_loss, 'cat_loss': cat_loss}

    @torch.no_grad()
    def sampling(self, model, num_steps, active):
        B, L = active.shape
        device = active.device
        xt = torch.zeros((B, L, self.geom_dim), device=device)
        yt = torch.full((B, L), self.mask_token, dtype=torch.long, device=device)
        is_masked = active.clone()
        ts = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        dt = 1.0 / num_steps
        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(B)
            is_last_step = i == num_steps - 1
            geom_pred, cat_logits = model(xt, yt, is_masked, t)

            revealed = active & ~is_masked
            drift = (geom_pred - xt) / (1 - self.alpha(t).view(-1, 1, 1)).clamp(min=1e-5)
            xt = torch.where(revealed.unsqueeze(-1), xt + drift * dt, xt)

            masked = active & is_masked
            if is_last_step:
                reveal = masked
            else:
                rate = (self.dalpha(t) / (1 - self.alpha(t)).clamp(min=1e-5)).view(-1, 1)
                num_events = torch.distributions.Poisson(rate * dt).sample()
                reveal = masked & (num_events > 0)

            alpha_t = self.alpha(t).view(-1, 1, 1)
            new_geom = alpha_t * geom_pred + (1 - alpha_t) * torch.randn_like(xt)
            xt = torch.where(reveal.unsqueeze(-1), new_geom, xt)

            cat_sample = torch.distributions.Categorical(logits=cat_logits).sample()
            yt = torch.where(reveal, cat_sample, yt)

            is_masked = is_masked & ~reveal
        return xt, yt
