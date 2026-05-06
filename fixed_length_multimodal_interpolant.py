import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class FixedLengthMultimodalModelPrediction:
    # This handles denoising
    clean_data: Tensor
    # This handles the unmasking
    label_logits: Tensor

@dataclass
class SamplingTrajectoryResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    x_mask_t: Tensor # Shape [Batch, Length]
    y_mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]

@dataclass
class SamplingResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    x_mask_t: Tensor # Shape [Batch, Length]
    y_mask_t: Tensor # Shape [Batch, Length]
    trajectory: List[SamplingTrajectoryResult]

    def __getitem__(self, index: int) -> SamplingTrajectoryResult:
        trajectory_slice = []
        for step in self.trajectory:
             trajectory_slice.append(SamplingTrajectoryResult(
                xt=step.xt[index],
                yt=step.yt[index],
                x_mask_t=step.x_mask_t[index],
                y_mask_t=step.y_mask_t[index],
                t=step.t[index]
             ))

        return SamplingResult(
            xt=self.xt[index],
            yt=self.yt[index],
            x_mask_t=self.x_mask_t[index],
            y_mask_t=self.y_mask_t[index],
            trajectory=trajectory_slice
        )

    def __len__(self) -> int:
        return self.xt.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]



@dataclass
class JointMultimodalInterpolantResult:
    # Joint Interpolant
    pad_token: int
    t: Tensor # Shape [Batch]
    # X stuff
    x1: Tensor
    x_mask_1: Tensor
    xt: Tensor  # Shape [Batch, Length, D] (D is the euclidean dimension)
    # Y stuff
    y1: Tensor
    y_mask_1: Tensor
    yt: Tensor  # Shape [Batch, Length]


class FixedLengthMultimodalInterpolant():
    def __init__(
        self,
        max_length: int,
        non_special_tokens: int = 5,
        vocab_size: int = 10000,
        mask_token: int = 10000,
        pad_token: int = 10001,
        bos_token: int = 10002,
        euclidean_dim: int = 3,
    ):
        super().__init__()
        self.max_length = max_length
        self.euclidean_dim = euclidean_dim
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.non_special_tokens = non_special_tokens

    def dalpha(self, t):
        return torch.ones_like(t)

    def alpha(self, t):
        return t

    def pad_sequence(self, x, y, mask_x, mask_y):
        """Prepend a BOS position so position 0 is always the start token."""
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        y = torch.cat([torch.full_like(y[:, :1], self.bos_token), y], dim=1)
        mask_x = torch.cat([torch.ones_like(mask_x[:, :1]), mask_x], dim=1)
        mask_y = torch.cat([torch.ones_like(mask_y[:, :1]), mask_y], dim=1)
        return x, y, mask_x, mask_y

    def get_unmasking_time(self, y1):
        return torch.rand_like(y1, dtype=torch.float32)

    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor, attn_mask_x: Tensor, attn_mask_y: Tensor) -> JointMultimodalInterpolantResult:
        t_shaped_disc = t.view(-1,1)
        t_shaped_euc = t.view(-1,1,1)

        # Add noise to euclidean data
        full_xt = self.alpha(t_shaped_euc) * x1 + (1 - self.alpha(t_shaped_euc)) * torch.randn_like(x1)
        full_xt = torch.where(attn_mask_x.unsqueeze(-1), full_xt, 0.)
        full_xt[:,0] = 0. # Don't corrupt the BOS position

        # Masking discrete data
        # Deletion and masking times are independent for every position, thats why we pass y1
        unmasking_time_disc = self.get_unmasking_time(y1)

        # Discrete data
        mask_positions = (t_shaped_disc <= unmasking_time_disc) & attn_mask_y
        mask_positions[:, 0] = False  # BOS is never masked
        yt = torch.where(mask_positions, self.mask_token, y1) # Change to mask id

        return JointMultimodalInterpolantResult(
            pad_token=self.pad_token,
            t=t,
            x1=x1,
            x_mask_1=attn_mask_x,
            xt=full_xt,
            y1=y1,
            y_mask_1=attn_mask_y,
            yt=yt,
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)

    def compute_loss(self, model, batch):
        _x1 = batch["x"]
        _y1 = batch["y"]
        _mask_x = batch["mask_x"]
        _mask_y = batch["mask_y"]
        x1, y1, mask_x, mask_y = self.pad_sequence(_x1, _y1, _mask_x, _mask_y)
        t = self.sample_time(x1.shape[0], x1.device)
        interpolant_sample = self.sample_interpolant(t, x1, y1, mask_x, mask_y)

        prediction: FixedLengthMultimodalModelPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            symbols_mask=interpolant_sample.y_mask_1,
            pos_mask=interpolant_sample.x_mask_1,
            symbols_time=t,
            pos_time=t
        )
        x_mask_shaped = interpolant_sample.x_mask_1.unsqueeze(-1)

        # Euclidean loss
        dsm_loss = (interpolant_sample.x1 - prediction.clean_data)**2 * x_mask_shaped
        dsm_loss[:,0] = 0. # Don't take loss at the BOS position
        dsm_loss = dsm_loss.sum(dim=-1)
        dsm_loss = dsm_loss.mean() / x1.shape[-1]

        # Unmasking loss
        masked_positions = (interpolant_sample.yt == self.mask_token)
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y1[masked_positions]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
        }

    def get_drift(self, prediction: FixedLengthMultimodalModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data
        return (clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1))

    def get_unmasking_rate(self, prediction: FixedLengthMultimodalModelPrediction, t: Tensor) -> Tensor:
        # Subtracting 4 because we are not considering the start of sequence token, end of sequence token, the mask token and the pad token
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1, 1) * prediction.label_logits[:, :, :self.vocab_size-4].softmax(dim=-1)
        return rate

    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tensor:
        xt = torch.randn((batch_size, max_length, self.euclidean_dim), device=device)
        yt = torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.mask_token
        yt[:,0] = self.bos_token
        return xt, yt

    def update_xt_yt(
        self,
        prediction: FixedLengthMultimodalModelPrediction,
        xt: Tensor,
        yt: Tensor,
        x_mask_t: Tensor,
        y_mask_t: Tensor,
        t: Tensor,
        dt: float,
        is_last_step: bool,
    ) -> tuple[Tensor, Tensor]:
        """Denoise and unmask both euclidean and discrete. Does not modify masks."""
        # Denoise euclidean
        drift = self.get_drift(prediction, xt, t)
        xt = xt + (drift * dt)
        xt = torch.where(x_mask_t.unsqueeze(-1), xt, 0.)
        xt[:, 0] = 0.

        # Unmask discrete
        masked_positions_disc = (yt == self.mask_token)
        unmasking_rate_disc = self.get_unmasking_rate(prediction, t)
        unmasking_nums_disc = torch.distributions.poisson.Poisson(unmasking_rate_disc * dt).sample()
        num_jumps = unmasking_nums_disc.sum(dim=-1)
        if not is_last_step:
            change_pos = (num_jumps == 1) & (y_mask_t) & (masked_positions_disc)
            unmasking_nums_disc = unmasking_nums_disc * change_pos.unsqueeze(-1)
            new_sample = unmasking_nums_disc.argmax(dim=-1)
        else:
            change_pos = masked_positions_disc & y_mask_t
            new_sample = unmasking_rate_disc.argmax(dim=-1)
        yt = torch.where(change_pos, new_sample, yt)
        yt[:, 0] = self.bos_token

        return xt, yt

    @torch.no_grad()
    def sampling(
        self,
        model: torch.nn.Module,
        steps: int,
        batch_size: int,
        x_length: int,
        y_length: int,
        max_length: int,
        device: torch.device,
        return_trace: bool = False,
    ) -> SamplingResult:
        max_length = max_length + 1  # Plus one for the BOS token
        xt, yt = self.get_prior_distribution(batch_size, max_length, device)

        # Masks: BOS position (0) plus x_length / y_length content positions
        mask_x = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_y = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_x[:, :x_length + 1] = True
        mask_y[:, :y_length + 1] = True

        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), x_mask_t=mask_x.clone(), y_mask_t=mask_y.clone(), t=t
            ))
        torch.set_printoptions(precision=2, sci_mode=False)

        for i in tqdm(range(steps), leave=False):
            t0 = t
            t1 = t0 + dt
            is_last_step = (i == steps - 1)

            prediction: FixedLengthMultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_y,
                pos_mask=mask_x,
                symbols_time=t0,
                pos_time=t0
            )
            xt, yt = self.update_xt_yt(
                prediction=prediction,
                xt=xt,
                yt=yt,
                x_mask_t=mask_x,
                y_mask_t=mask_y,
                t=t0,
                dt=dt,
                is_last_step=is_last_step,
            )

            t = t1

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), x_mask_t=mask_x.clone(), y_mask_t=mask_y.clone(), t=t
                ))

        return SamplingResult(
            xt=xt, yt=yt, x_mask_t=mask_x, y_mask_t=mask_y, trajectory=trajectory
        )


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
