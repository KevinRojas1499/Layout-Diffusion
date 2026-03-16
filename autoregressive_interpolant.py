import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class MultimodalModelPrediction:
    # This handles denoising
    clean_data: Tensor
    # This handles the unmasking
    label_logits: Tensor
    clean_data_unmasking: Tensor
    # This handles the insertions
    insertion_rate: Tensor

@dataclass
class SamplingTrajectoryResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]

@dataclass
class SamplingResult:
    xt: Tensor # Shape [Batch, Length]
    yt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    trajectory: List[SamplingTrajectoryResult]

    def __getitem__(self, index: int) -> SamplingResult:
        trajectory_slice = []
        for step in self.trajectory:
             trajectory_slice.append(SamplingTrajectoryResult(
                xt=step.xt[index],
                yt=step.yt[index],
                mask_t=step.mask_t[index],
                t=step.t[index]
             ))

        return SamplingResult(
            xt=self.xt[index],
            yt=self.yt[index],
            mask_t=self.mask_t[index],
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
    xt: Tensor  # Shape [Batch, Length]
    yt: Tensor  # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    x1: Tensor
    y1: Tensor

    @property
    def y1_length(self) -> Tensor:
        return (self.y1 != self.pad_token).sum(dim=1)
    
    @property
    def yt_length(self) -> Tensor:
        return (self.yt != self.pad_token).sum(dim=1)
    


class MultimodalInterpolant():
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
    
    def get_len_time_t(self, length_1, t):
        rate = -torch.log(1 - t)
        length = torch.poisson(rate)
        return torch.min(length, length_1)
    
    def get_masking_time_t(self,t):
        return torch.rand_like(t)
    
    def pad_sequence(self, x: Tensor, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mask_shaped = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # This is actually doubled because the dataset already handles it
        x = torch.where(mask_shaped, x, torch.zeros_like(x)) 
        y = torch.where(mask, y, torch.full_like(y, self.pad_token))

        # Add at the start of the sequence
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        y = torch.cat([torch.full_like(y[:, :1], self.bos_token), y], dim=1)
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)

        return x, y, mask

    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor,attn_mask: Tensor) -> JointMultimodalInterpolantResult:
        t_shaped_disc = t.view(-1,1)
        t_shaped_euc = t.view(-1,1,1)

        # Add noise to euclidean data
        full_xt = self.alpha(t_shaped_euc) * x1 + (1 - self.alpha(t_shaped_euc)) * torch.randn_like(x1)
        full_xt = torch.where(attn_mask.unsqueeze(-1), full_xt, 0.)
        full_xt[:,0] = 0.  # Don't corrupt beginning of the sequence

        # Masking data
        # Deletion and masking times are independent for every position, thats why we pass y1
        len_1 = torch.sum(y1 != self.pad_token, dim=-1)
        len_t = self.get_len_time_t(len_1, t)
        masking_time = self.get_masking_time_t(t)

        # Discrete data
        mask_positions = (t_shaped_disc <= masking_time) & attn_mask
        mask_positions[:, 0] = False  # Keep BOS visible; sampling keeps BOS fixed.
        yt = torch.where(mask_positions, self.mask_token, y1) # Change to mask id

        # Euclidean data
        full_xt = torch.where(mask_positions.unsqueeze(-1), 0., full_xt) # Change to mask id

        # Set up attention mask of deleted data
        deleted_positions = (torch.arange(y1.shape[1], device=y1.device) >= len_t)
        new_mask = attn_mask & ~deleted_positions
        new_mask[:,0] = True # Don't delete the start of the sequence
        yt = torch.where(new_mask, yt, self.pad_token)
        full_xt = torch.where(new_mask.unsqueeze(-1), full_xt, 0.)

        return JointMultimodalInterpolantResult(
            pad_token=self.pad_token,
            xt=full_xt, yt=yt, mask_t=new_mask, t=t, x1=x1, y1=y1, 
        )

    def jump_kernel_elbo(self, x, y, eps=1e-6):
        # x_safe: true length
        # y_safe: predicted length
        x_safe = torch.clamp(x, min=eps)
        y_safe = torch.clamp(y, min=eps)

        return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
 
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps)
    
    def compute_loss(self, model, batch):
        _x1 = batch["x"]
        _y1 = batch["y"]
        _mask_1 = batch["mask"]
        x1, y1, mask_1 = self.pad_sequence(_x1, _y1, _mask_1)
        t = self.sample_time(_x1.shape[0], _x1.device)
        interpolant_sample = self.sample_interpolant(t, x1, y1, mask_1)

        prediction: MultimodalModelPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            symbols_mask=interpolant_sample.mask_t,
            pos_mask=interpolant_sample.mask_t,
            symbols_time=t,
            pos_time=t
        )
        mask_t_shaped = interpolant_sample.mask_t.unsqueeze(-1)

        masked_positions = (interpolant_sample.yt == self.mask_token)
        
        # Euclidean loss
        # We must only compute the loss for positions that are:
        # not deleted: mask_t_shaped
        # not masked: masked_positions.unsqueeze(-1)
        dsm_loss = (interpolant_sample.x1 - prediction.clean_data)**2 * mask_t_shaped
        dsm_loss[:,0] = 0. # Don't take loss at the start of the sequence
        dsm_loss = dsm_loss.sum(dim=-1)[~masked_positions]
        dsm_loss = dsm_loss.mean() / x1.shape[-1]
        # Insertion loss: gaps = insertions needed (positive when we've deleted in forward)
        insertion_rate = prediction.insertion_rate
        gaps = interpolant_sample.y1_length - interpolant_sample.yt_length
        insertion_loss = self.jump_kernel_elbo(gaps.unsqueeze(-1).clamp(min=1e-6), insertion_rate)
        insertion_loss = insertion_loss.sum() / (y1.shape[0] * y1.shape[1]) # This is not the best scaling factor
        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y1[masked_positions]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        # Predicted euclidean loss
        # Gather along V dimension: clean_data_unmasking is [B, L, V, D], y1 is [B, L] with vocab indices
        y1_indices = interpolant_sample.y1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x1.shape[-1])  # [B, L] -> [B, L, 1, D]
        predicted_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=y1_indices).squeeze(2)  # [B, L, 1, D] -> [B, L, D]
        euclidean_loss = (predicted_cond_y1 - interpolant_sample.x1)**2
        euclidean_loss = euclidean_loss.sum(dim=-1)[masked_positions]
        euclidean_loss = euclidean_loss.mean() / x1.shape[-1]


        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
            "euclidean_unmasking_loss": euclidean_loss,
            "insertion_loss": insertion_loss,
        }
    
    def get_drift(self, prediction: MultimodalModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        return (clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1))
    
    def get_insertion_rate(self, prediction: MultimodalModelPrediction, t: Tensor) -> Tensor:
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1) * prediction.insertion_rate
        return rate
    
    def get_unmasking_rate(self, prediction: MultimodalModelPrediction, t: Tensor) -> Tensor:
        # Subtracting 4 because we are not considering the start of sequence token, end of sequence token, the mask token and the pad token
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1, 1) * prediction.label_logits[:, :, :self.vocab_size-4].softmax(dim=-1)
        return rate
    
    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tensor:
        xt = torch.zeros((batch_size, max_length, self.euclidean_dim), device=device)
        yt = torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.pad_token
        yt[:,0] = self.bos_token
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_t[:, 0] = True # Only the start is active
        return xt, yt, mask_t

    def update_xt_yt(
        self,
        prediction: MultimodalModelPrediction,
        xt: Tensor,
        yt: Tensor,
        mask_t: Tensor,
        t: Tensor,
        dt: float,
        is_last_step: bool,
    ) -> tuple[Tensor, Tensor]:
        # Denoise
        masked_positions = (yt == self.mask_token)
        drift = self.get_drift(prediction, xt, t)
        xt = xt + (drift * dt)
        xt = torch.where(mask_t.unsqueeze(-1), xt, 0.)
        xt[:,0] = 0.  # Don't corrupt beginning of the sequence
        xt = torch.where(masked_positions.unsqueeze(-1), 0., xt)

        # Unmasking
        unmasking_rate = self.get_unmasking_rate(prediction, t)
        unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate * dt).sample()
        num_jumps = unmasking_nums.sum(dim=-1)
        if not is_last_step:
            change_pos = (num_jumps == 1) & (mask_t) & (masked_positions)
            unmasking_nums = unmasking_nums * change_pos.unsqueeze(-1)
            new_sample = unmasking_nums.argmax(dim=-1)
        else:
            change_pos = masked_positions
            new_sample = unmasking_rate.argmax(dim=-1)

        indices = new_sample.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, xt.shape[-1])  # [B, L] -> [B, L, 1, D]
        mean_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=indices).squeeze(2)

        yt = torch.where(change_pos, new_sample, yt)
        alpha_t = self.alpha(t).view(-1, 1, 1)
        new_xt = alpha_t * mean_cond_y1 + (1 - alpha_t) * torch.randn_like(xt)
        xt = torch.where(change_pos.unsqueeze(-1), new_xt, xt)

        return xt, yt
    
    def perform_insertions(
        self,
        prediction: MultimodalModelPrediction,
        xt: Tensor,
        yt: Tensor,
        mask_t: Tensor,
        t: Tensor,
        dt: float,
        max_length: int,
        is_last_step: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        insertion_rate = self.get_insertion_rate(prediction, t)
        ext = torch.distributions.poisson.Poisson(insertion_rate * dt).sample()

        if is_last_step:
            return xt, yt, mask_t

        batch_size, seq_len = xt.shape[:2]
        device = xt.device
        for j in range(batch_size):
            # Build sequence j with sampled insertions in autoregressive order:
            # for each position k, emit token k then insert mask tokens after it (before k+1).
            new_xt_j = []
            new_yt_j = []
            for k in range(seq_len):
                if not mask_t[j, k]:
                    break
                new_xt_j.append(xt[j, k:k+1, :].clone())
                new_yt_j.append(yt[j, k].item())
                if ext[j, k] > 0:
                    n_insert = int(ext[j, k].item())
                    for _ in range(n_insert):
                        new_sample = torch.zeros_like(xt[j, :1, :], device=device)
                        new_xt_j.append(new_sample)
                        new_yt_j.append(self.mask_token)

            # Ensure we don't exceed the tensor size
            new_len = min(len(new_xt_j), max_length)
            new_xt_tensor = torch.cat(new_xt_j[:new_len], dim=0)
            new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
            xt[j, :new_len, :] = new_xt_tensor
            yt[j, :new_len] = new_yt_tensor
            xt[j, new_len:, :] = 0.0
            yt[j, new_len:] = self.pad_token
            mask_t[j, :new_len] = True
            mask_t[j, new_len:] = False

        return xt, yt, mask_t
    
    @torch.no_grad()
    def sampling(
        self,
        model: torch.nn.Module,
        steps: int,
        batch_size: int,
        max_length: int,
        device: torch.device,
        return_trace: bool = False,
        sampler: str = 'split',
    ) -> SamplingResult:
        if sampler == 'staggered':
            return self.staggered_sampler(model, steps, batch_size, max_length, device, return_trace)

        
        max_length = max_length + 1 # Plus one for the BOS token 
        # 1) Initialize all‑pad sequence and trace
        xt, yt, mask_t = self.get_prior_distribution(batch_size, max_length, device)
        
        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
            ))
        torch.set_printoptions(precision=2, sci_mode=False)
        for i in tqdm(range(steps), leave=False):
            t0 = t
            t_mid = t0 + dt / 2
            t1 = t0 + dt

            is_last_step = (i == steps - 1)

            prediction: MultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t0,
                pos_time=t0
            )
            if sampler == 'euler':
                step_size = dt
            else:
                step_size = dt if is_last_step else dt/2
            xt, yt = self.update_xt_yt(
                prediction=prediction,
                xt=xt,
                yt=yt,
                mask_t=mask_t,
                t=t0,
                dt=step_size,
                is_last_step=is_last_step,
            )
            if sampler == 'euler':
                xt, yt, mask_t = self.perform_insertions(
                    prediction=prediction,
                    xt=xt,
                    yt=yt,
                    mask_t=mask_t,
                    t=t0,
                    dt=dt,
                    max_length=max_length,
                    is_last_step=is_last_step,
                )
            elif not is_last_step:
                prediction_2: MultimodalModelPrediction = model(
                    cat_tokens=yt,
                    euclidean_tokens=xt,
                    symbols_mask=mask_t,
                    pos_mask=mask_t,
                    symbols_time=t_mid,
                    pos_time=t_mid
                )
                xt, yt, mask_t = self.perform_insertions(
                    prediction=prediction_2,
                    xt=xt,
                    yt=yt,
                    mask_t=mask_t,
                    t=t_mid,
                    dt=dt,
                    max_length=max_length,
                    is_last_step=is_last_step,
                )
                prediction_3: MultimodalModelPrediction = model(
                    cat_tokens=yt,
                    euclidean_tokens=xt,
                    symbols_mask=mask_t,
                    pos_mask=mask_t,
                    symbols_time=t1,
                    pos_time=t1
                )
                xt, yt = self.update_xt_yt(
                    prediction=prediction_3,
                    xt=xt,
                    yt=yt,
                    mask_t=mask_t,
                    t=t1,
                    dt=dt/2,
                    is_last_step=is_last_step,
                )
            t = t1

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
                ))

        return SamplingResult(
            xt=xt, yt=yt, mask_t=mask_t, trajectory=trajectory
        )

    @torch.no_grad()
    def staggered_sampler(
        self,
        model: torch.nn.Module,
        steps: int,
        batch_size: int,
        max_length: int,
        device: torch.device,
        return_trace: bool = False,
    ) -> SamplingResult:
        max_length = max_length + 1 # Plus one for the BOS token 
        # 1) Initialize all‑pad sequence and trace
        xt, yt, mask_t = self.get_prior_distribution(batch_size, max_length, device)
        
        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
            ))
        torch.set_printoptions(precision=2, sci_mode=False)

        t1 = t 
        t2 = t + dt/2
        prediction_0: MultimodalModelPrediction = model(
            cat_tokens=yt,
            euclidean_tokens=xt,
            symbols_mask=mask_t,
            pos_mask=mask_t,
            symbols_time=t1,
            pos_time=t1
        )
        xt, yt = self.update_xt_yt(
            prediction=prediction_0,
            xt=xt,
            yt=yt,
            mask_t=mask_t,
            t=t1,
            dt=dt/2,
            is_last_step=False,
        )


        for i in tqdm(range(steps), leave=False):
            is_last_step = (i == steps - 1)
            # is_last_step = False # Perhaps this is not needed now
            # print(f'Staggered sampler t1: {t1[0].item()}, t2: {t2[0].item()}')

            prediction: MultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t2,
                pos_time=t2
            )
            xt, yt, mask_t = self.perform_insertions(
                prediction=prediction,
                xt=xt,
                yt=yt,
                mask_t=mask_t,
                t=t2,
                dt=dt,
                max_length=max_length,
                is_last_step=is_last_step,
            )
            prediction_2: MultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t1,
                pos_time=t1
            )
            step_size = dt/2 if is_last_step else dt
            xt, yt = self.update_xt_yt(
                prediction=prediction_2,
                xt=xt,
                yt=yt,
                mask_t=mask_t,
                t=t1,
                dt=step_size,
                is_last_step=is_last_step,
            )
            t1 = t1 + step_size
            t2 = t2 + dt

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t1
                ))

        return SamplingResult(
            xt=xt, yt=yt, mask_t=mask_t, trajectory=trajectory
        )
def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")