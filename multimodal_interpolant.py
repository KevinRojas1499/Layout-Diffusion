import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
import torch.nn.functional as F



@dataclass
class MultimodalModelPrediction:
    clean_data: Tensor
    insertion_logits: Tensor
    insertion_rate: Tensor
    label_logits: Tensor

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

    def __getitem__(self, index: int) -> SamplingTrajectoryResult:
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
    xt: Tensor  # Shape [Batch, Length]
    yt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    xt_original_order: Tensor # Shape [Batch, Length]
    mask_original_order: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    x0: Tensor
    x0_ordered: Tensor
    y0_ordered: Tensor
    yt_original_order: Tensor
    eos_bos_mask_t: Tensor

    @property
    def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
        x0_len = self.mask_t.sum(dim=-1)
        gaps = self.st.clone()

        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([gaps, pad_back], dim=1)  # Add a leading zero

        xt_length = self.mask_t.sum(dim=-1)
        gaps.scatter_(
            1, xt_length.unsqueeze(1) + 1, x0_len.unsqueeze(1)
        )  # Fill the last position with x1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= xt_length.unsqueeze(1)
        gaps[~mask] = 0

        return gaps, mask

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


class MultimodalInterpolant():
    def __init__(
        self,
        max_length: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        train_only_dsm : bool = False,
        vocab_size: int = 10000,
        mask_token: int = 10000,
        pad_token: int = 10001,
        bos_token: int = 10002,
        eos_token: int = 10003,
        euclidean_dim: int = 3,
    ):
        super().__init__()
        self.max_length = max_length
        self.euclidean_dim = euclidean_dim
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.train_only_dsm = train_only_dsm
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def beta(self, t):
        return 500 * (self.linear_start**.5 * (1-t) + t * self.linear_end**.5)**2

    def beta_int(self, t):
        dif = self.linear_end**.5 - self.linear_start**.5
        return 500 * ( (self.linear_start**.5 * (1-t) + t * self.linear_end**.5)**3 /(3 * dif) - self.linear_start**1.5/(3 * dif) )    
    
    def scale(self, t):
        big_beta = self.beta_int(t)
        return torch.exp(-big_beta)
    
    def sigma(self,t):
        big_beta = self.beta_int(t)
        return (1 - torch.exp(-2 * big_beta))**.5
    
    def deletion_time(self, t: Tensor, x0: Tensor, masking_time: Tensor) -> Tensor:
        deletion_time = masking_time + torch.rand_like(x0, dtype=torch.float32) * (1 - masking_time)

        return deletion_time

    def masking_time(self, t: Tensor, x0: Tensor) -> Tensor:
        masking_time = torch.rand_like(x0, dtype=torch.float32)
        return masking_time

    def pad_sequence(self, x: Tensor, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask_shaped = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.where(mask_shaped, x, torch.zeros_like(x)) # This is to represent the end of the sequence with zeros
        y = torch.where(mask, y, torch.full_like(y, self.pad_token))
        # Add at the start of the sequence
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        y = torch.cat([torch.full_like(y[:, :1], self.bos_token), y], dim=1)
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)

        # Add at the end of the sequence
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        y = torch.cat([y, torch.full_like(y[:, :1], self.pad_token)], dim=1)
        mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        eos_bos_mask = (mask[:, :-1] != mask[:, 1:])
        mask[:,1:] = mask[:, 1:] | eos_bos_mask
        eos_bos_mask = torch.cat([torch.ones_like(mask[:, :1]), eos_bos_mask], dim=1)
        y = torch.where(eos_bos_mask, torch.full_like(y, self.eos_token), y)
        y[:,0] = self.bos_token

        return x, y, mask, eos_bos_mask

    def sample_interpolant(self, t: Tensor, x0: Tensor, y0: Tensor,mask_0: Tensor, eos_bos_mask: Tensor) -> JointMultimodalInterpolantResult:
        t = t.view(-1,1,1)
        # Add noise to euclidean data
        full_xt = self.scale(t) * x0 + self.sigma(t) * torch.randn_like(x0)
        mask_shaped = mask_0.unsqueeze(-1).expand(-1, -1, full_xt.shape[-1])
        full_xt = full_xt * mask_shaped
        eos_bos_mask_shaped = eos_bos_mask.unsqueeze(-1).expand(-1, -1, full_xt.shape[-1])
        full_xt = torch.where(eos_bos_mask_shaped, torch.zeros_like(full_xt), full_xt)

        # Masking data
        masking_time = self.masking_time(t, y0) 
        t_shaped = t.view(-1,1).expand(-1, full_xt.shape[1])
        yt = torch.where(t_shaped >= masking_time, y0, self.mask_token) # Change to mask id
        deletion_time = self.deletion_time(t, y0, masking_time)
        new_mask = mask_0 & (t_shaped < deletion_time) & ((t_shaped < deletion_time) | (t_shaped >= masking_time))
        new_mask = new_mask | eos_bos_mask

        t_big = t.view(-1,1,1).expand(-1, full_xt.shape[1], full_xt.shape[2])
        masking_time_big = masking_time.unsqueeze(-1).expand(-1, -1, full_xt.shape[2])
        full_xt = torch.where(t_big >= masking_time_big, full_xt, 0.) # Change to mask id

        # Reorder the data according to the active positions
        st = self.get_st(new_mask)
        xt = self.get_active_positions(full_xt, st)
        yt = self.get_active_positions(yt, st)
        y0_reordered = self.get_active_positions(y0, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order
        x0_ordered = self.get_active_positions(x0, st)
        eos_bos_mask_ordered = self.get_active_positions(eos_bos_mask, st)
        st[:,0] = -1 # Small hack to make the cum sums work nicely
        return JointMultimodalInterpolantResult(
            xt=xt, yt=yt, st=st, mask_t=mask_t, t=t, x0=x0, 
            x0_ordered=x0_ordered, 
            xt_original_order=full_xt, 
            mask_original_order=new_mask, 
            y0_ordered=y0_reordered,
            yt_original_order=yt,
            eos_bos_mask_t=eos_bos_mask_ordered
        )

    # TODO: check if this formulation is correct
    def jump_kernel_elbo(self, x, y, eps=1e-6):
        # x_safe: true length
        # y_safe: predicted length
        x_safe = torch.clamp(x, min=eps)
        y_safe = torch.clamp(y, min=eps)

        return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
 
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps) + eps
    
    def get_st(self, mask_t: Tensor) -> Tensor:
        return mask_t.argsort(dim=1, descending=True, stable=True)
    
    def get_active_positions(self, xt: Tensor, st: Tensor) -> Tensor:
        if xt.dim() == 2:
            return torch.gather(xt, 1, st)
        else:
            # Expand st to match xt's shape along all dimensions except dimension 1
            # st: [B, L] -> [B, L, 1, 1, ...] to match xt: [B, L, D, ...]
            index_shape = list(st.shape) + [1] * (xt.dim() - 2)
            st_expanded = st.view(*index_shape).expand_as(xt)
            return torch.gather(xt, 1, st_expanded)
    

    def compute_loss(self, model, batch):
        _x0 = batch["x"]
        _y0 = batch["y"]
        _mask_0 = batch["mask"]

        x0, y0, mask_0, eos_bos_mask = self.pad_sequence(_x0, _y0, _mask_0)
        t = self.sample_time(_x0.shape[0], _x0.device)
        interpolant_sample = self.sample_interpolant(t, x0, y0, mask_0, eos_bos_mask)
        eos_bos_mask_t = interpolant_sample.eos_bos_mask_t

        prediction: MultimodalModelPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            text_mask=interpolant_sample.mask_t,
            image_mask=interpolant_sample.mask_t,
            text_time_cond=t,
            image_time_cond=t
        )
        mask_t_shaped = interpolant_sample.mask_t.unsqueeze(-1)
        eos_bos_mask_shaped = eos_bos_mask_t.unsqueeze(-1)
        lengths = interpolant_sample.mask_t.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Euclidean loss
        dsm_loss = (interpolant_sample.x0_ordered - prediction.clean_data)**2 * mask_t_shaped * ~eos_bos_mask_shaped
        dsm_loss = dsm_loss.sum(dim=-1) / lengths
        dsm_loss = dsm_loss.mean()
        

        # Insertion loss
        gaps, gaps_mask = interpolant_sample.gaps_and_mask
        insertion_loss = self.jump_kernel_elbo(
            gaps[gaps_mask], prediction.insertion_rate[gaps_mask]
        )
        insertion_loss = insertion_loss.sum(dim=-1) / lengths.sum()
        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        batch_size, seq_len, num_classes = prediction.label_logits.shape
        logits_flat = prediction.label_logits.view(-1, num_classes)
        targets_flat = interpolant_sample.y0_ordered.view(-1).long()
        tokens_loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        # Reshape back to [batch, seq_len]
        tokens_loss = tokens_loss_flat.view(batch_size, seq_len)
        tokens_loss = tokens_loss * interpolant_sample.mask_t # Only consider the valid tokens
        tokens_loss = tokens_loss.sum(dim=-1) / lengths
        tokens_loss = tokens_loss.mean()

        return {
            "dsm_loss": dsm_loss,
            "tokens_loss": tokens_loss,
            "insertion_loss": insertion_loss,
        }
    
    def get_score(self, prediction: MultimodalModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        # clean_data = torch.arange(clean_data.shape[1], device=clean_data.device).repeat(clean_data.shape[0], 1)
        # Isolating score effect
        return - (xt - clean_data * self.scale(t).view(-1, 1, 1)) / self.sigma(t).view(-1, 1, 1)**2
    
    def get_insertion_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        lambda_t = self.beta(t).view(-1, 1)
        predicted_rate = prediction.insertion_rate

        # Subtracting 2 because we are not considering the start and end of sequence tokens
        rate = lambda_t * predicted_rate

        return rate
    
    def get_unmasking_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        lambda_t = self.beta(t).view(-1, 1)
        big_beta = self.beta_int(t).view(-1, 1)
        return 1/(torch.exp(big_beta) - 1) * lambda_t
    
    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tensor:
        xt = torch.zeros((batch_size, max_length + 2, self.euclidean_dim), device=device)
        yt = torch.cat([
            torch.ones((batch_size, 1), device=device, dtype=torch.long) * self.bos_token, 
            torch.ones((batch_size, 1), device=device, dtype=torch.long) * self.eos_token], dim=1)
        yt = torch.cat([yt, torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.pad_token], dim=1)
        mask_t = torch.zeros((batch_size, max_length + 2), dtype=torch.bool, device=device)
        mask_t[:, :2] = True
        eos_bos_mask = (yt == self.eos_token) | (yt == self.bos_token)
        return xt, yt, mask_t, eos_bos_mask
    
    @torch.no_grad()
    def euclidean_sampling(
        self,
        model: torch.nn.Module,
        steps: int,
        batch_size: int,
        max_length: int,
        device: torch.device,
        return_trace: bool = False,
    ) -> SamplingTrajectoryResult:
        # 1) Initialize all‑pad sequence and trace
        xt, yt, mask_t, eos_bos_mask = self.get_prior_distribution(batch_size, max_length, device)
        
        dt = 1.0 / steps
        t = torch.ones(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
            ))
        for i in range(steps):
            prediction: MultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                text_mask=mask_t,
                image_mask=mask_t,
                text_time_cond=t,
                image_time_cond=t
            )
            # Denoise
            score = self.get_score(prediction, xt, t)
            beta = self.beta(t).view(-1, 1, 1)
            beta_int = self.beta_int(t).view(-1, 1, 1)
            mask_shaped = mask_t.unsqueeze(-1).expand(-1, -1, xt.shape[-1])
            eos_bos_mask_shaped = eos_bos_mask.unsqueeze(-1).expand(-1, -1, xt.shape[-1])
            xt = xt + (beta * (xt + score) * dt) * mask_shaped * ~eos_bos_mask_shaped

            unmasking_rate = self.get_unmasking_rate(prediction, mask_t, t)
            # Expand unmasking_rate to match sequence length: [batch, 1] -> [batch, seq_len]
            seq_len = mask_t.shape[1]
            unmasking_rate = unmasking_rate.expand(-1, seq_len)
            unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate * dt).sample()

            # Unmasking
            dist = prediction.label_logits[:, :, :self.vocab_size]
            new_sample = sample_categorical(dist.softmax(dim=-1), method="hard")  # [batch, seq_len]

            change_pos = (unmasking_nums > 0) & (mask_t == True) & (yt == self.mask_token)
            new_xt = torch.exp(-beta_int) * prediction.clean_data + torch.sqrt(1 - torch.exp(-2 * beta_int)) * torch.rand_like(xt)
            xt[change_pos] = new_xt[change_pos]

            yt[change_pos] = new_sample[change_pos]


            # Perform insertions
            insertion_rate = self.get_insertion_rate(prediction, mask_t, t)
            ext = torch.bernoulli((insertion_rate * dt).clamp(0.0, 1.0)).long()  # (B, L) where L is seq_len after padding

            seq_len = xt.shape[1]  # After padding, this is max_length + 2
            if i != steps - 1:
                for j in range(batch_size):
                    # Add dimensions
                    new_xt_j = []
                    new_yt_j = []
                    for k in range(seq_len):
                        if not mask_t[j, k]:
                            break
                        # Insert new token before position k if ext[j, k-1] == 1 (for k > 0)
                        # insertion_rate[:, k-1] represents the rate for inserting before position k
                        if k > 0 and ext[j, k-1] == 1:
                            new_sample = torch.zeros_like(xt[0, :1, :])
                            new_xt_j.append(new_sample)
                            new_yt_j.append(self.mask_token)
                        # Always append the current token at position k
                        new_xt_j.append(xt[j, k:k+1, :])
                        new_yt_j.append(yt[j, k].item())
                    
                    # Ensure we don't exceed the tensor size
                    new_len = min(len(new_xt_j), max_length) 
                    new_xt_tensor = torch.cat(new_xt_j[:new_len], dim=0)
                    new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
                    xt[j, :new_len, :] = new_xt_tensor
                    yt[j, :new_len] = new_yt_tensor
                    xt[j, 0] = 0.
                    xt[j, new_len-1] = 0.
                    yt[j, 0] = self.bos_token
                    yt[j, new_len-1] = self.eos_token
                    mask_t[j, :new_len] = True
                    mask_t[j, new_len:] = False
                    eos_bos_mask[j,:] = False
                    eos_bos_mask[j, 0] = True
                    eos_bos_mask[j, new_len-1] = True

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
                ))
            t = t - dt

        return SamplingResult(
            xt=xt, yt=yt, mask_t=mask_t, trajectory=trajectory
        )

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")