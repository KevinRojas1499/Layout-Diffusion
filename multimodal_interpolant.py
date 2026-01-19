import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
import torch.nn.functional as F



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
        gaps = self.st.clone()


        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)
        idx = torch.arange(self.st.shape[1], device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx < self.mask_t.sum(dim=-1).unsqueeze(1)
        gaps = torch.cat([torch.zeros_like(gaps[:, :1]), gaps], dim=1)
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
        delta : float = .9999,
        train_only_dsm : bool = False,
        non_special_tokens: int = 5,
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
        self.delta = delta
        self.train_only_dsm = train_only_dsm
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.non_special_tokens = non_special_tokens

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
    
    def alpha(self, t):
        return self.delta/(1-self.delta * t)
    
    def gamma(self, t):
        return self.delta/(1-self.delta * t)
    
    def alpha_bar(self, t):
        return -torch.log1p(-self.delta * t)
    
    def prob_mask(self, t):
        # return torch.exp(-self.alpha_bar(t))
        return -(1 - self.delta * t) * torch.log1p(-self.delta * t)
    
    def prob_empty(self, t):
        return self.delta * t + (1-self.delta * t) * torch.log1p(-self.delta * t)
    
    def get_w(self, t):
        t = t.clamp(min=1e-5, max=1.0 - 1e-5)  # Clamp t to avoid edge cases
        prob_empty = self.prob_empty(t).clamp(min=1e-8, max=1.0 - 1e-8)
        prob_mask = self.prob_mask(t).clamp(min=1e-8)  # Ensure prob_mask is positive
        
        # Compute in log-space
        log_w = torch.log(prob_mask) + torch.log1p(-prob_empty) - torch.log(prob_empty)
        return torch.exp(log_w)

    def get_masking_and_deletion_time(self, y0):
        u1 = torch.rand_like(y0, dtype=torch.float32)
        masking_time = (1-u1) / self.delta
        u2 = torch.rand_like(y0, dtype=torch.float32)
        # deletion_time = (1 - u2 * (1-self.delta * masking_time)) / self.delta
        deletion_time = (1 - u1 * u2) / self.delta
        return masking_time, deletion_time

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

    def sample_interpolant(self, t: Tensor, x0: Tensor, y0: Tensor,attn_mask: Tensor, eos_bos_mask: Tensor) -> JointMultimodalInterpolantResult:
        t_shaped_disc = t.view(-1,1).expand(-1, y0.shape[1])
        t_shaped_euc = t.view(-1,1,1).expand(-1, x0.shape[1], x0.shape[2])

        # Add noise to euclidean data
        full_xt = self.scale(t_shaped_euc) * x0 + self.sigma(t_shaped_euc) * torch.randn_like(x0)
        full_xt = full_xt * attn_mask.unsqueeze(-1)
        eos_bos_mask_shaped = eos_bos_mask.unsqueeze(-1).expand(-1, -1, full_xt.shape[-1])
        full_xt = torch.where(eos_bos_mask_shaped, torch.zeros_like(full_xt), full_xt)

        # Masking data
        # Deletion and masking times are independent for every position, thats why we pass y0
        masking_time, deletion_time = self.get_masking_and_deletion_time(y0)

        # Discrete data
        mask_positions = (t_shaped_disc >= masking_time) & attn_mask
        yt = torch.where(mask_positions, self.mask_token, y0) # Change to mask id

        # Euclidean data
        full_xt = torch.where(mask_positions.unsqueeze(-1), 0., full_xt) # Change to mask id

        # Set up attention mask of deleted data
        new_mask = attn_mask & (t_shaped_disc < deletion_time)
        new_mask = new_mask | eos_bos_mask

        # Reorder the data according to the active positions
        st = self.get_st(new_mask)
        xt = self.get_active_positions(full_xt, st)
        yt = self.get_active_positions(yt, st)
        y0_reordered = self.get_active_positions(y0, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order
        x0_ordered = self.get_active_positions(x0, st)
        eos_bos_mask_ordered = self.get_active_positions(eos_bos_mask, st)

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
            symbols_mask=interpolant_sample.mask_t,
            pos_mask=interpolant_sample.mask_t,
            symbols_time=t,
            pos_time=t
        )
        mask_t_shaped = interpolant_sample.mask_t.unsqueeze(-1)
        eos_bos_mask_shaped = eos_bos_mask_t.unsqueeze(-1)
        lengths = interpolant_sample.mask_t.sum(dim=-1, keepdim=True).clamp(min=1)

        masked_positions = (interpolant_sample.yt == self.mask_token)
        
        # Euclidean loss
        # We must only compute the loss for positions that are:
        # not deleted: mask_t_shaped
        # not masked: masked_positions.unsqueeze(-1)
        # Not padding: ~eos_bos_mask_shaped
        dsm_loss = (interpolant_sample.x0_ordered - prediction.clean_data)**2 * mask_t_shaped * ~eos_bos_mask_shaped
        dsm_loss = dsm_loss.sum(dim=-1)[~masked_positions]
        dsm_loss = dsm_loss.mean() / x0.shape[-1]
        # Insertion loss
        insertion_rate = prediction.insertion_rate
        gaps, gaps_mask = interpolant_sample.gaps_and_mask
        insertion_loss = self.jump_kernel_elbo(gaps[gaps_mask], insertion_rate[gaps_mask])
        insertion_loss = insertion_loss.sum() / (y0.shape[0] * self.max_length)
        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y0_ordered[masked_positions]
        tokens_loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        tokens_loss = tokens_loss_flat.mean()

        # Predicted euclidean loss
        # Gather along V dimension: clean_data_unmasking is [B, L, V, D], y0 is [B, L] with vocab indices
        y0_indices = interpolant_sample.y0_ordered.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x0.shape[-1])  # [B, L] -> [B, L, 1, D]
        predicted_cond_y0 = prediction.clean_data_unmasking.gather(dim=2, index=y0_indices).squeeze(2)  # [B, L, 1, D] -> [B, L, D]
        euclidean_loss = (predicted_cond_y0 - interpolant_sample.x0_ordered)**2
        euclidean_loss = euclidean_loss.sum(dim=-1)[masked_positions]
        euclidean_loss = euclidean_loss.mean() / x0.shape[-1]


        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
            "euclidean_unmasking_loss": euclidean_loss,
            "insertion_loss": insertion_loss,
        }
    
    def get_score(self, prediction: MultimodalModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        # clean_data = torch.arange(clean_data.shape[1], device=clean_data.device).repeat(clean_data.shape[0], 1)
        return - (xt - clean_data * self.scale(t).view(-1, 1, 1)) / self.sigma(t).view(-1, 1, 1)**2
    
    def get_insertion_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        gamma_t = self.gamma(t).view(-1, 1)
        rate = prediction.insertion_rate

        return gamma_t * rate
    
    def get_unmasking_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        alpha_t = self.alpha(t).view(-1, 1)
        alpha_bar = self.alpha_bar(t).view(-1, 1)
        return 1/(torch.exp(alpha_bar) - 1) * alpha_t
    
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
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t,
                pos_time=t
            )
            # Denoise
            # This has a mistake, it currently denoises positions that are masked
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
            indices = new_sample.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, xt.shape[-1])  # [B, L] -> [B, L, 1, D]
            mean_cond_y0 = prediction.clean_data_unmasking.gather(dim=2, index=indices).squeeze(2)

            change_pos = (unmasking_nums > 0) & (mask_t == True) & (yt == self.mask_token)
            new_xt = torch.exp(-beta_int) * mean_cond_y0 + torch.sqrt(1 - torch.exp(-2 * beta_int)) * torch.randn_like(xt)
            xt[change_pos] = new_xt[change_pos]
            yt[change_pos] = new_sample[change_pos]


            # Perform insertions
            insertion_rate = self.get_insertion_rate(prediction, mask_t, t)
            ext = torch.distributions.poisson.Poisson(insertion_rate * dt).sample().floor().long()

            seq_len = xt.shape[1]  # After padding, this is max_length + 2
            if i != steps - 1:
                for j in range(batch_size):
                    # Add dimensions
                    new_xt_j = []
                    new_yt_j = []
                    for k in range(seq_len):
                        if not mask_t[j, k]:
                            break
                        # Insert new token after position k if ext[j, k] == 1 (for k > 0)
                        if k > 0 and ext[j, k] > 0:
                            new_sample = torch.zeros_like(xt[0, :1, :])
                            # for _ in range(ext[j, k]):
                            # We should only insert one token at a time
                            new_xt_j.append(new_sample.clone())
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