import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
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
    pad_token: int
    xt: Tensor  # Shape [Batch, Length]
    yt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    xt_original_order: Tensor # Shape [Batch, Length]
    mask_original_order: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    x1: Tensor
    y1: Tensor
    x1_ordered: Tensor
    y1_ordered: Tensor
    yt_original_order: Tensor

    @property
    def y1_length(self) -> Tensor:
        return (self.y1 != self.pad_token).sum(dim=1)
    
    @property
    def yt_length(self) -> Tensor:
        return (self.yt != self.pad_token).sum(dim=1)
    
    @property
    def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
        y1_len = self.y1_length
        gaps = self.st.clone()

        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([gaps, pad_back], dim=1)  # Add a leading zero

        gaps.scatter_(
            1, self.yt_length.unsqueeze(1), y1_len.unsqueeze(1)
        )  # Fill the last position with y1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= self.yt_length.unsqueeze(1)
        gaps[~mask] = 0

        return gaps, mask


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
    
    def get_masking_and_deletion_time(self, y1):
        u1 = torch.rand_like(y1, dtype=torch.float32)
        deletion_time = (1-u1)
        u2 = torch.rand_like(y1, dtype=torch.float32)
        masking_time = (1 - u1 * u2) 
        return masking_time, deletion_time

    def pad_sequence(self, x: Tensor, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        full_xt[:,0] = 0. # Don't corrupt begginning of the sequence

        # Masking data
        # Deletion and masking times are independent for every position, thats why we pass y1
        masking_time, deletion_time = self.get_masking_and_deletion_time(y1)

        # Discrete data
        mask_positions = (t_shaped_disc <= masking_time) & attn_mask
        yt = torch.where(mask_positions, self.mask_token, y1) # Change to mask id

        # Euclidean data
        full_xt = torch.where(mask_positions.unsqueeze(-1), 0., full_xt) # Change to mask id

        # Set up attention mask of deleted data
        new_mask = attn_mask & (t_shaped_disc >= deletion_time)
        new_mask[:,0] = True # Don't delete the start of the sequence
        yt = torch.where(new_mask, yt, self.pad_token)
        full_xt = torch.where(new_mask.unsqueeze(-1), full_xt, 0.)

        # Reorder the data according to the active positions
        st = self.get_st(new_mask)
        xt = self.get_active_positions(full_xt, st)
        yt = self.get_active_positions(yt, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order
        y1_ordered = self.get_active_positions(y1, st)
        x1_ordered = self.get_active_positions(x1, st)

        return JointMultimodalInterpolantResult(
            pad_token=self.pad_token,
            xt=xt, yt=yt, st=st, mask_t=mask_t, t=t, x1=x1, y1=y1, 
            x1_ordered=x1_ordered, 
            xt_original_order=full_xt, 
            mask_original_order=new_mask, 
            y1_ordered=y1_ordered,
            yt_original_order=yt,
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
        return torch.rand(batch_size, device=device) * (1 - eps)
    
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
        dsm_loss = (interpolant_sample.x1_ordered - prediction.clean_data)**2 * mask_t_shaped
        dsm_loss[:,0] = 0. # Don't take loss at the start of the sequence
        dsm_loss = dsm_loss.sum(dim=-1)[~masked_positions]
        dsm_loss = dsm_loss.mean() / x1.shape[-1]
        # Insertion loss
        insertion_rate = prediction.insertion_rate
        gaps, gaps_mask = interpolant_sample.gaps_and_mask
        insertion_loss = self.jump_kernel_elbo(gaps[gaps_mask], insertion_rate[gaps_mask])
        insertion_loss = insertion_loss.sum() / (y1.shape[0] * self.max_length)
        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y1_ordered[masked_positions]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        # Predicted euclidean loss
        # Gather along V dimension: clean_data_unmasking is [B, L, V, D], y1 is [B, L] with vocab indices
        y1_indices = interpolant_sample.y1_ordered.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x1.shape[-1])  # [B, L] -> [B, L, 1, D]
        predicted_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=y1_indices).squeeze(2)  # [B, L, 1, D] -> [B, L, D]
        euclidean_loss = (predicted_cond_y1 - interpolant_sample.x1_ordered)**2
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
        return - (xt - clean_data) / self.alpha(t).view(-1, 1, 1)
    
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
        xt, yt, mask_t = self.get_prior_distribution(batch_size, max_length, device)
        
        dt = 1.0 / steps
        t = torch.zeros(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
            ))
        torch.set_printoptions(precision=2, sci_mode=False)
        for i in tqdm(range(steps), leave=False, disable=True):
            prediction: MultimodalModelPrediction = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t,
                pos_time=t
            )
            # Denoise
            masked_positions = (yt == self.mask_token)
            drift = self.get_drift(prediction, xt, t)
            xt = xt + (drift * dt)
            xt = torch.where(mask_t.unsqueeze(-1), xt, 0.)
            xt[:,0] = 0. # Don't corrupt begginning of the sequence
            xt = torch.where(masked_positions.unsqueeze(-1), 0., xt)

            # Unmasking
            unmasking_rate = self.get_unmasking_rate(prediction, t)
            unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate * dt).sample()
            num_jumps = unmasking_nums.sum(dim=-1)
            if i != steps - 1:
                change_pos = (num_jumps == 1) & (mask_t) & (masked_positions)
                unmasking_nums = unmasking_nums * change_pos.unsqueeze(-1)
                new_sample = unmasking_nums.argmax(dim=-1)
            else:
                change_pos = masked_positions
                new_sample = unmasking_rate.argmax(dim=-1)
            
            indices = new_sample.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, xt.shape[-1])  # [B, L] -> [B, L, 1, D]
            mean_cond_y1 = prediction.clean_data_unmasking.gather(dim=2, index=indices).squeeze(2)

            yt = torch.where(change_pos, new_sample, yt)
            new_xt = self.alpha(t).view(-1, 1, 1) * mean_cond_y1 + (1 - self.alpha(t).view(-1, 1, 1)) * torch.randn_like(xt)
            xt = torch.where(change_pos.unsqueeze(-1), new_xt, xt)


            # Perform insertions
            insertion_rate = self.get_insertion_rate(prediction, t)
            ext = torch.distributions.poisson.Poisson(insertion_rate * dt).sample()

            seq_len = xt.shape[1]  # After padding, this is max_length + 2
            if i != steps - 1:
                for j in range(batch_size):
                    # Add dimensions
                    new_xt_j = []
                    new_yt_j = []
                    for k in range(seq_len):
                        if not mask_t[j, k]:
                            break
                        new_xt_j.append(xt[j, k:k+1, :])
                        new_yt_j.append(yt[j, k].item())
                        # Insert new token after position k if ext[j, k] > 0
                        if ext[j, k] > 0:
                            # Consider doing a for loop here
                            # print('------------- Position ', k, ' -------------')
                            for _ in range(int(ext[j, k].item())):
                                new_sample = torch.zeros_like(xt[0, :1, :])
                                new_xt_j.append(new_sample.clone())
                                new_yt_j.append(self.mask_token)
                    
                    # Ensure we don't exceed the tensor size
                    new_len = min(len(new_xt_j), max_length) 
                    new_xt_tensor = torch.cat(new_xt_j[:new_len], dim=0)
                    new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
                    xt[j, :new_len, :] = new_xt_tensor
                    yt[j, :new_len] = new_yt_tensor
                    xt[j, 0] = 0.
                    yt[j, 0] = self.bos_token
                    mask_t[j, :new_len] = True
                    mask_t[j, new_len:] = False
                else:
                    # Do something here 
                    pass

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt.clone(), yt=yt.clone(), mask_t=mask_t.clone(), t=t
                ))
            t = t + dt

        return SamplingResult(
            xt=xt, yt=yt, mask_t=mask_t, trajectory=trajectory
        )

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")