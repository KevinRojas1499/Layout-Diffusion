import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class MultimodalModelPredictionBothVar:
    # This handles denoising
    clean_data: Tensor
    euc_insertion_rate: Tensor
    # This handles the unmasking
    label_logits: Tensor
    disc_insertion_rate: Tensor

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
    t: Tensor # Shape [Batch]
    # X stuff
    x1: Tensor
    x_mask_1: Tensor
    xt: Tensor  # Shape [Batch, Length, D] (D is the euclidean dimension)
    x_targets: Tensor # Shape [Batch, Length, D]
    x_st : Tensor # Shape [Batch, Length]
    x_mask_t: Tensor # Shape [Batch, Length]
    # Y stuff
    y1: Tensor
    y_mask_1: Tensor
    yt: Tensor  # Shape [Batch, Length]
    y_targets: Tensor # Shape [Batch, Length]
    y_st : Tensor # Shape [Batch, Length]
    y_mask_t: Tensor # Shape [Batch, Length]

    def get_st(self, mask_t: Tensor) -> Tensor:
        return mask_t.argsort(dim=1, descending=True, stable=True)

    def gaps_and_mask(self, mask_1: Tensor, mask_t: Tensor) -> tuple[Tensor, Tensor]:
        # Mask_t is the mask without reordeing, the returned mask is a contigous array
        y1_len = mask_1.sum(dim=1)
        st = self.get_st(mask_t)
        gaps = st.clone()

        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([gaps, pad_back], dim=1)  # Add a leading zero

        yt_len = mask_t.sum(dim=1)
        gaps.scatter_(
            1, yt_len.unsqueeze(1), y1_len.unsqueeze(1)
        )  # Fill the last position with y1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=mask_t.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= yt_len.unsqueeze(1)
        gaps[~mask] = 0

        return gaps, mask


class MultimodalInterpolantBoth():
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
    
    def get_unmasking_and_insertion(self, y1):
        u1 = torch.rand_like(y1, dtype=torch.float32)
        insertion_time = (1-u1)
        u2 = torch.rand_like(y1, dtype=torch.float32)
        umasking_time = (1 - u1 * u2) 
        return umasking_time, insertion_time

    def pad_sequence(self, x: Tensor, y: Tensor, mask_x: Tensor, mask_y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask_shaped_x = mask_x.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # This is actually doubled because the dataset already handles it
        x = torch.where(mask_shaped_x, x, torch.zeros_like(x)) 
        y = torch.where(mask_y, y, torch.full_like(y, self.pad_token))

        # Add at the start of the sequence
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        y = torch.cat([torch.full_like(y[:, :1], self.bos_token), y], dim=1)
        mask_x = torch.cat([torch.ones_like(mask_x[:, :1]), mask_x], dim=1)
        mask_y = torch.cat([torch.ones_like(mask_y[:, :1]), mask_y], dim=1)

        return x, y, mask_x, mask_y

    def sample_interpolant(self, t: Tensor, x1: Tensor, y1: Tensor,attn_mask_x: Tensor, attn_mask_y: Tensor) -> JointMultimodalInterpolantResult:
        t_shaped_disc = t.view(-1,1)
        t_shaped_euc = t.view(-1,1,1)

        # Add noise to euclidean data
        full_xt = self.alpha(t_shaped_euc) * x1 + (1 - self.alpha(t_shaped_euc)) * torch.randn_like(x1)
        full_xt = torch.where(attn_mask_x.unsqueeze(-1), full_xt, 0.)
        full_xt[:,0] = 0. # Don't corrupt begginning of the sequence

        # Masking euclidean data
        unmasking_time, insertion_time = self.get_unmasking_and_insertion(x1[:,:,0])
        mask_positions_euc = (t_shaped_euc.squeeze(-1) <= unmasking_time) & attn_mask_x
        full_xt = torch.where(mask_positions_euc.unsqueeze(-1), 0., full_xt) # Change to mask id

        # Set up attention mask of deleted data
        new_mask_euc = attn_mask_x & (t_shaped_euc.squeeze(-1) >= insertion_time)
        new_mask_euc[:,0] = True # Don't delete the start of the sequence


        # Masking discrete data
        # Deletion and masking times are independent for every position, thats why we pass y1
        unmasking_time_disc, deletion_time_disc = self.get_unmasking_and_insertion(y1)

        # Discrete data
        mask_positions = (t_shaped_disc <= unmasking_time_disc) & attn_mask_y
        yt = torch.where(mask_positions, self.mask_token, y1) # Change to mask id

        # Set up attention mask of deleted data
        new_mask = attn_mask_y & (t_shaped_disc >= deletion_time_disc)
        new_mask[:,0] = True # Don't delete the start of the sequence
        yt = torch.where(new_mask, yt, self.pad_token)

        # Reorder the data according to the active positions
        y_st = self.get_st(new_mask)
        x_st = self.get_st(new_mask_euc)
        xt = self.get_active_positions(full_xt, x_st)
        yt = self.get_active_positions(yt, y_st)
        mask_t = self.get_active_positions(new_mask, y_st) # This will reorder the mask according to the new order
        mask_t_euc = self.get_active_positions(new_mask_euc, x_st) # This will reorder the mask according to the new order
        y1_ordered = self.get_active_positions(y1, y_st)
        x1_ordered = self.get_active_positions(x1, x_st)

        return JointMultimodalInterpolantResult(
            pad_token=self.pad_token,
            t=t,
            x1=x1,
            x_mask_1=attn_mask_x,
            xt=xt,
            x_targets=x1_ordered,
            x_st=x_st,
            x_mask_t=mask_t_euc,
            y1=y1,
            y_mask_1=attn_mask_y,
            yt=yt,
            y_targets=y1_ordered,
            y_st=y_st,
            y_mask_t=mask_t,
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
        _mask_x = batch["mask_x"]
        _mask_y = batch["mask_y"]
        x1, y1, mask_x, mask_y = self.pad_sequence(_x1, _y1, _mask_x, _mask_y)
        t = self.sample_time(_x1.shape[0], _x1.device)
        interpolant_sample = self.sample_interpolant(t, x1, y1, mask_x, mask_y)

        prediction: MultimodalModelPredictionBothVar = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            symbols_mask=interpolant_sample.y_mask_t,
            pos_mask=interpolant_sample.x_mask_t,
            symbols_time=t,
            pos_time=t
        )
        x_mask_t_shaped = interpolant_sample.x_mask_t.unsqueeze(-1)

        masked_positions = (interpolant_sample.yt == self.mask_token)
        
        # Euclidean loss
        # We must only compute the loss for positions that are:
        # not deleted: mask_t_shaped
        # we compute the loss even for masked positions, to model the jumps like that
        dsm_loss = (interpolant_sample.x_targets - prediction.clean_data)**2 * x_mask_t_shaped
        dsm_loss[:,0] = 0. # Don't take loss at the start of the sequence
        dsm_loss = dsm_loss.sum(dim=-1)[~masked_positions]
        dsm_loss = dsm_loss.mean() / x1.shape[-1]
        # Insertion loss
        gaps_x, gaps_mask_x = interpolant_sample.gaps_and_mask(interpolant_sample.x_mask_1, interpolant_sample.x_mask_t)
        insertion_loss_x = self.jump_kernel_elbo(gaps_x[gaps_mask_x], prediction.euc_insertion_rate[gaps_mask_x])
        insertion_loss_x = insertion_loss_x.sum() / (x1.shape[0] * x1.shape[1])
        gaps_y, gaps_mask_y = interpolant_sample.gaps_and_mask(interpolant_sample.y_mask_1, interpolant_sample.y_mask_t)
        insertion_loss_y = self.jump_kernel_elbo(gaps_y[gaps_mask_y], prediction.disc_insertion_rate[gaps_mask_y])
        insertion_loss_y = insertion_loss_y.sum() / (y1.shape[0] * y1.shape[1])

        # Unmasking loss
        # Reshape for cross_entropy: [batch, seq_len, num_classes] -> [batch * seq_len, num_classes]
        # and [batch, seq_len] -> [batch * seq_len]
        logits_flat = prediction.label_logits[masked_positions]
        targets_flat = interpolant_sample.y_targets[masked_positions]
        tokens_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none").mean()

        return {
            "dsm_loss": dsm_loss,
            "discrete_unmasking_loss": tokens_loss,
            "euc_insertion_loss": insertion_loss_x,
            "disc_insertion_loss": insertion_loss_y,
        }
    
    def get_drift(self, prediction: MultimodalModelPredictionBothVar, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        return (clean_data - xt) / (1 - self.alpha(t).view(-1, 1, 1))
    
    def get_insertion_rate(self, prediction: MultimodalModelPredictionBothVar, t: Tensor) -> Tensor:
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate_euc = coeff.view(-1, 1) * prediction.euc_insertion_rate
        rate_disc = coeff.view(-1, 1) * prediction.disc_insertion_rate
        return rate_euc, rate_disc
    
    def get_unmasking_rate(self, prediction: MultimodalModelPredictionBothVar, t: Tensor) -> Tensor:
        # Subtracting 4 because we are not considering the start of sequence token, end of sequence token, the mask token and the pad token
        coeff = self.dalpha(t) / (1 - self.alpha(t))
        rate = coeff.view(-1, 1, 1) * prediction.label_logits[:, :, :self.vocab_size-4].softmax(dim=-1)
        return coeff, rate # This is the rate for cont, rate discrete
    
    def get_prior_distribution(self, batch_size: int, max_length: int, device: torch.device) -> Tensor:
        xt = torch.zeros((batch_size, max_length, self.euclidean_dim), device=device)
        yt = torch.ones((batch_size, max_length), device=device, dtype=torch.long) * self.pad_token
        yt[:,0] = self.bos_token
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        mask_t[:, 0] = True # Only the start is active
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
        for i in tqdm(range(steps), leave=False):
            prediction: MultimodalModelPredictionBothVar = model(
                cat_tokens=yt,
                euclidean_tokens=xt,
                symbols_mask=mask_t,
                pos_mask=mask_t,
                symbols_time=t,
                pos_time=t
            )
            # Denoise
            masked_positions_disc = (yt == self.mask_token)
            masked_positions_euc = (xt == 0.) # TODO: Keep better track of the mask
            drift = self.get_drift(prediction, xt, t)
            xt = xt + (drift * dt)
            xt = torch.where(mask_t.unsqueeze(-1), xt, 0.)
            xt[:,0] = 0. # Don't corrupt begginning of the sequence
            xt = torch.where(masked_positions_euc, 0., xt)

            # Unmasking
            unmasking_rate_euc, unmasking_rate_disc = self.get_unmasking_rate(prediction, t)
            unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate_euc * dt).sample()
            # Unmask continuous data
            new_xt = self.alpha(t).view(-1, 1, 1) * prediction.clean_data + (1 - self.alpha(t).view(-1, 1, 1)) * torch.randn_like(xt)

            if i != steps - 1:
                change_pos = (unmasking_nums >= 1) & (mask_t) & (masked_positions_euc)
                unmasking_nums = unmasking_nums * change_pos.unsqueeze(-1)
            else:
                change_pos = masked_positions_euc
            xt = torch.where(change_pos, new_xt, xt)

            # Unmask discrete data
            unmasking_nums_disc = torch.distributions.poisson.Poisson(unmasking_rate_disc * dt).sample()
            num_jumps = unmasking_nums_disc.sum(dim=-1)
            if i != steps - 1:
                change_pos = (num_jumps == 1) & (mask_t) & (masked_positions_disc)
                unmasking_nums_disc = unmasking_nums_disc * change_pos.unsqueeze(-1)
                new_sample = unmasking_nums_disc.argmax(dim=-1)
            else:
                change_pos = masked_positions_disc
                new_sample = unmasking_rate_disc.argmax(dim=-1)
            yt = torch.where(change_pos, new_sample, yt)

            # Perform insertions
            insertion_rate_euc, insertion_rate_disc = self.get_insertion_rate(prediction, t)
            ext_euc = torch.distributions.poisson.Poisson(insertion_rate_euc * dt).sample()
            ext_disc = torch.distributions.poisson.Poisson(insertion_rate_disc * dt).sample()

            seq_len = xt.shape[1]
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
                        if ext_euc[j, k] > 0:
                            for _ in range(int(ext_euc[j, k].item())):
                                new_sample = torch.zeros_like(xt[0, :1, :])
                                new_xt_j.append(new_sample.clone())
                        if ext_disc[j, k] > 0:
                            for _ in range(int(ext_disc[j, k].item())):
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