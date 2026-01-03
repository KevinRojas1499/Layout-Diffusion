import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator, Tuple
from utils.tokenizer import IntervalTokenizer
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




class MultimodalInterpolant():
    def __init__(
        self,
        max_length: int,
        interval_tokenizer: IntervalTokenizer,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        train_only_dsm : bool = False,
        vocab_size: int = 10000,
        mask_token: int = 10000,
        pad_token: int = 10001,
    ):
        super().__init__()
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.train_only_dsm = train_only_dsm
        self.tokenizer = interval_tokenizer
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token

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
        deletion_time = masking_time + torch.rand_like(x0) * (1 - masking_time)

        return deletion_time

    def masking_time(self, t: Tensor, x0: Tensor) -> Tensor:
        masking_time = torch.rand_like(x0)
        return masking_time

    def pad_sequence(self, x: Tensor, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.where(mask, x, torch.zeros_like(x)) # This is to represent the end of the sequence with zeros
        # Add at the start of the sequence
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        y = torch.cat([torch.full_like(y[:, :1], self.pad_token), y], dim=1)
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)

        # Add at the end of the sequence
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        y = torch.cat([y, torch.full_like(y[:, :1], self.pad_token)], dim=1)
        mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        eos_bos_mask = (mask[:, :-1] != mask[:, 1:])
        mask[:,1:] = mask[:, 1:] | eos_bos_mask
        eos_bos_mask = torch.cat([torch.ones_like(mask[:, :1]), eos_bos_mask], dim=1)
        return x, y, mask, eos_bos_mask

    def sample_interpolant(self, t: Tensor, x0: Tensor, y0: Tensor,mask_0: Tensor, eos_bos_mask: Tensor) -> JointMultimodalInterpolantResult:
        t = t.view(-1,1)
        full_xt = self.scale(t) * x0 + self.sigma(t) * torch.randn_like(x0)
        full_xt = full_xt * mask_0
        full_xt = torch.where(eos_bos_mask, torch.zeros_like(full_xt), full_xt)

        masking_time = self.masking_time(t, x0) # This is for the text modality, but I am using x0 for dtype 
        yt = torch.where(t < masking_time, y0, self.vocab_size) # Change to mask id
        t_shaped = t.expand(-1, full_xt.shape[1])
        deletion_time = self.deletion_time(t, x0, masking_time)
        new_mask = mask_0 & (t_shaped < deletion_time) & ((t_shaped < deletion_time) | (t_shaped >= masking_time))
        new_mask = new_mask | eos_bos_mask

        st = self.get_st(new_mask)
        xt = self.get_active_positions(full_xt, st)
        yt = self.get_active_positions(yt, st)
        y0_reordered = self.get_active_positions(y0, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order
        x0_ordered = self.get_active_positions(x0, st)
        st[:,0] = -1 # Small hack to make the cum sums work nicely

        return JointMultimodalInterpolantResult(
            xt=xt, yt=yt, st=st, mask_t=mask_t, t=t, x0=x0, 
            x0_ordered=x0_ordered, 
            xt_original_order=full_xt, 
            mask_original_order=new_mask, 
            y0_ordered=y0_reordered,
            yt_original_order=yt
        )
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps) + eps
    
    def get_st(self, mask_t: Tensor) -> Tensor:
        return mask_t.argsort(dim=1, descending=True, stable=True)
    
    def get_active_positions(self, xt: Tensor, st: Tensor) -> Tensor:
        return torch.gather(xt, 1, st)
    
    def interval_mixture(self, probs: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        """
        probs: [B, D, K]  (densities on K-grid for each of D dims)
        st:    [B, D]     (partially ordered indices with a "reset")

        returns:
        mixture: [B, D, K], where mixture[:, d, :] = average of probs over
                i in [st[:, d]+1, st[:, d+1]-1] for d in the first increasing run,
                else 0.
        """
        B, D, K = probs.shape
        assert st.shape == (B, D)

        # Pairs that are strictly increasing
        # st = torch.cat([st, -torch.ones_like(st[:, :1]) * D], dim=1)
        inc = st[:, 1:] > st[:, :-1]          # [B, D-1]

        # Keep only the first increasing prefix
        prefix = torch.cumprod(inc.to(torch.int64), dim=1).to(torch.bool)  # [B, D-1]

        start = st[:, :-1] + 1   # [B, D-1]
        end   = st[:, 1:]  - 1   # [B, D-1]

        nonempty = end >= start
        mask = prefix & nonempty  # [B, D-1]

        # Interval lengths (clamp to 1 so empty intervals are safe)
        lengths = (end - start + 1).clamp(min=1)  # [B, D-1]

        # Range-sum over dim=1 using cumsum
        cs = probs.cumsum(dim=1)  # [B, D, K]
        cs_pad = torch.cat(
            [torch.zeros(B, 1, K, device=probs.device, dtype=probs.dtype), cs],
            dim=1
        )  # [B, D+1, K]
        start_cl = start.clamp(0, D)
        endp1_cl = (end + 1).clamp(0, D)

        start_idx = start_cl.unsqueeze(-1).expand(-1, -1, K)
        endp1_idx = endp1_cl.unsqueeze(-1).expand(-1, -1, K)

        interval_sum = cs_pad.gather(1, endp1_idx) - cs_pad.gather(1, start_idx)
        interval_sum = interval_sum * mask.unsqueeze(-1)

        # Divide by interval length
        interval_avg = interval_sum / lengths.unsqueeze(-1)

        # Put into [B, D, K]
        mixture = probs.new_zeros(B, D, K)
        mixture[:, :-1, :] = interval_avg
        return mixture

    def compute_loss(self, model, batch):
        _x0 = batch["data"]
        _y0 = batch["label"]
        _mask_0 = batch["mask"]

        x0, y0, mask_0, eos_bos_mask = self.pad_sequence(_x0, _y0, _mask_0)
        t = self.sample_time(_x0.shape[0], _x0.device)
        interpolant_sample = self.sample_interpolant(t, x0, y0, mask_0, eos_bos_mask)

        prediction: MultimodalModelPrediction = model(
            euclidean_tokens=interpolant_sample.xt,
            cat_tokens=interpolant_sample.yt,
            text_mask=interpolant_sample.mask_t,
            image_mask=interpolant_sample.mask_t,
            text_time_cond=t,
            image_time_cond=t
        )

        lengths = (interpolant_sample.mask_t.sum(dim=-1) - 2).clamp(min=1) # Subtracting 2 because we are not considering the end of sequence token and the start of sequence token
        dsm_loss = (interpolant_sample.x0_ordered - prediction.clean_data)**2 * interpolant_sample.mask_t * ~eos_bos_mask
        dsm_loss = dsm_loss.sum(dim=-1) / lengths
        dsm_loss = dsm_loss.mean()
        
        means = (self.scale(t).view(-1, 1) * interpolant_sample.x0).unsqueeze(-1)
        variances = self.sigma(t).view(-1,1,1) ** 2

        grid_points = torch.linspace(
            self.tokenizer.left_endpoint, 
            self.tokenizer.right_endpoint, 
            self.tokenizer.num_bins, device=x0.device
        ).view(1,1,-1)
        potentials = -.5 * (grid_points - means)**2 / variances
        probs = 1 / (variances * 2 * torch.pi).sqrt() * torch.exp(potentials)

        mixture_prob = self.interval_mixture(probs, interpolant_sample.st)
        insertion_rate = prediction.insertion_rate
        insertion_logits = prediction.insertion_logits.log_softmax(dim=-1)
        euclidean_insertion_loss = insertion_rate - (mixture_prob * (insertion_logits + insertion_rate.log().unsqueeze(-1))).sum(dim=-1)

        euclidean_insertion_loss = euclidean_insertion_loss.sum(dim=-1) / lengths
        euclidean_insertion_loss = euclidean_insertion_loss.mean()

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
            "euclidean_insertion_loss": euclidean_insertion_loss,
        }
    
    def get_score(self, prediction: MultimodalModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        # clean_data = torch.arange(clean_data.shape[1], device=clean_data.device).repeat(clean_data.shape[0], 1)
        # Isolating score effect
        return - (xt - clean_data * self.scale(t).view(-1, 1)) / self.sigma(t).view(-1, 1)**2
    
    def get_insertion_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        lambda_t = self.beta(t).view(-1, 1)
        predicted_rate = prediction.insertion_rate

        # Subtracting 2 because we are not considering the end of sequence token and the start of sequence token
        rate = lambda_t / (mask_t.sum(dim=-1, keepdim=True) - 2).clamp(min=1)
        rate = rate * predicted_rate

        return rate
    
    def get_unmasking_rate(self, prediction: MultimodalModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        lambda_t = self.beta(t).view(-1, 1)
        big_beta = self.beta_int(t).view(-1, 1)
        return 1/(torch.exp(big_beta) - 1) * lambda_t
    
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
        xt = torch.randn((batch_size, max_length), device=device)
        yt = torch.full((batch_size, max_length), self.pad_token, device=device)
        xt, yt, mask_t, eos_bos_mask = self.pad_sequence(xt, yt, torch.zeros((batch_size, max_length), dtype=torch.bool, device=device))
        mask_t = mask_t | eos_bos_mask
        
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
            beta = self.beta(t).view(-1, 1)
            xt = xt + (beta * (xt + score) * dt) * mask_t * ~eos_bos_mask

            unmasking_rate = self.get_unmasking_rate(prediction, mask_t, t)
            # Expand unmasking_rate to match sequence length: [batch, 1] -> [batch, seq_len]
            seq_len = mask_t.shape[1]
            unmasking_rate = unmasking_rate.expand(-1, seq_len)
            unmasking_nums = torch.distributions.poisson.Poisson(unmasking_rate * dt).sample()
            new_sample = sample_categorical(prediction.label_logits.softmax(dim=-1), method="hard")  # [batch, seq_len]

            change_pos = (unmasking_nums > 0) & (mask_t == True) & (yt == self.mask_token)

            yt[change_pos] = new_sample[change_pos]


            # Perform insertions
            insertion_rate = self.get_insertion_rate(prediction, mask_t, t)
            ext = torch.bernoulli((insertion_rate * dt).clamp(0.0, 1.0)).long()  # (B, L) where L is seq_len after padding

            probabilities = F.softmax(prediction.label_logits, dim=-1)
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
                            new_sample = torch.multinomial(probabilities[j, k], 1).float()
                            new_sample = self.tokenizer.decode(new_sample)
                            new_xt_j.append(new_sample.item())
                            new_yt_j.append(self.mask_token)
                        # Always append the current token at position k
                        new_xt_j.append(xt[j, k].item())
                        new_yt_j.append(yt[j, k].item())
                    
                    # Ensure we don't exceed the tensor size
                    new_len = min(len(new_xt_j), seq_len)
                    new_xt_tensor = torch.tensor(new_xt_j[:new_len], device=device, dtype=xt.dtype)
                    new_yt_tensor = torch.tensor(new_yt_j[:new_len], device=device, dtype=yt.dtype)
                    xt[j, :new_len] = new_xt_tensor
                    yt[j, :new_len] = new_yt_tensor
                    xt[j, new_len-1] = 0.
                    yt[j, new_len-1] = self.pad_token
                    mask_t[j, :new_len] = True
                    mask_t[j, new_len:] = False
                    eos_bos_mask[j,:] = False
                    if new_len > 0:
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