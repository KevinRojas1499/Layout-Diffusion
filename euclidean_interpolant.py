import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator
from utils.tokenizer import IntervalTokenizer
import torch.nn.functional as F



@dataclass
class EuclideanModelPrediction:
    clean_data: Tensor
    logits: Tensor
    rate: Tensor

    def __init__(
        self,
        clean_data: Tensor,
        logits: Tensor,
        rate: Tensor,
    ):
        self.clean_data = clean_data
        self.logits = logits
        self.rate = rate

@dataclass
class SamplingTrajectoryResult:
    xt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]

@dataclass
class SamplingResult:
    xt: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    trajectory: List[SamplingTrajectoryResult]

    def __getitem__(self, index: int) -> SamplingTrajectoryResult:
        trajectory_slice = []
        for step in self.trajectory:
             trajectory_slice.append(SamplingTrajectoryResult(
                xt=step.xt[index],
                mask_t=step.mask_t[index],
                t=step.t[index]
             ))

        return SamplingResult(
            xt=self.xt[index],
            mask_t=self.mask_t[index],
            trajectory=trajectory_slice
        )

    def __len__(self) -> int:
        return self.xt.shape[0]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]



@dataclass
class JointEuclideanInterpolantResult:
    # Joint Interpolant
    xt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    xt_original_order: Tensor # Shape [Batch, Length]
    mask_original_order: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    x0: Tensor
    x0_ordered: Tensor




class EuclideanInterpolant():
    def __init__(
        self,
        max_length: int,
        interval_tokenizer: IntervalTokenizer,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        train_only_dsm : bool = False
    ):
        super().__init__()
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.train_only_dsm = train_only_dsm
        self.tokenizer = interval_tokenizer

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
    
    def deletion_time(self, t: Tensor, x0: Tensor) -> Tensor:
        deletion_time = torch.rand_like(x0)
        return deletion_time

    def pad_sequence(self, x: Tensor, mask: Tensor) -> Tensor:
        x = torch.where(mask, x, torch.zeros_like(x)) # This is to represent the end of the sequence with zeros
        # Add at the start of the sequence
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)

        # Add at the end of the sequence
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        eos_bos_mask = (mask[:, :-1] != mask[:, 1:])
        mask[:,1:] = mask[:, 1:] | eos_bos_mask
        eos_bos_mask = torch.cat([torch.ones_like(mask[:, :1]), eos_bos_mask], dim=1)
        return x, mask, eos_bos_mask

    def sample_interpolant(self, t: Tensor, x0: Tensor, mask_0: Tensor, eos_bos_mask: Tensor) -> JointEuclideanInterpolantResult:
        full_xt = self.scale(t).view(-1, 1) * x0 + self.sigma(t).view(-1, 1) * torch.randn_like(x0)
        full_xt = full_xt * mask_0
        full_xt = torch.where(eos_bos_mask, torch.zeros_like(full_xt), full_xt)
        t_shaped = t.unsqueeze(1).expand(-1, full_xt.shape[1])
        deletion_time = self.deletion_time(t, x0)
        new_mask = mask_0 & (t_shaped < deletion_time)
        new_mask = new_mask | eos_bos_mask

        st = self.get_st(new_mask)
        xt = self.get_active_positions(full_xt, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order
        x0_ordered = self.get_active_positions(x0, st)
        st[:,0] = -1 # Small hack to make the cum sums work nicely

        return JointEuclideanInterpolantResult(
            xt=xt, st=st, mask_t=mask_t, t=t, x0=x0, x0_ordered=x0_ordered, xt_original_order=full_xt, mask_original_order=new_mask
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
        _mask_0 = batch["mask"]

        x0, mask_0, eos_bos_mask = self.pad_sequence(_x0, _mask_0)
        t = self.sample_time(_x0.shape[0], _x0.device)
        interpolant_sample = self.sample_interpolant(t, x0, mask_0, eos_bos_mask)

        prediction: EuclideanModelPrediction = model(interpolant_sample.xt, interpolant_sample.mask_t, t)

        lengths = (interpolant_sample.mask_t.sum(dim=-1) - 2).clamp(min=1) # Subtracting 2 because we are not considering the end of sequence token and the start of sequence token
        dsm_loss = (interpolant_sample.x0_ordered - prediction.clean_data)**2 * interpolant_sample.mask_t * ~eos_bos_mask
        dsm_loss = dsm_loss.sum(dim=-1) / lengths
        dsm_loss = dsm_loss.mean()
        
        means = (self.scale(t).view(-1, 1) * interpolant_sample.x0).unsqueeze(-1)
        variances = self.sigma(t).view(-1,1,1) ** 2

        grid_points = torch.linspace(self.tokenizer.left_endpoint, self.tokenizer.right_endpoint, self.tokenizer.num_bins, device=x0.device).view(1,1,-1)
        potentials = -.5 * (grid_points - means)**2 / variances
        probs = 1 / (variances * 2 * torch.pi).sqrt() * torch.exp(potentials)

        mixture_prob = self.interval_mixture(probs, interpolant_sample.st)
        # torch.set_printoptions(precision=2, sci_mode=False)
        # k = 4
        # print('Time' , t[k])
        # print('X0' , interpolant_sample.x0[k])
        # print('XT' , interpolant_sample.xt[k])
        # print('ST' , interpolant_sample.st[k])
        # print('Mixture prob' , mixture_prob[k])
        # print('Mixture prob shape' , mixture_prob.shape)
        rate = prediction.rate
        logits = prediction.logits.log_softmax(dim=-1)
        tokens_loss = rate - (mixture_prob * (logits + rate.log().unsqueeze(-1))).sum(dim=-1)

        tokens_loss = tokens_loss.sum(dim=-1) / lengths
        tokens_loss = tokens_loss.mean()
        
        return {
            "dsm_loss": dsm_loss,
            "tokens_loss": tokens_loss,
        }
    
    def get_score(self, prediction: EuclideanModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        clean_data = prediction.clean_data 
        # clean_data = torch.arange(clean_data.shape[1], device=clean_data.device).repeat(clean_data.shape[0], 1)
        # Isolating score effect
        return - (xt - clean_data * self.scale(t).view(-1, 1)) / self.sigma(t).view(-1, 1)**2
    
    def get_rate(self, prediction: EuclideanModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        lambda_t = self.beta(t).view(-1, 1)
        predicted_rate = prediction.rate

        # Subtracting 2 because we are not considering the end of sequence token and the start of sequence token
        rate = lambda_t / (mask_t.sum(dim=-1, keepdim=True) - 2).clamp(min=1)
        rate = rate * predicted_rate

        return rate
    
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
        xt = torch.randn((batch_size, max_length + 1), device=device)
        xt, mask_t, eos_bos_mask = self.pad_sequence(xt, torch.zeros((batch_size, max_length + 1), dtype=torch.bool, device=device))
        mask_t = mask_t | eos_bos_mask
        
        dt = 1.0 / steps
        t = torch.ones(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt, mask_t=mask_t.clone(), t=t
            ))
        for i in range(steps):
            prediction: EuclideanModelPrediction = model(xt, mask_t, t)
            # Denoise
            score = self.get_score(prediction, xt, t)
            beta = self.beta(t).view(-1, 1)
            xt = xt + (beta * (xt + score) * dt) * mask_t * ~eos_bos_mask

            insertion_rate = self.get_rate(prediction, mask_t, t)
            # Need to cheeck if I need to multiply by dt here
            ext = torch.bernoulli((insertion_rate * dt).clamp(0.0, 1.0)).long()  # (B, L+1)

            probabilities = F.softmax(prediction.logits, dim=-1)
            if i != steps - 1:
                for j in range(batch_size):
                    # Add dimensions
                    new_xt_j = []
                    for k in range(max_length + 1):
                        if not mask_t[j, k]:
                            break
                        if ext[j,k] == 1 and k != 0:
                            new_sample = torch.multinomial(probabilities[j, k], 1).float()
                            new_sample = self.tokenizer.decode(new_sample)
                            new_xt_j.append(new_sample)
                        new_xt_j.append(xt[j, k])
                    xt[j, :len(new_xt_j)] = torch.tensor(new_xt_j, device=device)
                    xt[j, len(new_xt_j)-1] = 0.
                    mask_t[j, :len(new_xt_j)] = True
                    eos_bos_mask[j,:] = False
                    eos_bos_mask[j, 0] = True
                    eos_bos_mask[j, len(new_xt_j)-1] = True
            
                    

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt, mask_t=mask_t.clone(), t=t
                ))
            t = t - dt

        return SamplingResult(
            xt=xt, mask_t=mask_t, trajectory=trajectory
        )

class EuclideanFixedSizeInterpolant():
    def __init__(
        self,
        max_length: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        train_only_dsm : bool = False
    ):
        super().__init__()
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.train_only_dsm = train_only_dsm

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
    
    def deletion_time(self, t: Tensor, x0: Tensor) -> Tensor:
        deletion_time = torch.rand_like(x0)
        return deletion_time

    def sample_interpolant(self, t: Tensor, x0: Tensor, mask_0: Tensor) -> JointEuclideanInterpolantResult:
        full_xt = self.scale(t).view(-1, 1) * x0 + self.sigma(t).view(-1, 1) * torch.randn_like(x0)
        deletion_time = self.deletion_time(t, x0)
        t_shaped = t.unsqueeze(1).expand(-1, x0.shape[1])

        new_mask = mask_0 & (t_shaped < deletion_time)
        st = new_mask.argsort(dim=1, descending=True, stable=True)

        return JointEuclideanInterpolantResult(
            xt=full_xt, st=st, mask_t=new_mask, t=t, x0=x0, xt_original_order=full_xt, mask_original_order=new_mask
        )
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps) + eps
    
    def compute_loss(self, model, batch):
        x0 = batch["data"]
        mask_0 = batch["mask"]

        t = self.sample_time(x0.shape[0], x0.device)
        interpolant_sample = self.sample_interpolant(t, x0, mask_0)
        if self.train_only_dsm:
            interpolant_sample.mask_t = mask_0

        prediction: EuclideanModelPrediction = model(interpolant_sample.xt, interpolant_sample.mask_t, t)

        lengths = interpolant_sample.mask_t.sum(dim=-1).clamp(min=1)
        dsm_loss = (x0 - prediction.clean_data)**2 * interpolant_sample.mask_t
        dsm_loss = dsm_loss.sum(dim=-1) / lengths
        dsm_loss = dsm_loss.mean()

        ai = prediction.std 
        bi = prediction.mean * self.scale(t).view(-1,1)
        ci = prediction.rate

        
        # Calculate deleted particles count (masked_lengths)
        # We need the number of particles that are currently masked (deleted) but were originally valid.
        # mask_original_order is TRUE for ACTIVE particles, FALSE for DELETED or PADDING.
        # mask_0 is TRUE for VALID particles (active or deleted), FALSE for PADDING.
        
        # Particles that are deleted but valid:
        # (~interpolant_sample.mask_original_order) & mask_0
        
        valid_deleted_mask = (~interpolant_sample.mask_t) & mask_0
        masked_lengths = valid_deleted_mask.sum(dim=-1).clamp(min=1)
        
        prediction_loss = (.5 * ai * (self.sigma(t).view(-1,1)**2 + (interpolant_sample.xt - bi)**2) - ci )* valid_deleted_mask
        prediction_loss = prediction_loss.sum(dim=-1) / masked_lengths
        prediction_loss = prediction_loss.mean()

        rate_loss = torch.exp(ci) * (2 * torch.pi/ ai).sqrt() * valid_deleted_mask
        rate_loss = rate_loss.sum(dim=-1) / masked_lengths
        rate_loss = rate_loss.mean()

        return {
            "dsm_loss": dsm_loss,
            "prediction_loss": prediction_loss,
            "rate_loss": rate_loss,
        }
    
    def get_score(self, prediction: EuclideanModelPrediction, xt: Tensor, t: Tensor) -> Tensor:
        return - (xt - prediction.clean_data * self.scale(t).view(-1, 1)) / self.sigma(t).view(-1, 1)**2
    
    def get_actual_rate(self, prediction: EuclideanModelPrediction, mask_t: Tensor, t: Tensor) -> Tensor:
        ai = prediction.std
        # bi = prediction.mean
        ci = prediction.rate

        lambda_t = self.beta(t).view(-1, 1)

        rate = ci.exp() * (2 * torch.pi/ ai).sqrt() * lambda_t / mask_t.sum(dim=-1, keepdim=True).clamp(min=1)

        return rate
    
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
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        if self.train_only_dsm:
            mask_t = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)
        st = torch.arange(max_length, device=device).repeat(batch_size, 1)

        dt = 1.0 / steps
        t = torch.ones(batch_size, device=device)

        trajectory = []
        if return_trace:
            trajectory.append(SamplingTrajectoryResult(
                xt=xt, st=st, mask_t=mask_t, t=t
            ))
        for i in range(steps):
            # ——— predict and convert rates ———
            prediction: EuclideanModelPrediction = model(xt, mask_t, t)
            # Ablating the leaarning
            # clean_data = torch.arange(prediction.clean_data.shape[1], device=prediction.clean_data.device).repeat(prediction.clean_data.shape[0], 1)
            # prediction.clean_data = clean_data
            # prediction.mean = clean_data
            print('Mean prediction')
            print(prediction.mean[0])
            print('Clean data prediction')
            print(prediction.clean_data[0] * mask_t[0])
            
            # Denoise
            score = self.get_score(prediction, xt, t)
            beta = self.beta(t).view(-1, 1)
            xt = xt + (beta * (xt + score) * dt) * mask_t
            
            # Add dimensions
            insertion_rate = self.get_actual_rate(prediction, mask_t, t)
            insertion_nums = torch.distributions.poisson.Poisson(insertion_rate * dt).sample()
            # insertion_nums[insertion_nums.sum(dim = -1) > 1] = 0 
            

            new_coordinate = prediction.mean * self.scale(t).view(-1,1) + (1/prediction.std.sqrt()) * torch.randn_like(xt)
            
            # Only update positions that are newly inserted (insertion > 0 AND currently empty)
            change_pos = (insertion_nums > 0) & ~mask_t
            xt[change_pos] = new_coordinate[change_pos]
            
            # Update mask after writing new values
            mask_t = mask_t | (insertion_nums > 0)

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt, st=st, mask_t=mask_t, t=t
                ))
            t = t - dt

        return SamplingResult(
            xt=xt, st=st, mask_t=mask_t, trajectory=trajectory
        )