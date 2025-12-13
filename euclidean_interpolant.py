import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Iterator

@dataclass
class EuclideanModelPrediction:
    clean_data: Tensor
    mean: Tensor
    std: Tensor
    rate: Tensor

    def __init__(
        self,
        clean_data: Tensor,
        mean: Tensor,
        std: Tensor,
        rate: Tensor,
    ):
        self.clean_data = clean_data
        self.mean = mean
        self.std = std
        self.rate = rate

@dataclass
class SamplingTrajectoryResult:
    xt: Tensor # Shape [Batch, Length]
    st: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]

@dataclass
class SamplingResult:
    xt: Tensor # Shape [Batch, Length]
    st: Tensor # Shape [Batch, Length]
    mask_t: Tensor # Shape [Batch, Length]
    xt_original_order: Tensor # Shape [Batch, Length]
    mask_original_order: Tensor # Shape [Batch, Length]
    t: Tensor # Shape [Batch]
    trajectory: List[SamplingTrajectoryResult]

    def __getitem__(self, index: int) -> SamplingTrajectoryResult:
        trajectory_slice = []
        for step in self.trajectory:
             trajectory_slice.append(SamplingTrajectoryResult(
                xt=step.xt[index],
                st=step.st[index],
                mask_t=step.mask_t[index],
                t=step.t[index]
             ))

        return SamplingResult(
            xt=self.xt[index],
            st=self.st[index],
            mask_t=self.mask_t[index],
            xt_original_order=self.xt_original_order[index],
            mask_original_order=self.mask_original_order[index],
            t=self.t[index],
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




class EuclideanInterpolant():
    def __init__(
        self,
        max_length: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
    ):
        super().__init__()
        self.linear_start = linear_start
        self.linear_end = linear_end

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
        xt = self.get_active_positions(full_xt, st)
        mask_t = self.get_active_positions(new_mask, st) # This will reorder the mask according to the new order

        return JointEuclideanInterpolantResult(
            xt=xt, st=st, mask_t=mask_t, t=t, x0=x0, xt_original_order=full_xt, mask_original_order=new_mask
        )
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-5
        return torch.rand(batch_size, device=device) * (1 - eps) + eps
    
    def get_active_positions(self, xt: Tensor, st: Tensor) -> Tensor:
        return torch.gather(xt, 1, st)
    
    def compute_loss(self, model, batch):
        x0 = batch["data"]
        mask_0 = batch["mask"]

        t = self.sample_time(x0.shape[0], x0.device)
        interpolant_sample = self.sample_interpolant(t, x0, mask_0)

        prediction: EuclideanModelPrediction = model(interpolant_sample.xt, interpolant_sample.mask_t, t)

        lengths = interpolant_sample.mask_t.sum(dim=-1).clamp(min=1)
        x0_ordered = self.get_active_positions(x0, interpolant_sample.st)
        torch.set_printoptions(precision=2, sci_mode=False)
        print('Time: ', t[0])
        print('X0: ', x0_ordered[0], sep='\n')
        print('Prediction: ', prediction.clean_data[0], sep='\n')
        dsm_loss = (x0_ordered - prediction.clean_data)**2 * interpolant_sample.mask_t
        dsm_loss = dsm_loss.sum(dim=-1) / lengths
        dsm_loss = dsm_loss.mean()

        ai = prediction.std
        bi = prediction.mean
        ci = prediction.rate

        
        # Calculate deleted particles count (masked_lengths)
        # We need the number of particles that are currently masked (deleted) but were originally valid.
        # mask_original_order is TRUE for ACTIVE particles, FALSE for DELETED or PADDING.
        # mask_0 is TRUE for VALID particles (active or deleted), FALSE for PADDING.
        
        # Particles that are deleted but valid:
        # (~interpolant_sample.mask_original_order) & mask_0
        
        valid_deleted_mask = (~interpolant_sample.mask_original_order) & mask_0
        masked_lengths = valid_deleted_mask.sum(dim=-1).clamp(min=1)
        
        prediction_loss = .5 * (ai * (interpolant_sample.xt_original_order - bi)**2 - ci) * valid_deleted_mask
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
        clean_data = prediction.clean_data 
        # clean_data = torch.arange(clean_data.shape[1], device=clean_data.device).repeat(clean_data.shape[0], 1)
        # Isolating score effect
        return - (xt - clean_data * self.scale(t).view(-1, 1)) / self.sigma(t).view(-1, 1)**2
    
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
        unordered_xt = xt.clone()
        mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
        unordered_mask_t = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
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
            
            # Denoise
            score = self.get_score(prediction, xt, t)
            beta = self.beta(t).view(-1, 1)
            xt = xt + (beta * (xt + score) * dt) * mask_t
            
            # Save the updated active particles back to the main memory
            unordered_xt.scatter_(1, st, xt)

            # Add dimensions
            insertion_rate = self.get_actual_rate(prediction, mask_t, t)
            insertion_nums = torch.distributions.poisson.Poisson(insertion_rate).sample()
            insertion_nums[insertion_nums.sum(dim = -1) > 1] = 0 
            unordered_mask_t = unordered_mask_t | (insertion_nums > 0)
            st = unordered_mask_t.argsort(dim=1, descending=True, stable=True)
            new_coordinate = prediction.mean + (1/prediction.std**.5) * torch.randn_like(xt)
            unordered_xt[insertion_nums > 0] = new_coordinate[insertion_nums > 0]
            xt = self.get_active_positions(unordered_xt, st)
            mask_t = self.get_active_positions(unordered_mask_t, st)

            if return_trace:
                trajectory.append(SamplingTrajectoryResult(
                    xt=xt, st=st, mask_t=mask_t, t=t
                ))
            t = t - dt

        return SamplingResult(
            xt=xt, st=st, mask_t=mask_t, xt_original_order=unordered_xt, mask_original_order=unordered_mask_t, t=t, trajectory=trajectory
        )