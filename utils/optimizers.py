import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        return [base_lr * min(step / self.warmup_steps, 1.0) for base_lr in self.base_lrs]


class CosineDecayScheduler(_LRScheduler):
    """Cosine decay from `base_lr` to `min_lr` over `total_steps`, no warmup.

    Designed for resume-from-checkpoint runs: the model is already warm, so we
    skip warmup and decay the LR over the remaining training budget.
    """

    def __init__(self, optimizer, total_steps, min_lr=0.0, last_epoch=-1):
        self.total_steps = max(1, int(total_steps))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch + 1)
        if step >= self.total_steps:
            return [self.min_lr for _ in self.base_lrs]
        progress = step / self.total_steps
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cos_val for base_lr in self.base_lrs]

class CombinedOptimizer(Optimizer):
    def __init__(self, optimizer1, optimizer2):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
    
    @property
    def param_groups(self):
        # Return combined param_groups so scheduler can access them
        return self.optimizer1.param_groups + self.optimizer2.param_groups
    
    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
    
    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()
    
    def state_dict(self):
        return {
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        self.optimizer2.load_state_dict(state_dict['optimizer2'])