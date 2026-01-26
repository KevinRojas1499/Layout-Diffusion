from torch.optim.lr_scheduler import _LRScheduler

class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        return [base_lr * min(step / self.warmup_steps, 1.0) for base_lr in self.base_lrs]

class CombinedOptimizer:
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