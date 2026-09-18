'''
Exponential moving average of the backbone weights (the usual diffusion-model trick).

- updated after every optimizer step: ema = d * ema + (1 - d) * w, with d warmed up from 0
  as min(decay, (1 + step) / (10 + step));
- validation runs on the EMA weights (swapped in / out around the validation loop);
- every checkpoint's `state_dict` holds the EMA weights, so test.py and the sampling scripts
  use them unchanged; the raw weights are stored under `ema_raw_state_dict` for resuming.
'''
import copy

import torch
from lightning.pytorch.callbacks import Callback


class EMA(Callback):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = None          # {name: tensor} of the EMA weights (self.model.* of the LightningModule)
        self.step = 0
        self.swapped = False

    def _params(self, pl_module):
        return dict(pl_module.model.named_parameters())

    def on_fit_start(self, trainer, pl_module):
        if self.ema is None:
            self.ema = {k: v.detach().clone() for k, v in self._params(pl_module).items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in self._params(pl_module).items():
            self.ema[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @torch.no_grad()
    def _swap(self, pl_module):
        for k, v in self._params(pl_module).items():
            tmp = v.detach().clone()
            v.copy_(self.ema[k])
            self.ema[k].copy_(tmp)
        self.swapped = not self.swapped

    def on_validation_start(self, trainer, pl_module):
        if self.ema is not None and not self.swapped:
            self._swap(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.swapped:
            self._swap(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is None:
            return
        sd = checkpoint['state_dict']
        if self.swapped:                   # saved during validation: state_dict already holds the EMA weights
            raw = {f'model.{k}': self.ema[k].clone() for k in self.ema}         # (the raw ones sit in self.ema)
        else:
            raw = {f'model.{k}': sd[f'model.{k}'].clone() for k in self.ema}
            for k in self.ema:
                sd[f'model.{k}'] = self.ema[k].clone()
        checkpoint['ema_raw_state_dict'] = raw
        checkpoint['ema_step'] = self.step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_raw_state_dict' in checkpoint:
            self.ema = {k[len('model.'):]: v.clone() for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
            self.step = checkpoint.get('ema_step', 0)
            self._restore_raw = checkpoint['ema_raw_state_dict']

    def on_train_start(self, trainer, pl_module):
        params = self._params(pl_module)
        if self.ema is not None:
            self.ema = {k: v.to(params[k].device) for k, v in self.ema.items()}
        raw = getattr(self, '_restore_raw', None)
        if raw is not None:               # resumed: put the raw weights back into the model, keep the EMA aside
            with torch.no_grad():
                for k, v in self._params(pl_module).items():
                    v.copy_(raw[f'model.{k}'])
            self._restore_raw = None
