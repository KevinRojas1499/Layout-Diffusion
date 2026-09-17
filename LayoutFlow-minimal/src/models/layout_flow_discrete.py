import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseGenModel
from src.fid import FID_score


class LayoutFlowDiscrete(BaseGenModel):
    '''
    LayoutFlow with the category modelled by masked (absorbing-state) discrete
    diffusion instead of AnalogBit flow matching. Both modalities are corrupted
    jointly at the same time t:

      geometry x : x_t = t x_1 + (1-t) x_0, x_0 ~ N(0, I), at EVERY element for
                   the whole trajectory (never blanked) -- exactly LayoutFlow's
                   linear conditional flow, velocity-MSE (+ geom L1) loss.
      category y : each element is independently [MASK] with prob 1-t (masked
                   iff t < u_i, u_i ~ U(0,1)); cross-entropy on masked elements.

    Sampling: Euler steps on x for all elements; a masked y unmasks during
    [t, t+dt] with the exact posterior prob dt/(1-t), its value drawn from the
    model's categorical. Conditioning (random4) reuses BaseGenModel's cond mask
    with one flag for the category: "category given" = never masked.
    '''

    def __init__(
        self, backbone_model, sampler, optimizer, scheduler=None, loss_fcn='mse',
        pretrained_dir='./pretrained', format='xywh', fid_calc_every_n=20,
        dataset='RICO', num_cat=6, inference_steps=100, cond='uncond',
        add_loss='', add_loss_weight=1, cat_loss_weight=1.0, vis_dir=None,
    ):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = cond
        fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n) if fid_calc_every_n else None
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model,
                          vis_dir=vis_dir)
        self.attr_dim = 1            # one cond flag for the (discrete) category
        self.num_cat = num_cat
        self.mask_id = num_cat
        self.model = backbone_model
        self.sampler = sampler
        self.inference_steps = inference_steps
        self.loss_fcn = nn.MSELoss() if loss_fcn != 'l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.cat_loss_weight = cat_loss_weight
        self.save_hyperparameters(ignore=['backbone_model', 'sampler'])

    def forward(self, xt, yt, cond_mask, t):
        return self.model(xt, yt, cond_mask, t)

    def _clean_geom(self, batch):
        x1 = self.sampler.preprocess(batch['bbox'])
        return batch['mask'] * x1 + (~batch['mask']) * batch['bbox']   # pad stays 0, as in LayoutFlow

    def training_step(self, batch, batch_idx):
        cond_mask = self.get_cond_mask(batch)
        cg, cc = cond_mask[..., :self.geom_dim], cond_mask[..., -1].bool()
        active = batch['mask'].squeeze(-1)

        x1 = self._clean_geom(batch)
        x0 = self.sampler.sample(batch)
        t = self.sample_t(x0)
        tpad = t.view(-1, 1, 1)
        xt = tpad * x1 + (1 - tpad) * x0
        ut = x1 - x0
        xt = (1 - cg) * x1 + cg * xt

        y1 = batch['type'].long()
        masked = active & cc & (t.view(-1, 1) < torch.rand_like(x0[..., 0]))
        yt = torch.where(masked, torch.full_like(y1, self.mask_id), y1)

        vt, logits = self(xt, yt, cond_mask, t)

        flow_loss = self.loss_fcn(cg * vt, cg * ut)
        cat_loss = F.cross_entropy(logits[masked], y1[masked]) if masked.any() else logits.sum() * 0
        loss = flow_loss + self.cat_loss_weight * cat_loss
        self.log('flow_loss', flow_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('cat_loss', cat_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.add_loss:
            assert self.add_loss == 'geom_l1_loss'
            l1 = F.l1_loss(cg * vt, cg * ut)
            self.log(self.add_loss, l1, on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self.add_loss_weight * l1
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def inference(self, batch):
        cond_mask = self.get_cond_mask(batch)
        cg, cc = cond_mask[..., :self.geom_dim], cond_mask[..., -1].bool()
        active = batch['mask'].squeeze(-1)

        ref = self._clean_geom(batch)
        x = self.sampler.sample(batch)
        y1 = batch['type'].long()
        y = torch.where(active & cc, torch.full_like(y1, self.mask_id), y1)

        N = self.inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = (1 - cg) * ref + cg * x
            v, logits = self(x, y, cond_mask, t)

            is_masked = y == self.mask_id
            p_unmask = 1.0 if i == N - 1 else dt / (1 - i * dt)
            reveal = is_masked & (torch.rand_like(x[..., 0]) < p_unmask)
            logits[..., 0] = float('-inf')            # id 0 is the dataset's pad label, never a class
            sample = torch.distributions.Categorical(logits=logits).sample()
            y = torch.where(reveal, sample, y)
            x = x + v * dt

        x = self.sampler.preprocess(x, reverse=True)
        geom = (1 - cg) * batch['bbox'] + cg * x
        return geom, y.clamp(0, self.num_cat - 1)
