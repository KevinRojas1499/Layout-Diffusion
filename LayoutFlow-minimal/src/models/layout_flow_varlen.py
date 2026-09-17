import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseGenModel
from src.fid import FID_score
from src.backbone_varlen import gmm_nll, gmm_sample, gmm_mean


class LayoutFlowVarLen(BaseGenModel):
    '''
    Insert -> unmask -> denoise generation of whole layout elements (geometry x
    and category y together), built up in two stages:

      insertion=False  fixed length. Every element starts as a mask token M and is
                       unmasked at its own random time; from then on it is denoised.
      insertion=True   variable length. The layout starts empty, masked elements are
                       inserted, then unmasked, then denoised. Length is generated.

    Conditional path for one element of the data (x_1, y_1), with reveal schedule
    kappa(t) = min(t / t_max, 1) and u_1, u_2 ~ U(0,1):

      exists   iff kappa(t) >= 1 - u_1          (always, when insertion=False)
      visible  iff kappa(t) >= 1 - u_1 u_2      (>= u_2 when insertion=False)
      masked   = exists and not visible  ->  (0, M)
      visible                            ->  (t x_1 + (1-t) x_0, y_1),  x_0 ~ N(0, I)

    so a masked element unmasks, and a missing one is inserted, with the same hazard
    kappa'/(1-kappa). Unmasking draws the element's value from p_t(v | x_1): the
    category is revealed clean and the geometry at the *current* noise level. The
    model therefore needs a sample from the posterior of the clean box, not its mean:
    the unmask head is a Gaussian mixture over x_1 (given the sampled class), trained
    by NLL, and the revealed value is t x_1 + (1-t) eps. unmask_geom='mean' is the
    point-estimate ablation (MSE-trained mean in place of a posterior sample).

    Losses: velocity MSE (+ L1) on visible elements, cross-entropy + geometry NLL on
    masked elements, Poisson Bregman divergence between the predicted insertion rate
    and the number of elements still missing.
    '''

    def __init__(
        self, backbone_model, sampler, optimizer, scheduler=None, loss_fcn='mse',
        pretrained_dir='./pretrained', format='xywh', fid_calc_every_n=20,
        dataset='RICO', num_cat=6, inference_steps=100, insertion=False, t_max=1.0,
        unmask_geom='gmm', add_loss='', add_loss_weight=1, cat_loss_weight=0.25,
        geom_unmask_weight=0.05, ins_loss_weight=0.1, vis_dir=None,
    ):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = 'uncond'
        fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n) if fid_calc_every_n else None
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model,
                          vis_dir=vis_dir)
        assert unmask_geom in ('gmm', 'mean')
        self.num_cat = num_cat
        self.mask_id = num_cat
        self.model = backbone_model
        self.sampler = sampler
        self.inference_steps = inference_steps
        self.insertion = insertion
        self.t_max = t_max
        self.unmask_geom = unmask_geom
        self.loss_fcn = nn.MSELoss() if loss_fcn != 'l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.cat_loss_weight = cat_loss_weight
        self.geom_unmask_weight = geom_unmask_weight
        self.ins_loss_weight = ins_loss_weight
        self.save_hyperparameters(ignore=['backbone_model', 'sampler'])

    def forward(self, xt, yt, exists, t):
        return self.model(xt, yt, exists, t)

    def kappa(self, t):
        return (t / self.t_max).clamp(max=1.0)

    def sample_state(self, batch, t):
        '''Draw (x_t, y_t, exists, visible) from the conditional path at time t (B,).'''
        active = batch['mask'].squeeze(-1)
        x1 = batch['mask'] * self.sampler.preprocess(batch['bbox'])
        y1 = batch['type'].long()
        x0 = self.sampler.sample(batch)

        k = self.kappa(t).view(-1, 1)
        u1, u2 = torch.rand_like(x0[..., 0]), torch.rand_like(x0[..., 0])
        if self.insertion:
            exists = active & (k >= 1 - u1)
            visible = active & (k >= 1 - u1 * u2)
        else:
            exists = active
            visible = active & (k >= u2)

        tpad = t.view(-1, 1, 1)
        vis = visible.unsqueeze(-1)
        xt = vis * (tpad * x1 + (1 - tpad) * x0)
        yt = torch.where(visible, y1, torch.full_like(y1, self.mask_id))
        return xt, yt, exists, visible, x0, x1, y1

    def training_step(self, batch, batch_idx):
        t = torch.rand(batch['bbox'].shape[0], device=self.device)
        xt, yt, exists, visible, x0, x1, y1 = self.sample_state(batch, t)
        masked = exists & ~visible
        vt, logits, h, ins_rate = self(xt, yt, exists, t)

        vis = visible.unsqueeze(-1)
        ut = x1 - x0
        flow_loss = self.loss_fcn(vis * vt, vis * ut)
        loss = flow_loss
        self.log('flow_loss', flow_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.add_loss:
            assert self.add_loss == 'geom_l1_loss'
            l1 = F.l1_loss(vis * vt, vis * ut)
            self.log(self.add_loss, l1, on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self.add_loss_weight * l1

        if masked.any():
            cat_loss = F.cross_entropy(logits[masked], y1[masked])
            log_pi, mu, sigma = self.model.unmask_geom(h[masked], y1[masked])
            if self.unmask_geom == 'gmm':
                geom_unmask_loss = gmm_nll(log_pi, mu, sigma, x1[masked]).mean()
            else:
                geom_unmask_loss = F.mse_loss(gmm_mean(log_pi, mu), x1[masked])
        else:
            cat_loss = geom_unmask_loss = logits.sum() * 0
        loss = loss + self.cat_loss_weight * cat_loss + self.geom_unmask_weight * geom_unmask_loss
        self.log('cat_loss', cat_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('geom_unmask_loss', geom_unmask_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.insertion:
            missing = (batch['mask'].squeeze(-1).sum(1) - exists.sum(1)).float()
            rate = ins_rate.clamp(min=1e-6)
            ins_loss = (rate - missing * rate.log()).mean()
            loss = loss + self.ins_loss_weight * ins_loss
            self.log('ins_loss', ins_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log('ins_mae', (ins_rate - missing).abs().mean(), on_step=False, on_epoch=True, sync_dist=True)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def inference(self, batch):
        '''Returns (bbox, label, pad_mask). With insertion the length is generated, not read from the batch.'''
        B, S = batch['type'].shape
        dev = batch['bbox'].device
        exists = torch.zeros(B, S, dtype=torch.bool, device=dev) if self.insertion else batch['mask'].squeeze(-1).clone()
        x = torch.zeros(B, S, self.geom_dim, device=dev)
        y = torch.full((B, S), self.mask_id, dtype=torch.long, device=dev)

        N = self.inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((B,), i * dt, device=dev)
            t_next = (i + 1) * dt
            k, k_next = min(i * dt / self.t_max, 1.0), min(t_next / self.t_max, 1.0)
            # exact per-step jump probability of the hazard kappa'/(1-kappa)
            p = 1.0 if (k_next >= 1.0 or i == N - 1) else (k_next - k) / (1 - k)

            v, logits, h, ins_rate = self(x, y, exists, t)
            masked = exists & (y == self.mask_id)
            visible = exists & ~masked

            # denoise
            x = torch.where(visible.unsqueeze(-1), x + v * dt, x)

            # unmask: class from the categorical, clean box from the mixture, noised to t_next
            reveal = masked & (torch.rand(B, S, device=dev) < p)
            if reveal.any():
                logits = logits[reveal]
                logits[:, 0] = float('-inf')              # id 0 is the dataset's pad label, never a class
                cls = torch.distributions.Categorical(logits=logits).sample()
                log_pi, mu, sigma = self.model.unmask_geom(h[reveal], cls)
                g1 = gmm_sample(log_pi, mu, sigma) if self.unmask_geom == 'gmm' else gmm_mean(log_pi, mu)
                x[reveal] = t_next * g1 + (1 - t_next) * torch.randn_like(g1)
                y[reveal] = cls

            # insert: Poisson number of new masked elements into free slots
            if self.insertion and p < 1.0:
                n_new = torch.poisson(ins_rate * p)
                free_rank = (~exists).cumsum(1)
                exists = exists | (~exists & (free_rank <= n_new.view(-1, 1)))

        # pack the generated elements to the front, as the dataset does
        order = exists.int().argsort(dim=1, descending=True, stable=True)
        x, y, exists = x.gather(1, order.unsqueeze(-1).expand_as(x)), y.gather(1, order), exists.gather(1, order)
        geom = exists.unsqueeze(-1) * self.sampler.preprocess(x, reverse=True)
        label = torch.where(exists, y, torch.zeros_like(y)).clamp(0, self.num_cat - 1)
        return geom, label, exists
