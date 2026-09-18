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
        geom_unmask_weight=0.05, ins_loss_weight=0.1, cat_drop=0.0, cond='uncond', vis_dir=None,
    ):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        assert cond in ('uncond', 'random4')
        self.cond = cond            # training mix; validation always generates unconditionally
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
        self.cat_drop = cat_drop
        self.save_hyperparameters(ignore=['backbone_model', 'sampler'])
        self.sampling = dict(self.DEFAULT_SAMPLING)

    def forward(self, xt, yt, exists, t, ctx=None, cmask=None):
        return self.model(xt, yt, exists, t, ctx, cmask)

    def kappa(self, t):
        return (t / self.t_max).clamp(max=1.0)

    def cond_mask(self, batch, task):
        '''
        (B,S,5) float mask over [x, y, w, h, cat]; 1 = free, 0 = given. Same conventions as
        LayoutFlow: cat_cond gives the categories, size_cond categories + sizes, elem_compl a
        random ~20% of the elements entirely, random4 mixes those three with uncond by batch
        quarter (training). Pad slots are never "given".
        '''
        B, S = batch['type'].shape
        active = batch['mask'].squeeze(-1)
        m = torch.ones(B, S, 5, device=self.device)
        if task == 'uncond':
            return m
        sl = {'elem_compl': slice(0, B // 4), 'cat_cond': slice(B // 4, B // 2), 'size_cond': slice(B // 2, 3 * B // 4)} \
            if task == 'random4' else {task: slice(0, B)}
        if 'cat_cond' in sl:
            m[sl['cat_cond'], :, 4] = 0
        if 'size_cond' in sl:
            m[sl['size_cond'], :, 2:] = 0
        if 'elem_compl' in sl:
            s = sl['elem_compl']
            L = batch['length'][s].view(-1, 1).float()
            n_given = (L * 0.2 * torch.rand_like(L)).floor() + 1           # 1 + U(0, 0.2 L) elements, if L > 1
            rank = torch.rand(L.shape[0], S, device=self.device).masked_fill(~active[s], 2).argsort(1).argsort(1)
            given = (rank < n_given) & (L > 1)
            m[s] = torch.where(given.unsqueeze(-1), torch.zeros_like(m[s]), m[s])
        return m * active.unsqueeze(-1).float() + (1 - active.unsqueeze(-1).float())

    def sample_state(self, batch, t, cmask=None):
        '''
        Draw (x_t, y_t, exists, visible) from the conditional path at time t (B,). A conditioned
        element (its category given, cmask[..., 4] == 0) is visible from t = 0; its given
        coordinates are held at their clean values. Returns the free-coordinate mask too.
        '''
        active = batch['mask'].squeeze(-1)
        x1 = batch['mask'] * self.sampler.preprocess(batch['bbox'])
        y1 = batch['type'].long()
        x0 = batch['mask'] * torch.randn_like(x1)
        given = active & (cmask[..., 4] == 0) if cmask is not None else torch.zeros_like(active)

        k = self.kappa(t).view(-1, 1)
        u1, u2 = torch.rand_like(x0[..., 0]), torch.rand_like(x0[..., 0])
        if self.insertion:
            exists = active & (k >= 1 - u1)
            visible = active & (k >= 1 - u1 * u2)
        else:
            exists = active
            visible = active & (k >= u2)
        exists, visible = exists | given, visible | given

        tpad = t.view(-1, 1, 1)
        vis = visible.unsqueeze(-1)
        xt = vis * (tpad * x1 + (1 - tpad) * x0)
        free = cmask[..., :4] if cmask is not None else torch.ones_like(xt)
        xt = free * xt + (1 - free) * x1
        yt = torch.where(visible, y1, torch.full_like(y1, self.mask_id))
        return xt, yt, exists, visible, x0, x1, y1, free

    def training_step(self, batch, batch_idx):
        t = torch.rand(batch['bbox'].shape[0], device=self.device)
        cmask = self.cond_mask(batch, self.cond) if self.cond != 'uncond' else None
        xt, yt, exists, visible, x0, x1, y1, free = self.sample_state(batch, t, cmask)
        masked = exists & ~visible
        if self.cat_drop > 0:
            # classifier-free guidance training: hide every category of a layout w.p. cat_drop, so the
            # model also learns the category-unconditional velocity (the boxes stay visible)
            drop = (torch.rand(xt.shape[0], device=self.device) < self.cat_drop).unsqueeze(1)
            yt = torch.where(drop & visible, torch.full_like(yt, self.mask_id), yt)
        vt, logits, h, ins_rate = self(xt, yt, exists, t, batch.get('ctx'), cmask)

        vis = visible.unsqueeze(-1) * free           # no velocity target on given coordinates
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

    # Sampler knobs, not hyperparameters: the defaults reproduce the sampler the reported numbers
    # were produced with; test.py overrides them with `+sampling.<key>=<value>`.
    DEFAULT_SAMPLING = dict(
        gmm_temp=1.0,      # scale on the mixture component's sigma when drawing the clean box at reveal
        reveal_eps=1.0,    # scale on the fresh noise mixed into the revealed box, t x_1 + (1-t) eps
        solver='euler',    # 'heun': second-order steps once everything is revealed (kappa = 1)
        cfg_w=1.0,         # classifier-free guidance on the categories: v_u + w (v_c - v_u); 1 = off
        snap_grid=0,       # > 0: round the final ltrb edges to multiples of 1/snap_grid
    )

    def velocity(self, x, y, exists, t, ctx=None, cmask=None):
        v = self(x, y, exists, t, ctx, cmask)[0]
        w = self.sampling['cfg_w']
        if w != 1.0:
            # "unconditional" = same boxes, every category hidden behind the mask token
            v_u = self(x, torch.where(exists, torch.full_like(y, self.mask_id), y), exists, t, ctx, cmask)[0]
            v = v_u + w * (v - v_u)
        return v

    @torch.no_grad()
    def inference(self, batch, task='uncond', given_length=False):
        '''
        Returns (bbox, label, pad_mask). task: uncond | cat_cond | size_cond | elem_compl, with the
        given values read from the batch. With insertion the number of elements is generated unless
        given_length (LayoutFlow's protocol: the remaining slots start as masked elements).
        '''
        B, S = batch['type'].shape
        dev = batch['bbox'].device
        active = batch['mask'].squeeze(-1)
        given_length = given_length or task in ('cat_cond', 'size_cond')   # the whole element set is given
        cmask = self.cond_mask(batch, task)
        given = active & (cmask[..., 4] == 0)
        held = (1 - cmask[..., :4]) * given.unsqueeze(-1)              # coordinates pinned to the batch values
        x1 = self.sampler.preprocess(batch['bbox'])
        exists = active.clone() if (not self.insertion or given_length) else given.clone()
        # a given element is visible from t = 0: given coordinates clean, the rest at pure noise
        x = torch.zeros(B, S, self.geom_dim, device=dev)
        if given.any():
            x = given.unsqueeze(-1) * (held * x1 + (1 - held) * torch.randn_like(x))
        y = torch.where(given, batch['type'].long(), torch.full((B, S), self.mask_id, dtype=torch.long, device=dev))
        s = self.sampling
        ctx = batch.get('ctx')

        N = self.inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((B,), i * dt, device=dev)
            t_next = (i + 1) * dt
            k, k_next = min(i * dt / self.t_max, 1.0), min(t_next / self.t_max, 1.0)
            # exact per-step jump probability of the hazard kappa'/(1-kappa)
            p = 1.0 if (k_next >= 1.0 or i == N - 1) else (k_next - k) / (1 - k)

            v, logits, h, ins_rate = self(x, y, exists, t, ctx, cmask)
            masked = exists & (y == self.mask_id)
            visible = exists & ~masked
            vis = visible.unsqueeze(-1)
            if s['cfg_w'] != 1.0:
                v = self.velocity(x, y, exists, t, ctx, cmask)

            # denoise (Euler; Heun once the layout is complete and nothing can change discontinuously)
            if s['solver'] == 'heun' and k >= 1.0 and i < N - 1:
                x_e = torch.where(vis, x + v * dt, x)
                v2 = self.velocity(x_e, y, exists, t + dt, ctx, cmask)
                x = torch.where(vis, x + 0.5 * (v + v2) * dt, x)
            else:
                x = torch.where(vis, x + v * dt, x)
            if task != 'uncond':
                x = held * x1 + (1 - held) * x

            # unmask: class from the categorical, clean box from the mixture, noised to t_next
            reveal = masked & (torch.rand(B, S, device=dev) < p)
            if reveal.any():
                logits = logits[reveal]
                logits[:, 0] = float('-inf')              # id 0 is the dataset's pad label, never a class
                cls = torch.distributions.Categorical(logits=logits).sample()
                log_pi, mu, sigma = self.model.unmask_geom(h[reveal], cls)
                g1 = gmm_sample(log_pi, mu, sigma * s['gmm_temp']) if self.unmask_geom == 'gmm' else gmm_mean(log_pi, mu)
                x[reveal] = t_next * g1 + (1 - t_next) * s['reveal_eps'] * torch.randn_like(g1)
                y[reveal] = cls

            # insert: Poisson number of new masked elements into free slots
            if self.insertion and not given_length and p < 1.0:
                n_new = torch.poisson(ins_rate * p)
                free_rank = (~exists).cumsum(1)
                exists = exists | (~exists & (free_rank <= n_new.view(-1, 1)))

        # pack the generated elements to the front, as the dataset does
        order = exists.int().argsort(dim=1, descending=True, stable=True)
        x, y, exists = x.gather(1, order.unsqueeze(-1).expand_as(x)), y.gather(1, order), exists.gather(1, order)
        geom = exists.unsqueeze(-1) * self.sampler.preprocess(x, reverse=True)
        if s['snap_grid']:
            g = s['snap_grid']
            ltrb = torch.cat([geom[..., :2] - geom[..., 2:] / 2, geom[..., :2] + geom[..., 2:] / 2], -1)
            ltrb = (ltrb * g).round() / g
            geom = torch.cat([(ltrb[..., :2] + ltrb[..., 2:]) / 2, ltrb[..., 2:] - ltrb[..., :2]], -1)
        label = torch.where(exists, y, torch.zeros_like(y)).clamp(0, self.num_cat - 1)
        return geom, label, exists
