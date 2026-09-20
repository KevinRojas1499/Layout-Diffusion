import math
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
        geom_unmask_weight=0.05, ins_loss_weight=0.1, cat_drop=0.0, cond='uncond', ralf_cache=None, fid_empty_id=None,
        ctx_token_drop=0.0, ctx_drop=0.0, vis_dir=None, clock='coupled', ret_drop=0.0,
        relation_loss_weight=0.0, relation_text_id=2, relation_underlay_id=3, relation_margin=0.01,
    ):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        assert cond in ('uncond', 'random4', 'random5')   # random5 = random4 + a refinement fifth (t in [0.9, 1])
        self.cond = cond            # training mix; validation always generates unconditionally
        if not fid_calc_every_n:
            fid_model = None
        elif ralf_cache:      # content-aware datasets: RALF's FIDNetV3 against their precomputed val features
            from src.fid_ralf import RalfFID
            fid_model = RalfFID(dataset, ralf_cache, calc_every_n=fid_calc_every_n, empty_id=fid_empty_id)
        else:
            fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n)
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model,
                          vis_dir=vis_dir)
        assert unmask_geom in ('gmm', 'mean')
        self.num_cat = num_cat
        self.mask_id = num_cat
        self.model = backbone_model
        self.sampler = sampler
        self.inference_steps = inference_steps
        # insertion: False (fixed length) | True (insert as mask, reveal later) | 'editflow' (Edit-Flow-style
        # baseline: per-element rates, the new element arrives with its class and a box at the current noise level)
        self.editflow = insertion in ('editflow', 'oneflow')
        self.oneflow = insertion == 'oneflow'      # + per-element clocks: an inserted box starts from pure noise at its own t = 0
        # oneflow clocks at training time: 'coupled' = s_i is a function of t and the insertion time (the synchronised
        # sampling path); 'decoupled' = s_i ~ U(0,1) independently of t (Diffuse-Everything-style product of times:
        # every monotone clock policy at sampling time is then on-distribution, see DEFAULT_SAMPLING['clock'])
        assert clock in ('coupled', 'decoupled')
        self.clock = clock
        self.insertion = bool(insertion)
        self.t_max = t_max
        self.unmask_geom = unmask_geom
        self.loss_fcn = nn.MSELoss() if loss_fcn != 'l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.cat_loss_weight = cat_loss_weight
        self.geom_unmask_weight = geom_unmask_weight
        self.ins_loss_weight = ins_loss_weight
        self.cat_drop = cat_drop
        self.ctx_token_drop, self.ctx_drop = ctx_token_drop, ctx_drop   # canvas regularisation (training only)
        self.ret_drop = ret_drop                                          # hide the whole retrieved set for this fraction of samples
        # containment relation loss (content-aware datasets): for every ground-truth (underlay contains text) pair, a hinge
        # on the predicted clean boxes x1_hat = x_t + (1 - t) v so that the text stays strictly inside the underlay with a margin
        self.relation_loss_weight, self.relation_text_id, self.relation_underlay_id, self.relation_margin = \
            relation_loss_weight, relation_text_id, relation_underlay_id, relation_margin
        self.save_hyperparameters(ignore=['backbone_model', 'sampler'])
        self.sampling = dict(self.DEFAULT_SAMPLING)

    def forward(self, xt, yt, exists, t, ctx=None, cmask=None, t_elem=None, ctx_hide=None, ret=None, ret_hide=None):
        return self.model(xt, yt, exists, t, ctx, cmask, t_elem, ctx_hide, ret, ret_hide)

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
        if task == 'random4':
            sl = {'elem_compl': slice(0, B // 4), 'cat_cond': slice(B // 4, B // 2), 'size_cond': slice(B // 2, 3 * B // 4)}
        elif task == 'random5':
            q = B // 5
            sl = {'elem_compl': slice(0, q), 'cat_cond': slice(q, 2 * q), 'size_cond': slice(2 * q, 3 * q), 'refinement': slice(3 * q, 4 * q)}
        else:
            sl = {task: slice(0, B)}
        for k in ('cat_cond', 'refinement'):      # categories given, boxes free
            if k in sl:
                m[sl[k], :, 4] = 0
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

    def sample_state(self, batch, t, cmask=None, s_lo=None):
        '''
        Draw (x_t, y_t, exists, visible) from the conditional path at time t (B,). A conditioned
        element (its category given, cmask[..., 4] == 0) is visible from t = 0; its given
        coordinates are held at their clean values. Returns the free-coordinate mask too.
        s_lo (B,): lower bound of the decoupled per-element clocks (the refinement fifth of random5).
        '''
        active = batch['mask'].squeeze(-1)
        x1 = batch['mask'] * self.sampler.preprocess(batch['bbox'])
        y1 = batch['type'].long()
        x0 = batch['mask'] * torch.randn_like(x1)
        given = active & (cmask[..., 4] == 0) if cmask is not None else torch.zeros_like(active)

        k = self.kappa(t).view(-1, 1)
        u1, u2 = torch.rand_like(x0[..., 0]), torch.rand_like(x0[..., 0])
        if self.editflow:                 # no masked stage: an element is visible from the moment it is inserted
            exists = visible = active & (k >= 1 - u1)
        elif self.insertion:
            exists = active & (k >= 1 - u1)
            visible = active & (k >= 1 - u1 * u2)
        else:
            exists = active
            visible = active & (k >= u2)
        exists, visible = exists | given, visible | given

        tpad = t.view(-1, 1, 1)
        if self.oneflow and self.clock == 'decoupled':
            # every element's clock is its own time variable, independent of t and of when it was inserted
            lo = s_lo.view(-1, 1) if s_lo is not None else torch.zeros_like(t).view(-1, 1)
            s = (lo + (1 - lo) * torch.rand_like(u1)) * visible
            self._t_elem = s
            tpad = s.unsqueeze(-1)
        elif self.oneflow:
            # insertion time tau_i = t_max (1 - u1) <= t for existing elements; own clock s_i = (t - tau_i) / (1 - tau_i)
            tau = (self.t_max * (1 - u1)).clamp(max=t.view(-1, 1)) * (~given) # given elements: tau = 0
            s = ((t.view(-1, 1) - tau) / (1 - tau)).clamp(0, 1) * visible
            self._t_elem = s
            tpad = s.unsqueeze(-1)
        vis = visible.unsqueeze(-1)
        xt = vis * (tpad * x1 + (1 - tpad) * x0)
        free = cmask[..., :4] if cmask is not None else torch.ones_like(xt)
        xt = free * xt + (1 - free) * x1
        yt = torch.where(visible, y1, torch.full_like(y1, self.mask_id))
        return xt, yt, exists, visible, x0, x1, y1, free

    def relation_loss(self, xt, vt, x1, y1, visible, s):
        '''
        Hinge on strict containment of the predicted clean boxes, for the (underlay, text) pairs that are contained in
        the data. Boxes are cx, cy, w, h in the preprocessed [-1, 1] space (an affine map, so containment is preserved;
        the margin is given in canvas units and scaled by 2). Pairs are weighted by the elements' clocks (a prediction at
        s ~ 0 is noise) and the loss is averaged over pairs.
        '''
        x1_hat = xt + (1 - s).unsqueeze(-1) * vt
        ltrb = lambda b: torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)
        p, g = ltrb(x1_hat), ltrb(x1)
        under = visible & (y1 == self.relation_underlay_id)
        text = visible & (y1 == self.relation_text_id)
        # ground-truth containment (B, S_under, S_text)
        inside = ((g[:, None, :, 0] >= g[:, :, None, 0]) & (g[:, None, :, 1] >= g[:, :, None, 1]) &
                  (g[:, None, :, 2] <= g[:, :, None, 2]) & (g[:, None, :, 3] <= g[:, :, None, 3]))
        pair = under[:, :, None] & text[:, None, :] & inside
        if not pair.any():
            return vt.sum() * 0
        m = 2 * self.relation_margin
        viol = (F.relu(m + p[:, :, None, 0] - p[:, None, :, 0]) + F.relu(m + p[:, :, None, 1] - p[:, None, :, 1]) +
                F.relu(m + p[:, None, :, 2] - p[:, :, None, 2]) + F.relu(m + p[:, None, :, 3] - p[:, :, None, 3]))
        w = (s[:, :, None] * s[:, None, :]).sqrt() * pair
        return (viol * w).sum() / w.sum().clamp(min=1e-6)

    def training_step(self, batch, batch_idx):
        t = torch.rand(batch['bbox'].shape[0], device=self.device)
        cmask = self.cond_mask(batch, self.cond) if self.cond != 'uncond' else None
        s_lo = None
        if self.cond == 'random5':      # refinement fifth: nearly clean layouts, the state the refinement task starts from
            q = t.shape[0] // 5
            t[3 * q:4 * q] = 0.9 + 0.1 * t[3 * q:4 * q]
            s_lo = torch.zeros_like(t); s_lo[3 * q:4 * q] = 0.9
        xt, yt, exists, visible, x0, x1, y1, free = self.sample_state(batch, t, cmask, s_lo)
        masked = exists & ~visible
        if self.cat_drop > 0:
            # classifier-free guidance training: hide every category of a layout w.p. cat_drop, so the
            # model also learns the category-unconditional velocity (the boxes stay visible)
            drop = (torch.rand(xt.shape[0], device=self.device) < self.cat_drop).unsqueeze(1)
            yt = torch.where(drop & visible, torch.full_like(yt, self.mask_id), yt)
        ctx_hide = None
        if batch.get('ctx') is not None and (self.ctx_token_drop > 0 or self.ctx_drop > 0):
            # hide a random subset of canvas tokens, and the whole canvas for some samples: the layout must not be
            # keyed on the exact canvas (the un-regularised model memorises the 48k training canvases)
            B_, L_ = batch['ctx'].shape[:2]
            ctx_hide = torch.rand(B_, L_, device=self.device) < self.ctx_token_drop
            ctx_hide |= (torch.rand(B_, 1, device=self.device) < self.ctx_drop)
        ret_hide = None
        if batch.get('ret') is not None and self.ret_drop > 0:
            ret_hide = torch.rand(xt.shape[0], device=self.device) < self.ret_drop
        vt, logits, h, ins_rate, extra = self(xt, yt, exists, t, batch.get('ctx'), cmask, self._t_elem if self.oneflow else None, ctx_hide, batch.get('ret'), ret_hide)

        vis = visible.unsqueeze(-1) * free           # no velocity target on given coordinates
        ut = x1 - x0
        flow_loss = self.loss_fcn(vis * vt, vis * ut)
        loss = flow_loss
        if self.relation_loss_weight > 0:
            rel_loss = self.relation_loss(xt, vt, x1, y1, visible, self._t_elem if self.oneflow else t.view(-1, 1).expand_as(visible))
            self.log('relation_loss', rel_loss, on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self.relation_loss_weight * rel_loss
        self.log('flow_loss', flow_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.add_loss:
            assert self.add_loss == 'geom_l1_loss'
            l1 = F.l1_loss(vis * vt, vis * ut)
            self.log(self.add_loss, l1, on_step=True, on_epoch=True, sync_dist=True)
            loss = loss + self.add_loss_weight * l1

        if masked.any() and not self.editflow:
            cat_loss = F.cross_entropy(logits[masked], y1[masked])
            log_pi, mu, sigma = self.model.unmask_geom(h[masked], y1[masked])
            if self.unmask_geom == 'gmm':
                geom_unmask_loss = gmm_nll(log_pi, mu, sigma, x1[masked]).mean()
            else:
                geom_unmask_loss = F.mse_loss(gmm_mean(log_pi, mu), x1[masked])
        else:
            cat_loss = geom_unmask_loss = logits.sum() * 0
        loss = loss + self.cat_loss_weight * cat_loss + self.geom_unmask_weight * geom_unmask_loss
        if not self.editflow:
            self.log('cat_loss', cat_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('geom_unmask_loss', geom_unmask_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.editflow:
            ins_loss, val_loss = self.editflow_losses(batch, exists, x1, y1, h, ins_rate, extra)
            loss = loss + self.ins_loss_weight * ins_loss + val_loss
            self.log('ins_loss', ins_loss, on_step=True, on_epoch=True, sync_dist=True)
        elif self.insertion:
            missing = (batch['mask'].squeeze(-1).sum(1) - exists.sum(1)).float()
            rate = ins_rate.clamp(min=1e-6)
            ins_loss = (rate - missing * rate.log()).mean()
            loss = loss + self.ins_loss_weight * ins_loss
            self.log('ins_loss', ins_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log('ins_mae', (ins_rate - missing).abs().mean(), on_step=False, on_epoch=True, sync_dist=True)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def editflow_parents(self, active, exists, x1):
        '''
        The set analogue of Edit Flows' alignment: every missing element (active & ~exists) is assigned to
        its nearest existing element by clean box centre as the "parent" that will insert it; layouts
        with no existing element assign to the global token (parent index S).
        Returns parent (B,S) long (S = global), valid only where missing.
        '''
        B, S = active.shape
        d = torch.cdist(x1[..., :2], x1[..., :2])                                   # (B,S,S)
        d = d.masked_fill(~exists.unsqueeze(1), float('inf'))                      # only existing elements can be parents
        parent = d.argmin(-1)
        parent = torch.where(exists.any(1, keepdim=True), parent, torch.full_like(parent, S))
        return parent

    def editflow_losses(self, batch, exists, x1, y1, h, ins_rate, extra):
        '''Per-parent Poisson rate loss + class / box NLL of each missing element under its parent's heads.'''
        active = batch['mask'].squeeze(-1)
        missing = active & ~exists
        B, S = active.shape
        parent = self.editflow_parents(active, exists, x1)
        # rate targets: number of children per existing element, per global token
        tgt = torch.zeros(B, S + 1, device=self.device)
        tgt.scatter_add_(1, torch.where(missing, parent, torch.full_like(parent, S)), missing.float())
        tgt[:, S] = (missing & (parent == S)).sum(1).float()
        rates = torch.cat([extra['elem_rate'], ins_rate.unsqueeze(1)], 1).clamp(min=1e-6)      # (B,S+1)
        live = torch.cat([exists, torch.ones(B, 1, dtype=torch.bool, device=self.device)], 1)
        ins_loss = ((rates - tgt * rates.log()) * live).sum() / live.sum()
        self.log('ins_mae', ((rates * live).sum(1) - missing.sum(1).float()).abs().mean(), on_step=False, on_epoch=True, sync_dist=True)
        if not missing.any():
            return ins_loss, ins_loss * 0
        # value heads of the parent predict the child's class and clean box
        hp = torch.cat([h, extra['h_glob'].unsqueeze(1)], 1)                          # (B,S+1,d)
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)[missing]
        hpar = hp[b_idx, parent[missing]]
        cat_loss = F.cross_entropy(self.model.cat_head(hpar), y1[missing])
        self.log('cat_loss', cat_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.oneflow:                  # the child's box starts from pure noise: nothing to predict at insertion
            return ins_loss, self.cat_loss_weight * cat_loss
        log_pi, mu, sigma = self.model.unmask_geom(hpar, y1[missing])
        geom_loss = gmm_nll(log_pi, mu, sigma, x1[missing]).mean() if self.unmask_geom == 'gmm' else F.mse_loss(gmm_mean(log_pi, mu), x1[missing])
        self.log('geom_unmask_loss', geom_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return ins_loss, self.cat_loss_weight * cat_loss + self.geom_unmask_weight * geom_loss

    # Sampler knobs, not hyperparameters: the defaults reproduce the sampler the reported numbers
    # were produced with; test.py overrides them with `+sampling.<key>=<value>`.
    DEFAULT_SAMPLING = dict(
        gmm_temp=1.0,      # scale on the mixture component's sigma when drawing the clean box at reveal
        reveal_eps=1.0,    # scale on the fresh noise mixed into the revealed box, t x_1 + (1-t) eps
        solver='euler',    # 'heun': second-order steps once everything is revealed (kappa = 1)
        cfg_w=1.0,         # classifier-free guidance on the categories: v_u + w (v_c - v_u); 1 = off
        snap_grid=0,       # > 0: round the final ltrb edges to multiples of 1/snap_grid
        clock='sync',      # oneflow clock policy: 'sync' (all boxes clean at t = 1, ds = dt / (1 - tau)) | 'unit'
                           # (ds = dt, the run continues past t = 1 until every box is clean, OneFlow-style) |
                           # 'insert_first' (clocks frozen until kappa = 1, then everything is denoised together)
    )

    @torch.no_grad()
    def editflow_insert(self, x, y, exists, h, extra, ins_rate, p, t_next, s, tau):
        '''
        Edit-Flow-style insertion step: each existing element (and the global token) inserts
        Poisson(rate * p) new elements, each arriving with a class sampled from the parent's category
        head and a box from the parent's mixture head, noised to t_next. New elements fill free slots.
        '''
        B, S = exists.shape
        rates = torch.cat([extra['elem_rate'] * exists, ins_rate.unsqueeze(1)], 1)          # (B,S+1), global last
        n = torch.poisson(rates * p).long()
        free = S - exists.sum(1)
        # cap the total per layout at the free slots: drop excess children (global ones first, then by index)
        n_total = n.sum(1)
        if (n_total > free).any():
            for b in torch.nonzero(n_total > free).flatten().tolist():
                over = int(n_total[b] - free[b])
                for j in [S] + list(range(S)):
                    take = min(over, int(n[b, j])); n[b, j] -= take; over -= take
                    if over == 0:
                        break
        if n.sum() == 0:
            return x, y, exists, tau
        b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, S + 1)
        parent = torch.arange(S + 1, device=x.device).unsqueeze(0).expand(B, S + 1)
        b_new, p_new = b_idx.repeat_interleave(n.flatten()), parent.repeat_interleave(n.flatten())   # one row per child
        hp = torch.cat([h, extra['h_glob'].unsqueeze(1)], 1)[b_new, p_new]
        logits = self.model.cat_head(hp); logits[:, 0] = float('-inf')
        cls = torch.distributions.Categorical(logits=logits).sample()
        if self.oneflow:                                                    # box from pure noise, own clock starts now
            xn = torch.randn(len(cls), self.geom_dim, device=x.device)
        else:
            log_pi, mu, sigma = self.model.unmask_geom(hp, cls)
            g1 = gmm_sample(log_pi, mu, sigma * s['gmm_temp']) if self.unmask_geom == 'gmm' else gmm_mean(log_pi, mu)
            xn = t_next * g1 + (1 - t_next) * s['reveal_eps'] * torch.randn_like(g1)
        # k-th child of layout b goes to the k-th free slot of b
        first = torch.zeros(B, dtype=torch.long, device=x.device).scatter_add_(0, b_new, torch.ones_like(b_new)).cumsum(0) - torch.bincount(b_new, minlength=B)
        rank = torch.arange(len(b_new), device=x.device) - first[b_new]
        free_slots = torch.nonzero(~exists)                                                   # (n_free, 2) sorted by (b, slot)
        free_first = torch.zeros(B, dtype=torch.long, device=x.device).scatter_add_(0, free_slots[:, 0], torch.ones_like(free_slots[:, 0])).cumsum(0) - torch.bincount(free_slots[:, 0], minlength=B)
        slot = free_slots[free_first[b_new] + rank, 1]
        x[b_new, slot] = xn; y[b_new, slot] = cls; exists[b_new, slot] = True
        if self.oneflow:
            tau[b_new, slot] = t_next
        return x, y, exists, tau

    def velocity(self, x, y, exists, t, ctx=None, cmask=None, ret=None):
        v = self(x, y, exists, t, ctx, cmask, ret=ret)[0]
        w = self.sampling['cfg_w']
        if w != 1.0:
            # "unconditional" = same boxes, every category hidden behind the mask token
            v_u = self(x, torch.where(exists, torch.full_like(y, self.mask_id), y), exists, t, ctx, cmask, ret=ret)[0]
            v = v_u + w * (v - v_u)
        return v

    @torch.no_grad()
    def inference(self, batch, task='uncond', given_length=False, t_start=0.0, renoise=False):
        '''
        Returns (bbox, label, pad_mask). task: uncond | cat_cond | size_cond | elem_compl | refinement,
        with the given values read from the batch. With insertion the number of elements is generated
        unless given_length (LayoutFlow's protocol: the remaining slots start as masked elements).
        refinement: every element is given with its (noisy) box and integrated from t_start (upstream
        uses 0.97) to 1, i.e. the flow is used as a denoiser of the batch's layout.
        renoise: put the batch's boxes on the path first, x = t_start x + (1 - t_start) eps (SDEdit-style),
        so that starting at e.g. t_max is on-distribution.
        '''
        B, S = batch['type'].shape
        dev = batch['bbox'].device
        active = batch['mask'].squeeze(-1)
        given_length = given_length or task in ('cat_cond', 'size_cond', 'refinement')   # the whole element set is given
        if task == 'refinement':
            t_start = t_start or 0.97
        cmask = self.cond_mask(batch, task)
        given = active & (cmask[..., 4] == 0)
        held = (1 - cmask[..., :4]) * given.unsqueeze(-1)              # coordinates pinned to the batch values
        x1 = self.sampler.preprocess(batch['bbox'])
        exists = active.clone() if (not self.insertion or given_length) else given.clone()
        # a given element is visible from t = 0: given coordinates clean, the rest at pure noise
        x = torch.zeros(B, S, self.geom_dim, device=dev)
        if task == 'refinement':
            x = given.unsqueeze(-1) * x1                                 # start from the batch's (noisy) boxes
            if renoise:
                x = given.unsqueeze(-1) * (t_start * x1 + (1 - t_start) * torch.randn_like(x))
        elif given.any():
            x = given.unsqueeze(-1) * (held * x1 + (1 - held) * torch.randn_like(x))
        y = torch.where(given, batch['type'].long(), torch.full((B, S), self.mask_id, dtype=torch.long, device=dev))
        s = self.sampling
        ctx, ret = batch.get('ctx'), batch.get('ret')
        tau = torch.zeros(B, S, device=dev)                                  # oneflow: insertion time per slot
        clock = s['clock'] if self.oneflow else None
        # per-element clocks (oneflow): given elements start at t_start on their own clock
        sc = torch.zeros(B, S, device=dev)
        if clock == 'unit':
            sc = t_start * given.float()
        elif clock == 'insert_first':
            sc = max(0.0, (t_start - self.t_max) / (1 - self.t_max)) * given.float()

        N = self.inference_steps
        dt = (1.0 - t_start) / N
        # unit-rate clocks: a box inserted at tau needs 1/dt more steps, so the run goes on past t = 1
        N_tot = N + (math.ceil(self.t_max / dt) if clock == 'unit' and not given_length else 0)
        for i in range(N_tot):
            t_i = t_start + i * dt
            t = torch.full((B,), min(t_i, 1.0), device=dev)
            t_next = t_start + (i + 1) * dt
            k, k_next = min(t_i / self.t_max, 1.0), min(t_next / self.t_max, 1.0)
            # exact per-step jump probability of the hazard kappa'/(1-kappa)
            p = 1.0 if (k_next >= 1.0 or i == N - 1) else (k_next - k) / (1 - k)
            if i >= N:
                p = 0.0                                                      # tail: no more insertions

            if clock == 'sync':
                sc = ((t_i - tau) / (1 - tau)).clamp(0, 1) * exists
                ds = (dt / (1 - tau)) * exists
            elif clock == 'unit':
                ds = (1 - sc).clamp(max=dt) * exists
            elif clock == 'insert_first':
                ds = (1 - sc).clamp(max=(0.0 if t_next <= self.t_max else dt / (1 - self.t_max))) * exists
            t_elem = sc * exists if self.oneflow else None
            v, logits, h, ins_rate, extra = self(x, y, exists, t, ctx, cmask, t_elem, ret=ret)
            masked = exists & (y == self.mask_id)
            visible = exists & ~masked
            vis = visible.unsqueeze(-1)
            dt_elem = ds.unsqueeze(-1) if self.oneflow else dt              # own clock
            if s['cfg_w'] != 1.0:
                v = self.velocity(x, y, exists, t, ctx, cmask, ret)

            # denoise (Euler; Heun once the layout is complete and nothing can change discontinuously)
            if s['solver'] == 'heun' and k >= 1.0 and i < N - 1:
                x_e = torch.where(vis, x + v * dt, x)
                v2 = self.velocity(x_e, y, exists, t + dt, ctx, cmask, ret)
                x = torch.where(vis, x + 0.5 * (v + v2) * dt, x)
            else:
                x = torch.where(vis, x + v * dt_elem, x)
            if self.oneflow:
                sc = ((sc + ds).clamp(max=1.0)) * exists
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
            if self.editflow and not given_length and 0.0 < p < 1.0:
                x, y, exists, tau = self.editflow_insert(x, y, exists, h, extra, ins_rate, p, t_next, s, tau)
            elif self.insertion and not given_length and p < 1.0:
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
