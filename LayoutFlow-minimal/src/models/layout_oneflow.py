'''
OneFlow (Nguyen, Havasi, Berrada, Zettlemoyer, Chen 2025, arXiv:2510.03506) on layouts: the mixed-modal
variable-length baseline, following the paper's Sections 2.1-2.3, Appendix D.1 and Algorithms 1-2.

A layout is a *sequence* of elements in the dataset's order (learned positions + BOS, as in the Edit Flows
baseline). Text tokens = the classes; every element is OneFlow's `<|image|>` mechanism applied per token: the
class is the discrete token value and the box is its continuous latent, inserted as N(0, I) at t_box = 0 and
flow-matched on its own clock (eq. 10).
- Insertion-only Edit Flow with the linear schedule kappa_t = t (Sec. 2.1; OneFlow has no delete / substitute).
- Per-gap heads (Sec. 2.1.1-2.1.3): pi_i (probability of zero missing tokens, BCE), lambda_nonzero (Poisson
  NLL on k_i > 0, eq. 4-5) and the bag-of-tokens distribution Q_i (cross-entropy over the missing tokens, eq. 6);
  the ratio kappa'/(1-kappa) is factored out of the rates and *no time value is fed to the network for the
  insertion heads* (Sec. 2.1.1); the loss is unweighted (eq. 7).
- Interleaved time schedule (Sec. 2.3.2, App. D.1): tau_text ~ U(0, 2), t_text = clip(tau_text); per element
  tau_box = tau_text - kappa^{-1}(u), u ~ U(0, 1); the element is missing iff tau_box < 0 (it enters the insertion
  targets), otherwise present with t_box = clip(tau_box) and the flow-matching loss (eq. 9) at t_box.
- Sampling = Algorithms 1-2: Euler with step dt; each present box moves min(1 - t_box, dt) * v; while t_text < 1
  every gap inserts one token w.p. Bernoulli(1 - pi_i) and Bernoulli(dt kappa'/(1-kappa) lambda_i); the loop
  continues until t_text = 1 and every box has t_box = 1 (so it runs to t = 2 at unit rate).
Nothing from LayoutFlowVarLen is used (no set structure, no reveal schedule, no synchronised clocks).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseGenModel
from src.fid import FID_score


class OneFlowTransformer(nn.Module):
    '''Class token + box latent + learned position (+ BOS); per-token box time only (the insertion heads see no time).'''

    def __init__(self, num_cls, max_len, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, geom_dim=4):
        super().__init__()
        self.BOS = num_cls
        self.tok = nn.Embedding(num_cls + 1, d_model)
        self.box = nn.Linear(geom_dim, d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.time = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pi = nn.Linear(d_model, 1)             # P(k_i = 0)
        self.lam = nn.Linear(d_model, 1)            # lambda_nonzero
        self.Q = nn.Linear(d_model, num_cls)        # bag of tokens
        self.v = nn.Linear(d_model, geom_dim)       # box velocity

    def forward(self, cls, box, t_box, pad_mask):
        '''cls (B,L) incl. BOS at 0, box (B,L,4), t_box (B,L), pad_mask True = pad -> pi (B,L), lam (B,L), logQ (B,L,C), v (B,L,4)'''
        B, L = cls.shape
        x = self.tok(cls) + self.box(box) + self.pos(torch.arange(L, device=cls.device)).unsqueeze(0) + self.time(t_box.unsqueeze(-1))
        h = self.norm(self.layers(x, src_key_padding_mask=pad_mask))
        return torch.sigmoid(self.pi(h)).squeeze(-1), F.softplus(self.lam(h)).squeeze(-1), F.log_softmax(self.Q(h), -1), self.v(h)


class LayoutOneFlow(BaseGenModel):
    def __init__(self, backbone_model=None, sampler=None, optimizer=None, scheduler=None, pretrained_dir='./pretrained',
                 format='xywh', fid_calc_every_n=20, dataset='RICO', num_cat=6, max_len=20, inference_steps=50,
                 d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, box_loss_weight=1.0,
                 ralf_cache=None, vis_dir=None):
        self.format = format
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = 'uncond'
        if not fid_calc_every_n:
            fid_model = None
        elif ralf_cache:
            from src.fid_ralf import RalfFID
            fid_model = RalfFID(dataset, ralf_cache, calc_every_n=fid_calc_every_n)
        else:
            fid_model = FID_score(dataset, pretrained_dir, calc_every_n=fid_calc_every_n)
        super().__init__(optimizer=optimizer, scheduler=scheduler, dataset=dataset, fid_model=fid_model, vis_dir=vis_dir)
        self.num_cat, self.max_len, self.inference_steps = num_cat, max_len, inference_steps
        self.C = num_cat - 1                        # dataset categories are 1..num_cat-1 (0 = pad id)
        self.sampler = sampler                      # the same [0,1] -> [-1,1] box preprocessing as LayoutFlowVarLen
        self.box_loss_weight = box_loss_weight
        self.loss_fcn = nn.MSELoss()                # only for base.validation_step's val_loss diagnostic
        self.model = OneFlowTransformer(self.C, max_len, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.save_hyperparameters(ignore=['backbone_model', 'sampler'])

    # linear kappa: kappa(t) = t, kappa^{-1}(u) = u, hazard kappa'/(1-kappa) = 1/(1-t)
    def _with_bos(self, cls, box, t_box, mask):
        B = cls.shape[0]
        dev = cls.device
        one = torch.ones(B, 1, dtype=torch.bool, device=dev)
        return (torch.cat([torch.full((B, 1), self.model.BOS, dtype=torch.long, device=dev), cls], 1),
                torch.cat([torch.zeros(B, 1, 4, device=dev), box], 1),
                torch.cat([torch.ones(B, 1, device=dev), t_box], 1), torch.cat([one, mask], 1))

    def training_step(self, batch, batch_idx):
        m1 = batch['mask'].squeeze(-1)
        B, S = m1.shape
        cls1 = (batch['type'].long() - 1).clamp(min=0)
        x1 = m1.unsqueeze(-1) * self.sampler.preprocess(batch['bbox'])
        # interleaved time schedule (eqs. 27-29): extended text time, per-element box time
        tau_text = 2 * torch.rand(B, device=self.device)
        tau_box = tau_text.view(-1, 1) - torch.rand(B, S, device=self.device)          # kappa^{-1}(u) = u
        present = m1 & (tau_box >= 0)
        t_box = tau_box.clamp(0, 1)
        x0 = torch.randn_like(x1)
        xt = t_box.unsqueeze(-1) * x1 + (1 - t_box.unsqueeze(-1)) * x0
        # pack the present elements (order kept); gap i = right of the i-th present token, BOS = gap 0
        order = torch.argsort((~present).int(), dim=1, stable=True)
        n_t = present.sum(1)
        live = torch.arange(S, device=self.device).unsqueeze(0) < n_t.unsqueeze(1)
        g = lambda a: torch.gather(a, 1, order if a.dim() == 2 else order.unsqueeze(-1).expand_as(a))
        cls_t, box_t, tb_t = g(cls1) * live, g(xt) * live.unsqueeze(-1), g(t_box) * live
        cls_b, box_b, tb_b, mask_b = self._with_bos(cls_t, box_t, tb_t, live)
        pi, lam, logQ, v = self.model(cls_b, box_b, tb_b, ~mask_b)
        # insertion targets: a missing element j is pending in the gap after the number of present elements before j
        missing = m1 & ~present
        pos = present.int().cumsum(1) - present.int()
        k = torch.zeros(B, S + 1, device=self.device).scatter_add_(1, pos, missing.float())       # (B, gaps)
        n_gap = (n_t + 1).float()
        l_bce = F.binary_cross_entropy(pi.clamp(1e-6, 1 - 1e-6), (k == 0).float(), reduction='none') * mask_b
        l_pois = (lam - k * lam.clamp(min=1e-6).log()) * (k > 0) * mask_b
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, S)[missing]
        l_tok = torch.zeros(B, device=self.device).index_add_(0, b_idx, -logQ[b_idx, pos[missing], cls1[missing]])
        loss_text = ((l_bce + l_pois).sum(1) + l_tok) / n_gap                                       # eq. 7
        # flow matching on the present boxes at their own times (eq. 9)
        u1 = g(x1 - x0) * live.unsqueeze(-1)
        loss_box = (((v[:, 1:] - u1) ** 2).sum(-1) * live).sum() / live.sum().clamp(min=1)
        loss = loss_text.mean() + self.box_loss_weight * loss_box
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('text_loss', loss_text.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('flow_loss', loss_box, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('ins_mae', (((1 - pi) * lam * mask_b).sum(1) - missing.sum(1).float()).abs().mean(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def inference(self, batch):
        '''Algorithms 1-2. Returns (bbox, label, pad_mask) with the generated elements packed to the front.'''
        B = batch['type'].shape[0]
        dev = self.device
        S = self.max_len
        cls = torch.zeros(B, S, dtype=torch.long, device=dev)
        box = torch.zeros(B, S, 4, device=dev)
        tb = torch.zeros(B, S, device=dev)
        n = torch.zeros(B, dtype=torch.long, device=dev)
        dt = 1.0 / self.inference_steps
        t = 0.0
        while t < 1.0 - 1e-9 or bool(((tb < 1) & (torch.arange(S, device=dev).unsqueeze(0) < n.unsqueeze(1))).any()):
            live = torch.arange(S, device=dev).unsqueeze(0) < n.unsqueeze(1)
            cls_b, box_b, tb_b, mask_b = self._with_bos(cls, box, tb, live)
            pi, lam, logQ, v = self.model(cls_b, box_b, tb_b, ~mask_b)
            # boxes: one Euler step on their own clocks
            d = (1 - tb).clamp(min=0, max=dt) * live
            box = box + d.unsqueeze(-1) * v[:, 1:]
            tb = tb + d
            # insertions while t_text < 1: one token per gap, after the token at that position (BOS = front)
            dt_text = min(1 - t, dt)
            if dt_text > 0:
                p_lam = (dt_text / (1 - t) * lam).clamp(max=1.0) if t < 1 - 1e-9 else torch.ones_like(lam)
                ins = (torch.rand_like(pi) < (1 - pi)) & (torch.rand_like(pi) < p_lam) & mask_b
                tok = torch.distributions.Categorical(logits=logQ).sample()
                keep = mask_b.clone(); keep[:, 0] = False
                n_ins_before = ins.long().cumsum(1) - ins.long()
                n_keep_before = keep.long().cumsum(1) - keep.long()
                pos_keep = n_keep_before + n_ins_before
                pos_ins = n_keep_before + keep.long() + n_ins_before
                new_cls = torch.zeros(B, S, dtype=torch.long, device=dev); new_box = torch.zeros(B, S, 4, device=dev); new_tb = torch.zeros(B, S, device=dev)
                bi = torch.arange(B, device=dev).unsqueeze(1).expand_as(cls_b)
                ok_k = keep & (pos_keep < S); ok_i = ins & (pos_ins < S)                    # capped at max_len
                new_cls[bi[ok_k], pos_keep[ok_k]] = cls_b[ok_k]; new_box[bi[ok_k], pos_keep[ok_k]] = box_b[ok_k]; new_tb[bi[ok_k], pos_keep[ok_k]] = tb_b[ok_k]
                new_cls[bi[ok_i], pos_ins[ok_i]] = tok[ok_i]
                new_box[bi[ok_i], pos_ins[ok_i]] = torch.randn(int(ok_i.sum()), 4, device=dev)   # inserted at t_box = 0 (eq. 10)
                cls, box, tb = new_cls, new_box, new_tb
                n = (ok_k.sum(1) + ok_i.sum(1)).clamp(max=S)
                t += dt_text
        live = torch.arange(S, device=dev).unsqueeze(0) < n.unsqueeze(1)
        geom = live.unsqueeze(-1) * self.sampler.preprocess(box, reverse=True)
        label = torch.where(live, cls + 1, torch.zeros_like(cls))
        return geom, label, live
