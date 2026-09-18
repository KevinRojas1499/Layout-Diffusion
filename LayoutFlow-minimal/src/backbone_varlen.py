import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.backbone import Block, TransformerEncoder


class VarLenBackbone(nn.Module):
    '''
    Backbone for whole-element masking / insertion (src.models.layout_flow_varlen).
    Same embedding + AdaLN transformer as DiscreteBackbone, with three changes:

      * slots that do not exist yet are hidden with a key-padding mask instead of
        being fed as visible pad tokens, so the final length cannot leak;
      * a learned global token is prepended. It is always attended to (the layout
        may be empty at t=0) and carries the insertion head;
      * besides the velocity (denoise) and category-logits (unmask) heads there is
        an unmask-geometry head: a K-component diagonal Gaussian mixture over the
        *clean* box of a masked element, conditioned on its class.

    The transformer has no positional encoding -- a layout is a set -- so "where
    to insert" carries no information and the insertion head is a single rate per
    layout: the expected number of elements still missing.
    '''

    def __init__(self, latent_dim=128, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, num_cat=6, gmm_components=16, sigma_min=0.01, ctx_dim=0, ctx_len=64,
                 cond_input=False, elem_rates=False):
        super().__init__()
        self.geom_dim = 4
        self.mask_id = num_cat
        self.K = gmm_components
        self.sigma_min = sigma_min

        self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        self.type_embed = nn.Embedding(num_cat + 1, latent_dim)      # num_cat = [MASK]
        self.elem_embed = nn.Linear(2 * latent_dim, d_model)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # conditional training (cond=random4): the network must see which elements / coordinates are given,
        # otherwise "all visible because given" and "all visible because insertion is done" are indistinguishable
        self.cond_input = cond_input
        if cond_input:
            self.cond_embed = nn.Linear(5, d_model)
        # content-aware: canvas feature tokens (precomputed, frozen) prepended as always-visible context;
        # they carry a learned position embedding because, unlike the elements, the grid cells are ordered
        self.ctx_dim = ctx_dim
        if ctx_dim:
            self.ctx_embed = nn.Linear(ctx_dim, d_model)
            self.ctx_pos = nn.Parameter(torch.randn(1, ctx_len, d_model) * 0.02)

        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        self.geom_head = nn.Linear(d_model, self.geom_dim)
        self.cat_head = nn.Linear(d_model, num_cat)
        self.ins_head = nn.Linear(d_model, 1)
        # Edit-Flow-style baseline: an insertion rate per existing element (the global one covers the empty layout)
        self.elem_rates = elem_rates
        if elem_rates:
            self.elem_ins_head = nn.Linear(d_model, 1)
        self.cls_cond = nn.Embedding(num_cat, d_model)
        self.gmm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, self.K * (1 + 2 * self.geom_dim)))

    def forward(self, geom: Tensor, cat: Tensor, exists: Tensor, t: Tensor, ctx: Tensor = None, cmask: Tensor = None,
                t_elem: Tensor = None):
        '''
        geom (B,S,4), cat (B,S) long, exists (B,S) bool, t (B,), ctx (B,L,ctx_dim) canvas tokens or None,
        cmask (B,S,5) conditioning mask over [x,y,w,h,cat] (1 = free, 0 = given) or None
        t_elem (B,S) per-element clock (OneFlow-style baseline) or None: the global/canvas tokens then use t
        -> velocity (B,S,4), logits (B,S,num_cat), h (B,S,d_model), ins_rate (B,), extra {h_glob (B,d_model), elem_rate (B,S) or None}
        '''
        x = self.elem_embed(torch.cat([self.geom_embed(geom), self.type_embed(cat)], dim=-1))
        if self.cond_input:
            given = torch.zeros(*cat.shape, 5, device=x.device) if cmask is None else 1 - cmask
            x = x + self.cond_embed(given)
        pre = [self.global_token.expand(x.shape[0], -1, -1)]
        if self.ctx_dim:
            pre.append(self.ctx_embed(ctx) + self.ctx_pos)
        n_pre = sum(p.shape[1] for p in pre)
        x = torch.cat(pre + [x], dim=1)
        hidden = torch.cat([torch.zeros(x.shape[0], n_pre, dtype=torch.bool, device=x.device), ~exists], dim=1)
        if t_elem is not None:
            t = torch.cat([t.unsqueeze(1).expand(-1, n_pre), t_elem], dim=1)
        h = self.transformer(x, timestep=t, key_padding_mask=hidden)
        h_glob, h = h[:, 0], h[:, n_pre:]
        ins_rate = F.softplus(self.ins_head(h_glob)).squeeze(-1)
        extra = {'h_glob': h_glob, 'elem_rate': F.softplus(self.elem_ins_head(h)).squeeze(-1) if self.elem_rates else None}
        return self.geom_head(h), self.cat_head(h), h, ins_rate, extra

    def unmask_geom(self, h: Tensor, cls: Tensor):
        '''h (N,d_model), cls (N,) -> mixture (log_pi (N,K), mu (N,K,4), sigma (N,K,4)) over the clean box.'''
        out = self.gmm_head(h + self.cls_cond(cls))
        log_pi = F.log_softmax(out[:, :self.K], dim=-1)
        mu, raw = out[:, self.K:].view(-1, self.K, 2 * self.geom_dim).chunk(2, dim=-1)
        return log_pi, mu, F.softplus(raw) + self.sigma_min


def gmm_nll(log_pi: Tensor, mu: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
    '''Negative log-likelihood of target (N,4) under the diagonal mixture, per element.'''
    z = (target.unsqueeze(1) - mu) / sigma
    log_comp = (-0.5 * z ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
    return -torch.logsumexp(log_pi + log_comp, dim=-1)


def gmm_sample(log_pi: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    k = torch.distributions.Categorical(logits=log_pi).sample()
    idx = k.view(-1, 1, 1).expand(-1, 1, mu.shape[-1])
    mu_k, sigma_k = mu.gather(1, idx).squeeze(1), sigma.gather(1, idx).squeeze(1)
    return mu_k + sigma_k * torch.randn_like(mu_k)


def gmm_mean(log_pi: Tensor, mu: Tensor) -> Tensor:
    return (log_pi.exp().unsqueeze(-1) * mu).sum(1)
