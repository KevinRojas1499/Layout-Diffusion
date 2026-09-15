'''
Alternative to src/backbone.py's Backbone: same forward(geom, attr, cond_flags, t)
contract, but the per-element joint modeling is done by our own project's MMDiT
joint-attention architecture (src/mmdit.py, ported from Layout-Diffusion's
models/mmdit.py / MMDiTQM9) instead of LayoutDM's single-stream transformer.

Geometry and category are treated as two separate modalities (each with their
own embedding, adaptive-layernorm time-conditioning, and output head) that
exchange information only through the shared joint-attention layers -- as
opposed to Backbone, which concatenates them into one token per element and
runs a single-stream transformer over that.

Since the two streams never get fused into one token the way Backbone's do,
something has to tell the model which geom-token and attr-token are the same
layout element. `pos_encoding` selects how:
  - 'rotary' (default): the mechanism MMDiTQM9 actually uses -- the same
    (parameter-free) rotary phase applied to both streams' queries/keys at
    each sequence index, via src/rotary.py. Doesn't require matching dims.
  - 'additive': one learned per-position embedding, added identically to both
    streams. Requires dim_modalities to match.
Both were added after diagnosing a first MMDiT run (no binding signal at all)
that plateaued far short of Backbone's FID. A head-to-head RICO run then
picked rotary over additive: faster and lower long-term convergence, at the
cost of a noisier FID curve early in training. See git log for both writeups.
'''
import math
import torch
from torch import nn, Tensor

from src.mmdit import MMDiT, FinalLayer, TimestepEmbedder
from src.rotary import Rotary


class MMDiTBackbone(nn.Module):
    def __init__(self, dim_modalities=(256, 256), dim_joint_attn=256, depth=4,
                 dim_head=64, heads=8, num_cat=6, num_bits=None, max_len=20,
                 pos_encoding='rotary'):
        super().__init__()
        assert pos_encoding in ('additive', 'rotary')
        self.pos_encoding = pos_encoding
        self.geom_dim = 4
        num_bits = num_bits if num_bits is not None else int(math.ceil(math.log2(num_cat)))
        dim_geom, dim_attr = dim_modalities

        # geom_cond: how many of the 4 geometry dims are free (0-4); attr_cond: 0/1
        # flag -- same conditioning signal Backbone uses, just embedded per-modality
        # instead of through one shared table.
        self.cond_enc_geom = nn.Embedding(6, dim_geom)
        self.cond_enc_attr = nn.Embedding(2, dim_attr)
        self.geom_embed = nn.Linear(self.geom_dim, dim_geom)
        self.attr_embed = nn.Linear(num_bits, dim_attr)

        if pos_encoding == 'additive':
            assert dim_geom == dim_attr, 'additive shared position embedding needs equal modality dims'
            self.pos_embed = nn.Embedding(max_len, dim_geom)
        else:
            # One shared instance reused for both streams: token i in geom and
            # token i in attr get the identical rotation, applied to q/k inside
            # JointAttention (dim_head-sized, independent of modality embedding dim).
            self.rotary = Rotary(dim=dim_head)

        self.time_embed_geom = TimestepEmbedder(dim_geom)
        self.time_embed_attr = TimestepEmbedder(dim_attr)

        self.mmdit = MMDiT(
            depth=depth, dim_modalities=(dim_geom, dim_attr), dim_conds=(dim_geom, dim_attr),
            dim_joint_attn=dim_joint_attn, dim_head=dim_head, heads=heads,
        )

        self.geom_out = FinalLayer(dim_geom, self.geom_dim)
        self.attr_out = FinalLayer(dim_attr, num_bits)

    def forward(self, geom: Tensor, attr: Tensor, cond_flags: Tensor, t: Tensor) -> Tensor:
        geom_cond = cond_flags[:, :, :self.geom_dim].sum(-1)
        attr_cond = cond_flags[:, :, -1]

        geom_tok = self.geom_embed(geom) + self.cond_enc_geom(geom_cond)
        attr_tok = self.attr_embed(attr) + self.cond_enc_attr(attr_cond)

        rotary_pos_emb = None
        if self.pos_encoding == 'additive':
            pos_ids = torch.arange(geom.shape[1], device=geom.device)
            pos = self.pos_embed(pos_ids)[None]  # (1, S, dim) -- broadcasts over batch
            geom_tok = geom_tok + pos
            attr_tok = attr_tok + pos
        else:
            cos, sin = self.rotary(geom_tok, seq_dim=1)
            rotary_pos_emb = ((cos, sin), (cos, sin))  # same phase for both streams

        t_geom = self.time_embed_geom(t)
        t_attr = self.time_embed_attr(t)

        geom_tok, attr_tok = self.mmdit(
            modality_tokens=(geom_tok, attr_tok), time_cond=(t_geom, t_attr),
            rotary_pos_emb=rotary_pos_emb,
        )

        geom_pred = self.geom_out(geom_tok, t_geom)
        attr_pred = self.attr_out(attr_tok, t_attr)
        return torch.cat([geom_pred, attr_pred], dim=-1)
