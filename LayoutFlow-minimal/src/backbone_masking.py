'''
Backbone for the masking-diffusion (continuous_masking_interpolant.py) path:
geometry and category are already concatenated into one continuous vector by
the caller (see ContinuousMaskingModel), so there's only one token stream per
element -- no joint attention between separate modalities needed the way
src/backbone.py / src/backbone_mmdit.py have it, since there's nothing to
join anymore.

Reuses src/backbone.py's Block/TransformerEncoder/AdaLayerNorm as-is (they're
already generic single-stream, scalar-timestep building blocks, not tied to
that file's geom/attr split at the block level).

`use_pos_embed`: this backbone originally had no positional signal at all --
matching src/backbone.py's Backbone (`use_pos_enc: False` in LayoutFlow's own
config), which works fine for flow-matching. Masking diffusion leans much
more heavily on "infer this masked position from other revealed positions,"
though, so an explicit per-position identity may matter more here than it
does for Backbone. Opt-in, off by default so it's a clean single-variable
comparison against runs without it.
'''
import torch
from torch import nn, Tensor

from src.backbone import Block, TransformerEncoder


class MaskingBackbone(nn.Module):
    def __init__(self, x_dim, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, use_pos_embed=False, max_len=20):
        super().__init__()
        self.x_embed = nn.Linear(x_dim, d_model)
        # 0 = revealed, 1 = still masked -- the only per-position signal telling
        # the model which state a position is in (x itself is exactly zero for
        # both padding and masked positions, so this is not otherwise recoverable).
        self.mask_embed = nn.Embedding(2, d_model)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Embedding(max_len, d_model)

        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.linear = nn.Linear(d_model, x_dim)

    def forward(self, x: Tensor, is_masked: Tensor, t: Tensor) -> Tensor:
        '''x: (B, S, x_dim), is_masked: (B, S) bool, t: (B,).'''
        tok = self.x_embed(x) + self.mask_embed(is_masked.long())
        if self.use_pos_embed:
            pos_ids = torch.arange(x.shape[1], device=x.device)
            tok = tok + self.pos_embed(pos_ids)[None]
        tok = self.transformer(tok, timestep=t)
        return self.linear(tok)
