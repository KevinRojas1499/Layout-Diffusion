'''
The swappable per-element transformer backbone. This is the piece to replace first
when debugging why our own pipeline's numbers diverge from LayoutFlow's: it predicts
the flow-matching vector field for every (geometry, category) element jointly,
conditioned on which entries are held fixed (`cond_flags`) and on the diffusion time.

Trimmed from upstream's LayoutDMBackbone: that file also supported a "discrete"
category encoding and two alternate sequence layouts (`seq`, `seq_cond`) that are
never selected by LayoutFlow's own config (`attr_encoding=AnalogBit`,
`seq_type=stacked`) — dropped here to keep the forward pass to one path.
'''
import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import pack


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.pe[:, :x.shape[1]])


class AdaLayerNorm(nn.Module):
    '''Layer norm modulated by the diffusion/flow timestep (adaptive scale + shift).'''

    def __init__(self, n_embd):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(1, n_embd // 2), nn.ReLU(), nn.Linear(n_embd // 2, n_embd))
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        # timestep (B,) = one time per layout, or (B, L) = one per token (decoupled per-element clocks)
        emb = self.linear(self.silu(self.emb(timestep.unsqueeze(-1))))
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class Block(nn.Module):
    '''Pre-norm transformer block (self-attn + MLP) with AdaLayerNorm on the first norm.'''

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, cross=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # optional cross-attention to a context sequence (canvas tokens), kept out of the self-attention
        self.cross = cross
        if cross:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm_c = nn.LayerNorm(d_model, eps=1e-5)
            self.dropout_c = nn.Dropout(dropout)
            self.cross_gate = nn.Parameter(torch.zeros(1))     # tanh-gated, zero-initialised: starts as the context-free model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, timestep, key_padding_mask=None, ctx=None, ctx_padding_mask=None, attn_bias=None):
        x = self.norm1(x, timestep)
        x = x + self.dropout1(self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_bias, need_weights=False)[0])
        if self.cross and ctx is not None:
            q = self.norm_c(x)
            x = x + torch.tanh(self.cross_gate) * self.dropout_c(self.cross_attn(q, ctx, ctx, key_padding_mask=ctx_padding_mask, need_weights=False)[0])
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x, timestep, key_padding_mask=None, ctx=None, ctx_padding_mask=None, attn_bias=None):
        for layer in self.layers:
            x = layer(x, timestep, key_padding_mask=key_padding_mask, ctx=ctx, ctx_padding_mask=ctx_padding_mask, attn_bias=attn_bias)
        return self.norm(x)


class Backbone(nn.Module):
    '''
    Embeds geometry (4-dim bbox) and category (AnalogBit-encoded) per element,
    adds a per-position "is this conditioned on / held fixed" embedding, stacks
    both into one token sequence, runs a transformer encoder, and projects back
    down to (geometry, category) for the vector-field prediction.
    '''

    def __init__(self, latent_dim=128, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, num_cat=6, num_bits=None):
        super().__init__()
        self.geom_dim = 4
        num_bits = num_bits if num_bits is not None else int(math.ceil(math.log2(num_cat)))

        # geom_cond = how many of the 4 geometry dims are free (0-4); attr_cond is
        # a plain 0/1 flag. Both index into the same shared table (matches
        # upstream, which does the identical sum-then-lookup with one table).
        self.cond_enc = nn.Embedding(6, latent_dim)
        self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        self.type_embed = nn.Linear(num_bits, latent_dim)
        self.elem_embed = nn.Linear(2 * latent_dim, d_model)

        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.linear = nn.Linear(d_model, self.geom_dim + num_bits)

    def forward(self, geom: Tensor, attr: Tensor, cond_flags: Tensor, t: Tensor) -> Tensor:
        '''
        geom (B, S, 4), attr (B, S, num_bits), cond_flags (B, S, 4+num_bits) all 0/1
        (0 = held fixed / conditioned on, 1 = free / to be predicted), t (B,)
        '''
        geom_cond = cond_flags[:, :, :self.geom_dim].sum(-1)
        attr_cond = cond_flags[:, :, -1]
        geom = self.geom_embed(geom) + self.cond_enc(geom_cond)
        attr = self.type_embed(attr) + self.cond_enc(attr_cond)
        x, ps = pack([geom, attr], 'b s *')

        x = self.elem_embed(x)
        x = self.transformer(x, timestep=t)
        return self.linear(x)
