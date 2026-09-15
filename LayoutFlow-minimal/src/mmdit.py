'''
Joint-attention MMDiT core, ported from our own Layout-Diffusion project's
models/mmdit.py (same architecture used by MMDiTQM9) so it can be dropped into
LayoutFlow-minimal as an alternative to src/backbone.py's Backbone, for
single-variable NN-swap comparisons.

Trimmed relative to the source: dropped the rotary-position-embedding and
per-layer attention-bias plumbing (both optional there, `None` by default) --
LayoutFlow's own backbone doesn't use positional encoding either
(`use_pos_enc: False`), so this keeps the comparison apples-to-apples and
avoids pulling in the unrelated model/rotary.py dependency. Everything else
(JointAttention, adaptive layernorm time-conditioning, MMDiTBlock, MMDiT) is
architecturally the same.
'''
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange
from x_transformers.attend import Attend
from x_transformers import RMSNorm, FeedForward


def exists(v):
    return v is not None


class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class JointAttention(Module):
    '''Runs one shared attention over tokens from all modalities concatenated
    together, with separate per-modality qkv/output projections.'''

    def __init__(self, *, dim, dim_inputs, dim_head=64, heads=8, qk_rmsnorm=True, flash=True):
        super().__init__()
        dim_inner = dim_head * heads
        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs
        self.heads = heads

        self.to_qkv = ModuleList([nn.Linear(d, dim_inner * 3, bias=False) for d in dim_inputs])
        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h=heads, qkv=3)
        self.attend = Attend(flash=flash)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = ModuleList([nn.Linear(dim_inner, d, bias=False) for d in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in range(num_inputs)])

    def forward(self, inputs, masks=None):
        masks = masks or (None,) * self.num_inputs

        all_qkvs, all_masks = [], []
        for i, (x, mask, to_qkv) in enumerate(zip(inputs, masks, self.to_qkv)):
            qkv = self.split_heads(to_qkv(x))
            q, k, v = qkv
            if self.qk_rmsnorm:
                q = self.q_rmsnorms[i](q)
                k = self.k_rmsnorms[i](k)
            all_qkvs.append(torch.stack((q, k, v)))
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            all_masks.append(mask)

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        q, k, v = all_qkvs
        outs, *_ = self.attend(q, k, v, mask=all_masks)

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')
        return tuple(to_out(out) for out, to_out in zip(outs, self.to_out))


class AdaptiveLayerNorm(Module):
    '''LayerNorm modulated by a per-modality time-conditioning vector.'''

    def __init__(self, dim, dim_cond):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        cond_linear = nn.Linear(dim_cond, dim * 2)
        self.to_cond = nn.Sequential(Rearrange('b d -> b 1 d'), nn.SiLU(), cond_linear)
        nn.init.zeros_(cond_linear.weight)
        nn.init.constant_(cond_linear.bias[:dim], 1.)
        nn.init.zeros_(cond_linear.bias[dim:])

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return self.ln(x) * gamma + beta


class MMDiTBlock(Module):
    def __init__(self, *, dim_joint_attn, dim_modalities, dim_conds, dim_head=64, heads=8):
        super().__init__()
        self.num_modalities = len(dim_modalities)

        # Post-branch residual gates (one pair of scalars-per-channel per modality,
        # from a *separate* projection than the adaptive layernorms below).
        self.post_branch_gates = ModuleList([
            nn.Sequential(Rearrange('b d -> b 1 d'), nn.SiLU(), nn.Linear(dim_cond, dim_modality * 2))
            for dim_modality, dim_cond in zip(dim_modalities, dim_conds)
        ])
        for gate_proj in self.post_branch_gates:
            linear = gate_proj[-1]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, 1.)

        self.attn_layernorms = ModuleList([AdaptiveLayerNorm(d, c) for d, c in zip(dim_modalities, dim_conds)])
        self.joint_attn = JointAttention(dim=dim_joint_attn, dim_inputs=dim_modalities, dim_head=dim_head, heads=heads)
        self.ff_layernorms = ModuleList([AdaptiveLayerNorm(d, c) for d, c in zip(dim_modalities, dim_conds)])
        self.feedforwards = ModuleList([FeedForward(d) for d in dim_modalities])

    def forward(self, *, modality_tokens, time_cond, modality_masks=None):
        attn_gammas, ff_gammas = zip(*[
            gate_proj(cond).chunk(2, dim=-1) for gate_proj, cond in zip(self.post_branch_gates, time_cond)
        ])

        residual = list(modality_tokens)
        tokens = [ln(t, c) for t, c, ln in zip(modality_tokens, time_cond, self.attn_layernorms)]
        tokens = self.joint_attn(tokens, masks=modality_masks)
        tokens = [t * g + r for t, g, r in zip(tokens, attn_gammas, residual)]

        residual = list(tokens)
        normed = [ln(t, c) for t, c, ln in zip(tokens, time_cond, self.ff_layernorms)]
        tokens = [ff(t) for t, ff in zip(normed, self.feedforwards)]
        tokens = [t * g + r for t, g, r in zip(tokens, ff_gammas, residual)]
        return tokens


class MMDiT(Module):
    '''Stack of MMDiTBlocks: many rounds of "attend jointly, then per-modality FF".'''

    def __init__(self, *, depth, dim_modalities, dim_conds, **block_kwargs):
        super().__init__()
        self.blocks = ModuleList([
            MMDiTBlock(dim_modalities=dim_modalities, dim_conds=dim_conds, **block_kwargs) for _ in range(depth)
        ])
        self.norms = ModuleList([RMSNorm(d) for d in dim_modalities])

    def forward(self, *, modality_tokens, time_cond, modality_masks=None):
        for block in self.blocks:
            modality_tokens = block(modality_tokens=modality_tokens, time_cond=time_cond, modality_masks=modality_masks)
        return tuple(norm(t) for t, norm in zip(modality_tokens, self.norms))


class FinalLayer(nn.Module):
    '''DiT-style output head: adaptive-scale-shift LayerNorm, then a linear projection.'''

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'), nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * scale + shift)


class TimestepEmbedder(nn.Module):
    '''Sinusoidal timestep embedding + small MLP, standard DiT-style time conditioning.'''

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        import math
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t.flatten(), self.frequency_embedding_size))
