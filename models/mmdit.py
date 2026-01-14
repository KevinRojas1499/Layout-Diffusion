from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

# mlp 
def mlp(dim, dim_hidden, dim_out):
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
    )

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_inputs: Tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True,
        flash = True,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: Tuple[Tensor],
        masks: Tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# adaptive layernorm
# aim for clarity in generalized version

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)

            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            x = x * gamma + beta

        return x

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_modalities: Tuple[int, ...],
        dim_conds: Tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True,
        flash_attn = True,
        softclamp = False,
        softclamp_value = 50.,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # handle optional time conditioning

        has_cond_array = [exists(dim_cond) for dim_cond in dim_conds]
        self.has_cond_array = has_cond_array
        self.cond_dict = nn.ModuleDict()
        
        for i, cond in enumerate(dim_conds):
            if cond is not None:
                self.cond_dict[f'cond_linear_{i}'] = nn.Linear(dim_conds[i], dim_modalities[i] * 2)

                self.cond_dict[f'to_post_branch_gammas_{i}'] = nn.Sequential(
                    Rearrange('b d -> b 1 d'),
                    nn.SiLU(),
                    self.cond_dict[f'cond_linear_{i}']
                )

                nn.init.zeros_(self.cond_dict[f'cond_linear_{i}'].weight)
                nn.init.constant_(self.cond_dict[f'cond_linear_{i}'].bias, 1.)

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim, dim_cond in zip(dim_modalities, dim_conds)]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.joint_attn = JointAttention(
            dim = dim_joint_attn,
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn,
            qk_rmsnorm = qk_rmsnorm,
            softclamp = softclamp,
            softclamp_value = softclamp_value,
        )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim, dim_cond in zip(dim_modalities, dim_conds)]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = Tuple[Tensor | None, ...]
    ):
        assert len(modality_tokens) == self.num_modalities and len(time_cond) == self.num_modalities

        attn_gammas = [1.] * len(time_cond) # Default to 1. if no condition
        ff_gammas = [1.] * len(time_cond) # Default to 1. if no condition
        for i, cond in enumerate(time_cond):
            if cond is not None:
                attn_gammas[i], ff_gammas[i] = self.cond_dict[f'to_post_branch_gammas_{i}'](cond).chunk(2, dim = -1)

        # attention layernorms
        modality_tokens_residual = list(modality_tokens)  # Create new list instead of modifying in place
        modality_tokens = [ln(tokens, cond) for tokens, cond, ln in zip(modality_tokens, time_cond, self.attn_layernorms)]

        # attention
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # post attention gammas
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, attn_gammas)]

        # add attention residual
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # handle feedforward adaptive layernorm
        modality_tokens_residual = list(modality_tokens)
        modality_tokens = [ln(tokens, cond) for tokens, cond, ln in zip(modality_tokens, time_cond, self.ff_layernorms)]
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, ff_gammas)]

        # add feedforward residual
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # returns

        return modality_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities: Tuple[int, ...],
        dim_conds: Tuple[int, ...],
        **block_kwargs
    ):
        super().__init__()
        blocks = [MMDiTBlock(dim_modalities = dim_modalities, dim_conds = dim_conds, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = Tuple[Tensor | None, ...]
    ):
        for block in self.blocks:
            modality_tokens = block(
                time_cond = time_cond,
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        return tuple(modality_tokens)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size,out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * scale + shift
        x = self.linear(x)
        return x
    