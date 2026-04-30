"""Single-stream transformer (PyTorch port of a Julia `Toy` model).

This module mirrors the architecture in:

    struct Toy{L}
        layers::L
    end
    function Toy(dim, depth, vocab_size; cont_dim::Int = CONT_DIM)
        ...
        layers = (;
            loc_rff   = RandomFourierFeatures(cont_dim => 2dim, 1f0),
            loc_rff2  = RandomFourierFeatures(cont_dim => 2dim, 0.1f0),
            t_rff     = RandomFourierFeatures(1 => 4dim, 1f0),
            t_embed   = Dense(4dim => dim, bias=false),
            loc_encoder = Dense(4dim + cont_dim => dim, bias=false),
            d_encoder = Embedding(vocab_size => dim),
            rope      = RoPE(head_dim, 1000),
            transformers = transformers,
            loc_decoder   = Dense(dim => cont_dim, bias=false),
            count_decoder = Dense(dim => 1, bias=false),
            del_decoder   = Dense(dim => 1, bias=false),
            d_decoder     = Dense(dim => vocab_size, bias=false),
        )
        ...
    end

The output is wired to ``MultimodalModelPrediction`` (see
``multimodal_interpolant.py``) instead of returning a deletion rate:

    clean_data           <- loc_decoder
    label_logits         <- d_decoder
    insertion_rate       <- count_decoder
    clean_data_unmasking <- extra FinalLayer head (matches MMDiTQM9)

The forward signature matches ``MMDiTQM9`` / ``MMDiTBothVar`` in
``models/mmdit_qm9.py`` so this class can be used as a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

from model.rotary import Rotary, apply_rotary_pos_emb
from models.mmdit import MultiHeadRMSNorm, FinalLayer
from multimodal_interpolant import MultimodalModelPrediction


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def exists(v):
    return v is not None


class RandomFourierFeatures(nn.Module):
    """Random Fourier features matching ``Onion.RandomFourierFeatures``.

    Maps ``x`` of shape ``(..., in_dim)`` to ``(..., out_dim)`` via
    ``[sin(2π x W), cos(2π x W)]`` where ``W`` is a fixed (non-trainable)
    Gaussian random matrix of shape ``(in_dim, out_dim // 2)`` scaled by
    ``scale``.
    """

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        W = torch.randn(in_dim, out_dim // 2) * scale
        self.register_buffer("W", W, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_dim)
        proj = (x @ self.W) * 2 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# -----------------------------------------------------------------------------
# Adaptive transformer block (single-stream DiT block w/ RoPE + QK norm)
# -----------------------------------------------------------------------------

class AdaTransformerBlock(nn.Module):
    """Single-stream transformer block with adaptive layernorm conditioning.

    Mirrors the behavior of ``Onion.AdaTransformerBlock(dim, dim, nheads;
    head_dim, qk_norm=true)``: a standard DiT block where the conditioning
    vector produces (shift, scale, gate) modulations for both the attention
    and feedforward sub-layers.
    """

    def __init__(
        self,
        dim: int,
        dim_cond: int,
        num_heads: int,
        head_dim: int = 64,
        qk_norm: bool = True,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim_inner = num_heads * head_dim

        # Attention sub-layer.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, self.dim_inner * 3, bias=False)
        self.proj = nn.Linear(self.dim_inner, dim, bias=False)

        if qk_norm:
            self.q_rmsnorm = MultiHeadRMSNorm(head_dim, heads=num_heads)
            self.k_rmsnorm = MultiHeadRMSNorm(head_dim, heads=num_heads)
        else:
            self.q_rmsnorm = nn.Identity()
            self.k_rmsnorm = nn.Identity()

        # Feedforward sub-layer.
        ff_dim = int(ff_mult * dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        # AdaLN modulation: (shift_msa, scale_msa, gate_msa,
        #                    shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, 6 * dim, bias=True),
        )
        # Zero init so that the block initially acts as identity (DiT trick).
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _attention(
        self,
        x: Tensor,
        rotary_emb: Tuple[Tensor, Tensor] | None = None,
        kpad_mask: Tensor | None = None,
    ) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x)  # (B, L, 3 * H * D)
        # Use the same layout as `models.mmdit.JointAttention` so that we can
        # reuse `apply_rotary_pos_emb` directly.
        qkv = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d",
            qkv=3, h=self.num_heads, d=self.head_dim,
        )
        q, k, v = qkv  # each: (B, H, L, D)

        # Optional QK RMS norm (per-head).
        q = self.q_rmsnorm(q)
        k = self.k_rmsnorm(k)

        # Rotary positional embeddings (applied to q, k; identity on v).
        if exists(rotary_emb):
            cos, sin = rotary_emb
            qkv_stacked = torch.stack((q, k, v), dim=0)  # (3, B, H, L, D)
            qkv_reshaped = rearrange(qkv_stacked, "qkv b h n d -> b n qkv h d")
            qkv_rotated = apply_rotary_pos_emb(qkv_reshaped, cos, sin)
            qkv_rotated = rearrange(qkv_rotated, "b n qkv h d -> qkv b h n d")
            q, k, v = qkv_rotated

        # Build attention mask (True = attend) with shape (B, 1, 1, L).
        attn_mask = None
        if exists(kpad_mask):
            attn_mask = kpad_mask[:, None, None, :].to(dtype=torch.bool)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)

    def forward(
        self,
        x: Tensor,
        *,
        cond: Tensor,
        rotary_emb: Tuple[Tensor, Tensor] | None = None,
        kpad_mask: Tensor | None = None,
    ) -> Tensor:
        # cond: (B, dim_cond) -> chunked into 6 modulation vectors of dim ``dim``
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )

        # Attention sub-layer.
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        h = self._attention(h, rotary_emb=rotary_emb, kpad_mask=kpad_mask)
        x = x + gate_msa.unsqueeze(1) * h

        # Feedforward sub-layer.
        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.ff(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


# -----------------------------------------------------------------------------
# Toy model (single-stream transformer for QM9-style problems)
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    """Single-stream transformer port of the Julia ``Toy`` model.

    Output is wired to ``MultimodalModelPrediction`` so this class is
    interchangeable with ``MMDiTQM9`` (when ``branching_flows=False`` and
    ``autoregressive=False``).

    Args:
        euclidean_dim: Dimensionality of the continuous coordinates (``cont_dim``).
        vocab_size: Size of the discrete token vocabulary.
        dim: Hidden dimension of the transformer.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension for attention.
        max_seq_len: Maximum sequence length used to pre-compute RoPE caches.
        qk_norm: Whether to apply per-head RMS norm to Q and K.

    Forward signature (kwargs-only) matches ``MMDiTQM9.forward``:
        cat_tokens:        (B, L) integer tokens
        euclidean_tokens:  (B, L, euclidean_dim) coordinates (or (B, L) if D=1)
        symbols_mask:      (B, L) bool mask (True = valid)
        pos_mask:          (B, L) bool mask (True = valid)
        pos_time:          (B,) per-batch scalar timesteps
        symbols_time:      (B,) per-batch scalar timesteps
        detach_hidden:     ignored (kept for API parity with MMDiTQM9)

    Returns ``MultimodalModelPrediction`` with:
        clean_data:           (B, L, euclidean_dim)
        label_logits:         (B, L, vocab_size)
        clean_data_unmasking: (B, L, vocab_size, euclidean_dim)
        insertion_rate:       (B, L)
    """

    def __init__(
        self,
        euclidean_dim: int,
        vocab_size: int,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 12,
        head_dim: int = 64,
        max_seq_len: int = 1000,
        qk_norm: bool = True,
        **kwargs,  # absorb (and ignore) extra kwargs for API parity
    ):
        super().__init__()
        # ``Rotary`` (model.rotary) requires an even head dimension because it
        # splits the head into two halves for the rotation.
        assert head_dim % 2 == 0, "head_dim must be even (RoPE requirement)"
        self.euclidean_dim = euclidean_dim
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim

        # ------- Continuous (location) encoder -------
        # Two RFFs with different scales, concatenated with the raw locations,
        # then projected to ``dim`` (mirrors the Julia loc_encoder).
        self.loc_rff = RandomFourierFeatures(euclidean_dim, 2 * dim, scale=1.0)
        self.loc_rff2 = RandomFourierFeatures(euclidean_dim, 2 * dim, scale=0.1)
        self.loc_encoder = nn.Linear(4 * dim + euclidean_dim, dim, bias=False)

        # ------- Discrete encoder -------
        self.d_encoder = nn.Embedding(vocab_size, dim)

        # ------- Time encoders -------
        # Julia uses a single shared time. ``MMDiTQM9.forward`` provides two
        # times; we embed each independently and sum them so the API stays
        # compatible while the underlying conditioning is single-stream.
        self.t_rff_pos = RandomFourierFeatures(1, 4 * dim, scale=1.0)
        self.t_embed_pos = nn.Linear(4 * dim, dim, bias=False)
        self.t_rff_sym = RandomFourierFeatures(1, 4 * dim, scale=1.0)
        self.t_embed_sym = nn.Linear(4 * dim, dim, bias=False)

        # ------- Rotary positional embeddings -------
        # Reuse the existing ``Rotary`` module so the rotary layout matches
        # what `apply_rotary_pos_emb` expects elsewhere in this codebase.
        self.rope = Rotary(dim=head_dim, base=10_000)
        self.max_seq_len = max_seq_len

        # ------- Transformer stack -------
        self.transformers = nn.ModuleList([
            AdaTransformerBlock(
                dim=dim,
                dim_cond=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                qk_norm=qk_norm,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

        # ------- Output heads (FinalLayer = AdaLN + Linear, matches MMDiTQM9) -------
        self.loc_decoder = FinalLayer(dim, euclidean_dim)
        self.d_decoder = FinalLayer(dim, vocab_size)
        self.count_decoder = FinalLayer(dim, 1)
        # Like ``MMDiTQM9.positions_unmask_pred``: predicts a coordinate per
        # vocab token, reshaped to (B, L, vocab_size, euclidean_dim).
        self.unmask_decoder = FinalLayer(dim, euclidean_dim * vocab_size)

    # ------------------------------------------------------------------
    # Optimizer parameter grouping (Muon vs AdamW), matching MMDiTQM9.
    # ------------------------------------------------------------------
    @staticmethod
    def _split_params_by_size(params):
        muon_params, adam_params = [], []
        for p in params:
            (muon_params if p.ndim == 2 else adam_params).append(p)
        return muon_params, adam_params

    def get_muon_adam_params(self):
        adam_params, muon_params = [], []

        # Embedding / projection layers stay on AdamW.
        adam_params.extend(self.d_encoder.parameters())
        adam_params.extend(self.loc_rff.parameters())
        adam_params.extend(self.loc_rff2.parameters())
        adam_params.extend(self.loc_encoder.parameters())
        adam_params.extend(self.t_rff_pos.parameters())
        adam_params.extend(self.t_embed_pos.parameters())
        adam_params.extend(self.t_rff_sym.parameters())
        adam_params.extend(self.t_embed_sym.parameters())
        adam_params.extend(self.final_norm.parameters())

        # Output heads.
        for head in (self.loc_decoder, self.d_decoder,
                     self.count_decoder, self.unmask_decoder):
            adam_params.extend(head.parameters())

        # Transformer body: split 2D weights to Muon, the rest to AdamW.
        muon_, adam_ = self._split_params_by_size(self.transformers.parameters())
        muon_params.extend(muon_)
        adam_params.extend(adam_)

        return {"muon_params": muon_params, "adam_params": adam_params}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        cat_tokens: Tensor,
        euclidean_tokens: Tensor,
        symbols_mask: Tensor | None = None,
        pos_mask: Tensor | None = None,
        pos_time: Tensor | None = None,
        symbols_time: Tensor | None = None,
        detach_hidden=(False, False),  # accepted for API parity, unused
    ) -> MultimodalModelPrediction:
        # Normalize coordinate shape to (B, L, D).
        if euclidean_tokens.dim() == 2:
            euclidean_tokens = euclidean_tokens.unsqueeze(-1)
        B, L, D = euclidean_tokens.shape
        assert D == self.euclidean_dim, (
            f"Expected euclidean_tokens to have {self.euclidean_dim} dims, got {D}"
        )

        # ---- Token features: loc encoding + discrete embedding ----
        loc_feats = torch.cat([
            self.loc_rff(euclidean_tokens),
            self.loc_rff2(euclidean_tokens),
            euclidean_tokens,
        ], dim=-1)
        x = self.loc_encoder(loc_feats) + self.d_encoder(cat_tokens)

        # ---- Time conditioning (sum of two embeddings) ----
        # Times come in as (B,) scalars; reshape to (B, 1) for the RFF.
        if pos_time is None and symbols_time is None:
            raise ValueError("At least one of pos_time / symbols_time must be provided.")
        t_cond = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
        if exists(pos_time):
            t = pos_time.reshape(B, 1).to(dtype=x.dtype)
            t_cond = t_cond + self.t_embed_pos(self.t_rff_pos(t))
        if exists(symbols_time):
            t = symbols_time.reshape(B, 1).to(dtype=x.dtype)
            t_cond = t_cond + self.t_embed_sym(self.t_rff_sym(t))

        # ---- Single shared mask for the single-stream backbone ----
        # Symbol and position masks tend to be identical for QM9-style data.
        if exists(pos_mask) and exists(symbols_mask):
            kpad_mask = pos_mask & symbols_mask
        else:
            kpad_mask = pos_mask if exists(pos_mask) else symbols_mask

        # ---- RoPE: cache uses the largest seq len seen so far. ----
        # Match the Julia indexing `rope[1:size(locs, 2)]` by slicing the cache
        # to the current sequence length.
        rope_cos, rope_sin = self.rope(x, seq_dim=1)
        rope_cos = rope_cos[:, :L]
        rope_sin = rope_sin[:, :L]
        rotary_emb = (rope_cos, rope_sin)

        # ---- Transformer stack ----
        for block in self.transformers:
            x = block(x, cond=t_cond, rotary_emb=rotary_emb, kpad_mask=kpad_mask)
        x = self.final_norm(x)

        # ---- Output heads ----
        clean_data = self.loc_decoder(x, t_cond)                       # (B, L, D)
        label_logits = self.d_decoder(x, t_cond)                       # (B, L, V)
        clean_data_unmasking = self.unmask_decoder(x, t_cond).view(
            B, L, self.vocab_size, self.euclidean_dim
        )
        insertion_rate = self.count_decoder(x, t_cond).squeeze(-1)     # (B, L)
        insertion_rate = F.softplus(insertion_rate)
        if exists(pos_mask):
            insertion_rate = insertion_rate * pos_mask.to(insertion_rate.dtype)

        return MultimodalModelPrediction(
            clean_data=clean_data,
            label_logits=label_logits,
            clean_data_unmasking=clean_data_unmasking,
            insertion_rate=insertion_rate,
        )
