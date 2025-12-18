import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from euclidean_interpolant import EuclideanModelPrediction
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)

# TODO: This is giving me issues, I'll get back to it later
# flex_attention = torch.compile(flex_attention, mode="max-autotune")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for continuous inputs."""

    def __init__(self, embed_dim, scale=1.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: (B, L, 1)
        x_proj = x * self.W[None, None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# length scalar head
class ScalarLengthHead(nn.Module):
    def __init__(self, d_model: int, normalized_len: int, cond_dim: int | None = None):
        super().__init__()
        self.has_cond = cond_dim is not None
        if self.has_cond:
            self.adaLN = nn.Linear(cond_dim, 2 * d_model, bias=True)
            self.adaLN.weight.data.zero_()
            self.adaLN.bias.data.zero_()

        self.norm = LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()
        self.normalized_len = normalized_len

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x_fp32 = x.float()
        c_fp32 = c.float() if (self.has_cond and c is not None) else None
        if self.has_cond and c_fp32 is not None:
            shift, scale = self.adaLN(c_fp32)[:, None].chunk(2, dim=2)
            x_fp32 = modulate_fused(self.norm(x_fp32), shift, scale)
        else:
            x_fp32 = self.norm(x_fp32)
        s = self.proj2(self.act(self.proj1(x_fp32)))
        out = self.softplus(s).squeeze(-1) * self.normalized_len
        return out.to(x.dtype)


#################################################################################
#                                 Core Model                                    #
#################################################################################


def get_mask_mod_seq_len(seq_len: torch.Tensor):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx <= seq_len[b]) & (kv_idx <= seq_len[b])

    return mask_mod


def get_mask_mod_arbitrary(mask: torch.Tensor):
    def mask_mod(b, h, q_idx, kv_idx):
        return mask[b, kv_idx]

    return mask_mod


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, block_mask):
        batch_size = x.shape[0]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = rearrange(qkv, "b s three h d -> three b h s d", three=3)

        x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EuclideanTransformer(nn.Module):
    def __init__(self, hidden_size, cond_dim, n_heads, n_blocks, dropout, max_length, vocabulary_size, use_arbitrary_mask=True):
        # Max length also plays the role of the dimension, although only some entries of the output are active at a time
        super().__init__()
        self.use_arbitrary_mask = use_arbitrary_mask

        self.embedding = nn.Sequential(
            GaussianFourierProjection(hidden_size, scale=1.0),
            nn.Linear(hidden_size, hidden_size),
        )
        self.timestep_embedder = TimestepEmbedder(cond_dim)
        self.max_length = max_length
        self.rotary_emb = rotary.Rotary(
            hidden_size // n_heads
        )

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    hidden_size,
                    n_heads,
                    cond_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        
        self.output_layer = DDitFinalLayer(
            hidden_size, 1, cond_dim
        )

        self.logits_pred = DDitFinalLayer(
            hidden_size,
            vocabulary_size,
            cond_dim
        )
        self.rate_pred = DDitFinalLayer(
            hidden_size, 1, cond_dim
        )

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, indices: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        B, L = indices.shape
        if self.use_arbitrary_mask:
            mask_mod = get_mask_mod_arbitrary(mask)
        else:
            seq_lens = (mask).sum(dim=-1)
            mask_mod = get_mask_mod_seq_len(seq_lens)
            
        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=indices.shape[1],
            KV_LEN=indices.shape[1],
        )

        if indices.dim() == 2:
            x = indices.view(B, L, 1)
        else:
            x = indices
            
        x = self.embedding(x)
        x = x * mask.unsqueeze(-1).to(x.dtype) # Zero out masked embeddings
        c = F.silu(self.timestep_embedder(t))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, block_mask)

            # --- unmasking ---
            clean_data = self.output_layer(x, c)
            logits = self.logits_pred(x, c)
            rate = F.softplus(self.rate_pred(x,c))

            return EuclideanModelPrediction(
                clean_data=clean_data.squeeze(-1),
                logits=logits.squeeze(-1),
                rate=rate.squeeze(-1)
            )
