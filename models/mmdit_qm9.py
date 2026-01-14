from __future__ import annotations
from typing import Tuple
from euclidean_interpolant import EuclideanFixedSizeInterpolant
from huggingface_hub import PyTorchModelHubMixin

import torch
import math
import numpy as np
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

from models.mmdit import MMDiT, FinalLayer
from multimodal_interpolant import MultimodalModelPrediction
from model.rotary import Rotary


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

# rotary positional embedding helper
def get_rotary_emb(rotary_module, x):
    """
    Get rotary positional embeddings from the Rotary module.
    
    Args:
        rotary_module: Rotary module instance
        x: input tensor of shape (b, n, d) to determine sequence length
    
    Returns:
        cos, sin: tensors of shape (1, n, 3, 1, dim_head)
                 This format matches what apply_rotary_pos_emb expects
    """
    # The Rotary class generates embeddings with shape (1, seq_len, 3, 1, dim_head)
    # This is the exact format needed by apply_rotary_pos_emb from model.rotary
    cos, sin = rotary_module(x, seq_dim=1)
    return cos, sin


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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.flatten()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MMDiTQM9(nn.Module):
    def __init__(self, euclidean_dim, text_vocab_size, context_len, text_depth, image_depth, project_hidden=False, **kwargs):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.dim_modalities = kwargs['dim_modalities']
        self.dim_conds = kwargs['dim_conds']
        self.dim_text = self.dim_modalities[0]
        self.dim_image = self.dim_modalities[1]

        # Extract attention parameters for rotary embeddings
        self.dim_head = kwargs.get('dim_head', 64)
        self.heads = kwargs.get('heads', 8)

        # Time Encoders
        self.text_time_encoder = TimestepEmbedder(self.dim_text)
        self.image_time_encoder = TimestepEmbedder(self.dim_image)

        # Text Embeddings
        self.text_embedder = nn.Embedding(text_vocab_size, self.dim_text)

        # Euclidean Embeddings
        # Handle multi-dimensional coordinates [B, L, D] where D is the feature dimension
        # Each dimension is embedded separately using GaussianFourierProjection
        # Then combined and projected to dim_image
        # Each dimension gets embedded_dim features, so total is euclidean_dim * embedded_dim
        self.embedded_dim_per_feature = self.dim_image // euclidean_dim
        self.gaussian_fourier_projs = nn.ModuleList([
            GaussianFourierProjection(self.embedded_dim_per_feature, scale=1.0)
            for _ in range(euclidean_dim)
        ])
        # Project the concatenated embeddings to dim_image
        # Total embedding size after concatenation: euclidean_dim * embedded_dim_per_feature
        total_embed_dim = euclidean_dim * self.embedded_dim_per_feature
        self.euclidean_proj = nn.Linear(total_embed_dim, self.dim_image)

        # Rotary Positional Embeddings
        self.text_rotary = Rotary(dim=self.dim_head, base=10_000)
        self.image_rotary = Rotary(dim=self.dim_head, base=10_000)

        # Joint Embedding
        self.joint_embedding = MMDiT(**kwargs)

        # Single Embeddings
        self.has_text = text_depth > 0
        self.has_image = image_depth > 0
        sep_keys = ['depth', 'dim_modalities', 'dim_conds']
        block_kwargs = {k: v for k, v in kwargs.items() if k not in sep_keys}
        if self.has_text:
            self.text_dit = MMDiT(depth = text_depth, dim_modalities = [self.dim_text], dim_conds = [self.dim_text], **block_kwargs)
        else:
            self.freeze_last_block(0)
        if self.has_image:
            self.image_dit = MMDiT(depth = image_depth, dim_modalities = [self.dim_image], dim_conds = [self.dim_image], **block_kwargs)
        else:
            self.freeze_last_block(1)

        # Final Layers
        self.rate_pred = FinalLayer(self.dim_image, 1)
        if self.has_text:
            self.text_final_layer = FinalLayer(self.dim_text, text_vocab_size)
        if self.has_image:
            self.image_final_layer = FinalLayer(self.dim_image, euclidean_dim)


    def freeze_last_block(self, idx):
        last_block = self.joint_embedding.blocks[-1]
        feedforward = last_block.feedforwards[idx]
        layernorm = last_block.ff_layernorms[idx]
        last_block.joint_attn.to_out[idx].weight.requires_grad = False
        cond_block = last_block.cond_dict[f'cond_linear_{idx}']
        self.joint_embedding.norms[idx].g.requires_grad = False
        for param in feedforward.parameters():
            param.requires_grad = False
        for param in layernorm.parameters():
            param.requires_grad = False
        for param in cond_block.parameters():
            param.requires_grad = False
    
    def freeze_joint(self):
        layers_to_freeze = [self.text_embedder, self.text_time_encoder, 
                            self.image_embedder, self.image_time_encoder,
                            self.joint_embedding]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_image(self):
        layers_to_freeze = [self.image_embedder, self.image_time_encoder, self.image_dit, self.image_final_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def freeze_text(self):
        layers_to_freeze = [self.text_embedder, self.text_time_encoder, self.text_dit, self.text_final_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        *,
        cat_tokens,
        euclidean_tokens,
        text_mask = None,
        image_mask = None,
        image_time_cond = None,
        text_time_cond = None,
        detach_hidden = [False, False]
    ):
        # Shapes will be:
        # cat_tokens: [B, L]
        # euclidean_tokens: [B, L, D] where D is feature dimension (e.g., 3 for 3D coordinates)
        # text_mask: [B, L]
        # image_mask: [B, L]
        # text_time_cond: [B] - scalar timesteps per batch element
        # image_time_cond: [B] - scalar timesteps per batch element

        # Handle euclidean_tokens: [B, L, D] where D is the feature dimension
        # Embed each dimension separately and concatenate
        B, L = euclidean_tokens.shape[:2]
        
        if euclidean_tokens.dim() == 2:
            # If 2D [B, L], treat as single dimension and add feature dim
            euclidean_tokens = euclidean_tokens.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        
        D = euclidean_tokens.shape[-1]
        assert D == self.euclidean_dim, f"Expected euclidean_tokens to have {self.euclidean_dim} dimensions, got {D}"
        
        # Embed each dimension separately
        dim_embeddings = []
        for d in range(D):
            dim_token = euclidean_tokens[..., d:d+1]  # [B, L, 1]
            dim_emb = self.gaussian_fourier_projs[d](dim_token)  # [B, L, embedded_dim_per_feature]
            dim_embeddings.append(dim_emb)
        
        # Concatenate all dimension embeddings
        euclidean_tokens = torch.cat(dim_embeddings, dim=-1)  # [B, L, euclidean_dim * embedded_dim_per_feature]
        
        # Project to final embedding dimension
        euclidean_tokens = self.euclidean_proj(euclidean_tokens)  # [B, L, dim_image]
        
        cat_tokens = self.text_embedder(cat_tokens)
        text_time_cond = self.text_time_encoder(text_time_cond)  # [B] -> [B, dim_text]
        image_time_cond = self.image_time_encoder(image_time_cond)  # [B] -> [B, dim_image]

        # Generate rotary positional embeddings for each modality
        # The Rotary class returns (1, seq_len, 3, 1, dim_head) which is what we need
        text_cos, text_sin = get_rotary_emb(self.text_rotary, cat_tokens)
        image_cos, image_sin = get_rotary_emb(self.image_rotary, euclidean_tokens)
        rotary_pos_emb = ((text_cos, text_sin), (image_cos, image_sin))

        text_tokens_hidden, euclidean_tokens_hidden = self.joint_embedding(
            modality_tokens = (cat_tokens, euclidean_tokens),
            modality_masks = (text_mask, image_mask),
            time_cond = (text_time_cond, image_time_cond),
            rotary_pos_emb = rotary_pos_emb,
        )
        if detach_hidden[0]:
            text_tokens_hidden = text_tokens_hidden.detach()
        if detach_hidden[1]:
            euclidean_tokens_hidden = euclidean_tokens_hidden.detach()

        if self.has_text:
            cat_tokens = self.text_dit(
                modality_tokens = (text_tokens_hidden,),
                modality_masks = (text_mask,),
                time_cond = (text_time_cond,),
                rotary_pos_emb = ((text_cos, text_sin),),
            )[0]
            cat_tokens = self.text_final_layer(cat_tokens, text_time_cond)
            cat_tokens[:, :, :-1] = cat_tokens[:, :, :-1].log_softmax(dim=-1)

        if self.has_image:
            euclidean_tokens = self.image_dit(
                modality_tokens = (euclidean_tokens_hidden,),
                modality_masks = (image_mask,),
                time_cond = (image_time_cond,),
                rotary_pos_emb = ((image_cos, image_sin),),
            )[0]
            clean_data_pred = self.image_final_layer(euclidean_tokens, image_time_cond)
            insertion_rate = self.rate_pred(euclidean_tokens, image_time_cond).squeeze(-1)
            insertion_rate = F.softplus(insertion_rate)


        return MultimodalModelPrediction(
            clean_data=clean_data_pred,
            insertion_rate=insertion_rate,
            label_logits=cat_tokens,
            insertion_logits=None
        )

