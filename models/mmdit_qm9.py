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

from model.transformer import GaussianFourierProjection
from multimodal_interpolant import MultimodalModelPrediction

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

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class MMDiTQM9(nn.Module):
    def __init__(self, euclidean_dim, text_vocab_size, context_len, text_depth, image_depth, project_hidden=False, **kwargs):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.dim_modalities = kwargs['dim_modalities']
        self.dim_conds = kwargs['dim_conds']
        self.dim_text = self.dim_modalities[0]
        self.dim_image = self.dim_modalities[1]

        # Time Encoders
        self.text_time_encoder = TimestepEmbedder(self.dim_text)
        self.image_time_encoder = TimestepEmbedder(self.dim_image)

        # Text Embeddings
        self.text_embedder = nn.Embedding(text_vocab_size, self.dim_text)
        self.text_pos_embed = nn.Embedding(context_len, self.dim_text)
        nn.init.normal_(self.text_pos_embed.weight, std=0.02)

        # Image Embeddings
        grid = np.arange(context_len)
        self.register_buffer('pos_embed', torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.dim_image, grid)).float().unsqueeze(0))
        self.euclidean_embedding = nn.Sequential(
            GaussianFourierProjection(self.dim_image, scale=1.0),
            nn.Linear(self.dim_image, self.dim_image),
        )

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
        layers_to_freeze = [self.text_embedder, self.text_pos_embed, self.text_time_encoder, 
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
        layers_to_freeze = [self.text_embedder, self.text_pos_embed, self.text_time_encoder, self.text_dit, self.text_final_layer]
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
        # Create position indices based on sequence length
        B, L = cat_tokens.shape[:2]
        positions = torch.arange(L, device=cat_tokens.device)
        positions = positions.unsqueeze(0).expand(B, -1) # [batch, seq_len]
        
        # Handle euclidean_tokens: if 2D [B, L], add feature dim; if 3D [B, L, D], embed each dimension
        if euclidean_tokens.dim() == 2:
            euclidean_tokens = euclidean_tokens.unsqueeze(-1)  # [B, L] -> [B, L, 1]
            euclidean_tokens = self.euclidean_embedding(euclidean_tokens)  # [B, L, dim_image]
        elif euclidean_tokens.dim() == 3:
            D = euclidean_tokens.shape[-1]
            if D == 1:
                # Already correct shape [B, L, 1]
                euclidean_tokens = self.euclidean_embedding(euclidean_tokens)  # [B, L, dim_image]
            else:
                # For multi-dimensional coordinates (e.g., 3D: x, y, z), embed each dimension separately
                # and then combine using a linear projection
                # Split into individual dimensions: [B, L, D] -> D x [B, L, 1]
                dim_embeddings = []
                for d in range(D):
                    dim_token = euclidean_tokens[..., d:d+1]  # [B, L, 1]
                    dim_emb = self.euclidean_embedding(dim_token)  # [B, L, dim_image]
                    dim_embeddings.append(dim_emb)
                # Sum the embeddings from all dimensions (alternative: could concatenate and project)
                euclidean_tokens = sum(dim_embeddings) / D  # [B, L, dim_image] - average of embeddings
        else:
            raise ValueError(f"euclidean_tokens must be 2D or 3D, got {euclidean_tokens.dim()}D")
        
        # Slice or pad positional embedding to match actual sequence length
        pos_embed_len = self.pos_embed.shape[1]
        if L <= pos_embed_len:
            pos_embed_sliced = self.pos_embed[:, :L, :]  # [1, L, dim_image]
        else:
            # If sequence is longer than pos_embed, pad by repeating the last position
            padding = self.pos_embed[:, -1:, :].expand(1, L - pos_embed_len, -1)  # [1, L - pos_embed_len, dim_image]
            pos_embed_sliced = torch.cat([self.pos_embed, padding], dim=1)  # [1, L, dim_image]
        euclidean_tokens = euclidean_tokens + pos_embed_sliced
        
        cat_tokens = self.text_embedder(cat_tokens) + self.text_pos_embed(positions)
        text_time_cond = self.text_time_encoder(text_time_cond)  # [B] -> [B, dim_text]
        image_time_cond = self.image_time_encoder(image_time_cond)  # [B] -> [B, dim_image]

        text_tokens_hidden, euclidean_tokens_hidden = self.joint_embedding(
            modality_tokens = (cat_tokens, euclidean_tokens),
            modality_masks = (text_mask, image_mask),
            time_cond = (text_time_cond, image_time_cond),
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
            )[0]
            cat_tokens = self.text_final_layer(cat_tokens, text_time_cond)
            cat_tokens[:, :, :-1] = cat_tokens[:, :, :-1].log_softmax(dim=-1)

        if self.has_image:
            euclidean_tokens = self.image_dit(
                modality_tokens = (euclidean_tokens_hidden,),
                modality_masks = (image_mask,),
                time_cond = (image_time_cond,),
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

