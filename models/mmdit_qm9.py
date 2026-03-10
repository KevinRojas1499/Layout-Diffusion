from __future__ import annotations
from typing import Tuple
from branching_flows_interpolant import BranchingFlowsPrediction

import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

from models.mmdit import MMDiT, FinalLayer
from multimodal_interpolant import MultimodalModelPrediction
from multimodal_interpolant_both_var import MultimodalModelPredictionBothVar
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


class PairwiseSpatialBias(nn.Module):
    """
    Computes learnable, layer-specific, head-specific attention biases
    from pairwise spatial features (distances between atoms).
    """
    def __init__(self, num_heads, num_layers, spatial_feat_dim=32, max_distance=10.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.spatial_feat_dim = spatial_feat_dim
        self.max_distance = max_distance
        
        # Learnable MLP to convert pairwise distances to attention biases
        # Different for each layer and head
        self.bias_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, spatial_feat_dim),
                    nn.SiLU(),
                    nn.Linear(spatial_feat_dim, 1)
                )
                for _ in range(num_heads)
            ])
            for _ in range(num_layers)
        ])
        
    def compute_pairwise_distances(self, positions, mask=None):
        """
        Compute pairwise Euclidean distances between atom positions.
        
        Args:
            positions: (B, L, D) - atom positions
            mask: (B, L) - optional mask for valid positions
            
        Returns:
            distances: (B, L, L) - pairwise distance matrix
        """
        # Compute squared distances: ||p_i - p_j||^2
        # positions: (B, L, D)
        pos_i = positions.unsqueeze(2)  # (B, L, 1, D)
        pos_j = positions.unsqueeze(1)  # (B, 1, L, D)
        squared_dist = torch.sum((pos_i - pos_j) ** 2, dim=-1)  # (B, L, L)
        distances = torch.sqrt(squared_dist + 1e-8)  # (B, L, L) - add small epsilon for stability
        
        # Apply mask if provided (set masked distances to a large value)
        if mask is not None:
            # mask: (B, L) -> (B, L, 1) and (B, 1, L)
            mask_i = mask.unsqueeze(-1)  # (B, L, 1)
            mask_j = mask.unsqueeze(1)   # (B, 1, L)
            valid_mask = mask_i & mask_j  # (B, L, L)
            distances = torch.where(valid_mask, distances, torch.full_like(distances, self.max_distance * 2))
        
        return distances
    
    def forward(self, positions, layer_idx, mask=None):
        """
        Compute attention biases from pairwise spatial features.
        
        Args:
            positions: (B, L, D) - atom positions
            layer_idx: int - which layer this is (0-indexed)
            mask: (B, L) - optional mask for valid positions
            
        Returns:
            biases: (B, H, L, L) - attention biases for each head
        """
        B, L, D = positions.shape
        
        # Compute pairwise distances: (B, L, L)
        distances = self.compute_pairwise_distances(positions, mask)
        
        # Normalize/clamp distances
        distances = torch.clamp(distances / self.max_distance, 0, 1)
        
        # Reshape for MLP: (B, L, L, 1)
        distances = distances.unsqueeze(-1)
        
        # Compute biases for each head: (B, H, L, L)
        biases = []
        for head_idx in range(self.num_heads):
            # Apply layer-specific, head-specific MLP
            head_biases = self.bias_mlps[layer_idx][head_idx](distances)  # (B, L, L, 1)
            head_biases = head_biases.squeeze(-1)  # (B, L, L)
            biases.append(head_biases)
        
        # Stack: (H, B, L, L) -> (B, H, L, L)
        biases = torch.stack(biases, dim=1)
        
        return biases


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
    def __init__(self, euclidean_dim, vocab_size, symbols_depth, positions_depth, branching_flows: bool = False, improved=True, **kwargs):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.vocab_size = vocab_size
        self.dim_modalities = kwargs['dim_modalities']
        self.dim_conds = kwargs['dim_conds']
        self.dim_symbols = self.dim_modalities[0]
        self.dim_positions = self.dim_modalities[1]
        self.branching_flows = branching_flows
        # Extract attention parameters for rotary embeddings
        self.dim_head = kwargs.get('dim_head', 64)
        self.heads = kwargs.get('heads', 8)

        # Time Encoders
        self.symbols_time_encoder = TimestepEmbedder(self.dim_symbols)
        self.positions_time_encoder = TimestepEmbedder(self.dim_positions)

        # Text Embeddings
        self.symbols_embedder = nn.Embedding(vocab_size, self.dim_symbols)

        # Euclidean Embeddings        # Handle multi-dimensional coordinates [B, L, D] where D is the feature dimension
        # Each dimension is embedded separately using GaussianFourierProjection
        # Then combined and projected to dim_image
        # Each dimension gets embedded_dim features, so total is euclidean_dim * embedded_dim
        self.embedded_dim_per_feature = self.dim_positions // euclidean_dim
        self.gaussian_fourier_projs = nn.ModuleList([
            GaussianFourierProjection(self.embedded_dim_per_feature, scale=1.0)
            for _ in range(euclidean_dim)
        ])
        # Project the concatenated embeddings to dim_image
        # Total embedding size after concatenation: euclidean_dim * embedded_dim_per_feature
        total_embed_dim = euclidean_dim * self.embedded_dim_per_feature
        self.euclidean_proj = nn.Linear(total_embed_dim, self.dim_positions)

        # Rotary Positional Embeddings
        self.symbols_rotary = Rotary(dim=self.dim_head, base=10_000)
        self.positions_rotary = Rotary(dim=self.dim_head, base=10_000)

        # Pairwise spatial attention biases (layer-specific, head-specific)
        # We'll get depth from kwargs
        depth = kwargs.get('depth', 4)
        heads = kwargs.get('heads', 8)
        self.spatial_bias = PairwiseSpatialBias(
            num_heads=heads,
            num_layers=depth,
            spatial_feat_dim=32,
            max_distance=10.0
        )

        # Joint Embedding
        self.joint_embedding = MMDiT(**kwargs)

        # Single Embeddings
        sep_keys = ['depth', 'dim_modalities', 'dim_conds']
        block_kwargs = {k: v for k, v in kwargs.items() if k not in sep_keys}
        self.symbols_dit = MMDiT(depth = symbols_depth, dim_modalities = [self.dim_symbols], dim_conds = [self.dim_symbols], **block_kwargs)
        self.positions_dit = MMDiT(depth = positions_depth, dim_modalities = [self.dim_positions], dim_conds = [self.dim_positions], **block_kwargs)

        # Fusing layers
        self.label_fuse = nn.Sequential(nn.Linear(self.dim_symbols + self.dim_positions, self.dim_symbols), nn.SiLU())
        self.insert_fuse = nn.Sequential(nn.Linear(self.dim_symbols + self.dim_positions, self.dim_symbols), nn.SiLU())
        self.euclidean_fuse = nn.Sequential(nn.Linear(self.dim_symbols + self.dim_positions, self.dim_positions), nn.SiLU())

        # Final Layers
        self.insertion_rate = FinalLayer(self.dim_positions, 1)
        self.symbols_pred_layer = FinalLayer(self.dim_symbols, vocab_size)
        self.positions_pred_layer = FinalLayer(self.dim_positions, euclidean_dim)
        if not self.branching_flows:
            self.positions_unmask_pred = FinalLayer(self.dim_positions, euclidean_dim * vocab_size)

    def split_params_by_size(self, params):
        muon_params = []
        adam_params = []
        for p in params:
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        return muon_params, adam_params

    def get_muon_adam_params(self):
        adam_params = []
        # Embedding layers
        adam_params.extend(list(self.symbols_time_encoder.mlp.parameters()))
        adam_params.extend(list(self.positions_time_encoder.mlp.parameters()))
        adam_params.extend(list(self.symbols_embedder.parameters()))
        adam_params.extend(list(self.gaussian_fourier_projs.parameters()))
        adam_params.extend(list(self.euclidean_proj.parameters()))
        # Final layers
        adam_params.extend(list(self.insertion_rate.parameters()))
        adam_params.extend(list(self.symbols_pred_layer.parameters()))
        adam_params.extend(list(self.positions_pred_layer.parameters()))
        if not self.branching_flows:
            adam_params.extend(list(self.positions_unmask_pred.parameters()))

        muon_params = []
        # Joint Embedding
        muon_params_, adam_params_ = self.split_params_by_size(self.joint_embedding.parameters())
        muon_params.extend(muon_params_)
        adam_params.extend(adam_params_)
        # Single Embeddings
        muon_params_, adam_params_ = self.split_params_by_size(self.symbols_dit.parameters())
        muon_params.extend(muon_params_)
        adam_params.extend(adam_params_)
        muon_params_, adam_params_ = self.split_params_by_size(self.positions_dit.parameters())
        muon_params.extend(muon_params_)
        adam_params.extend(adam_params_)
        # Spatial Bias
        muon_params_, adam_params_ = self.split_params_by_size(self.spatial_bias.parameters())
        muon_params.extend(muon_params_)
        adam_params.extend(adam_params_)
        return {'muon_params': muon_params, 'adam_params': adam_params}

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
        layers_to_freeze = [self.symbols_embedder, self.symbols_time_encoder, 
                            self.image_embedder, self.positions_time_encoder,
                            self.joint_embedding]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_image(self):
        layers_to_freeze = [self.image_embedder, self.positions_time_encoder, self.positions_dit, self.positions_pred_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def freeze_text(self):
        layers_to_freeze = [self.symbols_embedder, self.symbols_time_encoder, self.symbols_dit, self.symbols_pred_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        *,
        cat_tokens,
        euclidean_tokens,
        symbols_mask = None,
        pos_mask = None,
        pos_time = None,
        symbols_time = None,
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
        
        # Store original positions for computing pairwise spatial biases
        atom_positions = euclidean_tokens  # [B, L, D]
        
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
        
        cat_tokens = self.symbols_embedder(cat_tokens)
        symbols_time = self.symbols_time_encoder(symbols_time)  # [B] -> [B, dim_text]
        pos_time = self.positions_time_encoder(pos_time)  # [B] -> [B, dim_image]

        # Generate rotary positional embeddings for each modality
        # The Rotary class returns (1, seq_len, 3, 1, dim_head) which is what we need
        text_cos, text_sin = get_rotary_emb(self.symbols_rotary, cat_tokens)
        image_cos, image_sin = get_rotary_emb(self.positions_rotary, euclidean_tokens)
        rotary_pos_emb = ((text_cos, text_sin), (image_cos, image_sin))

        # Compute pairwise spatial attention biases for each layer
        # Only apply to the euclidean/image modality
        # Get depth from joint_embedding
        depth = len(self.joint_embedding.blocks)
        spatial_biases = []
        for layer_idx in range(depth):
            # Compute spatial bias for this layer: (B, H, L, L)
            spatial_bias = self.spatial_bias(atom_positions, layer_idx, mask=pos_mask)
            spatial_biases.append(spatial_bias)
        
        # Format: (None for text, spatial_bias for image) for each layer
        attn_biases = [(None, spatial_biases[layer_idx]) for layer_idx in range(depth)]

        text_tokens_hidden, euclidean_tokens_hidden = self.joint_embedding(
            modality_tokens = (cat_tokens, euclidean_tokens),
            modality_masks = (symbols_mask, pos_mask),
            time_cond = (symbols_time, pos_time),
            rotary_pos_emb = rotary_pos_emb,
            attn_bias = attn_biases,
        )
        if detach_hidden[0]:
            text_tokens_hidden = text_tokens_hidden.detach()
        if detach_hidden[1]:
            euclidean_tokens_hidden = euclidean_tokens_hidden.detach()

        cat_tokens = self.symbols_dit(
            modality_tokens = (text_tokens_hidden,),
            modality_masks = (symbols_mask,),
            time_cond = (symbols_time,),
            rotary_pos_emb = ((text_cos, text_sin),),
        )[0]


        euclidean_tokens = self.positions_dit(
            modality_tokens = (euclidean_tokens_hidden,),
            modality_masks = (pos_mask,),
            time_cond = (pos_time,),
            rotary_pos_emb = ((image_cos, image_sin),),
        )[0]
        # Fuse the tokens
        hidden_representation = torch.cat([cat_tokens, euclidean_tokens], dim=-1)
        label_representation = self.label_fuse(hidden_representation)
        insert_representation = self.insert_fuse(hidden_representation)
        euclidean_representation = self.euclidean_fuse(hidden_representation)

        # Predict unmasking probabilities for symbols
        label_logits = self.symbols_pred_layer(label_representation, symbols_time)
        # Clean data prediction
        clean_data_pred = self.positions_pred_layer(euclidean_representation, pos_time)
        # Clean data prediction for the insertion
        if not self.branching_flows:
            clean_data_unmasking= self.positions_unmask_pred(euclidean_representation, pos_time).view(B, L, self.vocab_size, self.euclidean_dim)
        # Insertion rate prediction
        # TODO : Maybe the insertion rate could use both cat tokens and euclidean ones
        insertion_rate = self.insertion_rate(insert_representation, pos_time).squeeze(-1)  # [B, L]
        insertion_rate = F.softplus(insertion_rate)
        insertion_rate = insertion_rate * pos_mask

        if self.branching_flows:
            return BranchingFlowsPrediction(
                clean_data=clean_data_pred,
                label_logits=label_logits,
                split_rates=insertion_rate,
            )
        else:
            return MultimodalModelPrediction(
                clean_data=clean_data_pred,
                label_logits=label_logits,
                insertion_rate=insertion_rate,
                clean_data_unmasking=clean_data_unmasking
            )


class MMDiTBothVar(nn.Module):
    def __init__(self, euclidean_dim, vocab_size, symbols_depth, positions_depth, **kwargs):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.vocab_size = vocab_size
        self.dim_modalities = kwargs['dim_modalities']
        self.dim_conds = kwargs['dim_conds']
        self.dim_symbols = self.dim_modalities[0]
        self.dim_positions = self.dim_modalities[1]
        # Extract attention parameters for rotary embeddings
        self.dim_head = kwargs.get('dim_head', 64)
        self.heads = kwargs.get('heads', 8)

        # Time Encoders
        self.symbols_time_encoder = TimestepEmbedder(self.dim_symbols)
        self.positions_time_encoder = TimestepEmbedder(self.dim_positions)

        # Text Embeddings
        self.symbols_embedder = nn.Embedding(vocab_size, self.dim_symbols)

        # Euclidean Embeddings        # Handle multi-dimensional coordinates [B, L, D] where D is the feature dimension
        # Each dimension is embedded separately using GaussianFourierProjection
        # Then combined and projected to dim_image
        # Each dimension gets embedded_dim features, so total is euclidean_dim * embedded_dim
        self.embedded_dim_per_feature = self.dim_positions // euclidean_dim
        self.gaussian_fourier_projs = nn.ModuleList([
            GaussianFourierProjection(self.embedded_dim_per_feature, scale=1.0)
            for _ in range(euclidean_dim)
        ])
        # Project the concatenated embeddings to dim_image
        # Total embedding size after concatenation: euclidean_dim * embedded_dim_per_feature
        total_embed_dim = euclidean_dim * self.embedded_dim_per_feature
        self.euclidean_proj = nn.Linear(total_embed_dim, self.dim_positions)

        # Rotary Positional Embeddings
        self.symbols_rotary = Rotary(dim=self.dim_head, base=10_000)
        self.positions_rotary = Rotary(dim=self.dim_head, base=10_000)

        # Pairwise spatial attention biases (layer-specific, head-specific)
        # We'll get depth from kwargs
        depth = kwargs.get('depth', 4)
        heads = kwargs.get('heads', 8)
        self.spatial_bias = PairwiseSpatialBias(
            num_heads=heads,
            num_layers=depth,
            spatial_feat_dim=32,
            max_distance=10.0
        )

        # Joint Embedding
        self.joint_embedding = MMDiT(**kwargs)

        # Single Embeddings
        sep_keys = ['depth', 'dim_modalities', 'dim_conds']
        block_kwargs = {k: v for k, v in kwargs.items() if k not in sep_keys}
        self.symbols_dit = MMDiT(depth = symbols_depth, dim_modalities = [self.dim_symbols], dim_conds = [self.dim_symbols], **block_kwargs)
        self.positions_dit = MMDiT(depth = positions_depth, dim_modalities = [self.dim_positions], dim_conds = [self.dim_positions], **block_kwargs)

        # Final Layers
        self.euc_insertion_rate = FinalLayer(self.dim_positions, 1)
        self.disc_insertion_rate = FinalLayer(self.dim_symbols, 1)
        self.symbols_pred_layer = FinalLayer(self.dim_symbols, vocab_size)
        self.positions_pred_layer = FinalLayer(self.dim_positions, euclidean_dim)

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
        layers_to_freeze = [self.symbols_embedder, self.symbols_time_encoder, 
                            self.image_embedder, self.positions_time_encoder,
                            self.joint_embedding]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_image(self):
        layers_to_freeze = [self.image_embedder, self.positions_time_encoder, self.positions_dit, self.positions_pred_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def freeze_text(self):
        layers_to_freeze = [self.symbols_embedder, self.symbols_time_encoder, self.symbols_dit, self.symbols_pred_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        *,
        cat_tokens,
        euclidean_tokens,
        symbols_mask = None,
        pos_mask = None,
        pos_time = None,
        symbols_time = None,
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
        
        # Store original positions for computing pairwise spatial biases
        atom_positions = euclidean_tokens  # [B, L, D]
        
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
        
        cat_tokens = self.symbols_embedder(cat_tokens)
        symbols_time = self.symbols_time_encoder(symbols_time)  # [B] -> [B, dim_text]
        pos_time = self.positions_time_encoder(pos_time)  # [B] -> [B, dim_image]

        # Generate rotary positional embeddings for each modality
        # The Rotary class returns (1, seq_len, 3, 1, dim_head) which is what we need
        text_cos, text_sin = get_rotary_emb(self.symbols_rotary, cat_tokens)
        image_cos, image_sin = get_rotary_emb(self.positions_rotary, euclidean_tokens)
        rotary_pos_emb = ((text_cos, text_sin), (image_cos, image_sin))

        # Compute pairwise spatial attention biases for each layer
        # Only apply to the euclidean/image modality
        # Get depth from joint_embedding
        depth = len(self.joint_embedding.blocks)
        spatial_biases = []
        for layer_idx in range(depth):
            # Compute spatial bias for this layer: (B, H, L, L)
            spatial_bias = self.spatial_bias(atom_positions, layer_idx, mask=pos_mask)
            spatial_biases.append(spatial_bias)
        
        # Format: (None for text, spatial_bias for image) for each layer
        attn_biases = [(None, spatial_biases[layer_idx]) for layer_idx in range(depth)]

        text_tokens_hidden, euclidean_tokens_hidden = self.joint_embedding(
            modality_tokens = (cat_tokens, euclidean_tokens),
            modality_masks = (symbols_mask, pos_mask),
            time_cond = (symbols_time, pos_time),
            rotary_pos_emb = rotary_pos_emb,
            attn_bias = attn_biases,
        )
        if detach_hidden[0]:
            text_tokens_hidden = text_tokens_hidden.detach()
        if detach_hidden[1]:
            euclidean_tokens_hidden = euclidean_tokens_hidden.detach()

        cat_tokens = self.symbols_dit(
            modality_tokens = (text_tokens_hidden,),
            modality_masks = (symbols_mask,),
            time_cond = (symbols_time,),
            rotary_pos_emb = ((text_cos, text_sin),),
        )[0]

        # Predict unmasking probabilities for symbols
        label_logits = self.symbols_pred_layer(cat_tokens, symbols_time)

        euclidean_tokens = self.positions_dit(
            modality_tokens = (euclidean_tokens_hidden,),
            modality_masks = (pos_mask,),
            time_cond = (pos_time,),
            rotary_pos_emb = ((image_cos, image_sin),),
        )[0]
        # Clean data prediction
        clean_data_pred = self.positions_pred_layer(euclidean_tokens, pos_time)
        # Euclidean insertion rate prediction
        euc_insertion_rate = self.euc_insertion_rate(euclidean_tokens, pos_time).squeeze(-1)  # [B, L]
        euc_insertion_rate = F.softplus(euc_insertion_rate)
        euc_insertion_rate = euc_insertion_rate * pos_mask
        # Discrete insertion rate prediction
        disc_insertion_rate = self.disc_insertion_rate(cat_tokens, symbols_time).squeeze(-1)  # [B, L]
        disc_insertion_rate = F.softplus(disc_insertion_rate)
        disc_insertion_rate = disc_insertion_rate * symbols_mask

 
        return MultimodalModelPredictionBothVar(
            clean_data=clean_data_pred,
            label_logits=label_logits,
            euc_insertion_rate=euc_insertion_rate,
            disc_insertion_rate=disc_insertion_rate,
        )

