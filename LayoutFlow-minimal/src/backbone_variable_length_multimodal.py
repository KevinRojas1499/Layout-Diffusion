import torch
import torch.nn.functional as F
from torch import nn, Tensor
from src.backbone import Block, TransformerEncoder


class VariableLengthMultimodalBackbone(nn.Module):
    '''
    Like MultimodalMaskingBackbone, plus a third output head predicting the
    Poisson insertion rate for the variable-length interpolant's insertion
    mechanism (see src/variable_length_multimodal_interpolant.py). Category
    vocab now includes both a [MASK] id and a [PAD] id (for not-yet-born
    positions) and a [BOS] id.
    '''

    def __init__(self, geom_dim, num_cat, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, use_pos_embed=False, max_len=21):
        super().__init__()
        vocab_size = num_cat + 3   # + [MASK], [PAD], [BOS]
        self.geom_embed = nn.Linear(geom_dim, d_model)
        self.cat_embed = nn.Embedding(vocab_size, d_model)
        self.mask_embed = nn.Embedding(2, d_model)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Embedding(max_len, d_model)
        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.geom_head = nn.Linear(d_model, geom_dim)
        self.cat_head = nn.Linear(d_model, num_cat)
        self.insertion_head = nn.Linear(d_model, 1)

    def forward(self, xt: Tensor, yt: Tensor, is_masked: Tensor, alive: Tensor, t: Tensor):
        tok = self.geom_embed(xt) + self.cat_embed(yt) + self.mask_embed(is_masked.long())
        if self.use_pos_embed:
            pos_ids = torch.arange(xt.shape[1], device=xt.device)
            tok = tok + self.pos_embed(pos_ids)[None]
        tok = self.transformer(tok, timestep=t)
        geom_pred = self.geom_head(tok)
        cat_logits = self.cat_head(tok)
        insertion_rate = F.softplus(self.insertion_head(tok).squeeze(-1))
        insertion_rate = insertion_rate * alive.to(insertion_rate.dtype)
        return geom_pred, cat_logits, insertion_rate
