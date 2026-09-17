import torch
from torch import nn, Tensor
from src.backbone import Block, TransformerEncoder


class MultimodalMaskingBackbone(nn.Module):
    '''
    Like MaskingBackbone, but geometry and category are separate input/output
    channels rather than one combined analog-bit vector: category gets its own
    embedding table (with a real [MASK] id, index num_cat) instead of being
    zero-blanked as a continuous value, and its own classification head instead
    of sharing the regression head with geometry.
    '''

    def __init__(self, geom_dim, num_cat, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, use_pos_embed=False, max_len=20):
        super().__init__()
        self.geom_embed = nn.Linear(geom_dim, d_model)
        self.cat_embed = nn.Embedding(num_cat + 2, d_model)  # +2: reserved [MASK] and [PAD] ids
        self.mask_embed = nn.Embedding(2, d_model)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Embedding(max_len, d_model)
        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.geom_head = nn.Linear(d_model, geom_dim)
        self.cat_head = nn.Linear(d_model, num_cat)

    def forward(self, xt: Tensor, yt: Tensor, is_masked: Tensor, t: Tensor):
        tok = self.geom_embed(xt) + self.cat_embed(yt) + self.mask_embed(is_masked.long())
        if self.use_pos_embed:
            pos_ids = torch.arange(xt.shape[1], device=xt.device)
            tok = tok + self.pos_embed(pos_ids)[None]
        tok = self.transformer(tok, timestep=t)
        return self.geom_head(tok), self.cat_head(tok)
