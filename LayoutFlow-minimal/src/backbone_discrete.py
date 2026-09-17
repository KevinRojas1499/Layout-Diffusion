import torch
from torch import Tensor, nn
from einops import pack

from src.backbone import Block, TransformerEncoder


class DiscreteBackbone(nn.Module):
    '''
    src.backbone.Backbone with the category channel made discrete: the AnalogBit
    Linear embedding becomes an nn.Embedding over {0 = pad, 1..num_cat-1 = real
    classes, num_cat = [MASK]}, and the single vector-field head is split into a
    geometry velocity head (4) and a category logits head (num_cat). Everything
    else (cond-flag embedding, stacked geom+attr token, AdaLN transformer) is
    identical, so any FID gap vs LayoutFlow is attributable to the category
    modelling and not the network.
    '''

    def __init__(self, latent_dim=128, d_model=512, nhead=8, dim_feedforward=2048,
                 num_layers=4, dropout=0.1, num_cat=6):
        super().__init__()
        self.geom_dim = 4
        self.mask_id = num_cat
        self.cond_enc = nn.Embedding(6, latent_dim)
        self.geom_embed = nn.Linear(self.geom_dim, latent_dim)
        self.type_embed = nn.Embedding(num_cat + 1, latent_dim)
        self.elem_embed = nn.Linear(2 * latent_dim, d_model)

        layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.geom_head = nn.Linear(d_model, self.geom_dim)
        self.cat_head = nn.Linear(d_model, num_cat)

    def forward(self, geom: Tensor, cat: Tensor, cond_flags: Tensor, t: Tensor):
        '''geom (B,S,4), cat (B,S) long, cond_flags (B,S,5) 0/1 (last entry = category free), t (B,)'''
        geom_cond = cond_flags[:, :, :self.geom_dim].sum(-1)
        attr_cond = cond_flags[:, :, -1]
        g = self.geom_embed(geom) + self.cond_enc(geom_cond)
        a = self.type_embed(cat) + self.cond_enc(attr_cond)
        x, _ = pack([g, a], 'b s *')
        x = self.transformer(self.elem_embed(x), timestep=t)
        return self.geom_head(x), self.cat_head(x)
