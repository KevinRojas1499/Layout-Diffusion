"""Feature extractor used by LayoutFlow / LayoutDiffusion for layout FID.

The encoder portion of this network is what we run; the decoder/discriminator
heads are kept so the released checkpoint loads cleanly. Architecturally
identical to LayoutDiffusion's `LayoutNet`, with `num_label = num_classes + 1`
(0 reserved for padding) and `max_bbox = 20` for PubLayNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)
        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        B = x.size(1)
        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)
        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)
        x = self.core(x, src_key_padding_mask=padding_mask)
        return x


class FIDNetV3(nn.Module):
    def __init__(self, num_label, d_model=256, nhead=4, num_layers=4, max_bbox=50):
        super().__init__()
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)
        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.fc_out_disc = nn.Linear(d_model, 1)
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)
        te = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)
        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask):
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]
