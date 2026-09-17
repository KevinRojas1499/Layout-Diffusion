'''
Layout FID: a pretrained classifier-derived feature extractor (LayoutNet, frozen,
weights come from the upstream pretrained/ checkpoints) used to compute Frechet
distance between generated and real layout distributions. Trimmed from upstream's
fid_model.py, which also defines FIDNet/FIDNetV2/FIDNetV3 variants used by other
codebases (LayoutDM, LayoutGAN++) but not by LayoutFlow's own training/eval path
-- only LayoutNet is ever instantiated here.
'''
from collections import OrderedDict as OD

import numpy as np
import torch
import torch.nn as nn
from pytorch_fid.fid_score import calculate_frechet_distance

from src.utils import convert_bbox


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer('token_mask', torch.zeros(1, 1, dtype=torch.bool))
        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]; padding_mask: [B, N], False = valid, True = padded
        B = x.size(1)
        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)
        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)
        return self.core(x, src_key_padding_mask=padding_mask)


class LayoutNet(nn.Module):
    '''Pretrained feature extractor + auxiliary discriminator/reconstruction heads.
    Only `extract_features` is used at eval time; the other heads exist purely so the
    pretrained state_dict (trained with them) loads without key mismatches.'''

    def __init__(self, num_label=25, max_bbox=20):
        super().__init__()
        d_model, nhead, num_layers = 256, 4, 4
        num_label += 1

        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)
        self.enc_transformer = TransformerWithToken(d_model=d_model, dim_feedforward=d_model // 2,
                                                     nhead=nhead, num_layers=num_layers)
        self.fc_out_disc = nn.Linear(d_model, 1)

        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)
        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2)
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


class FID_score(nn.Module):
    '''Computes layout FID against precomputed val/test reference statistics.'''

    def __init__(self, dataset, pretrained_dir, calc_every_n=50):
        super().__init__()
        self.dataset = dataset
        self.calc_every_n = calc_every_n

        musig = torch.load(f'{pretrained_dir}/FIDNet_musig_val_{dataset.lower()}.pt', map_location='cpu')
        self.mu, self.sig = musig[0].numpy(), musig[1:].numpy()

        num_classes = 5 if dataset == 'PubLayNet' else 25
        self.fid_model = LayoutNet(num_classes, 20)
        state_dict = torch.load(f'{pretrained_dir}/fid_{dataset.lower()}.pth.tar', map_location='cpu')
        state = OD([(key.split('module.')[-1], v) for key, v in state_dict.items()])
        self.fid_model.load_state_dict(state)
        self.fid_model.requires_grad_(False)
        self.fid_model.eval()

    def calc_FID(self, data, format='xywh'):
        '''data: dict with bbox (N,S,4) in `format`, label (N,S), pad_mask (N,S) bool.'''
        with torch.no_grad():
            ltrb_bbox = convert_bbox(data['bbox'], f'{format}->ltrb') if format != 'ltrb' else data['bbox']
            ltrb_bbox = ltrb_bbox * data['pad_mask'][..., None]
            feats = self.fid_model.extract_features(ltrb_bbox, data['label'], ~data['pad_mask'])
            fake_mu = feats.cpu().numpy().mean(axis=0)
            fake_sig = np.cov(feats.cpu().numpy(), rowvar=False)
        return calculate_frechet_distance(fake_mu, fake_sig, self.mu, self.sig)
