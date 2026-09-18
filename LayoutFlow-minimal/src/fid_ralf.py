'''
Layout FID for the content-aware datasets, computed exactly as RALF's eval.py does it: FIDNetV3
features (the encoder half; order-invariant transformer with a class token over (cx, cy, w, h, label))
with RALF's released weights, against their precomputed ground-truth features of the validation
split. FIDNetV3 is copied from RALF (CyberAgentAILab/RALF, image2layout/train/fid/model.py,
Apache-2.0) so that the training venv does not need their package.
'''
import numpy as np
import torch
import torch.nn as nn
from pytorch_fid.fid_score import calculate_frechet_distance

# RALF's class ids are sorted(vocabulary): CGL -> embellishment 0, logo 1, text 2, underlay 3.
# Ours (src.data.CGL.LABELS, 1-indexed): logo 1, text 2, underlay 3, embellishment 4.
OURS_TO_RALF = {'CGL': {1: 1, 2: 2, 3: 3, 4: 0}}


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer('token_mask', torch.zeros(1, 1, dtype=torch.bool))
        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward), num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        B = x.size(1)
        x = torch.cat([self.token.expand(-1, B, -1), x], dim=0)
        padding_mask = torch.cat([self.token_mask.expand(B, -1), src_key_padding_mask], dim=1)
        return self.core(x, src_key_padding_mask=padding_mask)


class FIDNetV3(nn.Module):
    '''Encoder + decoder as in RALF (the decoder is only needed to load the state dict).'''

    def __init__(self, num_label, d_model=256, nhead=4, num_layers=4, max_bbox=10):
        super().__init__()
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)
        self.enc_transformer = TransformerWithToken(d_model=d_model, dim_feedforward=d_model // 2, nhead=nhead, num_layers=num_layers)
        self.fc_out_disc = nn.Linear(d_model, 1)
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)
        self.dec_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2), num_layers=num_layers)
        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, mask):
        '''bbox (B,N,4) cxcywh in [0,1], label (B,N) RALF ids, mask (B,N) True = real element -> (B, d_model)'''
        x = self.enc_fc_in(torch.cat([self.fc_bbox(bbox), self.emb_label(label)], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        return self.enc_transformer(x, ~mask)[0]


class RalfFID:
    '''Drop-in for src.fid.FID_score: calc_FID(gen_data, format) against the val-split GT features.'''

    def __init__(self, dataset, ralf_cache, calc_every_n=10, empty_id=None, split='val'):
        self.calc_every_n = calc_every_n
        self.empty_id = empty_id
        self.label_map = OURS_TO_RALF[dataset]
        name = dataset.lower()
        self.net = FIDNetV3(num_label=len(self.label_map))
        state = torch.load(f'{ralf_cache}/PRECOMPUTED_WEIGHT_DIR/fidnet/{name}/model_best.pth.tar', map_location='cpu')['state_dict']
        self.net.load_state_dict(state)
        self.net.eval().requires_grad_(False)
        feats = torch.load(f'{ralf_cache}/eval_gt_features/{name}_FIDNetV3_features.pth', map_location='cpu')['layout'][split].numpy()
        self.mu_real, self.cov_real = feats.mean(0), np.cov(feats, rowvar=False)

    @torch.no_grad()
    def calc_FID(self, gen, format='xywh'):
        assert format == 'xywh'
        bbox, label, mask = gen['bbox'], gen['label'], gen['pad_mask'].clone()
        if self.empty_id is not None:
            mask &= label != self.empty_id
        lut = torch.zeros(max(self.label_map) + 1, dtype=torch.long, device=label.device)
        for k, v in self.label_map.items():
            lut[k] = v
        label = lut[label.clamp(0, max(self.label_map))]
        self.net.to(bbox.device)
        feats = torch.cat([self.net.extract_features(bbox[i:i + 512], label[i:i + 512], mask[i:i + 512]) for i in range(0, len(bbox), 512)]).cpu().numpy()
        return float(calculate_frechet_distance(feats.mean(0), np.cov(feats, rowvar=False), self.mu_real, self.cov_real))
