'''
Frozen canvas features for content-aware layout generation (RALF's CGL / PKU parquet release).
Each canvas -> DINOv2-small patch tokens on a 224x224 resize, average-pooled to an 8x8 grid, with
the saliency map's mean over the same cell appended: (64, 385) fp16 per image. Writes one file per
split: {'id': [...], 'feats': (N, 64, 385) fp16}.
  python scripts/precompute_canvas_feats.py <parquet_dir> <out_dir> [splits...]
'''
import glob, io, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_split(parquet_dir, split):
    files = sorted(glob.glob(f'{parquet_dir}/{split}-*.parquet'))
    assert files, f'no parquet for split {split} in {parquet_dir}'
    return pq.ParquetDataset(files).read(columns=['id', 'image', 'saliency'])


@torch.no_grad()
def main(parquet_dir, out_dir, splits):
    dev = 'cuda'
    net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev).eval()
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        tab = load_split(parquet_dir, split)
        ids, feats = tab.column('id').to_pylist(), []
        imgs, sals = tab.column('image').to_pylist(), tab.column('saliency').to_pylist()
        for i in range(0, len(ids), 256):
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB').resize((224, 224), Image.BICUBIC)))
                             for r in imgs[i:i + 256]]).to(dev).permute(0, 3, 1, 2).float() / 255
            x = (x - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)
            s = torch.stack([torch.from_numpy(np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('L').resize((224, 224), Image.BILINEAR)))
                             for r in sals[i:i + 256]]).to(dev).float().unsqueeze(1) / 255
            tok = net.forward_features(x)['x_norm_patchtokens']                      # (B, 256, 384), 16x16 grid
            tok = tok.view(-1, 16, 16, 384).permute(0, 3, 1, 2)
            tok = F.avg_pool2d(tok, 2).flatten(2).transpose(1, 2)                    # (B, 64, 384), 8x8 grid
            sal = F.adaptive_avg_pool2d(s, 8).flatten(2).transpose(1, 2)             # (B, 64, 1)
            feats.append(torch.cat([tok, sal], -1).half().cpu())
        feats = torch.cat(feats)
        torch.save({'id': ids, 'feats': feats}, f'{out_dir}/{split}.pt')
        print(f'{split}: {tuple(feats.shape)} -> {out_dir}/{split}.pt')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ['train', 'val', 'test', 'with_no_annotations_test'])
