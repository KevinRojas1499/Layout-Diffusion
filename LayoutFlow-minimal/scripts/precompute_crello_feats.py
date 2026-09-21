'''
Frozen canvas features for content-aware Crello: DINOv2-small patch tokens on each plate
(scripts/prepare_crello_plates.py), average-pooled to an 8x8 grid -> (64, 384) fp16 per template.
Crello has no saliency map, so unlike the CGL features there is no 385th channel.
  python scripts/precompute_crello_feats.py <layout_dir> <plate_dir> <out_dir> [splits...]
'''
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@torch.no_grad()
def main(layout_dir, plate_dir, out_dir, splits):
    dev = 'cuda'
    net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev).eval()
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        recs = torch.load(f'{layout_dir}/{split}_canvas.pt', weights_only=False)
        ids, feats = [r['id'] for r in recs], []
        for i in range(0, len(recs), 128):
            batch = recs[i:i + 128]
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(f'{plate_dir}/{r["plate"]}').convert('RGB').resize((224, 224), Image.BICUBIC)))
                             for r in batch]).to(dev).permute(0, 3, 1, 2).float() / 255
            x = (x - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)
            tok = net.forward_features(x)['x_norm_patchtokens']                      # (B, 256, 384), 16x16
            tok = tok.view(-1, 16, 16, 384).permute(0, 3, 1, 2)
            tok = F.avg_pool2d(tok, 2).flatten(2).transpose(1, 2)                    # (B, 64, 384), 8x8
            feats.append(tok.half().cpu())
            if i % 2048 == 0:
                print(f'{split}: {i}/{len(recs)}', flush=True)
        feats = torch.cat(feats)
        torch.save({'id': ids, 'feats': feats}, f'{out_dir}/{split}.pt')
        print(f'{split}: {feats.shape} -> {out_dir}/{split}.pt', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or ['train', 'validation', 'test'])
