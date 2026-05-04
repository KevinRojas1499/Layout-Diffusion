"""Cross-validate eval pipelines: feed LayoutFlow's saved samples through our
FIDNetV3+musig path. Should match LayoutFlow's test.py FID (8-9 range).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Variable-Length-Diffusion-Toy")
from eval.layout.fidnet import FIDNetV3
from eval.layout.metrics import frechet_from_stats, compute_alignment, compute_overlap

LAYOUTFLOW_ROOT = "/workspace/LayoutFlow"
SAMPLES = f"{LAYOUTFLOW_ROOT}/results/checkpoint_PubLayNet_LayoutFlow_uncond_bbox.pt"
WEIGHTS = f"{LAYOUTFLOW_ROOT}/pretrained/fid_publaynet.pth.tar"
MUSIG = f"{LAYOUTFLOW_ROOT}/pretrained/FIDNet_musig_test_publaynet.pt"

device = torch.device("cuda")

data = torch.load(SAMPLES)[:2000].to(device)
bbox_xywh = data[..., :4].to(torch.float32)  # center (x, y, w, h) in [0, 1]
label = data[..., 4].to(torch.long)          # 1..5 at valid positions
mask = data[..., 5].to(torch.bool)           # True = valid

cx, cy, w, h = bbox_xywh.unbind(-1)
ltrb = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

label = label * mask.long()
padding_mask = ~mask

fid_model = FIDNetV3(num_label=6, max_bbox=20).to(device)
sd = torch.load(WEIGHTS, map_location="cpu")
fid_model.load_state_dict({k.split("module.")[-1]: v for k, v in sd.items()})
fid_model.eval()

with torch.no_grad():
    feats = fid_model.extract_features(ltrb, label, padding_mask).cpu().numpy()

mu_fake = feats.mean(axis=0)
sigma_fake = np.cov(feats, rowvar=False)

musig = torch.load(MUSIG, map_location="cpu")
mu_real, sigma_real = musig[0].numpy(), musig[1:].numpy()

fid = frechet_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)

bbox_cxcywh_unit = bbox_xywh.cpu()
align = compute_alignment(bbox_cxcywh_unit, mask.cpu())
over = compute_overlap(bbox_cxcywh_unit, mask.cpu())

print(f"Cross-eval of LayoutFlow's saved samples through OUR pipeline:")
print(f"  N samples: {data.shape[0]}")
print(f"  FID = {fid:.4f}")
print(f"  Align (LayoutGAN++): {align['alignment-LayoutGAN++'].mean().item():.4f}")
print(f"  Overlap (LayoutGAN++): {over['overlap-LayoutGAN++'].mean().item():.4f}")
print(f"  (LayoutFlow's own test.py gave: FID 8.78 ± 0.52 over 10 runs, single-seed 8.86 last)")
