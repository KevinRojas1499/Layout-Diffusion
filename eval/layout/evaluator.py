"""LayoutEvaluator — FID + Alignment evaluation against LayoutFlow's PubLayNet pipeline.

Wraps LayoutDiffusion's LayoutNet (the feature extractor LayoutFlow uses) and
the precomputed test-split (mu, sigma) shipped at
`<layoutflow_root>/pretrained/FIDNet_musig_test_publaynet.pt`. A generated batch
in this repo's format is run through LayoutNet and FID is computed against the
precomputed Gaussian — directly comparable to LayoutFlow's reported numbers.

Typical usage:

    evaluator = LayoutEvaluator.for_dataset(
        dataset_name="publaynet",
        tokenizer=tokenizer,
        device=device,
    )
    metrics = evaluator.evaluate_samples(xt, yt, mask)
    # {"fid": ..., "alignment-LayoutGAN++": ..., "overlap-LayoutGAN++": ..., ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import BoolTensor, LongTensor, Tensor

from custom_datasets.layout_labels import DATASET_REGISTRY
from utils.tokenizer import VocabTokenizer

from .adapter import build_class_index_map, prepare_for_fidnet
from .fidnet import FIDNetV3
from .metrics import compute_alignment, compute_overlap, frechet_from_stats


# LayoutFlow's evaluation pipeline (LayoutDiffusion's LayoutNet) uses max_bbox=20
# and labels 1..N for valid bboxes (0 is reserved for padding/background).
LAYOUTFLOW_MAX_BBOX = 20
LAYOUTFLOW_DEFAULT_ROOT = "/workspace/LayoutFlow"

# Our dataset_name → the suffix LayoutFlow uses on its pretrained asset filenames
# (`fid_<suffix>.pth.tar`, `FIDNet_musig_test_<suffix>.pt`). They drop the "25"
# from rico25.
_LAYOUTFLOW_ASSET_SUFFIX = {
    "publaynet": "publaynet",
    "rico25": "rico",
}


@dataclass
class LayoutEvaluator:
    fid_model: FIDNetV3
    id_to_class: LongTensor
    num_classes: int
    max_seq_length: int
    device: torch.device
    real_mu: np.ndarray
    real_sigma: np.ndarray
    label_offset: int = 1  # LayoutNet uses 0=pad, 1..N=classes

    @classmethod
    def for_dataset(
        cls,
        dataset_name: str,
        tokenizer: VocabTokenizer,
        device: torch.device,
        layoutflow_root: str | None = None,
    ) -> "LayoutEvaluator":
        """Build an evaluator that mirrors LayoutFlow's eval pipeline.

        Loads `LayoutNet` weights from `<root>/pretrained/fid_<dataset>.pth.tar`
        and the precomputed test-split (mu, sigma) from
        `<root>/pretrained/FIDNet_musig_test_<dataset>.pt`. Currently only
        publaynet ships the assets needed for this evaluator.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset_name {dataset_name!r}. Known: {list(DATASET_REGISTRY)}"
            )
        ordered_tokens, num_classes = DATASET_REGISTRY[dataset_name]
        id_to_class, _ = build_class_index_map(tokenizer, ordered_tokens)

        root = layoutflow_root or LAYOUTFLOW_DEFAULT_ROOT
        suffix = _LAYOUTFLOW_ASSET_SUFFIX[dataset_name]
        weight_path = os.path.join(root, "pretrained", f"fid_{suffix}.pth.tar")
        musig_path = os.path.join(root, "pretrained", f"FIDNet_musig_test_{suffix}.pt")
        if not os.path.exists(weight_path) or not os.path.exists(musig_path):
            raise FileNotFoundError(
                f"LayoutFlow assets not found under {root}/pretrained. "
                "Clone https://huggingface.co/JulianGuerreiro/LayoutFlow."
            )
        sd = torch.load(weight_path, map_location="cpu")
        state = {k.split("module.")[-1]: v for k, v in sd.items()}
        fid_model = FIDNetV3(num_label=num_classes + 1, max_bbox=LAYOUTFLOW_MAX_BBOX).to(device)
        fid_model.load_state_dict(state)
        fid_model.eval()
        musig = torch.load(musig_path, map_location="cpu")
        return cls(
            fid_model=fid_model,
            id_to_class=id_to_class,
            num_classes=num_classes,
            max_seq_length=LAYOUTFLOW_MAX_BBOX,
            device=device,
            real_mu=musig[0].numpy(),
            real_sigma=musig[1:].numpy(),
        )

    @torch.no_grad()
    def extract_features(
        self,
        xt: Tensor,
        yt: LongTensor,
        mask: BoolTensor,
    ) -> tuple[np.ndarray, BoolTensor, Tensor, LongTensor]:
        """Run a batch through LayoutNet and return (features, valid_mask, bbox01, label).

        Output shapes/dtypes mirror what's needed downstream:
        bbox01 in (cx, cy, w, h) [0,1], label as the offset class index, valid_mask True
        at non-padded positions.
        """
        bbox01, label, padding_mask, valid_mask = prepare_for_fidnet(
            xt.to(self.device),
            yt.to(self.device),
            mask.to(self.device),
            self.id_to_class,
            self.max_seq_length,
        )
        # LayoutNet uses labels 1..N for valid bboxes (0 = pad/bg).
        label = label + valid_mask.long() * self.label_offset
        # LayoutNet was trained on (l, t, r, b); convert from xywh.
        cx, cy, w, h = bbox01.unbind(-1)
        ltrb = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        feats = self.fid_model.extract_features(ltrb, label, padding_mask)
        return feats.cpu().numpy(), valid_mask.cpu(), bbox01.cpu(), label.cpu()

    @torch.no_grad()
    def evaluate_samples(
        self,
        xt: Tensor,
        yt: LongTensor,
        mask: BoolTensor,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Compute FID + Alignment + Overlap for a batch of generated samples.

        Args:
            xt: [B, L, 4] bbox in (cx, cy, w, h), normalized to [-1, 1] (this repo's convention).
            yt: [B, L] token IDs from VocabTokenizer.
            mask: [B, L] bool, True at active positions.

        Returns dict of scalar metrics: fid, alignment-LayoutGAN++ (recommended),
        overlap-LayoutGAN++, num_empty_layouts, num_samples.
        """
        all_feats = []
        bboxes_for_geom = []
        masks_for_geom = []
        n_empty = 0
        B = xt.shape[0]
        for i in range(0, B, batch_size):
            j = min(i + batch_size, B)
            feats, valid, bbox01, _ = self.extract_features(xt[i:j], yt[i:j], mask[i:j])
            all_feats.append(feats)
            bboxes_for_geom.append(bbox01)
            masks_for_geom.append(valid)
            n_empty += int((valid.sum(dim=1) == 0).sum().item())

        feats_fake = np.concatenate(all_feats, axis=0)
        bbox_all = torch.cat(bboxes_for_geom, dim=0)
        mask_all = torch.cat(masks_for_geom, dim=0)

        # Drop empty layouts from geometric metrics so they don't bias the per-layout averages.
        nonempty = mask_all.sum(dim=1) > 0
        if nonempty.any():
            align = {k: v.mean().item() for k, v in compute_alignment(bbox_all[nonempty], mask_all[nonempty]).items()}
            over = {k: v.mean().item() for k, v in compute_overlap(bbox_all[nonempty], mask_all[nonempty]).items()}
        else:
            align = {k: float("nan") for k in
                     ("alignment-LayoutGAN++", "alignment-ACLayoutGAN", "alignment-NDN")}
            over = {k: float("nan") for k in
                    ("overlap-LayoutGAN++", "overlap-ACLayoutGAN", "overlap-LayoutGAN")}

        mu_fake = feats_fake.mean(axis=0)
        sigma_fake = np.cov(feats_fake, rowvar=False)
        fid = frechet_from_stats(self.real_mu, self.real_sigma, mu_fake, sigma_fake)

        return {
            "fid": fid,
            **align,
            **over,
            "num_empty_layouts": n_empty,
            "num_samples": B,
        }
