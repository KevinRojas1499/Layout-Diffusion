"""Layout metrics — Alignment, Overlap, Frechet distance.

Alignment and Overlap are ported from
    layout-dm/src/trainer/trainer/helpers/metric.py
which itself credits AC-LayoutGAN [Lee+, TVCG'20], LayoutGAN++ [Kikuchi+, MM'21],
and NDN [Lee+, CVPR'20].

Frechet distance is the same numerical procedure as
`pytorch_fid.fid_score.calculate_frechet_distance` but inlined to avoid the dep.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from scipy import linalg
from torch import BoolTensor, FloatTensor


def convert_xywh_to_ltrb(bbox):
    """xywh (cx, cy, w, h) -> ltrb (xl, yt, xr, yb). Works on tensors or arrays."""
    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xl = cx - w / 2
    xr = cx + w / 2
    yt = cy - h / 2
    yb = cy + h / 2
    return xl, yt, xr, yb


def compute_alignment(bbox: FloatTensor, mask: BoolTensor) -> Dict[str, FloatTensor]:
    """Per-layout alignment scores (lower is better).

    Args:
        bbox: [B, S, 4] in (cx, cy, w, h), normalized to [0, 1].
        mask: [B, S] bool, True for valid bboxes.

    Returns dict of three variants:
        alignment-LayoutGAN++  (the one most papers including LayoutDM report)
        alignment-ACLayoutGAN  (un-normalized sum)
        alignment-NDN          (alternative formulation)
    """
    S = bbox.size(1)
    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log(1 - X)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask.float(), "b s -> b", reduction="sum").clamp(min=1)
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")
    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0
    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    return {
        "alignment-LayoutGAN++": score_normalized,
        "alignment-ACLayoutGAN": score,
        "alignment-NDN": score_Y,
    }


def compute_overlap(bbox: FloatTensor, mask: BoolTensor) -> Dict[str, FloatTensor]:
    """Per-layout overlap scores (lower is better).

    Args:
        bbox: [B, S, 4] in (cx, cy, w, h), normalized to [0, 1].
        mask: [B, S] bool, True for valid bboxes.

    Returns dict with LayoutGAN++ (normalized), ACLayoutGAN, and LayoutGAN (upper-tri sum).
    """
    B, S = mask.size()
    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=ai.device)
    batch_mask[:, idx, idx] = True
    ai = ai.masked_fill(batch_mask, 0)
    ar = torch.nan_to_num(ai / a1)

    score = reduce(ar, "b s1 s2 -> b", reduction="sum")
    score_normalized = score / reduce(mask.float(), "b s -> b", reduction="sum").clamp(min=1)
    score_normalized[torch.isnan(score_normalized)] = 0.0

    ids = torch.arange(S)
    ii, jj = torch.meshgrid(ids, ids, indexing="ij")
    ai = ai.clone()
    ai[repeat(ii >= jj, "s1 s2 -> b s1 s2", b=B)] = 0.0
    overlap = reduce(ai, "b s1 s2 -> b", reduction="sum")

    return {
        "overlap-LayoutGAN++": score_normalized,
        "overlap-ACLayoutGAN": score,
        "overlap-LayoutGAN": overlap,
    }


def frechet_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance from precomputed Gaussian stats.

    Same procedure as `pytorch_fid.fid_score.calculate_frechet_distance`.
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def frechet_distance(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between two Gaussians fit to feature sets."""
    return frechet_from_stats(
        feats_real.mean(axis=0),
        np.cov(feats_real, rowvar=False),
        feats_fake.mean(axis=0),
        np.cov(feats_fake, rowvar=False),
        eps=eps,
    )
