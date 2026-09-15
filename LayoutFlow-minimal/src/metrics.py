'''Layout quality metrics: alignment, overlap, and maximum IoU (Kikuchi+, ACMMM'21).
Copied close to verbatim from upstream -- this is dataset/model-agnostic geometry
math with no framework dependency, nothing to trim.'''
from itertools import chain

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def compute_iou(box_1, box_2):
    lib = np if isinstance(box_1, np.ndarray) else torch
    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    l_max, r_min = lib.maximum(l1, l2), lib.minimum(r1, r2)
    t_max, b_min = lib.maximum(t1, t2), lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))
    return ai / (a1 + a2 - ai)


def _max_iou_for_layout(layout_1, layout_2):
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi, _bj = bi[np.where(li == l)], bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        if True in torch.isnan(iou):
            continue
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def _max_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([_max_iou_for_layout(layouts_1[i], layouts_2[j]) for i, j in zip(ii, jj)]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def _group_by_cats(layout_list):
    out = {}
    for bs, ls in layout_list:
        out.setdefault(str(sorted(ls.tolist())), []).append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2):
    c2bl_1, c2bl_2 = _group_by_cats(layouts_1), _group_by_cats(layouts_2)
    keys = list(set(c2bl_1) & set(c2bl_2))
    scores = [_max_iou((c2bl_1[k], c2bl_2[k])) for k in keys]
    scores = np.asarray(list(chain.from_iterable(scores)))
    return scores.mean().item() if len(scores) else 0.0


def compute_overlap(bbox, mask, format='xywh'):
    '''Attribute-conditioned LayoutGAN, sec 3.6.3 (Overlapping Loss).'''
    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0).permute(2, 0, 1)
    conv = convert_xywh_to_ltrb if format == 'xywh' else (lambda x: x)
    l1, t1, r1, b1 = conv(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = conv(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    l_max, r_min = torch.maximum(l1, l2), torch.minimum(r1, r2)
    t_max, b_min = torch.maximum(t1, t2), torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)
    ar = torch.from_numpy(np.nan_to_num((ai / a1).numpy()))
    score = torch.from_numpy(np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy()))
    return score.mean().item()


def compute_overlap_ignore_bg(bbox, label, mask, format='xywh'):
    '''RICO-specific: some categories (list item/card/bg image/modal) are expected to
    overlap by design, so they're excluded from the overlap penalty.'''
    for cat in (4, 9, 11, 17):
        mask = torch.where(label == cat, False, mask)
    return compute_overlap(bbox, mask, format)


def compute_alignment(bbox, mask, format='xywh'):
    '''Attribute-conditioned LayoutGAN, sec 3.6.4 (Alignment Loss).'''
    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox) if format == 'xywh' else bbox
    xc, yc = (xr + xl) / 2, (yt + yb) / 2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)
    X = -torch.log(1 - X)
    score = torch.from_numpy(np.nan_to_num((X.sum(-1) / mask.float().sum(-1)).numpy()))
    return score.mean().item()
