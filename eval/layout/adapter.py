"""Adapters between this repo's layout format and FIDNet's expected format.

This repo represents each layout sample as:
    xt: [B, L, 4]  bbox in (cx, cy, w, h) normalized to [-1, 1]
    yt: [B, L]     token IDs from VocabTokenizer (vocab + special tokens)
    mask: [B, L]   bool, True at positions the model considers active

FIDNetV3 expects:
    bbox: [B, L, 4]      in (cx, cy, w, h) normalized to [0, 1]
    label: [B, L]        long, class indices in [0, num_classes)
    padding_mask: [B, L] bool, True at PADDED (non-valid) positions
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import BoolTensor, LongTensor, Tensor

from utils.tokenizer import VocabTokenizer


def bbox_neg1_1_to_01(bbox: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) from [-1, 1] to [0, 1] and clamp.

    The dataset normalizer in `custom_datasets/publaynet.py` maps both centers
    and sizes via `2 * x - 1`, so the inverse for both is `(x + 1) / 2`.
    """
    return ((bbox + 1.0) / 2.0).clamp(0.0, 1.0)


def build_class_index_map(
    tokenizer: VocabTokenizer,
    category_tokens: Iterable[str],
) -> tuple[LongTensor, int]:
    """Map token IDs -> contiguous class indices.

    Args:
        tokenizer: the VocabTokenizer used at training.
        category_tokens: ordered iterable of category token strings (e.g.
            ('<text>','<title>','<list>','<table>','<figure>') in the order the
            FID network was trained on).

    Returns:
        (id_to_class, num_classes)
        id_to_class: LongTensor of size tokenizer.vocab_size where entry i is
            the class index for token-id i, or -1 if token i is a special / non-
            category token.
        num_classes: len(category_tokens).
    """
    cat_list = list(category_tokens)
    num_classes = len(cat_list)
    table = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long)
    for cls_idx, tok_str in enumerate(cat_list):
        if tok_str not in tokenizer.atom_to_idx:
            raise ValueError(f"Category token {tok_str!r} not in tokenizer vocab.")
        table[tokenizer.atom_to_idx[tok_str]] = cls_idx
    return table, num_classes


def prepare_for_fidnet(
    xt: Tensor,
    yt: LongTensor,
    mask: BoolTensor,
    id_to_class: LongTensor,
    max_seq_length: int,
) -> tuple[Tensor, LongTensor, BoolTensor, BoolTensor]:
    """Prepare a batch of layouts for FIDNetV3 forward.

    Drops/pads to `max_seq_length`. Positions whose token maps to -1 (specials
    like BOS / pad / mask) are forced inactive regardless of the input `mask`.

    Args:
        xt: [B, L, 4] bbox in (cx, cy, w, h), normalized to [-1, 1].
        yt: [B, L] token IDs.
        mask: [B, L] bool, True at active positions.
        id_to_class: LongTensor[vocab_size] from `build_class_index_map`.
        max_seq_length: the L the FID network was trained for (20 for LayoutFlow's LayoutNet).

    Returns:
        bbox: [B, max_seq_length, 4] in [0, 1]
        label: [B, max_seq_length] class indices in [0, num_classes); 0 at padding
        padding_mask: [B, max_seq_length] bool, True at padded positions
        valid_mask: [B, max_seq_length] bool, True at non-padded positions
    """
    device = xt.device
    id_to_class = id_to_class.to(device)

    cls = id_to_class[yt.clamp(min=0)]
    valid = mask & (cls >= 0)

    # Compact valid entries to the front, then pad/truncate to max_seq_length.
    B, L, D = xt.shape
    out_bbox = torch.zeros(B, max_seq_length, D, device=device, dtype=xt.dtype)
    out_label = torch.zeros(B, max_seq_length, device=device, dtype=torch.long)
    out_valid = torch.zeros(B, max_seq_length, device=device, dtype=torch.bool)

    bbox01 = bbox_neg1_1_to_01(xt)
    for i in range(B):
        idx = valid[i].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > max_seq_length:
            idx = idx[:max_seq_length]
        n = idx.numel()
        if n > 0:
            out_bbox[i, :n] = bbox01[i, idx]
            out_label[i, :n] = cls[i, idx]
            out_valid[i, :n] = True

    padding_mask = ~out_valid
    return out_bbox, out_label, padding_mask, out_valid
