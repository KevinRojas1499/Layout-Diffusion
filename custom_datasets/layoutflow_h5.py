"""LayoutFlow's HDF5 PubLayNet/RICO dataset.

LayoutFlow (Guerreiro et al., ECCV'24) ships preprocessed HDF5 data on Hugging
Face with the LayoutFormer++/LayoutDiffusion split. Each top-level key is one
layout, with fields:
    bbox       : [L, 4]  (x_top_left, y_top_left, w, h)  in [0, 1]
    categories : [L]     int in 1..num_classes (0 reserved for padding)
    length     : []      int in [1, 20]

We expose layouts in the form expected by this repo's training loop:
    x : [max_length, 4]  bbox in (cx, cy, w, h) normalized to [-1, 1]
    y : [max_length]     token ids from VocabTokenizer
    mask : [max_length]  True at valid positions
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from custom_datasets.layout_labels import (
    PUBLAYNET_LABELS,
    RICO25_LABELS,
    label_to_token,
)
from utils.tokenizer import VocabTokenizer


# h5 split filenames per dataset_name (only the LayoutFormer++/LayoutDiffusion split for now).
_H5_FILENAMES = {
    "publaynet": {
        "train": "publaynet_train.h5",
        "val": "publaynet_val.h5",
        "validation": "publaynet_val.h5",
        "test": "publaynet_test.h5",
    },
    "rico25": {
        "train": "ldm_rico_train.h5",
        "val": "ldm_rico_val.h5",
        "validation": "ldm_rico_val.h5",
        "test": "ldm_rico_test.h5",
    },
}

_LABELS_BY_DATASET = {
    "publaynet": PUBLAYNET_LABELS,
    "rico25": RICO25_LABELS,
}


def _cat_to_token_id(tokenizer: VocabTokenizer, dataset_name: str) -> torch.LongTensor:
    """Map h5 category id (1..N) -> tokenizer's token id. Index 0 stays as pad."""
    labels = _LABELS_BY_DATASET[dataset_name]
    table = torch.full((len(labels) + 1,), tokenizer.pad_token_id, dtype=torch.long)
    for i, name in enumerate(labels):
        table[i + 1] = tokenizer.atom_to_idx[label_to_token(name)]
    return table


class LayoutFlowH5Dataset(Dataset):
    """LayoutFlow-distributed h5 dataset (LayoutFormer++/LayoutDiffusion split).

    Args:
        tokenizer: a VocabTokenizer built from the dataset's category vocab.
        dataset_name: 'publaynet' or 'rico25'.
        split: 'train' / 'val' / 'test'.
        max_length: pad/truncate every layout to this length (LayoutFlow uses 20).
        data_path: directory containing the per-split h5 files (e.g.
            `/workspace/LayoutFlow-data/dataset/publaynet`).
    """

    def __init__(
        self,
        tokenizer: VocabTokenizer,
        dataset_name: str,
        split: str = "train",
        max_length: int = 20,
        data_path: str | None = None,
    ):
        if dataset_name not in _H5_FILENAMES:
            raise ValueError(f"Unknown dataset_name {dataset_name!r}.")
        if split not in _H5_FILENAMES[dataset_name]:
            raise ValueError(f"Unknown split {split!r}.")
        if data_path is None:
            raise ValueError("data_path is required (path to dataset/<name> dir from LayoutFlow's HF release).")
        path = os.path.join(data_path, _H5_FILENAMES[dataset_name][split])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Clone https://huggingface.co/JulianGuerreiro/LayoutFlow."
            )

        # Read everything into memory once at construction. The PubLayNet train h5 is ~700MB
        # but flattened bboxes/labels are small (~100MB), and avoids per-worker h5 file handles.
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            n = len(keys)
            lens = np.zeros(n, dtype=np.int64)
            offsets = np.zeros(n + 1, dtype=np.int64)
            # First pass: collect lengths + offsets.
            for i, k in enumerate(keys):
                L = int(np.array(f[k]["length"]))
                lens[i] = min(L, max_length)
                offsets[i + 1] = offsets[i] + lens[i]
            total = int(offsets[-1])
            x_all = np.zeros((total, 4), dtype=np.float32)
            y_all = np.zeros((total,), dtype=np.int64)
            for i, k in enumerate(keys):
                L = int(lens[i])
                if L == 0:
                    continue
                bbox = np.array(f[k]["bbox"])[:L].astype(np.float32)
                cats = np.array(f[k]["categories"])[:L].astype(np.int64).flatten()
                # (x_topleft, y_topleft, w, h) in [0,1] -> (cx, cy, w, h) in [-1, 1]
                bbox[:, 0] = bbox[:, 0] + bbox[:, 2] / 2.0
                bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2.0
                bbox = 2.0 * bbox - 1.0
                x_all[offsets[i] : offsets[i] + L] = bbox
                y_all[offsets[i] : offsets[i] + L] = cats

        cat_to_token = _cat_to_token_id(tokenizer, dataset_name)
        self._x = torch.from_numpy(x_all)
        self._y = cat_to_token[torch.from_numpy(y_all)]
        self._lens = torch.from_numpy(lens)
        self._starts = torch.from_numpy(offsets[:-1])

        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.n_layouts = n
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return self.n_layouts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self._starts[idx])
        n = int(self._lens[idx])
        return {"x": self._x[s : s + n], "y": self._y[s : s + n]}

    def dynamic_collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad each batch to its own max length."""
        lens = [b["x"].size(0) for b in batch]
        L = max(max(lens), 1)
        B = len(batch)
        x = torch.zeros(B, L, 4, dtype=torch.float32)
        y = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, (b, n) in enumerate(zip(batch, lens)):
            if n == 0:
                continue
            x[i, :n] = b["x"]
            y[i, :n] = b["y"]
            mask[i, :n] = True
        return {"x": x, "y": y, "mask": mask, "mask_x": mask, "mask_y": mask}
