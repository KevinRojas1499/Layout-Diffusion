"""Dataset for PubLayNet document layout analysis.

Each sample returns:
    - x: normalized bounding boxes [max_length, 4] (x_center, y_center, w, h) in [0, 1]
    - y: tokenized category labels [max_length]
    - mask: valid positions [max_length]

Usage:
    from utils.tokenizer import VocabTokenizer
    from custom_datasets.publaynet import PublayNetDataset, PUBLAYNET_VOCAB

    tokenizer = VocabTokenizer(vocab=PUBLAYNET_VOCAB)
    dataset = PublayNetDataset(tokenizer, max_length=64, split="train")
"""

import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import VocabTokenizer

# PubLayNet category_id -> token string (for VocabTokenizer with <token> format)
CATEGORY_NAMES = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
PUBLAYNET_VOCAB = {"<text>", "<title>", "<list>", "<table>", "<figure>"}


class PublayNetDataset(Dataset):
    """PubLayNet document layout dataset.

    Loads from HuggingFace (jordanparker6/publaynet). Each sample has variable-length
    annotations (bbox + category). Bboxes are normalized to [0, 1] by image dimensions.
    """

    def __init__(
        self,
        tokenizer: VocabTokenizer,
        max_length: int = 64,
        split: str = "train",
        max_samples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.hf_dataset = datasets.load_dataset(
            "jordanparker6/publaynet",
            split=split,
        )
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _get_category_token(self, ann: dict) -> str:
        """Get category token string from annotation."""
        cat_id = ann.get("category_id")
        if cat_id is None:
            cat = ann.get("category", {})
            cat_id = cat.get("category_id", 0)
            name = cat.get("name")
            if name is not None:
                return f"<{name}>"
        name = CATEGORY_NAMES.get(cat_id, "text")
        return f"<{name}>"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.hf_dataset[index]
        img = sample["image"]
        annotations = sample["annotations"]

        w, h = img.size
        if w <= 0 or h <= 0:
            w, h = 1.0, 1.0

        bboxes = []
        categories = []

        for ann in annotations:
            bbox = ann["bbox"]  # [x_min, y_min, width, height]
            x_min, y_min, bw, bh = bbox
            # Normalize to [0, 1]
            x_center = (x_min + bw / 2) / w
            y_center = (y_min + bh / 2) / h
            nw = bw / w
            nh = bh / h
            bboxes.append([x_center, y_center, nw, nh])
            categories.append(self._get_category_token(ann))

        original_length = len(bboxes)

        # Pad bboxes to max_length
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if len(bboxes) < self.max_length:
            pad = np.zeros((self.max_length - len(bboxes), 4), dtype=np.float32)
            bboxes = np.concatenate([bboxes, pad], axis=0)
        elif len(bboxes) > self.max_length:
            bboxes = bboxes[: self.max_length]
            categories = categories[: self.max_length]
            original_length = self.max_length

        x = torch.from_numpy(bboxes).float()

        # Tokenize categories: "<text><title><figure>" -> tensor (pad by token count, not char count)
        cat_str = "".join(categories)
        tokens = self.tokenizer.tokenize(cat_str)
        n_tokens = len(tokens)
        if n_tokens < self.max_length:
            pad = torch.full(
                (self.max_length - n_tokens,),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            y = torch.cat([tokens, pad])
        else:
            y = tokens[: self.max_length]

        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:original_length] = True

        return {"x": x, "y": y, "mask": mask}
