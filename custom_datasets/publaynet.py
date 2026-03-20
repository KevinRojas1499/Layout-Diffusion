"""Dataset for PubLayNet document layout analysis.

Each sample returns:
    - x: normalized bounding boxes [max_length, 4] (x_center, y_center, w, h) in [-1, 1]
    - y: tokenized category labels [max_length]
    - mask: valid positions [max_length]

Usage:
    from utils.tokenizer import VocabTokenizer
    from custom_datasets.publaynet import PublayNetDataset, PUBLAYNET_VOCAB

    tokenizer = VocabTokenizer(vocab=PUBLAYNET_VOCAB)
    dataset = PublayNetDataset(tokenizer, max_length=64, split="train")

Annotations-only mode (avoids ~130GB image download):
    Pass annotations_dir pointing to a directory with train.json, val.json (COCO format).
    Sources for annotations (IBM DAX has been unreliable; try alternatives):
    - Kaggle: https://www.kaggle.com/datasets/captaintushar/publaynet-dataset
    - IBM DAX (if available): https://developer.ibm.com/exchanges/data/all/publaynet/
    - Or run extract_annotations_to_json() once to save from HuggingFace.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import datasets
from utils.tokenizer import VocabTokenizer

# PubLayNet category_id -> token string (for VocabTokenizer with <token> format)
CATEGORY_NAMES = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
PUBLAYNET_VOCAB = {"<text>", "<title>", "<list>", "<table>", "<figure>"}

# COCO split -> filename in labels.tar.gz
SPLIT_FILENAMES = {"train": "train.json", "validation": "val.json", "val": "val.json", "test": "test.json"}


def extract_annotations_to_json(
    output_dir: str,
    splits: tuple[str, ...] = ("train", "validation"),
    max_samples_per_split: int | None = None,
) -> None:
    """Extract annotations + image dimensions from HuggingFace to local COCO JSON.

    Run once on a machine with enough space (~130GB) to download the full dataset.
    Produces train.json, val.json etc. (~few hundred MB). Upload that directory
    to your cluster and use dataset.data_path=output_dir to train without images.

    Example:
        python -c "
        from custom_datasets.publaynet import extract_annotations_to_json
        extract_annotations_to_json('data/publaynet_labels', splits=('train', 'validation'))
        "
    """
    from tqdm import tqdm

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        hf_split = "validation" if split == "val" else split
        ds = datasets.load_dataset("jordanparker6/publaynet", split=hf_split)
        if max_samples_per_split is not None:
            ds = ds.select(range(min(max_samples_per_split, len(ds))))

        images = []
        annotations = []
        ann_id = 0

        for idx in tqdm(range(len(ds)), desc=f"Extracting {split}"):
            row = ds[idx]
            img = row["image"]
            w, h = img.size
            if w <= 0 or h <= 0:
                w, h = 1.0, 1.0

            images.append({"id": idx, "width": w, "height": h, "file_name": f"{idx}.png"})
            for ann in row["annotations"]:
                ann_copy = dict(ann)
                ann_copy["id"] = ann_id
                ann_copy["image_id"] = idx
                ann_id += 1
                annotations.append(ann_copy)

        coco = {"images": images, "annotations": annotations, "categories": []}
        out_file = output_path / SPLIT_FILENAMES.get(split, f"{split}.json")
        with open(out_file, "w") as f:
            json.dump(coco, f)
        print(f"Wrote {out_file} ({len(images)} images, {len(annotations)} annotations)")


def _main():
    """CLI for extract_annotations_to_json. Run: python -m custom_datasets.publaynet [output_dir]"""
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/publaynet_labels"
    print(f"Extracting to {output_dir} (requires ~130GB for HuggingFace download)")
    extract_annotations_to_json(output_dir, splits=("train", "validation"))
    print(f"Done. Copy {output_dir}/ to your cluster and set dataset.data_path={output_dir}")


class PublayNetDataset(Dataset):
    """PubLayNet document layout dataset.

    Supports two loading modes:
    1. annotations_dir: Load from COCO JSON only (no images). Use extract_annotations_to_json()
       on a machine with space, then upload the output dir to your cluster.
    2. HuggingFace: Full dataset with images (default, requires ~130GB).
    """

    def __init__(
        self,
        tokenizer: VocabTokenizer,
        max_length: int = 64,
        split: str = "train",
        max_samples: int | None = None,
        annotations_dir: str | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self._annotations_only = annotations_dir is not None

        if annotations_dir is not None:
            self._load_from_coco_json(annotations_dir, split, max_samples)
        else:
            self.hf_dataset = datasets.load_dataset(
                "jordanparker6/publaynet",
                split=split,
            )
            if max_samples is not None:
                self.hf_dataset = self.hf_dataset.select(
                    range(min(max_samples, len(self.hf_dataset)))
                )

    def _load_from_coco_json(self, annotations_dir: str, split: str, max_samples: int | None):
        """Load from COCO JSON annotations only (no images). Width/height come from metadata."""
        path = Path(annotations_dir)
        filename = SPLIT_FILENAMES.get(split, f"{split}.json")
        json_path = path / filename
        if not json_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {json_path}. "
                f"Run extract_annotations_to_json() on a machine with HuggingFace access, "
                f"then copy the output directory to this machine."
            )
        with open(json_path) as f:
            coco = json.load(f)

        # Build image_id -> {width, height, annotations}
        images_by_id = {img["id"]: {"width": img["width"], "height": img["height"]} for img in coco["images"]}
        anns_by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        # Build list of (width, height, annotations) for each image that has annotations
        self._samples = []
        for img in coco["images"]:
            img_id = img["id"]
            if img_id not in anns_by_image:
                continue
            w, h = img["width"], img["height"]
            if w <= 0 or h <= 0:
                w, h = 1.0, 1.0
            self._samples.append((w, h, anns_by_image[img_id]))

        if max_samples is not None:
            self._samples = self._samples[: max_samples]

    def __len__(self) -> int:
        if self._annotations_only:
            return len(self._samples)
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
        if self._annotations_only:
            w, h, annotations = self._samples[index]
        else:
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
            # Normalize to [-1, 1]
            x_center = 2 * (x_min + bw / 2) / w - 1
            y_center = 2 * (y_min + bh / 2) / h - 1
            nw = 2 * bw / w - 1
            nh = 2 * bh / h - 1
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


if __name__ == "__main__":
    _main()
