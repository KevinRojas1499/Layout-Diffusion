"""Shared label constants and tokenizer builder for the layout (PubLayNet/RICO) workflow.

Originally lived alongside the layout-dm preprocessed dataset; lifted out so the
LayoutFlow pipeline can use them without depending on the layout-dm path.
"""

from __future__ import annotations

from utils.tokenizer import VocabTokenizer


# Canonical class-name orderings. The LayoutFlow / LayoutDiffusion FID network
# (`pretrained/fid_publaynet.pth.tar`) was trained against these orderings;
# using a different one scrambles emb_label lookups.
PUBLAYNET_LABELS = ["text", "title", "list", "table", "figure"]
RICO25_LABELS = [
    "Text", "Image", "Icon", "Text Button", "List Item", "Input",
    "Background Image", "Card", "Web View", "Radio Button", "Drawer",
    "Checkbox", "Advertisement", "Modal", "Pager Indicator", "Slider",
    "On/Off Switch", "Button Bar", "Toolbar", "Number Stepper",
    "Multi-Tab", "Date Picker", "Map View", "Video", "Bottom Navigation",
]


def label_to_token(name: str) -> str:
    """'Text Button' -> '<text_button>', 'On/Off Switch' -> '<on_off_switch>'."""
    return "<" + name.lower().replace("/", "_").replace(" ", "_") + ">"


PUBLAYNET_FID_CATEGORIES = tuple(label_to_token(n) for n in PUBLAYNET_LABELS)
RICO25_FID_CATEGORIES = tuple(label_to_token(n) for n in RICO25_LABELS)

# Shape: dataset_name -> (ordered_token_tuple, num_classes)
DATASET_REGISTRY = {
    "publaynet": (PUBLAYNET_FID_CATEGORIES, 5),
    "rico25": (RICO25_FID_CATEGORIES, 25),
}


def build_tokenizer(dataset_name: str) -> VocabTokenizer:
    """Build the canonical VocabTokenizer for a layout dataset."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_name {dataset_name!r}.")
    ordered_tokens, _ = DATASET_REGISTRY[dataset_name]
    return VocabTokenizer(vocab=set(ordered_tokens))
