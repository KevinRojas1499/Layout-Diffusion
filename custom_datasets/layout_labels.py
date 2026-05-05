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
# Order matches LayoutFlow's RICO `TYPE_2_CAT` (positions 1..25). The h5 files
# encode element type as `type ∈ 1..25` using this exact order, and LayoutFlow's
# RICO LayoutNet was trained with the same label index → embedding mapping.
# Keeping the position alignment is what lets the FID metric round-trip.
RICO25_LABELS = [
    "Advertisement", "Video", "Checkbox", "Drawer", "Icon",
    "Image", "Input", "List Item", "Modal", "Pager Indicator",
    "Text", "Toolbar", "Web View", "Map View", "Text Button",
    "Background Image", "Slider", "Multi-Tab", "Radio Button", "Date Picker",
    "Number Stepper", "Card", "On/Off Switch", "Bottom Navigation", "Button Bar",
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
