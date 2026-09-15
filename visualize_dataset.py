from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.tokenizer import VocabTokenizer


def plot_layout_sample(
    xt: torch.Tensor,
    yt: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: VocabTokenizer,
    out_file_name: str,
    iter_num: int | None = None,
    title: str | None = None,
) -> None:
    """
    Visualize a generated layout sample (bounding boxes with category labels).

    Args:
        xt: Bbox coordinates [L, 4] - (x_center, y_center, w, h) normalized in [-1, 1]
        yt: Token IDs [L] for category labels
        mask: Valid positions [L] (True = valid bbox)
        tokenizer: VocabTokenizer for decoding labels (e.g. PUBLAYNET_VOCAB)
        out_file_name: Output path for the figure
        iter_num: Optional training iteration for the title
        title: Optional custom title
    """
    xt_np = xt.detach().cpu().numpy() if isinstance(xt, torch.Tensor) else np.array(xt)
    # Convert [-1, 1] to [0, 1] for display
    xt_np = (xt_np + 1) / 2
    yt_np = yt.detach().cpu().numpy() if isinstance(yt, torch.Tensor) else np.array(yt)
    mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask, dtype=bool)

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # y=0 at top (document convention)

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
    cat_colors = {f"<{c}>": colors[i % len(colors)] for i, c in enumerate(["text", "title", "list", "table", "figure"])}

    n_valid = 0
    for i in range(len(mask_np)):
        if not mask_np[i]:
            continue
        token_id = int(yt_np[i])
        if token_id in (bos_id, eos_id):
            continue

        x_center, y_center, w, h = xt_np[i]
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        x_min = np.clip(x_min, 0, 1)
        y_min = np.clip(y_min, 0, 1)
        w = np.clip(w, 0.01, 1 - x_min)
        h = np.clip(h, 0.01, 1 - y_min)

        if token_id in (pad_id, mask_id):
            label = "<pad>" if token_id == pad_id else "<M>"
            color = "#95a5a6"
        else:
            label = tokenizer.idx_to_atom.get(token_id, str(token_id))
            color = cat_colors.get(label, "#34495e")
        n_valid += 1

        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label_clean = label.replace("<", "").replace(">", "")
        ax.text(
            x_min,
            y_min - 0.01,
            label_clean,
            fontsize=7,
            color=color,
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor=color),
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    display_title = title or (f"Layout (iter {iter_num}, n={n_valid})" if iter_num is not None else f"Layout (n={n_valid})")
    ax.set_title(display_title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file_name, dpi=120, bbox_inches="tight")
    plt.close()
