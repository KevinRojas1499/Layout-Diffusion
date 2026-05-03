"""Layout evaluation against LayoutFlow's PubLayNet pipeline.

Public API:
    LayoutEvaluator: loads LayoutNet + precomputed test musig and exposes
        `.evaluate_samples(xt, yt, mask)`.
    compute_alignment, compute_overlap: pure-geometric metrics on bbox+mask.
    frechet_from_stats / frechet_distance: FID helpers.
"""

from .evaluator import LayoutEvaluator
from .metrics import compute_alignment, compute_overlap, frechet_distance, frechet_from_stats

__all__ = [
    "LayoutEvaluator",
    "compute_alignment",
    "compute_overlap",
    "frechet_distance",
    "frechet_from_stats",
]
