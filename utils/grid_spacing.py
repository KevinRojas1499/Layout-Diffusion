"""Generate "nice" numbers for grid search spacing (steps, NFEs)."""

from __future__ import annotations

import math


# Common round numbers for steps/NFE axes. Sorted ascending.
_NICE_NUMBERS = sorted(
    {
        1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 300,
        500, 750, 1000, 1250, 1500, 2000, 2500, 5000, 10000,
    }
)


def _nearest_nice(value: float) -> int:
    """Round value to the nearest nice number."""
    if value <= 0:
        return 1
    nice = min(_NICE_NUMBERS, key=lambda n: abs(n - value))
    return nice


def nice_spaced_values(
    min_val: int,
    max_val: int,
    n: int,
    log_space: bool = True,
) -> list[int]:
    """Return n values in [min_val, max_val], snapped to nice numbers.

    Uses log-spacing by default (better for NFEs/steps). Values are rounded
    to nearest "nice" numbers (50, 75, 100, 125, 250, etc.) for clean plots.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        n: Number of points to generate.
        log_space: If True, space points logarithmically; else linearly.

    Returns:
        Sorted list of unique nice integers in [min_val, max_val].
    """
    if n < 1:
        return []
    if min_val >= max_val:
        return [max(1, min_val)]

    if log_space:
        # Log-space: better for multiplicative ranges like 50..1500
        raw = [
            min_val * (max_val / min_val) ** (i / (n - 1))
            for i in range(n)
        ]
    else:
        # Linear
        raw = [
            min_val + (max_val - min_val) * i / (n - 1)
            for i in range(n)
        ]

    snapped = [_nearest_nice(v) for v in raw]
    # Clamp to [min_val, max_val] and deduplicate
    clamped = [max(min_val, min(max_val, x)) for x in snapped]
    seen: set[int] = set()
    out: list[int] = []
    for x in sorted(clamped):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
