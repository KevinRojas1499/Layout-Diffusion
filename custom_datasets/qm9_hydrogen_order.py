"""
Reorder QM9 atoms so heavy atoms stay in file (canonical SMILES) order, and each
hydrogen is placed immediately before its nearest heavy atom; hydrogens sharing
the same heavy are ordered by increasing distance to that heavy.
"""
from __future__ import annotations

import numpy as np


def _is_hydrogen(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    return s in ("H", "D", "T")


def reorder_hydrogens_nearest_heavy(
    atomic_symbols: list,
    pos: np.ndarray,
) -> tuple[list, np.ndarray]:
    """
    Returns new (atomic_symbols, pos) with the same atoms, permuted.

    Heavy (non-hydrogen) atoms keep their relative order. Each hydrogen is
    assigned to the geometrically nearest heavy atom; the sequence becomes
    [... H* ... heavy_k ...] with H* sorted by distance to heavy_k when there
    are several.
    """
    symbols = list(atomic_symbols)
    pos = np.asarray(pos, dtype=np.float64)
    n = len(symbols)
    if pos.shape != (n, 3):
        raise ValueError(f"pos must be (N,3), got {pos.shape} for N={n}")

    heavy_indices = [i for i in range(n) if not _is_hydrogen(symbols[i])]
    h_indices = [i for i in range(n) if _is_hydrogen(symbols[i])]

    if not h_indices or not heavy_indices:
        return symbols, pos.copy()

    n_heavy = len(heavy_indices)
    heavy_pos = pos[heavy_indices]

    # k -> list of (distance, h_atom_index)
    groups: list[list[tuple[float, int]]] = [[] for _ in range(n_heavy)]
    for h_idx in h_indices:
        d = np.linalg.norm(heavy_pos - pos[h_idx], axis=1)
        k = int(np.argmin(d))
        groups[k].append((float(d[k]), h_idx))

    order: list[int] = []
    for k in range(n_heavy):
        groups[k].sort(key=lambda t: (t[0], t[1]))
        for _, h_idx in groups[k]:
            order.append(h_idx)
        order.append(heavy_indices[k])

    if len(order) != n or len(set(order)) != n:
        raise RuntimeError("Internal error: invalid atom permutation")

    new_symbols = [symbols[i] for i in order]
    new_pos = pos[order].copy()
    return new_symbols, new_pos
