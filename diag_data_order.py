"""Probe whether RICO layouts have a natural deterministic ordering.

For each candidate sort key, count what fraction of (multi-element) layouts
are already monotonically sorted by that key. If one rule wins decisively
( >80–90%), the data has that order baked in. If nothing wins, layouts
are essentially in arbitrary order (authoring/rendering).
"""

import argparse

import h5py
import numpy as np


CANDIDATES = [
    ("cy_then_cx (raster)",   lambda b, c: list(zip(b[:, 1] + b[:, 3] / 2, b[:, 0] + b[:, 2] / 2))),
    ("cx_then_cy",            lambda b, c: list(zip(b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2))),
    ("cy_only",               lambda b, c: list(zip(b[:, 1] + b[:, 3] / 2,))),
    ("cx_only",               lambda b, c: list(zip(b[:, 0] + b[:, 2] / 2,))),
    ("top (y_topleft)",       lambda b, c: list(zip(b[:, 1],))),
    ("left (x_topleft)",      lambda b, c: list(zip(b[:, 0],))),
    ("area_desc",             lambda b, c: list(zip(-(b[:, 2] * b[:, 3]),))),
    ("area_asc",              lambda b, c: list(zip(b[:, 2] * b[:, 3],))),
    ("category_asc",          lambda b, c: list(zip(c,))),
    ("category_then_cy",      lambda b, c: list(zip(c, b[:, 1] + b[:, 3] / 2))),
]


def is_monotone(seq):
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/workspace/LayoutFlow-data/dataset/rico/ldm_rico_train.h5")
    p.add_argument("--n", type=int, default=2000, help="Number of layouts to inspect.")
    p.add_argument("--cat_field", default="type")
    args = p.parse_args()

    monotone_counts = {name: 0 for name, _ in CANDIDATES}
    total = 0
    skipped_singleton = 0

    with h5py.File(args.h5, "r") as f:
        keys = list(f.keys())[: args.n]
        for k in keys:
            L = int(np.array(f[k]["length"]))
            if L < 2:
                skipped_singleton += 1
                continue
            bbox = np.array(f[k]["bbox"])[:L].astype(np.float64)
            cats = np.array(f[k][args.cat_field])[:L].astype(np.int64).flatten()
            total += 1
            for name, rule in CANDIDATES:
                if is_monotone(rule(bbox, cats)):
                    monotone_counts[name] += 1

    print(f"Inspected {total} layouts (length ≥ 2). Skipped {skipped_singleton} singletons.\n")
    print(f"{'Rule':<24} | {'monotone':>10} | {'fraction':>10}")
    print("-" * 55)
    ranked = sorted(monotone_counts.items(), key=lambda kv: -kv[1])
    for name, cnt in ranked:
        print(f"{name:<24} | {cnt:>10} | {cnt / max(total,1) * 100:>9.2f}%")

    print()
    best, best_cnt = ranked[0]
    frac = best_cnt / max(total, 1)
    if frac > 0.80:
        verdict = f"DATA IS ORDERED by '{best}' ({frac*100:.1f}%). Already canonical — order mismatch is the inference-side problem."
    elif frac > 0.40:
        verdict = f"DATA WEAKLY PREFERS '{best}' ({frac*100:.1f}%). Canonical rule visible but not enforced."
    else:
        verdict = "DATA IS UNORDERED (no rule >40%). Model is being trained on arbitrary order — RoPE is encoding noise."
    print(f"VERDICT: {verdict}")

    # Random baseline: a random permutation of L items is monotone by any
    # total order with prob 1/L!. For L≥3, that's <17%; anything above ~25%
    # is meaningful signal.


if __name__ == "__main__":
    main()
