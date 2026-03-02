"""
Compare QM9 atom-count distributions before vs after validity filtering.

This script helps answer whether the carbon-mismatch could come from
evaluation-time filtering rather than model behavior.
"""

import argparse
import json
import os
import sys

import numpy as np

# Add project root to import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_datasets.qm9 import QM9Dataset
from eval.evaluate_distribution import (
    compute_atom_counts,
    compute_ks_statistic,
    compute_molecular_fingerprints,
    extract_molecules_from_qm9_dataset,
)
from utils.tokenizer import VocabTokenizer


def summarize(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure atom-count shift caused by validity filtering."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of QM9 molecules to sample (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save a JSON summary.",
    )
    args = parser.parse_args()

    tokenizer = VocabTokenizer(vocab={"H", "C", "N", "O", "F"})
    dataset = QM9Dataset(tokenizer, max_length=30, use_raw_dataset=True)

    print(f"Loading {args.n_samples} molecules from QM9Dataset...")
    symbols, positions, _ = extract_molecules_from_qm9_dataset(
        dataset, n_samples=args.n_samples, random_seed=args.seed
    )
    print(f"Extracted molecules: {len(symbols)}")

    print("\nComputing validity filter indices...")
    _, valid_indices, stats = compute_molecular_fingerprints(
        symbols,
        positions,
        fingerprint_type="rdkit",
        filter_invalid=True,
        return_stats=True,
    )
    filtered_symbols = [symbols[i] for i in valid_indices]

    unfiltered_counts = compute_atom_counts(symbols)
    filtered_counts = compute_atom_counts(filtered_symbols)

    atom_types = ["total", "C", "N", "O", "F", "H"]
    ks_results = {}
    for atom_type in atom_types:
        ks_results[atom_type] = float(
            compute_ks_statistic(
                np.array(unfiltered_counts[atom_type]),
                np.array(filtered_counts[atom_type]),
            )
        )

    print("\n=== Filter Effect (Unfiltered vs Filtered) ===")
    print(
        f"Valid ratio: {len(filtered_symbols)}/{len(symbols)} "
        f"({100.0 * len(filtered_symbols) / max(len(symbols), 1):.2f}%)"
    )
    print("1-KSD by atom count (higher is more similar):")
    for atom_type in atom_types:
        print(f"  {atom_type:>5}: {ks_results[atom_type]:.4f}")

    c_unf = summarize(unfiltered_counts["C"])
    c_flt = summarize(filtered_counts["C"])
    print("\nCarbon count summary:")
    print(
        f"  Unfiltered: mean={c_unf['mean']:.3f}, std={c_unf['std']:.3f}, "
        f"min={c_unf['min']}, max={c_unf['max']}"
    )
    print(
        f"  Filtered:   mean={c_flt['mean']:.3f}, std={c_flt['std']:.3f}, "
        f"min={c_flt['min']}, max={c_flt['max']}"
    )

    if args.output_json:
        payload = {
            "n_samples": len(symbols),
            "n_filtered_valid": len(filtered_symbols),
            "fingerprint_filter_stats": stats,
            "ks_1_minus_ksd": ks_results,
            "carbon_summary": {
                "unfiltered": c_unf,
                "filtered": c_flt,
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved summary to {args.output_json}")


if __name__ == "__main__":
    main()
