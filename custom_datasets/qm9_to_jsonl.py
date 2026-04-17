"""
Export the preprocessed QM9 dataset (same source as QM9Dataset) to JSONL.

Each line mirrors the multimodal equations format in multimodal_math.py:
  - symbols: list of element strings (like equations' "symbols")
  - pos: list of [x, y, z] per atom, **centroid-subtracted** (same as QM9Dataset
    before rotation; we do not apply random rotation here).

Optional fields: canonical_smiles, length (sequence length).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm


def row_to_json_obj(row: dict) -> dict:
    symbols = list(row["atomic_symbols"])
    pos = np.asarray(row["pos"], dtype=np.float64)
    assert pos.shape[0] == len(symbols) and pos.shape[1] == 3, (
        f"Expected pos (N,3), got {pos.shape} for {len(symbols)} symbols"
    )
    pos = pos - np.mean(pos, axis=0)
    out: dict = {
        "symbols": symbols,
        "pos": pos.tolist(),
        "length": len(symbols),
    }
    smiles = row.get("canonical_smiles") or row.get("smiles")
    if smiles is not None:
        out["canonical_smiles"] = smiles
    return out


def export_qm9_jsonl(
    dataset_path: str | Path,
    output_path: str | Path,
) -> None:
    ds = datasets.load_from_disk(str(dataset_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(len(ds)), desc="Writing JSONL"):
            line = json.dumps(row_to_json_obj(ds[i]), ensure_ascii=False)
            f.write(line + "\n")

    print(f"Wrote {len(ds)} rows to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export qm9_preprocessed to JSONL.")
    p.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/qm9_preprocessed"),
        help="Path from load_from_disk (same as QM9Dataset).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/qm9.jsonl"),
        help="Output JSONL path.",
    )
    args = p.parse_args()
    export_qm9_jsonl(args.dataset_path, args.output)


if __name__ == "__main__":
    main()
