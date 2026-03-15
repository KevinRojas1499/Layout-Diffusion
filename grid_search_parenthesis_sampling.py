#!/usr/bin/env python3
"""
Grid search for parenthesis sampling: runs sampling_toy.py for each combination of
sampler and num_steps, saving outputs as samples-parenthesis-new/samples-{sampler}-{steps}.jsonl

Usage:
    uv run python grid_search_parenthesis_sampling.py

Edit the constants below to change num_samples, steps, or checkpoint.
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples-parenthesis-new"
CHECKPOINT = "experiments/parenthesis_l1_longer_correct/final_checkpoint.pt"
DATA_PATH = "data/equations_l1.jsonl"  # or data/parenthesis/equations_l1.jsonl
NUM_SAMPLES = 10000  # Increase for better plots (was 1000)

SAMPLERS = ["euler", "staggered", "split"]
NUM_STEPS = [50, 125, 250, 500, 750]


def run_one(sampler: str, num_steps: int) -> None:
    tmp_dir = SAMPLES_DIR / f".tmp_{sampler}_{num_steps}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_file = SAMPLES_DIR / f"samples-{sampler}-{num_steps}.jsonl"

    print(f"\n--- {sampler} / {num_steps} steps ---")
    cmd = [
        "uv",
        "run",
        "sampling_toy.py",
        "--num_samples",
        str(NUM_SAMPLES),
        "--dataset",
        "parenthesis",
        "--data_path",
        DATA_PATH,
        "--model",
        "MMDiTBothVar",
        "--interpolant",
        "multimodal_both",
        "--sampler",
        sampler,
        "--num_steps",
        str(num_steps),
        "--dir",
        str(tmp_dir),
        "--load_checkpoint",
        CHECKPOINT,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    samples_path = tmp_dir / "samples.jsonl"
    if samples_path.exists():
        shutil.move(str(samples_path), str(out_file))
        print(f"  -> {out_file}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    total = len(SAMPLERS) * len(NUM_STEPS)
    print(f"Grid search: {len(SAMPLERS)} samplers x {len(NUM_STEPS)} steps = {total} runs")
    print(f"Output dir: {SAMPLES_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Num samples per run: {NUM_SAMPLES}")

    for sampler in SAMPLERS:
        for num_steps in NUM_STEPS:
            run_one(sampler, num_steps)

    print(f"\nDone. Outputs in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
