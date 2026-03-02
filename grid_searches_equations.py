#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# DATASETS = [1, 3, 5, 8, 10]
DATASETS = [8, 10]
# DATASETS = [5]
RUNS_DIR = ROOT / "experiments"


def run_one(length_value: int) -> None:
    dataset_path = ROOT / "data" / f"equations_l{length_value}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    run_dir = RUNS_DIR / f"equations_l{length_value}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[L={length_value}] dataset={dataset_path}")
    print(f"[L={length_value}] output_dir={run_dir}")

    # cmd = [
    #     "uv",
    #     "run",
    #     "torchrun",
    #     "toy_training.py",
    #     "--data_path",
    #     str(dataset_path.relative_to(ROOT)),
    #     "--dir",
    #     str(run_dir.relative_to(ROOT)),
    #     "--log_rate",
    #     "5000",
    #     "--num_iters",
    #     "50000",
    # ]
    # Sampling
    # uv run torchrun sampling_toy.py --num_samples 1000 --num_steps 125 --load_checkpoint experiments/equations_l8/itr_50000/snapshot.pt --dir testing_samplers/2ndorder_125_
    for steps in [500, 250, 125, 75, 50]:
        cmd = [
            "uv",
            "run",
            "torchrun",
            "sampling_toy.py",
            "--num_samples",
            "1000",
            "--num_steps",
            str(steps),
            "--load_checkpoint",
            f"{str(run_dir.relative_to(ROOT))}/final_checkpoint.pt",
            "--dir",
            str(f'samples_toy/equations_l{length_value}_euler_{steps}/'),
            "--sampler",
            "euler"
        ]
        subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy(), check=True)

#     uv run torchrun --master-port 29502 toy_training.py --interpolant multimodal_both --dataset parenthesis --dir experiments/parenthesis  --data_path data/parenthesis/equations_l5.jsonl --model MMDiTBothVar --log_rate 2500 -
# -num_iters 100000
    # cmd = [
    #     "uv",
    #     "run",
    #     "torchrun",
    #     "toy_training.py",
    #     "--interpolant",
    #     "multimodal_both",
    #     "--dataset",
    #     "parenthesis",
    #     "--dir",
    #     str(f'experiments/parenthesis_l{length_value}/'),
    #     "--data_path",
    #     str(f'data/parenthesis/equations_l{length_value}.jsonl'),
    #     "--model",
    #     "MMDiTBothVar",
    #     "--log_rate",
    #     "5000",
    #     "--num_iters",
    #     "100000",
    # ]
    # uv run torchrun sampling_toy.py --num_samples 10000 --dataset parenthesis --data_path data/parenthesis/equations_l10.jsonl 
    # --interpolant multimodal_both --num_steps 1000 --dir samples-parenthesis/l10 --load_checkpoint experiments/parenthesis_l10/itr_100000/snapshot.pt --model MMDiTBothVar
    # cmd = [
    #     "uv",
    #     "run",
    #     "torchrun",
    #     "sampling_toy.py",
    #     "--num_samples",
    #     "10000",
    #     "--dataset",
    #     "parenthesis",
    #     "--data_path",
    #     str(f'data/parenthesis/equations_l{length_value}.jsonl'),
    #     "--interpolant",
    #     "multimodal_both",
    #     "--num_steps",
    #     "1000",
    #     "--dir",
    #     str(f'samples-parenthesis/l{length_value}/'),
    #     "--load_checkpoint",
    #     str(f'experiments/parenthesis_l{length_value}/itr_100000/snapshot.pt'),
    #     "--model",
    #     "MMDiTBothVar",
    # ]
    # subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy(), check=True)


def main() -> None:
    for length_value in DATASETS:
        run_one(length_value)


if __name__ == "__main__":
    main()
