#!/usr/bin/env python3
"""Run grid search over equations dataset: length values, samplers, and steps.

Generates samples.jsonl files under samples_toy/ for each (length, sampler, steps)
combination. Use test.ipynb to compute acc-1, acc-2, valid and plot vs steps/NFE.

Folder naming matches test.ipynb collect_results_for_length:
  - equations_l{L}_{method_key}_{steps}/samples.jsonl
  - method_key: euler, 2ndorder (for split/2nd-order), staggered
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from utils.grid_spacing import nice_spaced_values

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "experiments"
SAMPLES_DIR = ROOT / "samples_toy"

# Map sampler (sampling_toy param) -> folder suffix (for notebook method_map "2nd-order")
SAMPLER_TO_FOLDER = {"euler": "euler", "split": "2ndorder", "staggered": "staggered"}

# NFE per step (matches test.ipynb NFE_PER_STEP)
NFE_PER_STEP = {"euler": 1, "split": 3, "staggered": 2}


def run_one(
    length_value: int,
    sampler: str,
    steps: int,
    seed: int,
    num_samples: int,
    checkpoint_path: Path,
    output_dir: Path,
    dry_run: bool,
    skip_existing: bool,
) -> bool:
    if skip_existing and (output_dir / "samples.jsonl").exists():
        print(f"[L={length_value}] {sampler} {steps} steps seed={seed}: skip (exists)")
        return True

    cmd = [
        "uv",
        "run",
        "python",
        "sampling_toy.py",
        "--dataset",
        "equations",
        "--num_samples",
        str(num_samples),
        "--num_steps",
        str(steps),
        "--load_checkpoint",
        str(checkpoint_path),
        "--dir",
        str(output_dir),
        "--sampler",
        sampler,
        "--seed",
        str(seed),
    ]

    print(f"[L={length_value}] {sampler} {steps} steps seed={seed}: {' '.join(cmd)}")
    if dry_run:
        return True
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search over equations: length, sampler, steps. Generates samples for test.ipynb plots.",
    )
    parser.add_argument(
        "--length",
        type=int,
        action="append",
        default=[8, 10],
        help="Length value(s) to sweep. Repeat for multiple.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "split", "staggered"],
        action="append",
        default=["euler", "split", "staggered"],
        help="Sampler(s) to evaluate. Repeat for multiple.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        action="append",
        default=[],
        help="Explicit step count(s). Ignored if --steps-min/--steps-max/--steps-num are set.",
    )
    parser.add_argument(
        "--steps-min",
        type=int,
        default=None,
        help="Min steps for range mode. With --steps-max and --steps-num, generates nice-spaced values.",
    )
    parser.add_argument(
        "--steps-max",
        type=int,
        default=None,
        help="Max steps for range mode.",
    )
    parser.add_argument(
        "--steps-num",
        type=int,
        default=None,
        help="Number of step values in range mode. Uses log-spacing + nice rounding.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        action="append",
        default=[],
        help="Explicit NFE value(s). Ignored if --nfe-min/--nfe-max/--nfe-num are set.",
    )
    parser.add_argument(
        "--nfe-min",
        type=int,
        default=None,
        help="Min NFE for range mode. With --nfe-max and --nfe-num, generates nice-spaced values.",
    )
    parser.add_argument(
        "--nfe-max",
        type=int,
        default=None,
        help="Max NFE for range mode.",
    )
    parser.add_argument(
        "--nfe-num",
        type=int,
        default=None,
        help="Number of NFE values in range mode. Uses log-spacing + nice rounding.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1100,
        help="Samples per config (test.ipynb typically uses ~1100).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Base dir for experiments/equations_l{L}.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=SAMPLES_DIR,
        help="Base dir for samples_toy output.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_checkpoint.pt",
        help="Checkpoint filename under experiments/equations_l{L}/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="Random seed(s) per config. Repeat for multiple (enables error bars). Default: 1 2 3.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    lengths = list(dict.fromkeys(args.length))
    samplers = list(dict.fromkeys(args.sampler))
    seeds = list(dict.fromkeys(args.seed)) if args.seed else [1, 2, 3]

    # Steps from steps range or explicit
    steps_from_range: list[int] = []
    if args.steps_min is not None and args.steps_max is not None and args.steps_num is not None:
        steps_from_range = nice_spaced_values(args.steps_min, args.steps_max, args.steps_num, log_space=True)
        print(f"Steps (range {args.steps_min}..{args.steps_max}, n={args.steps_num}): {steps_from_range}")
    elif args.steps:
        steps_from_range = sorted(set(args.steps))
    else:
        steps_from_range = [50, 75, 125, 250, 500]
        print(f"Steps (default): {steps_from_range}")

    # NFEs from NFE range or explicit
    nfe_list: list[int] = []
    if args.nfe_min is not None and args.nfe_max is not None and args.nfe_num is not None:
        nfe_list = nice_spaced_values(args.nfe_min, args.nfe_max, args.nfe_num, log_space=True)
        print(f"NFEs (range {args.nfe_min}..{args.nfe_max}, n={args.nfe_num}): {nfe_list}")
    elif args.nfe:
        nfe_list = sorted(set(args.nfe))

    # For each sampler, union of steps from steps range and steps implied by NFE range.
    # No redundant runs: each (sampler, steps) is computed exactly once.
    sampler_steps: dict[str, list[int]] = {}
    for sampler in samplers:
        step_set: set[int] = set(steps_from_range)
        nfe_per_step = NFE_PER_STEP[sampler]
        for nfe in nfe_list:
            if nfe <= 0:
                continue
            s = nfe // nfe_per_step
            if s > 0:
                step_set.add(s)
        sampler_steps[sampler] = sorted(step_set, reverse=True)

    for length_value in lengths:
        run_dir = args.runs_dir / f"equations_l{length_value}"
        checkpoint = run_dir / args.checkpoint_name
        if not checkpoint.exists():
            print(f"[L={length_value}] Skipping: checkpoint not found {checkpoint}")
            continue

        for sampler in samplers:
            folder_suffix = SAMPLER_TO_FOLDER[sampler]
            for steps in sampler_steps[sampler]:
                base_name = f"equations_l{length_value}_{folder_suffix}_{steps}"
                for seed in seeds:
                    if len(seeds) == 1:
                        output_dir = args.samples_dir / base_name
                    else:
                        output_dir = args.samples_dir / base_name / f"seed_{seed}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    run_one(
                        length_value=length_value,
                        sampler=sampler,
                        steps=steps,
                        seed=seed,
                        num_samples=args.num_samples,
                        checkpoint_path=checkpoint,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                        skip_existing=args.skip_existing,
                    )

    print(f"\nDone. Samples under {args.samples_dir}. Use test.ipynb collect_results_for_length + plot_results_vs_steps.")


if __name__ == "__main__":
    main()
