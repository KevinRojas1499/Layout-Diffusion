#!/usr/bin/env python3
"""Grid over checkpoints (recursive) × samplers × num_steps; runs sampling.py via torchrun."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parent
SAMPLERS = ("euler", "split", "staggered")


def _discover_checkpoints(root: Path) -> list[Path]:
    ckpts = sorted(root.rglob("*.pt"))
    return ckpts


@click.command()
@click.option("--checkpoints-dir", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--out-root", type=click.Path(path_type=Path), required=True, help="Root directory for run outputs (mirrors relative paths under checkpoints_dir).")
@click.option("--sampler", type=click.Choice(SAMPLERS), multiple=True, default=("euler",), show_default=True)
@click.option("--num-steps", type=int, multiple=True, default=(75,), show_default=True)
@click.option("--num-samples", type=int, default=2500, show_default=True)
@click.option("--batch-size", type=int, default=100, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--master-port", type=int, default=29502, show_default=True)
@click.option("--use-ema", is_flag=True, default=False)
@click.option("--model", type=click.Choice(["radd", "DiT", "Transformer"]), default="DiT", show_default=True)
def main(
    checkpoints_dir: Path,
    out_root: Path,
    sampler: tuple[str, ...],
    num_steps: tuple[int, ...],
    num_samples: int,
    batch_size: int,
    seed: int,
    master_port: int,
    use_ema: bool,
    model: str,
) -> None:
    checkpoints_dir = checkpoints_dir.resolve()
    out_root = out_root.resolve()
    sampling_py = REPO_ROOT / "sampling.py"
    if not sampling_py.is_file():
        raise click.ClickException(f"Missing {sampling_py}")

    ckpts = _discover_checkpoints(checkpoints_dir)
    if not ckpts:
        raise click.ClickException(f"No .pt files under {checkpoints_dir}")

    for ckpt in reversed(ckpts):
        rel = ckpt.relative_to(checkpoints_dir)
        for s in sampler:
            for steps in num_steps:
                run_dir = out_root / rel.parent / f"{rel.stem}_steps-{steps}_{s}"
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--master_port={master_port}",
                    str(sampling_py),
                    "--num_samples",
                    str(num_samples),
                    "--num_steps",
                    str(steps),
                    "--sampler",
                    s,
                    "--batch_size",
                    str(batch_size),
                    "--seed",
                    str(seed),
                    "--load_checkpoint",
                    str(ckpt),
                    "--dir",
                    str(run_dir),
                    "--model",
                    model,
                ]
                if use_ema:
                    cmd.append("--use_ema")

                click.echo(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
