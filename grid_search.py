from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import click


def _run(cmd: List[str], dry_run: bool) -> None:
    click.echo(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _discover_itrs(exp_dir: Path) -> List[int]:
    itrs: List[int] = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"itr_(\d+)", child.name)
        if match:
            itrs.append(int(match.group(1)))
    return sorted(set(itrs))


def _build_itr_list(
    exp_dir: Path,
    start_itr: Optional[int],
    end_itr: Optional[int],
    step: int,
) -> List[int]:
    available = _discover_itrs(exp_dir)
    if start_itr is None and end_itr is None:
        return available

    if available and start_itr is None:
        start_itr = min(available)
    if available and end_itr is None:
        end_itr = max(available)

    if start_itr is None or end_itr is None:
        raise click.ClickException("Unable to infer start/end iteration from exp dir.")

    if end_itr < start_itr:
        raise click.ClickException("--end-itr must be >= --start-itr.")

    return list(range(start_itr, end_itr + 1, step))


def _format_template(template: str, itr: int) -> str:
    return template.format(itr=itr, itr_k=itr // 1000)


@click.command()
@click.option(
    "--exp-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("experiments/qm9-matching-cluster"),
    show_default=True,
)
@click.option("--start-itr", type=int, default=None)
@click.option("--end-itr", type=int, default=None)
@click.option("--step", type=int, default=5000, show_default=True)
@click.option("--num-samples", type=int, default=10000, show_default=True)
@click.option("--n-real-samples", type=int, default=10000, show_default=True)
@click.option("--master-port", type=int, default=29502, show_default=True)
@click.option(
    "--sample-dir-template",
    type=str,
    default="samples/qm9-matching-{itr_k}k-10k",
    show_default=True,
)
@click.option(
    "--generated-template",
    type=str,
    default="samples/qm9-matching-{itr_k}k-10k/samples.json",
    show_default=True,
)
@click.option(
    "--comparison-output-template",
    type=str,
    default="results/results-matching-{itr_k}-10k",
    show_default=True,
)
@click.option("--use-ema", is_flag=True, default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False, show_default=True)
@click.option("--skip-existing", is_flag=True, default=True, show_default=True)
def main(
    exp_dir: Path,
    start_itr: Optional[int],
    end_itr: Optional[int],
    step: int,
    num_samples: int,
    n_real_samples: int,
    master_port: int,
    sample_dir_template: str,
    generated_template: str,
    comparison_output_template: str,
    use_ema: bool,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    itrs = _build_itr_list(exp_dir, start_itr, end_itr, step)
    if not itrs:
        raise click.ClickException("No checkpoint iterations found.")

    for itr in itrs:
        checkpoint = exp_dir / f"itr_{itr}" / "snapshot.pt"
        if not checkpoint.exists():
            click.echo(f"Skipping itr={itr}: missing checkpoint {checkpoint}")
            continue

        sample_dir = Path(_format_template(sample_dir_template, itr))
        generated = Path(_format_template(generated_template, itr))
        comparison_output = Path(_format_template(comparison_output_template, itr))

        if skip_existing and generated.exists() and comparison_output.exists():
            click.echo(f"Skipping itr={itr}: outputs already exist.")
            continue

        sampling_cmd = [
            "uv",
            "run",
            "torchrun",
            "--master-port",
            str(master_port),
            "sampling.py",
            "--num_samples",
            str(num_samples),
            "--load_checkpoint",
            str(checkpoint),
            "--dir",
            str(sample_dir),
        ]
        if use_ema:
            sampling_cmd.append("--use_ema")

        eval_cmd = [
            "uv",
            "run",
            "python",
            "eval/test_qm9_distribution.py",
            "--n_real_samples",
            str(n_real_samples),
            "--generated",
            str(generated),
            "--comparison_output",
            str(comparison_output),
        ]

        click.echo(f"\n=== Iteration {itr} ===")
        _run(sampling_cmd, dry_run)
        _run(eval_cmd, dry_run)


if __name__ == "__main__":
    main()
