#!/usr/bin/env python3
"""
Summarize grid search results from results-muon-fused and optionally delete checkpoints
and results whose Average 1-KSD was below a threshold.

Usage:
    uv run python summarize_results_muon_fused.py                    # Print summary
    uv run python summarize_results_muon_fused.py --summary-out summary.txt
    uv run python summarize_results_muon_fused.py --delete-below 0.90 --dry-run  # Preview
    uv run python summarize_results_muon_fused.py --delete-below 0.90              # Delete
"""
from __future__ import annotations

import re
from pathlib import Path

import click


RESULTS_DIR = Path("results-muon-fused")
DEFAULT_EXP_DIR = Path("experiments/muon-fused-residual")
# results itr_N corresponds to experiment itr_{N*1000}
ITR_MULTIPLIER = 1000


def _parse_ks_summary(ks_file: Path) -> float | None:
    """Parse Average 1-KSD from ks_statistics_summary.txt. Returns None if missing."""
    if not ks_file.exists():
        return None
    with open(ks_file) as f:
        for line in f:
            if line.strip().startswith("Average"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        return None
    return None


def collect_results(results_dir: Path) -> list[tuple[int, float | None]]:
    """Collect (itr, avg_1ksd) for each itr_* in results_dir."""
    rows: list[tuple[int, float | None]] = []
    for d in sorted(
        results_dir.iterdir(),
        key=lambda p: int(m.group(1)) if (m := re.search(r"itr_(\d+)", p.name)) else 0,
    ):
        if not d.is_dir():
            continue
        m = re.match(r"itr_(\d+)", d.name)
        if not m:
            continue
        itr = int(m.group(1))
        ks_file = d / "all_samples" / "ks_statistics_summary.txt"
        avg = _parse_ks_summary(ks_file)
        rows.append((itr, avg))
    return rows


def format_summary(rows: list[tuple[int, float | None]]) -> str:
    """Format results as a text table."""
    lines = [
        "Grid Search Summary (results-muon-fused)",
        "=" * 50,
        f"{'itr':>8}  {'Avg 1-KSD':>10}  {'>= 0.90':>8}",
        "-" * 50,
    ]
    for itr, avg in rows:
        status = "yes" if avg is not None and avg >= 0.90 else "no"
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"
        lines.append(f"{itr:>8}  {avg_str:>10}  {status:>8}")
    lines.append("-" * 50)
    n_total = len(rows)
    n_below = sum(1 for _, a in rows if a is not None and a < 0.90)
    n_above = sum(1 for _, a in rows if a is not None and a >= 0.90)
    n_na = sum(1 for _, a in rows if a is None)
    lines.append(f"Total: {n_total}  |  >= 0.90: {n_above}  |  < 0.90: {n_below}  |  N/A: {n_na}")
    return "\n".join(lines)


def get_dirs_to_delete(
    rows: list[tuple[int, float | None]],
    exp_dir: Path,
    results_dir: Path,
    threshold: float,
) -> tuple[list[Path], list[Path]]:
    """Return (checkpoint_dirs, results_dirs) to delete where avg < threshold."""
    ckpt_dirs: list[Path] = []
    result_dirs: list[Path] = []
    for itr, avg in rows:
        if avg is None:
            continue
        if avg < threshold:
            exp_itr = itr * ITR_MULTIPLIER
            ckpt_dir = exp_dir / f"itr_{exp_itr}"
            result_dir = results_dir / f"itr_{itr}"
            if ckpt_dir.exists():
                ckpt_dirs.append(ckpt_dir)
            if result_dir.exists():
                result_dirs.append(result_dir)
    return ckpt_dirs, result_dirs


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=RESULTS_DIR,
    show_default=True,
)
@click.option(
    "--exp-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_EXP_DIR,
    show_default=True,
    help="Experiment directory containing itr_*/snapshot.pt checkpoints.",
)
@click.option(
    "--summary-out",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    default=None,
    help="Write summary to this file.",
)
@click.option(
    "--delete-below",
    type=float,
    default=None,
    help="Delete checkpoints whose Average 1-KSD is below this value (e.g. 0.90).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Only print what would be deleted, do not delete.",
)
@click.option(
    "--yes",
    "-y",
    "skip_confirm",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt when deleting.",
)
def main(
    results_dir: Path,
    exp_dir: Path,
    summary_out: Path | None,
    delete_below: float | None,
    dry_run: bool,
    skip_confirm: bool,
) -> None:
    rows = collect_results(results_dir)
    if not rows:
        click.echo(f"No results found in {results_dir}")
        return

    summary = format_summary(rows)
    click.echo(summary)

    if summary_out:
        summary_out.write_text(summary)
        click.echo(f"\nSummary written to {summary_out}")

    if delete_below is not None:
        import shutil

        ckpt_dirs, result_dirs = get_dirs_to_delete(rows, exp_dir, results_dir, delete_below)
        to_delete = ckpt_dirs + result_dirs
        if not to_delete:
            click.echo(f"\nNothing to delete (all have avg >= {delete_below})")
            return

        click.echo(f"\nCheckpoints with avg < {delete_below} to delete ({len(ckpt_dirs)}):")
        for p in ckpt_dirs:
            click.echo(f"  {p}")
        click.echo(f"\nResults to delete ({len(result_dirs)}):")
        for p in result_dirs:
            click.echo(f"  {p}")

        if dry_run:
            click.echo("\n[DRY RUN] No files deleted.")
            return

        if not skip_confirm and not click.confirm(
            f"\nDelete {len(ckpt_dirs)} checkpoint dirs and {len(result_dirs)} result dirs?"
        ):
            click.echo("Aborted.")
            return

        for p in ckpt_dirs:
            shutil.rmtree(p)
            click.echo(f"Deleted {p}")
        for p in result_dirs:
            shutil.rmtree(p)
            click.echo(f"Deleted {p}")


if __name__ == "__main__":
    main()
