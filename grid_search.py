from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional

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
    use_all_folders: bool,
) -> List[int]:
    available = _discover_itrs(exp_dir)
    if use_all_folders:
        return available

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


def _format_template(
    template: str, itr: int, sampler: Optional[str] = None, steps: Optional[int] = None
) -> str:
    return template.format(
        itr=itr,
        itr_k=itr // 1000,
        sampler=sampler,
        steps=steps,
    )

@click.group()
def grid_search_equations():
    pass

@grid_search_equations.command()
@click.option(
    "--exp-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("experiments/qm9-matching-cluster"),
    show_default=True,
)
@click.option("--start-itr", type=int, default=None)
@click.option("--end-itr", type=int, default=None)
@click.option("--step", type=int, default=5000, show_default=True)
@click.option(
    "--use-all-folders",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use every discovered itr_* folder in --exp-dir and ignore start/end/step.",
)
@click.option("--num-samples", type=int, default=10000, show_default=True)
@click.option("--n-real-samples", type=int, default=10000, show_default=True)
@click.option("--master-port", type=int, default=29501, show_default=True)
@click.option(
    "--sample-dir-template",
    type=str,
    default="samples/qm9-matching-{itr_k}k-10k",
    show_default=True,
    help="Use {itr} for full iteration (e.g. 250000) or {itr_k} for thousands (e.g. 250).",
)
@click.option(
    "--comparison-output-template",
    type=str,
    default="results/results-matching-{itr_k}-10k",
    show_default=True,
)
@click.option("--use-ema", is_flag=True, default=False, show_default=True)
@click.option("--num-steps", type=int, default=125, show_default=True)
@click.option("--dry-run", is_flag=True, default=False, show_default=True)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip iterations where evaluation output already exists.",
)
@click.option(
    "--eval-only",
    is_flag=True,
    default=False,
    show_default=True,
    help="Only run evaluation; skip sampling (use when samples already exist).",
)
@click.option(
    "--ks-only/--no-ks-only",
    default=True,
    show_default=True,
    help="Skip UMAP and distribution plots; only KS statistics (faster).",
)
def main(
    exp_dir: Path,
    start_itr: Optional[int],
    end_itr: Optional[int],
    step: int,
    use_all_folders: bool,
    num_samples: int,
    n_real_samples: int,
    master_port: int,
    sample_dir_template: str,
    comparison_output_template: str,
    use_ema: bool,
    num_steps: int,
    dry_run: bool,
    skip_existing: bool,
    eval_only: bool,
    ks_only: bool,
) -> None:
    itrs = _build_itr_list(exp_dir, start_itr, end_itr, step, use_all_folders)
    if not itrs:
        raise click.ClickException("No checkpoint iterations found.")

    for itr in reversed(itrs):
        checkpoint = exp_dir / f"itr_{itr}" / "snapshot.pt"
        sample_dir = Path(_format_template(sample_dir_template, itr))
        # Sampling always writes to {sample_dir}/samples.json, so eval must read from there
        generated = sample_dir / "samples.json"
        comparison_output = Path(_format_template(comparison_output_template, itr))

        if eval_only:
            if not generated.exists():
                click.echo(f"Skipping itr={itr}: no samples at {generated}")
                continue
        else:
            if not checkpoint.exists():
                click.echo(f"Skipping itr={itr}: missing checkpoint {checkpoint}")
                continue

        if skip_existing and comparison_output.exists():
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
            "--sampler",
            "euler",
            "--num_steps",
            str(num_steps),
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
        if ks_only:
            eval_cmd.append("--ks_only")

        click.echo(f"\n=== Iteration {itr} ===")
        if not eval_only:
            _run(sampling_cmd, dry_run)
        _run(eval_cmd, dry_run)


@grid_search_equations.command()
@click.option(
    "--exp-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("experiments/qm9-matching-cluster"),
    show_default=True,
)
@click.option("--start-itr", type=int, default=None)
@click.option("--end-itr", type=int, default=None)
@click.option("--step", type=int, default=5000, show_default=True)
@click.option(
    "--use-all-folders",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use every discovered itr_* folder in --exp-dir and ignore start/end/step.",
)
@click.option("--num-samples", type=int, default=10000, show_default=True)
@click.option("--n-real-samples", type=int, default=10000, show_default=True)
@click.option("--master-port", type=int, default=29502, show_default=True)
@click.option(
    "--sample-dir-template",
    type=str,
    default="samples/qm9-matching-{itr_k}k-10k-{sampler}-{steps}steps",
    show_default=True,
)
@click.option(
    "--comparison-output-template",
    type=str,
    default="results/results-matching-{itr_k}-10k-{sampler}-{steps}steps",
    show_default=True,
)
@click.option(
    "--sampler",
    "samplers",
    multiple=True,
    default=("euler", "split"),
    show_default=True,
    help="Sampler(s) to evaluate. Can be passed multiple times.",
)
@click.option(
    "--num-steps",
    "step_counts",
    type=int,
    multiple=True,
    default=(75, 125, 250),
    show_default=True,
    help="Number(s) of sampling steps to evaluate. Can be passed multiple times.",
)
@click.option("--use-ema", is_flag=True, default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False, show_default=True)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip iterations where evaluation output already exists.",
)
@click.option(
    "--ks-only/--no-ks-only",
    default=True,
    show_default=True,
    help="Skip UMAP and distribution plots; only KS statistics (faster).",
)
def grid_search_equations_samplers(
    exp_dir: Path,
    start_itr: Optional[int],
    end_itr: Optional[int],
    step: int,
    use_all_folders: bool,
    num_samples: int,
    n_real_samples: int,
    master_port: int,
    sample_dir_template: str,
    comparison_output_template: str,
    samplers: tuple[str, ...],
    step_counts: tuple[int, ...],
    use_ema: bool,
    dry_run: bool,
    skip_existing: bool,
    ks_only: bool,
) -> None:
    itrs = _build_itr_list(exp_dir, start_itr, end_itr, step, use_all_folders)
    if not itrs:
        raise click.ClickException("No checkpoint iterations found.")

    for itr in reversed(itrs):
        checkpoint = exp_dir / f"itr_{itr}" / "snapshot.pt"
        if not checkpoint.exists():
            click.echo(f"Skipping itr={itr}: missing checkpoint {checkpoint}")
            continue

        for sampler in samplers:
            for steps in step_counts:
                sample_dir = Path(_format_template(sample_dir_template, itr, sampler, steps))
                # Sampling always writes to {sample_dir}/samples.json, so eval must read from there
                generated = sample_dir / "samples.json"
                comparison_output = Path(_format_template(comparison_output_template, itr, sampler, steps))
                if skip_existing and generated.exists() and comparison_output.exists():
                    click.echo(f"Skipping itr={itr} with sampler {sampler} and steps {steps}: outputs already exist.")
                    continue

                sampling_cmd = [
                    "uv",
                    "run",
                    "torchrun",
                    "--master-port",
                    str(master_port),
                    "sampling.py",
                    "--num_steps",
                    str(steps),
                    "--num_samples",
                    str(num_samples),
                    "--load_checkpoint",
                    str(checkpoint),
                    "--dir",
                    str(sample_dir),
                    "--sampler",
                    sampler,
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
                if ks_only:
                    eval_cmd.append("--ks_only")

                click.echo(f"\n=== Iteration {itr} with sampler {sampler} ===")
                _run(sampling_cmd, dry_run)
                _run(eval_cmd, dry_run)


@grid_search_equations.command()
@click.option(
    "--generated",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to samples.json with generated molecules (e.g. from 80K run).",
)
@click.option(
    "--n-gen-samples",
    "n_gen_sample_sizes",
    type=int,
    multiple=True,
    default=(1000, 2500, 5000, 10000, 20000, 40000, 80000),
    show_default=True,
    help="Generated sample sizes to evaluate. Pass multiple times to override default.",
)
@click.option(
    "--use-all-real-samples/--no-use-all-real-samples",
    default=True,
    show_default=True,
    help="Use all QM9 samples for the real distribution (default). Disable to use --n-real-samples.",
)
@click.option("--n-real-samples", type=int, default=10000, show_default=True)
@click.option(
    "--comparison-output-template",
    type=str,
    default="results/eval-n-gen-{n_gen}",
    show_default=True,
    help="Non-batch only: output dir template for test_qm9_distribution.py. "
         "Batch mode writes under <generated-dir>/<stem>/n_gen_<n>/ (stem = JSON basename).",
)
@click.option("--dry-run", is_flag=True, default=False, show_default=True)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip sizes where comparison output already exists.",
)
@click.option(
    "--batch/--no-batch",
    default=True,
    show_default=True,
    help="Use batch mode: load data once, evaluate all sizes in one process (faster). Disable to run test_qm9_distribution.py separately per size.",
)
@click.option(
    "--ks-only/--no-ks-only",
    default=True,
    show_default=True,
    help="Skip UMAP and plots; only KS statistics (faster). With --no-batch, passes --ks_only to test script.",
)
@click.option(
    "--n-repeats",
    type=int,
    default=1,
    show_default=True,
    help="For stability analysis: number of independent subsamples per size. Each repeat uses a different random 2500 (or n_gen) subset.",
)
def eval_sample_sizes(
    generated: Path,
    n_gen_sample_sizes: tuple[int, ...],
    use_all_real_samples: bool,
    n_real_samples: int,
    comparison_output_template: str,
    dry_run: bool,
    skip_existing: bool,
    batch: bool,
    ks_only: bool,
    n_repeats: int,
) -> None:
    """Evaluate the same generated samples at different subsample sizes.

    Use when you have many samples (e.g. 80K) and want to see how KS statistics
    change with 1k, 2.5k, 5k, 10k, 20k, etc. No re-sampling needed.
    """
    if batch:
        # Single process: load once, evaluate all sizes (much faster)
        n_real = 132008 if use_all_real_samples else n_real_samples
        batch_cmd = [
            "uv",
            "run",
            "python",
            "eval/eval_ks_batch.py",
            "--generated",
            str(generated),
            "--n-gen-samples",
            *[str(s) for s in n_gen_sample_sizes],
            "--n-real-samples",
            str(n_real),
            "--n-repeats",
            str(n_repeats),
        ]
        if skip_existing:
            batch_cmd.append("--skip-existing")
        else:
            batch_cmd.append("--no-skip-existing")
        click.echo(f"\n=== Batch evaluation (all sizes in one run) ===")
        _run(batch_cmd, dry_run)
        return

    # Per-size subprocess calls
    for n_gen in n_gen_sample_sizes:
        comparison_output = Path(comparison_output_template.format(n_gen=n_gen))
        if skip_existing and comparison_output.exists():
            click.echo(f"Skipping n_gen={n_gen}: {comparison_output} exists.")
            continue

        n_real = 132008 if use_all_real_samples else n_real_samples  # QM9 dataset size
        eval_cmd = [
            "uv",
            "run",
            "python",
            "eval/test_qm9_distribution.py",
            "--n_real_samples",
            str(n_real),
            "--n_gen_samples",
            str(n_gen),
            "--generated",
            str(generated),
            "--comparison_output",
            str(comparison_output),
        ]
        if ks_only:
            eval_cmd.append("--ks_only")

        click.echo(f"\n=== Evaluating with n_gen_samples={n_gen}, n_real={n_real} ===")
        _run(eval_cmd, dry_run)


if __name__ == "__main__":
    grid_search_equations()
