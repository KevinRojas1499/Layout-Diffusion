"""
Load QM9 KS summaries from eval output folders and plot curves in the same style
as ``sampling_figures.ipynb`` (Okabe–Ito palette, PDF-friendly fonts).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- style (aligned with sampling_figures.ipynb) ---
SAMPLER_STYLES = {
    "euler": {"color": "#0072B2", "marker": "o", "ls": "-"},
    "staggered": {"color": "#E69F00", "marker": "s", "ls": "-"},
    "split": {"color": "#009E73", "marker": "^", "ls": "-"},
}
METHOD_LABELS = {"euler": "Euler", "staggered": "Staggered", "split": "Split"}
NFE_PER_STEP = {"euler": 1, "staggered": 2, "split": 3}

BRANCHING_STYLES = {
    "permutation": {"color": "#CC79A7", "marker": "D", "ls": "-"},
    "no_permutation": {"color": "#D55E00", "marker": "v", "ls": "-"},
}
BRANCHING_LABELS = {
    "permutation": "Branching (permutation)",
    "no_permutation": "Branching (no permutation)",
}

# ``full``: one comfortable figure (~5–6" wide). ``compact``: appendix panels (~3.6–4.5"
# wide × ~3.1" tall), sized for readable multi-line legends below the axes.
LayoutKind = Literal["full", "compact"]

METRIC_LABELS = {
    "mw": "Molecular weight",
    "logp": "LogP",
    "hbd": "HBD",
    "hba": "HBA",
    "C": "Carbon",
    "H": "Hydrogen",
    "O": "Oxygen",
    "N": "Nitrogen",
    "F": "Fluorine",
    "atoms": "Atoms",
    "Average": "Average",
}

STEPS_DIR_RE = re.compile(r"^steps-(\d+)_(euler|split|staggered)$")
# ``samples-full/samples-old/qm9-matching-{ckpt}k-10k-{sampler}-{N}steps/``
SAMPLES_OLD_ABLATION_RE = re.compile(
    r"^qm9-matching-(\d+)k-10k-(euler|split|staggered)-(\d+)steps$"
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "axes.titleweight": "medium",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def _parse_dt_suffix(name: str) -> float | None:
    """Parse trailing ``dt0p00133333`` / ``dt0p008`` from a directory name."""
    m = re.search(r"dt(0p[\d]+)$", name)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_repo_root_on_path() -> None:
    rs = str(_repo_root())
    if rs not in sys.path:
        sys.path.insert(0, rs)


def _try_valid_frac_from_samples_json(run_dir: Path) -> float | None:
    """
    When KS summaries were written without the validity appendix, compute the same
    **valid / total** fraction as ``eval_ks_batch.count_valid_generated_like_ks_filter``
    from ``samples.json`` beside the run (ablation layout).
    """
    jp = run_dir / "samples.json"
    if not jp.exists():
        return None
    _ensure_repo_root_on_path()
    try:
        from eval.eval_ks_batch import (  # noqa: PLC0415
            count_valid_generated_like_ks_filter,
            try_load_valid_molecules_json,
        )
    except ImportError:
        return None
    loaded = try_load_valid_molecules_json(str(jp))
    if not loaded:
        return None
    syms, pos = loaded
    valid_ct, total_ct = count_valid_generated_like_ks_filter(syms, pos)
    if not total_ct:
        return None
    return valid_ct / total_ct


def _try_valid_frac_from_nearby_summaries(
    run_dir: Path,
    *,
    mode: str,
    n_gen_candidates: tuple[str, ...],
) -> float | None:
    """
    Some eval runs write KS under ``n_gen_* / valid_only /`` without the validity
    appendix, while a parallel ``samples/n_gen_* /`` summary includes it. The
    fraction is the same subsample; attach it so sampler curves appear on validity plots.
    """
    candidates: list[Path] = []
    for ng in n_gen_candidates:
        for m in (mode, "valid_only", "all_samples"):
            candidates.append(run_dir / ng / m / "ks_statistics_summary.txt")
            candidates.append(run_dir / "samples" / ng / m / "ks_statistics_summary.txt")
    seen: set[Path] = set()
    for p in candidates:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        d = parse_ks_summary(p)
        if d and "valid_frac" in d:
            return float(d["valid_frac"])
    return None


def _attach_valid_frac_if_missing(
    data: dict[str, float],
    run_dir: Path,
    *,
    mode: str,
    n_gen_candidates: tuple[str, ...],
) -> None:
    if "valid_frac" in data:
        return
    vf = _try_valid_frac_from_nearby_summaries(run_dir, mode=mode, n_gen_candidates=n_gen_candidates)
    if vf is None:
        vf = _try_valid_frac_from_samples_json(run_dir)
    if vf is not None:
        data["valid_frac"] = vf


def parse_ks_summary(path: Path) -> dict[str, float] | None:
    """
    Parse ``ks_statistics_summary.txt``: metric name -> 1-KSD, plus ``valid_frac`` in [0,1].
    Returns None if file missing or empty.
    """
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    out: dict[str, float] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-"):
            continue
        if "Metric" in line and "1-KSD" in line:
            continue
        if "Distribution Agreement" in line:
            continue
        if "Percent valid" in line or re.search(r"valid:\s*[\d.]+%", line):
            m = re.search(r"([\d.]+)\s*%", line)
            if m:
                out["valid_frac"] = float(m.group(1)) / 100.0
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                out[key] = float(parts[1])
            except ValueError:
                pass
    return out if out else None


def _find_summary_file(run_dir: Path, mode: str, n_gen_candidates: tuple[str, ...]) -> Path | None:
    """Prefer ``valid_only`` summaries (match training metrics); fall back to ``all_samples``."""
    for ng in n_gen_candidates:
        p = run_dir / ng / mode / "ks_statistics_summary.txt"
        if p.exists():
            return p
        p2 = run_dir / "samples" / ng / mode / "ks_statistics_summary.txt"
        if p2.exists():
            return p2
    return None


_DEFAULT_N_GEN = ("n_gen_10050", "n_gen_10000", "n_gen_5050", "n_gen_2500")


def collect_num_steps_samples_old(
    root: Path,
    *,
    ckpt_k: int,
    mode: str = "valid_only",
    n_gen_candidates: tuple[str, ...] = _DEFAULT_N_GEN,
) -> pd.DataFrame:
    """
    Ablations stored as ``qm9-matching-{ckpt}k-10k-{euler|split|staggered}-{N}steps/``
    (e.g. under ``samples-full/samples-old``), with KS under ``samples/n_gen_*/``.
    """
    rows: list[dict] = []
    if not root.is_dir():
        return pd.DataFrame()
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = SAMPLES_OLD_ABLATION_RE.match(d.name)
        if not m:
            continue
        if int(m.group(1)) != ckpt_k:
            continue
        method = m.group(2)
        n_steps = int(m.group(3))
        summary_path = _find_summary_file(d, mode, n_gen_candidates)
        if summary_path is None:
            continue
        data = parse_ks_summary(summary_path)
        if not data:
            continue
        _attach_valid_frac_if_missing(data, d, mode=mode, n_gen_candidates=n_gen_candidates)
        row = {"method": method, "num_steps": n_steps, "nfe": n_steps * NFE_PER_STEP[method]}
        row.update({k: v for k, v in data.items() if k != "valid_frac"})
        if "valid_frac" in data:
            row["valid_frac"] = data["valid_frac"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_num_steps_ablation(
    root: Path,
    *,
    mode: str = "valid_only",
    n_gen_candidates: tuple[str, ...] = _DEFAULT_N_GEN,
) -> pd.DataFrame:
    """
    One row per ``steps-{n}_{sampler}`` directory under ``root``.
    Columns: ``method``, ``num_steps``, ``nfe``, each KS metric, ``valid_frac``.
    """
    rows: list[dict] = []
    if not root.is_dir():
        return pd.DataFrame()
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = STEPS_DIR_RE.match(d.name)
        if not m:
            continue
        n_steps = int(m.group(1))
        method = m.group(2)
        summary_path = _find_summary_file(d, mode, n_gen_candidates)
        if summary_path is None:
            continue
        data = parse_ks_summary(summary_path)
        if not data:
            continue
        _attach_valid_frac_if_missing(data, d, mode=mode, n_gen_candidates=n_gen_candidates)
        row = {"method": method, "num_steps": n_steps, "nfe": n_steps * NFE_PER_STEP[method]}
        row.update({k: v for k, v in data.items() if k != "valid_frac"})
        if "valid_frac" in data:
            row["valid_frac"] = data["valid_frac"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_branching_dt_sweep(
    root: Path,
    series_key: str,
    *,
    mode: str = "valid_only",
    n_gen_candidates: tuple[str, ...] = _DEFAULT_N_GEN,
) -> pd.DataFrame:
    """
    Directories like ``qm9_gen_best_ckpt_permutation_dt0p008`` or ``..._dt0p008``.

    The folder suffix encodes **Δt**; the corresponding **number of steps** is
    ``num_steps = 1/Δt`` (same grid as ``steps-{N}_*`` ablations). Rows include
    both ``dt`` and ``num_steps`` (integer round) for plotting on a shared axis.
    ``series_key`` is ``permutation`` or ``no_permutation`` (used for style lookup).
    """
    rows: list[dict] = []
    if not root.is_dir():
        return pd.DataFrame()
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        dt = _parse_dt_suffix(d.name)
        if dt is None or dt <= 0:
            continue
        num_steps = int(round(1.0 / dt))
        summary_path = _find_summary_file(d, mode, n_gen_candidates)
        if summary_path is None:
            continue
        data = parse_ks_summary(summary_path)
        if not data:
            continue
        _attach_valid_frac_if_missing(data, d, mode=mode, n_gen_candidates=n_gen_candidates)
        row = {"method": series_key, "dt": dt, "num_steps": num_steps}
        row.update({k: v for k, v in data.items() if k != "valid_frac"})
        if "valid_frac" in data:
            row["valid_frac"] = data["valid_frac"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_combined_num_steps_grid(
    ablation_root: Path,
    perm_root: Path,
    no_perm_root: Path,
    *,
    mode: str = "valid_only",
    n_gen_candidates: tuple[str, ...] = _DEFAULT_N_GEN,
    ablation_layout: Literal["steps_dirs", "samples_old"] = "steps_dirs",
    samples_old_ckpt_k: int = 1350,
) -> pd.DataFrame:
    """
    Vertically stack **num-steps ablation** (Euler / Split / Staggered) and **branching**
    sweeps (permutation / no-permutation), all aligned on ``num_steps`` (branching: ``1/dt``
    from the directory suffix).

    Set ``ablation_layout="samples_old"`` and point ``ablation_root`` at ``samples-full/samples-old``
    to use folders ``qm9-matching-{ckpt}k-10k-*-{N}steps`` (full KS summaries + validity).
    """
    if ablation_layout == "samples_old":
        ab = collect_num_steps_samples_old(
            ablation_root, ckpt_k=samples_old_ckpt_k, mode=mode, n_gen_candidates=n_gen_candidates
        )
    else:
        ab = collect_num_steps_ablation(ablation_root, mode=mode, n_gen_candidates=n_gen_candidates)
    p = collect_branching_dt_sweep(perm_root, "permutation", mode=mode, n_gen_candidates=n_gen_candidates)
    np_ = collect_branching_dt_sweep(no_perm_root, "no_permutation", mode=mode, n_gen_candidates=n_gen_candidates)
    return pd.concat([ab, p, np_], ignore_index=True)


def _fraction_ylim(lows: list, highs: list) -> tuple[float, float]:
    arr_l = np.asarray([x for x in lows if np.isfinite(x)], dtype=float)
    arr_h = np.asarray([x for x in highs if np.isfinite(x)], dtype=float)
    if arr_l.size == 0 or arr_h.size == 0:
        return 0.0, 1.0
    dmin, dmax = float(arr_l.min()), float(arr_h.max())
    span = dmax - dmin
    pad = max(span * 0.15, 0.01)
    ymin = max(0.0, dmin - pad)
    ymax = min(1.0, dmax + pad)
    if ymax - ymin < 0.04:
        mid = 0.5 * (dmin + dmax)
        ymin = max(0.0, mid - 0.02)
        ymax = min(1.0, mid + 0.02)
    if ymax <= ymin:
        ymin, ymax = 0.0, 1.0
    return ymin, ymax


def _style_for_method(method: str) -> dict:
    if method in SAMPLER_STYLES:
        return SAMPLER_STYLES[method]
    if method in BRANCHING_STYLES:
        return BRANCHING_STYLES[method]
    return {"color": "#333333", "marker": "o", "ls": "-"}


def _label_for_method(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method in BRANCHING_LABELS:
        return BRANCHING_LABELS[method]
    return method


def plot_single_metric(
    df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    out_pdf: Path,
    out_png: Path | None = None,
    y_is_fraction: bool = False,
    layout: LayoutKind = "full",
) -> None:
    configure_matplotlib()
    df = df.sort_values([x_col, "method"]).copy()
    if df.empty or y_col not in df.columns:
        print(f"No data for {y_col} → skip {out_pdf}")
        return

    n_m = int(df["method"].nunique())
    if layout == "compact":
        # Appendix-oriented: readable legend (wider/taller than 3-up inset size).
        fig_w = 3.65 + 0.14 * max(0, n_m - 3)
        fig_h = 3.15
        ln_w, ms = 1.55, 4.85
        t_fs, lab_fs, leg_fs = 10.2, 9.35, 8.05
        tk = 8.35
    else:
        fig_w = 5.2 + 0.4 * max(0, n_m - 3)
        fig_h = 3.35
        ln_w, ms = 2.0, 6.5
        t_fs, lab_fs, leg_fs = 11.5, 11.0, 9.5
        tk = 10.0

    fig_w_use = min(fig_w, 4.55) if layout == "compact" else fig_w
    fig, ax = plt.subplots(figsize=(fig_w_use, fig_h))
    lows: list = []
    highs: list = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values(x_col)
        sty = _style_for_method(method)
        y = sub[y_col].to_numpy(dtype=float)
        label = _label_for_method(method)
        ax.plot(
            sub[x_col],
            y,
            label=label,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["ls"],
            linewidth=ln_w,
            markersize=ms,
        )
        if y_is_fraction:
            mask = np.isfinite(y)
            if mask.any():
                lows.extend(y[mask].tolist())
                highs.extend(y[mask].tolist())
    if layout == "compact":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20 - 0.032 * max(0, n_m - 4)),
            ncol=2,
            frameon=False,
            fontsize=leg_fs,
            columnspacing=0.95,
            handletextpad=0.55,
            handlelength=2.25,
            borderpad=0.35,
        )
    else:
        ax.legend(loc="best", frameon=False, fontsize=leg_fs)
    if layout == "compact":
        ax.set_xlabel(x_label, fontsize=lab_fs, labelpad=8.0)
    else:
        ax.set_xlabel(x_label, fontsize=lab_fs)
    ax.set_ylabel(y_label, fontsize=lab_fs)
    ax.set_title(title, fontsize=t_fs, pad=5 if layout == "compact" else 6)
    ax.tick_params(axis="both", labelsize=tk)
    ax.grid(True, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_is_fraction:
        ymin, ymax = _fraction_ylim(lows, highs)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    if layout == "full":
        fig.tight_layout()
    else:
        b = 0.30 + 0.045 * max(0, n_m - 4)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=b)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)


def plot_all_metrics(
    df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    file_prefix: str,
    out_dir: Path,
    png: bool = False,
    layout: LayoutKind = "full",
) -> None:
    """One PDF per metric: Average, each KS metric, and validity."""
    if df.empty:
        print(f"Empty dataframe for prefix {file_prefix}")
        return
    metric_cols = [c for c in df.columns if c not in ("method", x_col, "nfe", "train_step", "num_steps", "dt")]
    if "valid_frac" in metric_cols:
        metric_cols.remove("valid_frac")

    # Average first (column name from eval file)
    order = []
    if "Average" in metric_cols:
        order.append("Average")
    order.extend(sorted(m for m in metric_cols if m != "Average"))

    for m in order:
        y_label = r"$1-\mathrm{KS}$" if m != "Average" else r"Average $1-\mathrm{KS}$"
        plot_title = METRIC_LABELS.get(m, m)
        plot_single_metric(
            df,
            x_col=x_col,
            x_label=x_label,
            y_col=m,
            y_label=y_label,
            title=f"{plot_title} ({file_prefix})",
            out_pdf=out_dir / f"{file_prefix}_{m}.pdf",
            out_png=(out_dir / f"{file_prefix}_{m}.png") if png else None,
            y_is_fraction=False,
            layout=layout,
        )

    if "valid_frac" in df.columns:
        plot_single_metric(
            df,
            x_col=x_col,
            x_label=x_label,
            y_col="valid_frac",
            y_label="Fraction valid",
            title=f"Validity ({file_prefix})",
            out_pdf=out_dir / f"{file_prefix}_validity.pdf",
            out_png=(out_dir / f"{file_prefix}_validity.png") if png else None,
            y_is_fraction=True,
            layout=layout,
        )


__all__ = [
    "LayoutKind",
    "configure_matplotlib",
    "collect_branching_dt_sweep",
    "collect_combined_num_steps_grid",
    "collect_num_steps_ablation",
    "collect_num_steps_samples_old",
    "parse_ks_summary",
    "plot_all_metrics",
    "plot_single_metric",
]

