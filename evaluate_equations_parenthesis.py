import json
import math
import click
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def extract_number(pos) -> float:
    if isinstance(pos, (list, tuple)):
        if not pos:
            return float("nan")
        return float(pos[0])
    return float(pos)


def _normalize_symbols(symbols) -> List[str]:
    if isinstance(symbols, str):
        return [ch for ch in symbols if ch.strip()]
    return list(symbols)


def _trim_until_dot(symbols: List[str]) -> List[str]:
    out = []
    for sym in symbols:
        if sym == ".":
            break
        out.append(sym)
    return out


def _eval_implicit_numbers_expr(
    symbols: List[str],
    numbers: List[float],
    start_idx: int,
    ignore_trailing_ops: bool,
) -> Tuple[float, int]:
    i = 0
    n_idx = start_idx

    def next_number() -> float:
        nonlocal n_idx
        if n_idx >= len(numbers):
            raise ValueError("Not enough numbers for expression.")
        value = numbers[n_idx]
        n_idx += 1
        return value

    def parse_factor() -> float:
        nonlocal i
        if i < len(symbols) and symbols[i] == "(":
            i += 1
            value = parse_expr()
            if i >= len(symbols) or symbols[i] != ")":
                raise ValueError("Unmatched '('.")
            i += 1
            return value
        if i < len(symbols) and symbols[i] in {"+", "-"}:
            op = symbols[i]
            i += 1
            val = parse_factor()
            return val if op == "+" else -val
        return next_number()

    def parse_term() -> float:
        nonlocal i
        value = parse_factor()
        while i < len(symbols) and symbols[i] in {"*", "/"}:
            op = symbols[i]
            i += 1
            rhs = parse_factor()
            if op == "*":
                value *= rhs
            else:
                value /= rhs
        return value

    def parse_expr() -> float:
        nonlocal i
        value = parse_term()
        while i < len(symbols) and symbols[i] in {"+", "-"}:
            op = symbols[i]
            i += 1
            rhs = parse_term()
            if op == "+":
                value += rhs
            else:
                value -= rhs
        return value

    if not symbols:
        raise ValueError("Empty expression side.")

    value = parse_expr()
    if i != len(symbols):
        if ignore_trailing_ops:
            while i < len(symbols) and symbols[i] in {"+", "-", "*", "/"}:
                i += 1
        if i != len(symbols):
            raise ValueError(f"Could not parse symbols: {symbols[i:]}")
    return value, n_idx


def parse_equation(
    symbols: List[str],
    positions: List,
    ignore_trailing_ops: bool = False,
) -> Tuple[float, float]:
    numbers = [extract_number(pos) for pos in positions]
    if any(math.isnan(n) for n in numbers):
        raise ValueError("Found NaN number.")
    if not numbers:
        raise ValueError("No numbers available.")

    symbols_seq = _trim_until_dot(_normalize_symbols(symbols))
    eq_positions = [i for i, sym in enumerate(symbols_seq) if sym == "="]
    if len(eq_positions) != 1:
        raise ValueError("Equation must contain exactly one '=' before '.'.")

    eq_idx = eq_positions[0]
    left_syms = symbols_seq[:eq_idx]
    right_syms = symbols_seq[eq_idx + 1 :]

    used_idx = 0
    if left_syms:
        left_value, used_idx = _eval_implicit_numbers_expr(
            left_syms,
            numbers,
            start_idx=used_idx,
            ignore_trailing_ops=ignore_trailing_ops,
        )
    else:
        if used_idx >= len(numbers):
            raise ValueError("Missing numeric value on left side.")
        left_value = numbers[used_idx]
        used_idx += 1

    if right_syms:
        right_value, used_idx = _eval_implicit_numbers_expr(
            right_syms,
            numbers,
            start_idx=used_idx,
            ignore_trailing_ops=ignore_trailing_ops,
        )
    else:
        if used_idx >= len(numbers):
            raise ValueError("Missing numeric value on right side.")
        right_value = numbers[used_idx]
        used_idx += 1

    if used_idx > len(numbers):
        raise ValueError("Consumed more numbers than available.")
    return left_value, right_value


def count_equation_symbols(symbols: List[str]) -> int:
    symbols_seq = _normalize_symbols(symbols)
    count = 0
    for sym in symbols_seq:
        count += 1
        if sym == ".":
            break
    return count


def count_equation_symbols_from_text(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return sum(1 for ch in text if ch in {"+", "-", "*", "/", "=", "."})


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to samples.json or samples.jsonl produced by sampling_toy.py",
)
@click.option(
    "--delta",
    type=float,
    default=0.5,
    show_default=True,
    help="Absolute tolerance threshold for equality.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print reasons for invalid equations.",
)
@click.option(
    "--ignore-trailing-ops",
    is_flag=True,
    default=False,
    help="Ignore trailing operators without a following number.",
)
@click.option(
    "--hist-bins",
    type=int,
    default=20,
    show_default=True,
    help="Number of bins for the absolute error histogram.",
)
@click.option(
    "--hist-max",
    type=float,
    default=None,
    help="Max absolute error for histogram range (defaults to max observed).",
)
@click.option(
    "--hist-out",
    type=click.Path(dir_okay=False, writable=True),
    default="hist_abs_error.png",
    show_default=True,
    help="Output path for histogram plot.",
)
@click.option(
    "--len-hist-out",
    type=click.Path(dir_okay=False, writable=True),
    default="hist_equation_length.png",
    show_default=True,
    help="Output path for equation length distribution plot.",
)
@click.option(
    "--gt-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="data/equations.jsonl",
    show_default=True,
    help="Ground-truth equations file to compare length distribution.",
)
def main(
    input_path: str,
    delta: float,
    debug: bool,
    ignore_trailing_ops: bool,
    hist_bins: int,
    hist_max: float | None,
    hist_out: str,
    len_hist_out: str,
    gt_file: str,
) -> None:
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise RuntimeError("Input file is empty.")
        if "\n" in content and content.lstrip().startswith("{") and not content.lstrip().startswith("{\"molecules\""):
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        else:
            data = json.loads(content)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                if "samples" in data and isinstance(data["samples"], list):
                    samples = data["samples"]
                elif "molecules" in data and isinstance(data["molecules"], list):
                    samples = data["molecules"]
                else:
                    samples = [data]
            else:
                raise RuntimeError("Unsupported JSON format.")
    total = len(samples)
    if total == 0:
        raise RuntimeError("No samples found in input file.")

    valid = 0
    correct = 0
    invalid = 0
    diffs: List[float] = []
    gen_lengths: List[int] = []

    for idx, sample in enumerate(samples):
        symbols = sample.get("symbols", [])
        positions = sample.get("positions", sample.get("numbers", []))
        gen_lengths.append(count_equation_symbols(symbols))
        try:
            left, right = parse_equation(
                symbols,
                positions,
                ignore_trailing_ops=ignore_trailing_ops,
            )
        except Exception as exc:
            invalid += 1
            if debug:
                print(f"invalid_sample[{idx}]: {exc}")
                print(f"symbols={symbols}")
                print(f"positions={positions}")
            continue
        valid += 1
        diff = abs(left - right)
        diffs.append(diff)
        if diff < delta:
            correct += 1

    accuracy = correct / valid if valid > 0 else 0.0
    print(f"total_samples: {total}")
    print(f"valid_equations: {valid}")
    print(f"invalid_equations: {invalid}")
    print(f"correct_within_delta: {correct}")
    print(f"accuracy: {accuracy:.4f}")

    if diffs:
        max_diff = max(diffs) if hist_max is None else hist_max
        if max_diff <= 0:
            max_diff = 1.0
        diffs_arr = np.asarray(diffs, dtype=float)
        mean_diff = float(diffs_arr.mean())
        median_diff = float(np.median(diffs_arr))

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(8.5, 5.5))
        plt.hist(
            diffs_arr,
            bins=hist_bins,
            range=(0, max_diff),
            color="#4C78A8",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            density=True,
        )
        plt.axvline(mean_diff, color="#F58518", linestyle="--", linewidth=1.5, label=f"mean={mean_diff:.3f}")
        plt.axvline(median_diff, color="#54A24B", linestyle=":", linewidth=1.5, label=f"median={median_diff:.3f}")
        plt.title("Absolute Error Histogram")
        plt.xlabel("|left - right|")
        plt.ylabel("Count")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(hist_out, dpi=200, facecolor="white")
        plt.close()

    if gen_lengths:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(8.5, 5.5))
        max_len = max(gen_lengths)
        bins = np.arange(1, max_len + 2) - 0.5
        plt.hist(
            gen_lengths,
            bins=bins,
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            label="generated",
            density=True,
        )
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_lines = [line.strip() for line in f if line.strip()]
        gt_lengths = []
        for line in gt_lines:
            item = json.loads(line)
            if "symbols" in item:
                gt_lengths.append(count_equation_symbols(item["symbols"]))
            elif "numbers" in item:
                gt_lengths.append(len(item["numbers"]))
        if gt_lengths:
            max_len = max(max_len, max(gt_lengths))
            bins = np.arange(1, max_len + 2) - 0.5
            plt.hist(
                gt_lengths,
                bins=bins,
                color="#F58518",
                alpha=0.45,
                edgecolor="white",
                linewidth=0.8,
                label="ground_truth",
                density=True,
            )
        plt.title("Equation Length Distribution")
        plt.xlabel("Number of symbols (+, -, *, /, =, .)")
        plt.ylabel("Count")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(len_hist_out, dpi=200, facecolor="white")
        plt.close()


if __name__ == "__main__":
    main()