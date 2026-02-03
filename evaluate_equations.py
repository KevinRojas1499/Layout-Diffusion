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


def eval_linear(numbers: List[float], ops: List[str]) -> float:
    if not numbers:
        raise ValueError("No numbers to evaluate.")
    if len(numbers) != len(ops) + 1:
        raise ValueError("Mismatched numbers and operators.")

    # First pass: handle * and /.
    stack = [numbers[0]]
    for op, num in zip(ops, numbers[1:]):
        if op == "*":
            stack[-1] *= num
        elif op == "/":
            stack[-1] /= num
        elif op in {"+", "-"}:
            stack.append(num if op == "+" else -num)
        else:
            raise ValueError(f"Unsupported operator: {op}")

    return sum(stack)


def parse_equation(
    symbols: List[str],
    positions: List,
    ignore_trailing_ops: bool = False,
) -> Tuple[float, float]:
    numbers = [extract_number(pos) for pos in positions]
    if any(math.isnan(n) for n in numbers):
        raise ValueError("Found NaN number.")
    if len(symbols) != len(numbers):
        raise ValueError("Symbols and numbers length mismatch.")

    left_numbers: List[float] = []
    left_ops: List[str] = []
    right_numbers: List[float] = []
    right_ops: List[str] = []
    on_right = False

    if not numbers:
        raise ValueError("No numbers available.")
    left_numbers.append(numbers[0])

    for i, sym in enumerate(symbols):
        if sym == ".":
            break
        if sym == "=":
            if i + 1 >= len(numbers):
                raise ValueError("No number after '='.")
            on_right = True
            right_numbers.append(numbers[i + 1])
            continue
        if sym not in {"+", "-", "*", "/"}:
            raise ValueError(f"Unsupported symbol: {sym}")
        if i + 1 >= len(numbers):
            if ignore_trailing_ops:
                break
            raise ValueError(f"No number after operator {sym}.")
        if on_right:
            right_ops.append(sym)
            right_numbers.append(numbers[i + 1])
        else:
            left_ops.append(sym)
            left_numbers.append(numbers[i + 1])

    if not right_numbers:
        raise ValueError("No right-hand side found.")

    left_value = eval_linear(left_numbers, left_ops)
    right_value = eval_linear(right_numbers, right_ops)
    return left_value, right_value


def count_equation_symbols(symbols: List[str]) -> int:
    count = 0
    for sym in symbols:
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
    help="Path to samples.json produced by sampling_toy.py",
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
    default="data/equations_varlen.txt",
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
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("molecules", [])
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
        positions = sample.get("positions", [])
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
        gt_lengths = [count_equation_symbols_from_text(line) for line in gt_lines]
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