import argparse
import random
import bisect
from typing import Dict, List, Tuple


def eval_expr(terms: List[Tuple[int, int]]) -> int:
    total = 0
    for sign, value in terms:
        total += sign * value
    return total


def format_expr(terms: List[Tuple[int, int]]) -> str:
    if not terms:
        raise ValueError("Expression must contain at least one term.")
    pieces = [str(terms[0][1])]  # first term is always positive in our generator
    for sign, value in terms[1:]:
        op = "+" if sign > 0 else "-"
        pieces.append(f"{op}{value}")
    return "".join(pieces)


def generate_expression(
    num_terms: int,
    max_value: int,
    rng: random.Random,
    force_positive_first: bool = True,
) -> List[Tuple[int, int]]:
    if num_terms < 1:
        raise ValueError("num_terms must be >= 1")
    terms: List[Tuple[int, int]] = []
    first = rng.randint(1, max_value)
    terms.append((1, first))
    for _ in range(1, num_terms):
        value = rng.randint(1, max_value)
        sign = rng.choice([1, -1])
        terms.append((sign, value))
    return terms


def generate_expression_for_target(
    target: int,
    num_terms: int,
    max_value: int,
    rng: random.Random,
    max_attempts: int = 500,
) -> List[Tuple[int, int]]:
    if num_terms < 1:
        raise ValueError("num_terms must be >= 1")
    if num_terms == 1:
        if 1 <= target <= max_value:
            return [(1, target)]
        raise ValueError("Target out of range for single-term expression.")

    for _ in range(max_attempts):
        terms: List[Tuple[int, int]] = []
        first = rng.randint(1, max_value)
        terms.append((1, first))
        running = first
        for _ in range(num_terms - 2):
            value = rng.randint(1, max_value)
            sign = rng.choice([1, -1])
            terms.append((sign, value))
            running += sign * value

        remaining = target - running
        if remaining == 0:
            # Avoid zero terms; retry.
            continue

        sign = 1 if remaining > 0 else -1
        value = abs(remaining)
        if 1 <= value <= max_value:
            terms.append((sign, value))
            return terms

    raise RuntimeError("Failed to generate expression for target within constraints.")


def generate_all_expressions(
    min_terms: int,
    max_terms: int,
    max_value: int,
) -> Dict[int, List[str]]:
    if min_terms < 1 or max_terms < min_terms:
        raise ValueError("Invalid term range.")
    if max_value < 1:
        raise ValueError("max_value must be >= 1")

    expressions_by_value: Dict[int, List[str]] = {}

    def build_terms(remaining: int, current: List[Tuple[int, int]]) -> None:
        if remaining == 0:
            expr = format_expr(current)
            value = eval_expr(current)
            expressions_by_value.setdefault(value, []).append(expr)
            return

        if not current:
            for value in range(1, max_value + 1):
                build_terms(remaining - 1, current + [(1, value)])
        else:
            for sign in (1, -1):
                for value in range(1, max_value + 1):
                    build_terms(remaining - 1, current + [(sign, value)])

    for num_terms in range(min_terms, max_terms + 1):
        build_terms(num_terms, [])

    return expressions_by_value


def generate_equations_exhaustive(
    expressions_by_value: Dict[int, List[str]],
    num_equations: int,
    rng: random.Random,
) -> List[str]:
    values = sorted(expressions_by_value.keys())
    prefix_ends = []
    total_pairs = 0
    for value in values:
        m = len(expressions_by_value[value])
        total_pairs += m * m  # ordered pairs
        prefix_ends.append(total_pairs)

    if total_pairs == 0:
        raise RuntimeError("No expressions generated; check constraints.")
    if num_equations > total_pairs:
        raise ValueError(
            f"Requested {num_equations} equations, but only {total_pairs} unique pairs exist."
        )

    selected = rng.sample(range(total_pairs), num_equations)
    equations = []
    for idx in selected:
        group_idx = bisect.bisect_right(prefix_ends, idx)
        start = 0 if group_idx == 0 else prefix_ends[group_idx - 1]
        offset = idx - start
        value = values[group_idx]
        group = expressions_by_value[value]
        m = len(group)
        left_idx, right_idx = divmod(offset, m)
        equations.append(f"{group[left_idx]}={group[right_idx]}.")

    return equations


def build_equation(
    min_terms: int,
    max_terms: int,
    max_value: int,
    rng: random.Random,
) -> str:
    max_equation_attempts = 500
    for _ in range(max_equation_attempts):
        left_terms = rng.randint(min_terms, max_terms)
        right_terms = rng.randint(min_terms, max_terms)

        left = generate_expression(left_terms, max_value, rng)
        target = eval_expr(left)

        min_target = 1 - (right_terms - 1) * max_value
        max_target = right_terms * max_value
        if target < min_target or target > max_target:
            continue

        try:
            right = generate_expression_for_target(target, right_terms, max_value, rng)
        except RuntimeError:
            continue

        return f"{format_expr(left)}={format_expr(right)}."

    raise RuntimeError("Failed to build a valid equation within constraints.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variable-length arithmetic equations with + and -."
    )
    parser.add_argument("--num-equations", type=int, default=1000)
    parser.add_argument("--min-terms", type=int, default=2)
    parser.add_argument("--max-terms", type=int, default=4)
    parser.add_argument("--max-value", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["exhaustive", "random"],
        default="exhaustive",
        help="Exhaustive uses all expressions then samples pairs; random builds each equation on the fly.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/equations_varlen.txt",
        help="Output file path.",
    )
    args = parser.parse_args()

    if args.min_terms < 1 or args.max_terms < args.min_terms:
        raise ValueError("Invalid term range.")
    if args.max_value < 1:
        raise ValueError("max-value must be >= 1")

    rng = random.Random(args.seed)

    if args.mode == "exhaustive":
        expressions_by_value = generate_all_expressions(
            args.min_terms, args.max_terms, args.max_value
        )
        equations = generate_equations_exhaustive(
            expressions_by_value, args.num_equations, rng
        )
    else:
        equations = []
        for _ in range(args.num_equations):
            equations.append(
                build_equation(args.min_terms, args.max_terms, args.max_value, rng)
            )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(equations))
        f.write("\n")


if __name__ == "__main__":
    main()