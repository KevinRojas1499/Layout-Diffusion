"""Generate arithmetic equation datasets in JSONL format."""

import argparse
import json
import random
from pathlib import Path


def parse_length_dist(
    text: str | None,
    min_terms: int,
    max_terms: int,
) -> tuple[list[int], list[float]] | None:
    if not text:
        return None
    lengths: list[int] = []
    weights: list[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            length_str, weight_str = raw.split(":")
            length = int(length_str)
            weight = float(weight_str)
        except ValueError as exc:
            raise ValueError(f"Invalid length distribution entry: '{raw}'") from exc
        if length < min_terms or length > max_terms:
            raise ValueError(
                f"Length {length} outside range [{min_terms}, {max_terms}]"
            )
        if weight <= 0:
            raise ValueError(f"Weight must be positive for length {length}")
        lengths.append(length)
        weights.append(weight)
    if not lengths:
        return None
    total = sum(weights)
    weights = [w / total for w in weights]
    return lengths, weights


def generate_equation(
    min_terms: int,
    max_terms: int,
    min_value: float,
    max_value: float,
    rng: random.Random,
    decimals: int,
    length_dist: tuple[list[int], list[float]] | None,
) -> dict:
    """Generate a random equation with terms in [min_value, max_value].
    
    Returns dict with:
        - equation: string representation (e.g., "3-2=5-4.")
        - numbers: list of operands as parsed (first term signed, rest unsigned)
        - symbols: list of operators/punctuation
        - result: the value both sides equal
        - length: number of terms
    """
    max_attempts = 500
    
    min_abs = 10 ** (-decimals)

    def sample_value() -> float:
        val = rng.uniform(min_value, max_value)
        while abs(val) < min_abs:
            val = rng.uniform(min_value, max_value)
        return round(val, decimals)

    def format_value(val: float) -> str:
        if abs(val) < 1e-6:
            return "0"
        text = f"{abs(val):.{decimals}f}".rstrip("0").rstrip(".")
        return text if text else "0"

    for _ in range(max_attempts):
        if length_dist:
            lengths, weights = length_dist
            left_n = rng.choices(lengths, weights=weights, k=1)[0]
            right_n = rng.choices(lengths, weights=weights, k=1)[0]
        else:
            left_n = rng.randint(min_terms, max_terms)
            right_n = rng.randint(min_terms, max_terms)
        
        # Generate left side terms as (sign, abs_value)
        left_terms = []
        for _ in range(left_n):
            val = sample_value()
            left_terms.append((1 if val >= 0 else -1, abs(val)))
        
        target = sum(s * v for s, v in left_terms)
        
        # Try to generate right side that equals target
        right_terms = []
        running = 0
        for i in range(right_n - 1):
            val = sample_value()
            right_terms.append((1 if val >= 0 else -1, abs(val)))
            running += val
        
        # Last term must make right side equal target
        last = round(target - running, decimals)
        if abs(last) < min_abs or not (min_value <= last <= max_value):
            continue
        right_terms.append((1 if last >= 0 else -1, abs(last)))
        
        # Build equation string
        def format_side(terms):
            s, v = terms[0]
            pieces = [f"-{format_value(v)}" if s < 0 else format_value(v)]
            for sign, val in terms[1:]:
                pieces.append(f"+{format_value(val)}" if sign > 0 else f"-{format_value(val)}")
            return "".join(pieces)
        
        equation = f"{format_side(left_terms)}={format_side(right_terms)}."
        
        # Build parsed representation
        numbers = []
        symbols = []
        
        for i, (sign, val) in enumerate(left_terms):
            if i == 0:
                numbers.append(sign * val)
            else:
                symbols.append('+' if sign > 0 else '-')
                numbers.append(val)
        
        symbols.append('=')
        
        for i, (sign, val) in enumerate(right_terms):
            if i == 0:
                numbers.append(sign * val)
            else:
                symbols.append('+' if sign > 0 else '-')
                numbers.append(val)
        
        symbols.append('.')
        
        return {
            "equation": equation,
            "numbers": numbers,
            "symbols": symbols,
            "result": target,
            "length": len(numbers),
        }
    
    raise RuntimeError("Failed to generate equation")


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic equations dataset")
    parser.add_argument("-n", "--num-equations", type=int, default=50000)
    parser.add_argument("-L", "--value-range", type=float, default=10.0,
                        help="Values drawn from [-L, L]")
    parser.add_argument("--decimals", type=int, default=2,
                        help="Number of decimal places for sampled values.")
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument(
        "--length-dist",
        type=str,
        default=None,
        help="Comma-separated length:prob pairs, e.g. '1:0.1,2:0.3,3:0.6'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="data/equations.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    length_dist = parse_length_dist(args.length_dist, args.min_terms, args.max_terms)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_equations} equations with values in [-{args.value_range}, {args.value_range}]")
    
    with open(output, "w") as f:
        for _ in range(args.num_equations):
            eq = generate_equation(
                args.min_terms, args.max_terms,
                -args.value_range, args.value_range, rng,
                args.decimals,
                length_dist,
            )
            f.write(json.dumps(eq) + "\n")
    
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
