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


def tokenize_expression(text: str) -> list[float | str]:
    tokens: list[float | str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in {"+", "-", "(", ")"}:
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            start = i
            i += 1
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                i += 1
            tokens.append(float(text[start:i]))
            continue
        raise ValueError(f"Unexpected character in expression: '{ch}'")
    return tokens


class TokenStream:
    def __init__(self, tokens: list[float | str]):
        self.tokens = tokens
        self.idx = 0

    def peek(self) -> float | str | None:
        if self.idx >= len(self.tokens):
            return None
        return self.tokens[self.idx]

    def next(self) -> float | str:
        if self.idx >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok


def parse_factor(ts: TokenStream) -> float:
    tok = ts.peek()
    if tok in {"+", "-"}:
        op = ts.next()
        val = parse_factor(ts)
        return val if op == "+" else -val
    if tok == "(":
        ts.next()
        val = parse_expression(ts)
        if ts.next() != ")":
            raise ValueError("Expected ')'.")
        return val
    if isinstance(tok, float):
        return float(ts.next())
    raise ValueError(f"Unexpected token in expression: {tok}")


def parse_expression(ts: TokenStream) -> float:
    val = parse_factor(ts)
    while True:
        tok = ts.peek()
        if tok == "+":
            ts.next()
            val += parse_factor(ts)
        elif tok == "-":
            ts.next()
            val -= parse_factor(ts)
        else:
            break
    return val


def evaluate_expression(text: str) -> float:
    tokens = tokenize_expression(text)
    ts = TokenStream(tokens)
    val = parse_expression(ts)
    if ts.peek() is not None:
        raise ValueError("Unexpected trailing tokens in expression.")
    return val


def evaluate_equation_text(text: str) -> tuple[float, float]:
    text = text.strip()
    if "=" not in text:
        raise ValueError("Equation text missing '='.")
    left_text, right_text = text.split("=", 1)
    return evaluate_expression(left_text), evaluate_expression(right_text)


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
        
        def pick_group_spans(terms, max_groups: int = 2) -> list[tuple[int, int]]:
            n_terms = len(terms)
            if n_terms < 2:
                return []
            valid_starts = [
                i for i, (sign, _) in enumerate(terms) if i == 0 or sign > 0
            ]
            if not valid_starts:
                return []
            possible_groups = min(max_groups, n_terms // 2, len(valid_starts))
            n_groups = rng.randint(1, possible_groups)
            spans: list[tuple[int, int]] = []
            attempts = 0
            while len(spans) < n_groups and attempts < 50:
                attempts += 1
                start = rng.choice(valid_starts)
                if start >= n_terms - 1:
                    continue
                end = rng.randrange(start + 1, n_terms)
                if any(not (end < s or start > e) for s, e in spans):
                    continue
                spans.append((start, end))
            return sorted(spans)

        left_groups = pick_group_spans(left_terms)
        right_groups = pick_group_spans(right_terms)

        # Build equation string and symbols with parenthesized spans.
        def format_side(terms, groups):
            if not terms:
                return "", []
            start_set = {s for s, _ in groups}
            end_set = {e for _, e in groups}
            pieces = []
            side_symbols: list[str] = []
            for i, (sign, val) in enumerate(terms):
                value_text = format_value(val)
                if i == 0:
                    prefix = "-" if sign < 0 else ""
                    if i in start_set:
                        side_symbols.append("(")
                        pieces.append(f"({prefix}{value_text}")
                    else:
                        pieces.append(f"{prefix}{value_text}")
                else:
                    op = "+" if sign > 0 else "-"
                    side_symbols.append(op)
                    if i in start_set:
                        side_symbols.append("(")
                        pieces.append(f"{op}({value_text}")
                    else:
                        pieces.append(f"{op}{value_text}")
                if i in end_set:
                    side_symbols.append(")")
                    pieces[-1] += ")"
            return "".join(pieces), side_symbols
        
        left_text, left_symbols = format_side(left_terms, left_groups)
        right_text, right_symbols = format_side(right_terms, right_groups)
        equation = f"{left_text}={right_text}"

        # Build parsed representation
        numbers = []
        symbols = []

        for i, (sign, val) in enumerate(left_terms):
            if i == 0:
                numbers.append(sign * val)
            else:
                numbers.append(val)

        symbols.extend(left_symbols)
        symbols.append("=")

        for i, (sign, val) in enumerate(right_terms):
            if i == 0:
                numbers.append(sign * val)
            else:
                numbers.append(val)

        symbols.extend(right_symbols)
        
        return {
            "equation": equation,
            "numbers": numbers,
            "symbols": symbols,
            "result": target,
            "length": len(symbols),
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that generated equations satisfy equality.",
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=1e-6,
        help="Tolerance for equation verification.",
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
            if args.verify:
                left_val, right_val = evaluate_equation_text(eq["equation"])
                if abs(left_val - right_val) > args.verify_tol:
                    raise RuntimeError(
                        "Verification failed for equation "
                        f"{eq['equation']} (left={left_val}, right={right_val})."
                    )
            f.write(json.dumps(eq) + "\n")
    
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
