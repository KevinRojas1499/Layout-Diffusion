"""Grid search over samplers and step counts for a given checkpoint.

Usage:
    uv run python grid_search_samplers.py \
        --checkpoint runs/parenthesis/stage1-depth8-100k/itr_70000/snapshot.pt \
        --out runs/parenthesis/stage1-depth8-100k/grid_search.jsonl
"""

import argparse
import json
import os
import statistics
import sys
import torch

sys.path.insert(0, '.')
from parenthesis_training import VocabTokenizer
from multimodal_interpolant import MultimodalInterpolant
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from models.mmdit_qm9 import MMDiTQM9
from evaluate_equations_parenthesis import parse_equation


def evaluate_samples(samples, char_tok):
    n = samples.xt.shape[0]
    valid = invalid = correct = 0
    diffs, scale_vals = [], []

    for i in range(n):
        syms_str = char_tok.decode(samples.yt[i].cpu())
        syms = list(syms_str)
        pos = samples.xt[i].cpu()[1:len(syms) + 1, :]
        nums = pos.squeeze(-1).tolist()
        try:
            L, R = parse_equation(syms, nums)
            valid += 1
            diff = abs(L - R)
            diffs.append(diff)
            if diff < 0.5:
                correct += 1
            if nums:
                scale_vals.append(max(abs(x) for x in nums))
        except Exception:
            invalid += 1

    accuracy = correct / valid if valid > 0 else 0.0
    mae = statistics.mean(diffs) if diffs else float('inf')
    avg_max_abs = statistics.mean(scale_vals) if scale_vals else 0.0
    return {
        'valid': valid, 'invalid': invalid, 'correct': correct,
        'accuracy': accuracy, 'mean_abs_error': mae, 'avg_max_abs': avg_max_abs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--eval_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Samples per forward pass; batches are aggregated to reach eval_samples')
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--data_path', default='data/parenthesis/equations_l5.jsonl')
    parser.add_argument('--out', required=True)
    parser.add_argument('--fp16', action='store_true', help='Run inference in fp16')
    args = parser.parse_args()

    steps_list = [100, 200, 300, 400, 500, 600, 750]
    samplers = ['euler', 'split', 'staggered']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    char_tok = VocabTokenizer(vocab={'+', '-', '*', '=', '.', '(', ')'})
    dataset = ParenthesizedEquationsDataset(char_tok, data_path=args.data_path)

    model = MMDiTQM9(
        euclidean_dim=1, vocab_size=char_tok.vocab_size,
        symbols_depth=args.depth, positions_depth=args.depth, depth=args.depth,
        dim_modalities=[args.hidden_dim, args.hidden_dim],
        dim_joint_attn=args.hidden_dim,
        dim_conds=[args.hidden_dim, args.hidden_dim],
    ).to(device)

    snap = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(snap['ema'])
    model.eval()
    if args.fp16:
        print("Running with autocast (mixed precision).")

    interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=char_tok.vocab_size,
        mask_token=char_tok.mask_token_id,
        pad_token=char_tok.pad_token_id,
        bos_token=char_tok.bos_token_id,
        euclidean_dim=1,
    )

    # Load already-completed cells so we can skip them on resume
    done = set()
    results = []
    if os.path.exists(args.out):
        with open(args.out) as fin:
            for line in fin:
                row = json.loads(line)
                done.add((row['sampler'], row['steps']))
                results.append(row)
        print(f"Resuming — skipping {len(done)} already-completed cell(s).")

    header = f"{'sampler':<12} {'steps':>6} {'acc':>8} {'valid':>8} {'MAE':>8} {'scale':>8}"
    print(header)
    print('-' * len(header))

    # Print already-done rows so the table is complete
    for row in results:
        print(
            f"{row['sampler']:<12} {row['steps']:>6}  "
            f"{row['accuracy']:>8.1%} {row['valid']:>4}/{row['eval_samples']}"
            f" {row['mean_abs_error']:>8.3f} {row['avg_max_abs']:>8.3f}  (cached)"
        )

    with open(args.out, 'a') as fout:
        for sampler in samplers:
            for steps in steps_list:
                if (sampler, steps) in done:
                    continue
                print(f"{sampler:<12} {steps:>6}  ", end='', flush=True)
                # Run in batches to avoid OOM when training is running concurrently
                all_xt, all_yt = [], []
                remaining = args.eval_samples
                autocast_ctx = torch.autocast('cuda', dtype=torch.float16) if args.fp16 and device.type == 'cuda' else torch.autocast('cpu', enabled=False)
                while remaining > 0:
                    bs = min(args.batch_size, remaining)
                    with torch.no_grad(), autocast_ctx:
                        batch = interpolant.sampling(
                            model, steps, bs,
                            dataset.max_length, device,
                            sampler=sampler,
                        )
                    all_xt.append(batch.xt.cpu().float())
                    all_yt.append(batch.yt.cpu())
                    remaining -= bs

                # Stitch batches together into a single SamplingResult-like object
                import types
                merged = types.SimpleNamespace(
                    xt=torch.cat(all_xt, dim=0),
                    yt=torch.cat(all_yt, dim=0),
                )
                m = evaluate_samples(merged, char_tok)
                row = {'sampler': sampler, 'steps': steps, 'eval_samples': args.eval_samples, **m}
                fout.write(json.dumps(row) + '\n')
                fout.flush()
                results.append(row)
                print(
                    f"{m['accuracy']:>8.1%} {m['valid']:>4}/{args.eval_samples}"
                    f" {m['mean_abs_error']:>8.3f} {m['avg_max_abs']:>8.3f}"
                )

    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
