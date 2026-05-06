"""Compare model's sampled length distribution against the test-set
length distribution. If they match closely, rejection sampling on length
won't help — the FID gap is conditional, not marginal.
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from omegaconf import OmegaConf

from custom_datasets.layoutflow_h5 import LayoutFlowH5Dataset
from layout_training import _get_dataset_and_dims, _get_interpolant, _get_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="rico25")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=20)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_rope", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", default="split")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = {
        "publaynet": "/workspace/LayoutFlow-data/dataset/publaynet",
        "rico25": "/workspace/LayoutFlow-data/dataset/rico",
    }[args.dataset]

    cfg = OmegaConf.create({
        "dataset": {
            "name": args.dataset,
            "data_path": data_path,
            "max_length": args.max_length,
            "split": "val",
            "max_samples": None,
        },
        "model": {
            "name": "Transformer",
            "symbols_depth": 4,
            "positions_depth": 4,
            "depth": args.depth,
            "hidden_dim": args.hidden_dim,
            "use_spatial_bias": False,
            "use_rope": args.use_rope,
        },
        "interpolant": {
            "name": "multimodal",
            "dsm_t_reweight": False,
            "cfg_dropout_prob": 0.0,
            "cat_cond_prob": 0.0,
            "size_cond_prob": 0.0,
        },
    })

    dataset, tokenizer, euclidean_dim, hidden_dim, max_length = _get_dataset_and_dims(cfg)
    model = _get_model(cfg, tokenizer.vocab_size, euclidean_dim, hidden_dim, device)
    interpolant = _get_interpolant(cfg, dataset, tokenizer, euclidean_dim)

    snap = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = snap["ema"] if args.use_ema else snap["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sampled_lengths = []
    n_done = 0
    while n_done < args.num_samples:
        b = min(args.batch_size, args.num_samples - n_done)
        s = interpolant.sampling(
            model, args.num_steps, b, max_length + 1, device,
            return_trace=False, sampler=args.sampler,
        )
        # mask_t includes BOS at position 0; subtract 1 to get layout length.
        mask = s.y_mask_t if hasattr(s, "y_mask_t") else s.mask_t
        lens = mask.sum(dim=1).cpu().tolist()
        sampled_lengths += [int(l) - 1 for l in lens]
        n_done += b
        print(f"  sampled {n_done}/{args.num_samples}", flush=True)

    test_ds = LayoutFlowH5Dataset(
        tokenizer=tokenizer, dataset_name=args.dataset, split="test",
        max_length=args.max_length, data_path=data_path,
    )
    # __getitem__ returns {"x": [L,4], "y": [L]} with rows already trimmed to L.
    test_lengths = [int(test_ds[i]["x"].shape[0]) for i in range(len(test_ds))]

    s_counter = Counter(sampled_lengths)
    t_counter = Counter(test_lengths)
    n_s = sum(s_counter.values())
    n_t = sum(t_counter.values())

    all_lens = sorted(set(s_counter) | set(t_counter))
    print()
    print(f"{'len':>4} | {'sampled':>10} | {'test':>10} | {'sample %':>9} | {'test %':>9} | gap")
    print("-" * 70)
    s_pmf, t_pmf = {}, {}
    for L in all_lens:
        sp = s_counter.get(L, 0) / n_s
        tp = t_counter.get(L, 0) / n_t
        s_pmf[L] = sp
        t_pmf[L] = tp
        gap = sp - tp
        marker = "  ***" if abs(gap) > 0.05 else ""
        print(f"{L:>4} | {s_counter.get(L,0):>10} | {t_counter.get(L,0):>10} | "
              f"{sp*100:>8.2f}% | {tp*100:>8.2f}% | {gap*100:>+6.2f}pp{marker}")

    tvd = 0.5 * sum(abs(s_pmf.get(L, 0) - t_pmf.get(L, 0)) for L in all_lens)
    s_mean = sum(L * c for L, c in s_counter.items()) / n_s
    t_mean = sum(L * c for L, c in t_counter.items()) / n_t

    print()
    print(f"Sampled n={n_s}, test n={n_t}")
    print(f"Mean length:  sampled={s_mean:.2f}  test={t_mean:.2f}  diff={s_mean - t_mean:+.2f}")
    print(f"Total Variation Distance (length pmf): {tvd:.4f}")
    print()
    if tvd < 0.05:
        verdict = "VERY CLOSE — rejection sampling will not help; skip Option 1, do Option 2."
    elif tvd < 0.15:
        verdict = "MODERATE gap — rejection sampling may recover ~1 FID point; worth trying."
    else:
        verdict = "LARGE gap — rejection sampling has real headroom; Option 1 first."
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
