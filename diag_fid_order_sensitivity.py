"""Test whether FIDNet/LayoutNet is order-sensitive: compute FID on
samples as-is and again after sorting each sample by y_topleft = cy - h/2,
which is the canonical RICO order.

If the two FIDs differ noticeably, FIDNet sees order and we should sort
outputs as a free win. If they match, FIDNet is order-invariant and any
order-related fix has to happen inside the sampler.
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from eval.layout import LayoutEvaluator
from layout_training import _get_dataset_and_dims, _get_interpolant, _get_model


def sort_by_y_topleft(xt: torch.Tensor, yt: torch.Tensor, mask: torch.Tensor):
    """Permute active layout positions (excluding BOS at idx 0) by y_topleft.
    Padding positions are left untouched.
    """
    B, L, _ = xt.shape
    xt_sorted = xt.clone()
    yt_sorted = yt.clone()
    for b in range(B):
        active = mask[b].clone()
        active[0] = False  # never move BOS
        active_idx = active.nonzero(as_tuple=True)[0]
        if active_idx.numel() <= 1:
            continue
        cy = xt[b, active_idx, 1]
        h = xt[b, active_idx, 3]
        y_top = cy - h / 2
        order = torch.argsort(y_top)
        permuted_idx = active_idx[order]
        xt_sorted[b, active_idx] = xt[b, permuted_idx]
        yt_sorted[b, active_idx] = yt[b, permuted_idx]
    return xt_sorted, yt_sorted


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="rico25")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=20)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_rope", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", default="split")
    p.add_argument("--layoutflow_root", default="/workspace/Variable-Length-Diffusion-Toy/LayoutFlow")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = {
        "publaynet": "/workspace/LayoutFlow-data/dataset/publaynet",
        "rico25": "/workspace/LayoutFlow-data/dataset/rico",
    }[args.dataset]

    cfg = OmegaConf.create({
        "dataset": {
            "name": args.dataset, "data_path": data_path,
            "max_length": args.max_length, "split": "val", "max_samples": None,
        },
        "model": {
            "name": "Transformer", "symbols_depth": 4, "positions_depth": 4,
            "depth": args.depth, "hidden_dim": args.hidden_dim,
            "use_spatial_bias": False, "use_rope": args.use_rope,
        },
        "interpolant": {
            "name": "multimodal", "dsm_t_reweight": False,
            "cfg_dropout_prob": 0.0, "cat_cond_prob": 0.0, "size_cond_prob": 0.0,
        },
    })

    dataset, tokenizer, euclidean_dim, hidden_dim, max_length = _get_dataset_and_dims(cfg)
    model = _get_model(cfg, tokenizer.vocab_size, euclidean_dim, hidden_dim, device)
    interpolant = _get_interpolant(cfg, dataset, tokenizer, euclidean_dim)

    snap = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = snap["ema"] if args.use_ema else snap["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    evaluator = LayoutEvaluator.for_dataset(
        dataset_name=args.dataset, tokenizer=tokenizer, device=device,
        layoutflow_root=args.layoutflow_root,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    xt_chunks, yt_chunks, mask_chunks = [], [], []
    n_done = 0
    while n_done < args.num_samples:
        b = min(args.batch_size, args.num_samples - n_done)
        s = interpolant.sampling(
            model, args.num_steps, b, max_length + 1, device,
            return_trace=False, sampler=args.sampler,
        )
        mask = s.y_mask_t if hasattr(s, "y_mask_t") else s.mask_t
        xt_chunks.append(s.xt)
        yt_chunks.append(s.yt)
        mask_chunks.append(mask)
        n_done += b
        print(f"  sampled {n_done}/{args.num_samples}", flush=True)

    L = max(c.size(1) for c in xt_chunks)
    def _pad(c, fill):
        if c.size(1) == L:
            return c
        pad = torch.full((c.size(0), L - c.size(1), *c.shape[2:]), fill,
                         dtype=c.dtype, device=c.device)
        return torch.cat([c, pad], dim=1)
    xt = torch.cat([_pad(c, 0.0) for c in xt_chunks], dim=0)
    yt = torch.cat([_pad(c, tokenizer.pad_token_id) for c in yt_chunks], dim=0)
    mask = torch.cat([_pad(c, False) for c in mask_chunks], dim=0)

    print("\n=== AS-IS (insertion order) ===")
    metrics_asis = evaluator.evaluate_samples(xt, yt, mask, batch_size=args.batch_size)
    for k, v in metrics_asis.items():
        print(f"  {k}: {v}")

    print("\n=== SORTED (y_topleft = cy - h/2) ===")
    xt_sorted, yt_sorted = sort_by_y_topleft(xt, yt, mask)
    metrics_sorted = evaluator.evaluate_samples(xt_sorted, yt_sorted, mask, batch_size=args.batch_size)
    for k, v in metrics_sorted.items():
        print(f"  {k}: {v}")

    print("\n=== DELTA (sorted - asis) ===")
    for k in metrics_asis:
        if isinstance(metrics_asis[k], (int, float)) and isinstance(metrics_sorted[k], (int, float)):
            delta = metrics_sorted[k] - metrics_asis[k]
            print(f"  {k}: {delta:+.4f}")

    fid_asis = metrics_asis.get("fid", metrics_asis.get("FID", None))
    fid_sorted = metrics_sorted.get("fid", metrics_sorted.get("FID", None))
    if fid_asis is not None and fid_sorted is not None:
        print()
        if abs(fid_sorted - fid_asis) < 0.5:
            print("VERDICT: FIDNet appears ORDER-INVARIANT (Δ < 0.5). "
                  "Damage is internal to model trajectory; sampler-side fix needed.")
        elif fid_sorted < fid_asis:
            print(f"VERDICT: FIDNet is ORDER-SENSITIVE; sorting cuts FID by {fid_asis - fid_sorted:.2f}. "
                  "Apply sort as a free post-hoc fix.")
        else:
            print(f"VERDICT: Sorting INCREASED FID by {fid_sorted - fid_asis:.2f}. Don't sort.")


if __name__ == "__main__":
    main()
