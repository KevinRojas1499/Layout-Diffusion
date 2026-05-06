"""Standalone CLI: evaluate a checkpoint through LayoutFlow's FID pipeline.

LayoutFlow uses LayoutDiffusion's `LayoutNet` feature extractor + a precomputed
test-split (mu, sigma); see `eval/layout/evaluator.py` for the implementation.
This script generates `--num_samples` from a checkpoint and reports FID +
alignment + overlap against that reference.

Example:
    uv run python evaluate_with_layoutflow.py \\
        --checkpoint runs/layout/<run>/itr_<N>/snapshot.pt \\
        --use_ema --depth 8 --hidden_dim 256 --no-use_spatial_bias \\
        --num_steps 200 --num_samples 2000 --seed 0 \\
        --layoutflow_root /workspace/LayoutFlow
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from custom_datasets.layoutflow_h5 import LayoutFlowH5Dataset
from eval.layout import LayoutEvaluator
from layout_training import _get_dataset_and_dims, _get_interpolant, _get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_samples", type=int, default=2000, help="LayoutFlow uses 2000 for uncond.")
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", type=str, default="split")
    p.add_argument("--cfg_weight", type=float, default=0.0,
                   help="Classifier-free guidance weight. 0 = no CFG.")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_spatial_bias", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_length", type=int, default=20, help="Our model's training max_length.")
    p.add_argument("--dataset", choices=["publaynet", "rico25"], default="publaynet")
    p.add_argument("--cond_mode", choices=["uncond", "cat_cond"], default="uncond",
                   help="cat_cond: draw real test-set categories and only diffuse positions.")
    p.add_argument("--layoutflow_root", default="/workspace/Variable-Length-Diffusion-Toy/LayoutFlow")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _get_layout_mask(samples) -> torch.Tensor:
    if hasattr(samples, "y_mask_t"):
        return samples.y_mask_t
    return samples.mask_t


@torch.no_grad()
def main() -> None:
    args = parse_args()
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
            "use_spatial_bias": args.use_spatial_bias,
            "use_rope": args.use_rope,
        },
        "interpolant": {"name": "multimodal", "dsm_t_reweight": False, "cfg_dropout_prob": 0.0, "cat_cond_prob": 0.0},
    })

    dataset, tokenizer, euclidean_dim, hidden_dim, max_length = _get_dataset_and_dims(cfg)
    model = _get_model(cfg, tokenizer.vocab_size, euclidean_dim, hidden_dim, device)
    interpolant = _get_interpolant(cfg, dataset, tokenizer, euclidean_dim)

    snapshot = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = snapshot["ema"] if args.use_ema else snapshot["model"]
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

    if args.cond_mode == "cat_cond":
        # Pull real (y, mask) pairs from the TEST split — same source the FID
        # mu/sigma was computed against. This makes cat_cond FID a true
        # conditional metric (predict positions given test categories).
        test_ds = LayoutFlowH5Dataset(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            split="test",
            max_length=args.max_length,
            data_path=data_path,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=test_ds.dynamic_collate,
        )
        test_iter = iter(test_loader)
        while n_done < args.num_samples:
            try:
                batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                batch = next(test_iter)
            b = min(batch["x"].size(0), args.num_samples - n_done)
            x_real = batch["x"][:b].to(device)
            y_real = batch["y"][:b].to(device)
            mask_real = batch["mask"][:b].to(device)
            # Match the BOS-prepending the trainer does (interpolant.pad_sequence).
            _, y_padded, mask_padded = interpolant.pad_sequence(x_real, y_real, mask_real)
            samples = interpolant.cat_cond_sampling(
                model, y_padded, mask_padded, args.num_steps, device,
                return_trace=False,
            )
            xt_chunks.append(samples.xt)
            yt_chunks.append(samples.yt)
            mask_chunks.append(_get_layout_mask(samples))
            n_done += b
    else:
        while n_done < args.num_samples:
            b = min(args.batch_size, args.num_samples - n_done)
            samples = interpolant.sampling(
                model, args.num_steps, b, max_length + 1, device,
                return_trace=False, sampler=args.sampler,
                cfg_weight=args.cfg_weight,
            )
            xt_chunks.append(samples.xt)
            yt_chunks.append(samples.yt)
            mask_chunks.append(_get_layout_mask(samples))
            n_done += b

    # Concatenate along batch and pad to common L for evaluator.
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

    metrics = evaluator.evaluate_samples(xt, yt, mask, batch_size=args.batch_size)
    metrics.update({
        "num_steps": int(args.num_steps),
        "sampler": args.sampler,
        "seed": int(args.seed),
        "use_ema": bool(args.use_ema),
        "cfg_weight": float(args.cfg_weight),
        "cond_mode": args.cond_mode,
    })

    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "metrics_layoutflow.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
