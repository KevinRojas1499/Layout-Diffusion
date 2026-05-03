# Variable-Length Diffusion — Layout Training Loop (PubLayNet-first)

## Status

We have a **strong baseline**: single-stream `Transformer` + `multimodal`
interpolant on PubLayNet, EMA FID **16.72** at iter 190k (base 18.78). This
replaces the earlier `MMDiTBothVar` + `multimodal_both` setup, which had a
structural problem with separate x/y mask schedules. Switching the
interpolant alone unlocked a ~34-point FID drop. The training curve is still
trending downward at 200k iters, so a longer extension run is on the table,
but the baseline is now solid enough that we should **iterate around it**
rather than start over.

The remaining gap is to the real-data floor (FID 6.25). We expect to close
it with **regularization, optimizer/lr tuning, and width**, in that order.

---

## Goal

Unconditional layout generation that beats LayoutDM (Inoue+, CVPR'23) on at
least one of FID and Alignment, ideally both. Tune on PubLayNet first, then
promote winners to Rico-25.

| Dataset | Real-data floor (val→test) | LayoutDM conditional (paper) |
|---|---|---|
| **PubLayNet** (train here) | FID 6.25 / Align 0.000214 / Overlap 0.0032 | FID 4.04±0.08 / Align 0.15±0.00 / Overlap 3.73 |
| Rico (transfer target) | FID 1.85 / Align 0.00109 / Overlap 0.665 | FID 3.03±0.06 / Align 0.36±0.06 / Overlap 57.55 |

Real-data floor numbers come from `eval/layout/real_data_baseline.json`.
LayoutDM only reports *conditional* numbers; unconditional is strictly
harder, so the LayoutDM column is aspirational, not a head-to-head target.

---

## Dataset

LayoutDM's pre-processed splits at
`download/datasets/<name>-max25/processed/{train,val,test}.pt`. Counts match
the paper exactly (PubLayNet 315,757 / 16,619 / 11,142). Loader is
`custom_datasets.layoutdm_processed.LayoutDMDataset`; the YAML configs
`conf/dataset/{rico25,publaynet}.yaml` default to `data_path:
download/datasets`.

Populate from layout-dm's release if missing:
```bash
wget https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip
unzip layoutdm_starter.zip
```

---

## Evaluation library

`eval/layout/` provides FID + Alignment + Overlap, verified bit-identical to
LayoutDM's released code (see `eval/layout/verify_with_layoutdm_code.py`).
Wired into `layout_training.py`: every `log_rate` checkpoint draws
`eval.num_samples` samples and appends to `{run.dir}/metrics.jsonl` (and
wandb under `eval/*` if `run.enable_wandb`).

EMA evaluation matters past ~80-100k iters; in-training eval uses the base
model, so use `evaluate_layout.py --use_ema` offline to read the true floor.

---

## How to run training

Smoke run:
```bash
uv run python layout_training.py dataset=publaynet run=debug \
  run.dir=runs/layout/publaynet-smoke
```

```bash
Standard 50k-iter trial (eval every 10k):
uv run python layout_training.py \
  dataset=publaynet \
  run.dir=runs/layout/publaynet-trial-NNN-<knob> \
  run.run_name=publaynet-trial-NNN-<knob> \
  run.num_iters=50000 \
  run.batch_size=128 \
  run.log_rate=10000 \
  run.enable_wandb=true
```

Resume / extend the current best:
```bash
uv run python layout_training.py \
  dataset=publaynet \
  run.dir=runs/layout/publaynet-trial-006-transformer \
  run.load_checkpoint=runs/layout/publaynet-trial-006-transformer/itr_200000/snapshot.pt \
  run.num_iters=300000
```

Standalone offline eval (with EMA):
```bash
uv run python evaluate_layout.py \
  --checkpoint runs/layout/<run>/itr_<N>/snapshot.pt \
  --num_samples 1000 --num_steps 50 --use_ema
```

---

## Iteration plan

The architecture sweep (depth, MMDiT variants) has already been run and is
not where the headroom is. With a converged Transformer baseline in hand,
the productive directions are below — **change one knob at a time** and
launch a 15k-iter trial against the same iteration count for comparison.

### Phase A — Regularization (L1 on bbox)

Priority. The model jointly predicts symbol logits and 4-D bbox values; the
continuous head is trained with denoising score matching (`dsm_loss`).
Adding an **L1 reconstruction term on clean bbox predictions** typically
sharpens edges and encourages snap-to-grid behavior — both of which directly
attack the Alignment metric and likely help FID as well.

Two flavors to consider (held in reserve in
`project_layout_aux_loss_options.md`):
- **L1 on `x1` clean bbox** (Layout Flow Matching style) — straight L1
  between predicted clean bbox and ground truth, weighted alongside DSM.
- **LayoutDM problem-specific reg** — penalizes degenerate boxes / overlap.

Implementation point: aux losses are added in `layout_training.py` around
the `losses = interpolant.compute_loss(...)` call (line ~325). Add them to
the loss sum behind a small `loss_weights` config so trials can sweep
weights without code churn.

Suggested first sweep: `l1_weight ∈ {0.0, 0.1, 0.5, 1.0}` at the current
Transformer config, 50k iters each. Note that 0.0 is what we have already run.

### Phase B — Optimization (lr, schedule, optimizer)

The current config uses adamw `lr=1e-4`, `warmup_iters=100`, no decay
schedule, batch=128. With a long-running converged baseline this is almost
certainly under-tuned.

| Knob | Default | Try | Notes |
|---|---|---|---|
| `optimizer.lr` | 1e-4 | 3e-4, 5e-5 | Pair higher lr with longer warmup |
| `run.warmup_iters` | 100 | 1000, 2000 | Stability at higher lr / long runs |
| `optimizer=muon` | — | `optimizer=muon optimizer.muon_lr=5e-4` | Worth one trial against adamw |
| `run.batch_size` | 128 | 256 | Larger = lower FID variance, more stable |
| `run.ema_beta` | 0.9999 | 0.99995, 0.99999 | EMA tracks slower; useful for long runs |
| LR decay | none | cosine to 1e-5 over `num_iters` | Likely worth it given the trajectory still moves at 190k |

The trajectory (10k=52.80 → 30k=30.72 → 90k=23.49 → 130k=22.00 → 190k=18.78)
suggests we're not optimizer-limited at the start, but a cosine schedule
into a small final lr would extract more from a long run. Try cosine first, however assume the run will be of 500k
so that we can resume from that checkpoint.

### Phase C — Width

Architecture is the *last* lever, not the first. The Transformer is depth=8
hidden_dim=256 (~15M params). Width is the remaining axis worth sweeping;
deeper has not paid off in prior trials. Try:

| Try | Override | Cost |
|---|---|---|
| Wider | `model.hidden_dim=384` | ~1.5× |
| Wider+ | `model.hidden_dim=512` | ~2.5× |

Run width sweeps **after** Phase A has a winner, so the regularization
weight is locked in. Mixing width and aux-loss changes makes attribution
impossible.

### Phase D — Sampling and EMA-only refinements

Once a model is converged:
- `eval.num_steps=100` (vs 50)
- `eval.num_samples=1000` (already on for the baseline)
- Verify with `--use_ema` for the reported number

We should also implement a better sampler. For instance by using a second order Heunn method or similar.

---

## Trial-run protocol

1. Read the candidate run's `metrics.jsonl` (last 5 lines) and the current
   best (`runs/layout/publaynet-trial-006-transformer/metrics.jsonl`).
2. Pick **one** knob from the Phase ordering. Don't compound changes.
3. Launch a 50k-iter trial with a fresh `run.dir` and `run.run_name`,
   `run.enable_wandb=true`.
4. Compare at 50k against the baseline's *same iteration count*.
5. If it wins, consider a 50-100k extension before declaring it the new
   baseline and updating this file.

A 50k-iter trial is the standard early-signal length; do not down-shift to
5k. The first ~10k iters underrepresent slow-converging changes
(regularization especially).

---

## Loop workflow (`/loop` mode)

Each iteration:
1. Inspect the latest trial's `metrics.jsonl`. Did it improve over the
   baseline at the same iter count?
2. Pick the next Phase-A knob until that phase is exhausted, then Phase B,
   etc.
3. Launch a 50k-iter trial (`run.num_iters=50000 run.log_rate=10000`) with a
   descriptive `run.dir` and `run.run_name`, `run.enable_wandb=true`.
4. When it finishes, compare to current best at the same iter count.
5. Update **Current best** below and commit if it wins.

### Current best (PubLayNet)

```
Run dir: runs/layout/publaynet-trial-008-l1w1.0
Config: Transformer (single-stream), depth=8, hidden_dim=256, ~15M params,
        multimodal interpolant, adamw lr=1e-4, batch=128, 400k iters,
        + Phase A: aux_l1_weight=1.0 (L1 on clean-bbox prediction)
EMA  @ iter 350000: FID 10.51  Align 0.00316  Overlap 0.242  ← best
Base @ iter 290000: FID 10.93  Align 0.00378  Overlap 0.249  ← best
Base @ iter 400000: FID 11.78  Align 0.00394  Overlap 0.264

EMA sweep (1000 samples × 50 steps, all checkpoints from 100k→400k):
  Top-5 by FID:  350k=10.51, 340k=10.52, 390k=10.64, 300k=10.70, 360k=10.76
  Top-5 by Align:330k=0.00312, 300k=0.00313, 350k=0.00316, 200k=0.00331,
                 260k=0.00340
  EMA improves over base by 1-3 FID points consistently past 200k.

Floor gap to real-data FID 6.25: ~1.7×. Below LayoutDM-paper's 4.04
(conditional, easier task) by 6.5 points but in the same ballpark.

Phase A status: WINNER locked in at l1_weight=1.0.
  - Sweep results @ 50k: l1=0.1 (28.19 @40k, killed) > l1=0.5 (22.82) >
    l1=1.0 (20.02). Monotonic improvement up the weight axis.
  - The l1=1.0 trend held at long training: 200k FID 13.24 vs baseline
    trial-006's 19.59. ~5× speedup over baseline trajectory at matched iters.

Notes: Curve plateaus around iter 300-400k (EMA FID 10.5±0.5 across the
range). Promote to Rico (PubLayNet now well below the FID 12 threshold).
Phase B (optimization knobs) is the natural next direction.
```

### Previous baseline (superseded — kept for context)

```
Run dir: runs/layout/publaynet-trial-006-transformer
Config: same as above, but aux_l1_weight=0.0 (no L1 reg).
EMA  @ iter 190000: FID 16.72  Align 0.0051  Overlap 0.278
Base @ iter 200000: FID 19.59  Align 0.0052  Overlap 0.311
```

### Current best (Rico — transfer target)

_(Promote a tuned PubLayNet config here once PubLayNet is at or below FID 12.)_

```
Run dir: (not started)
```

---

## File map

```
layout_training.py                    # Hydra training script (dispatch on dataset.name)
evaluate_layout.py                    # Standalone offline eval CLI (--use_ema)
conf/
  config.yaml                         # Composes run/dataset/model/optimizer/interpolant/eval
  dataset/{rico25,publaynet}.yaml     # Default to layout-dm processed .pt files
  model/{transformer,mmdit_both_var}.yaml
  optimizer/{adamw,adam,muon}.yaml
  interpolant/{multimodal,multimodal_both,branching}.yaml
  eval/default.yaml                   # eval cadence + sampling settings
  schema.py                           # Hydra dataclass schemas
custom_datasets/layoutdm_processed.py
eval/layout/
  evaluator.py                        # LayoutEvaluator.for_dataset(...)
  fidnet.py                           # Vendored FIDNetV3
  metrics.py                          # compute_alignment / compute_overlap / frechet_distance
  adapter.py                          # bbox/label conversion to FIDNet's format
  sanity_check.py                     # Real-data baseline (run once)
  verify_with_layoutdm_code.py        # Side-by-side w/ layout-dm's released code
  real_data_baseline.json             # Reference floor
runs/layout/<run-name>/
  metrics.jsonl                       # one JSON per eval (FID, Align, Overlap, ...)
  itr_<N>/snapshot.pt                 # model + ema + optimizer + scheduler
  itr_<N>/layouts/layout_*.png        # qualitative samples
```
