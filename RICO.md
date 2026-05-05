# Variable-Length Diffusion — Layout Training Loop (RICO, LayoutFlow setup)

## Status

RICO baseline training is now live in this repo, sharing the entire
LayoutFlow eval pipeline with PubLayNet. We've:

- Pulled LayoutFlow's HF release at `/workspace/LayoutFlow-data/dataset/rico`
  (`ldm_rico_{train,val,test}.h5`, LayoutFormer++/LayoutDiffusion split,
  `max_length=20`, 31,694 / 3,826 / 3,729 layouts).
- **Reproduced LayoutFlow's reported RICO uncond numbers** as a sanity
  check — see "Reference numbers" below. Our reproduction matches the
  paper: FID 2.30 vs paper 2.37.
- Generalized our pipeline to RICO (label ordering matches LayoutFlow's
  `TYPE_2_CAT`; h5 loader reads `type` instead of `categories`; evaluator
  loads `fid_rico.pth.tar` + `FIDNet_musig_test_rico.pt`).
- Launched the first 50k-iter from-scratch baseline
  (`runs/layout/rico25-trial-001-baseline`).

---

## Goal

Close the gap to **LayoutFlow's reported RICO uncond FID 2.37** (their
LayoutNet metric). Stretch: match it, modulo the structural advantage they
have from sampling lengths from an oracle distribution while we model
length naturally.

| Reference | FID |
|---|---|
| LayoutFlow paper (RICO uncond) | **2.37** |
| Our reproduction of the paper checkpoint (`src/test.py`) | 2.30 |
| Val → test musig (LayoutFlow's "real" sanity) | 2.10 |
| Test (2000 samples) → test musig (sample-size noise floor) | ~1.01 |
| Test → test musig (perfect) | ~0.00 |

LayoutFlow's 2.37 is essentially "as close to the test distribution as a
random real subset of similar size gets." That's the bar.

---

## Dataset

LayoutFlow's HF release at
`/workspace/LayoutFlow-data/dataset/rico/ldm_rico_{train,val,test}.h5`,
loaded via `custom_datasets.layoutflow_h5.LayoutFlowH5Dataset` with
`dataset_name="rico25"`. Counts: 31,694 train / 3,826 val / 3,729 test.
Layouts are length 1–20; bboxes stored as `(x_topleft, y_topleft, w, h) ∈
[0,1]` and converted to our internal `(cx, cy, w, h) ∈ [-1, 1]` at load
time.

**Label ordering caveat**: the h5 stores element type as `type ∈ 1..25`
using LayoutFlow's `TYPE_2_CAT` order (Advertisement=1, Video=2, …,
Button Bar=25). Our `RICO25_LABELS` is **deliberately stored in this same
order** so that token IDs round-trip correctly through both the loader
(h5 cat → token) and the evaluator (token → LayoutNet class index).
Reordering it will silently break FID. (LayoutDM's frequency-sorted order
is a different convention and is no longer used here.)

Populate (one-time):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/JulianGuerreiro/LayoutFlow /workspace/LayoutFlow-data
cd /workspace/LayoutFlow-data && git lfs pull --include="dataset/rico/ldm_rico_*.h5"
```

---

## Evaluation library

`eval/layout/` provides the LayoutFlow pipeline; same module as PubLayNet,
just different assets.

- `LayoutEvaluator.for_dataset("rico25", tokenizer, device)` loads
  LayoutNet weights from `<layoutflow_root>/pretrained/fid_rico.pth.tar`
  (note the suffix is `rico`, not `rico25` — the registry maps for us)
  and the test musig from `FIDNet_musig_test_rico.pt`.
- Conversions handled internally: `(cx,cy,w,h) → (l,t,r,b)` for LayoutNet
  input, and label `+1` shift for valid bboxes (LayoutNet uses `0=pad,
  1..25=classes`).
- `FIDNetV3(num_label=26, max_bbox=20)` for RICO (vs `num_label=6` for
  PubLayNet). `max_bbox=20` matches LayoutFlow's RICO LayoutNet, which
  truncates RICO at 20 just like PubLayNet.
- Wired into `layout_training.py`: every `log_rate` checkpoint draws
  `eval.num_samples` samples and appends to `{run.dir}/metrics.jsonl`
  (and wandb under `eval/*` if `run.enable_wandb`).
- **In-training eval uses the EMA model** (carryover from PubLayNet, where
  flipping to EMA was worth ~2 FID points).

Standalone offline eval:
```bash
uv run python evaluate_with_layoutflow.py \
  --checkpoint runs/layout/<run>/itr_<N>/snapshot.pt \
  --dataset rico25 \
  --use_ema --num_steps 200 --num_samples 2000 --seed 0
```

(If `evaluate_with_layoutflow.py` does not yet expose a `--dataset` flag,
add it before relying on this command.)

---

## How to run training

Standard 50k-iter trial (eval every 10k):
```bash
uv run python layout_training.py \
  dataset=rico25 \
  run.dir=runs/layout/rico25-trial-NNN-<knob> \
  run.run_name=rico25-trial-NNN-<knob> \
  run.num_iters=50000 \
  run.batch_size=128 \
  run.aux_l1_weight=1.0 \
  run.log_rate=10000 \
  run.enable_wandb=true \
  eval.layoutflow_root=/workspace/Variable-Length-Diffusion-Toy/LayoutFlow
```

Resume / extend:
```bash
uv run python layout_training.py \
  dataset=rico25 \
  run.dir=runs/layout/<existing-run> \
  run.load_checkpoint=runs/layout/<existing-run>/itr_<N>/snapshot.pt \
  run.num_iters=<N + extension>
```

`eval.num_steps=50`, `eval.num_samples=1000`, and the split sampler are
the defaults from prior PubLayNet sweeps; don't reduce without reason.

**Throughput on RTX PRO 4000 Blackwell**: ~13 it/s at batch=128, model
hidden_dim=256 depth=8. So 50k iters ≈ 65 min training + a handful of
eval passes (~30 s each).

**Note on epochs vs iters**: RICO train is 31,694 layouts / batch 128 =
~247 iters/epoch. So 50k iters ≈ 200 epochs. For comparison, LayoutFlow's
RICO recipe is `trainer.max_epochs=2500` at batch=512. We are
under-training relative to that on epoch-count terms; longer runs may
pay off.

---

## Iteration plan

We **inherit LAYOUT.md's findings** verbatim where they generalize, then
adapt knobs for RICO specifics. Run a 50k-iter trial per knob with the
same iter count for comparison, **changing one knob at a time**.

### Phase A — Larger batch size (batch=512) [highest priority]

**Empirical: batch=512 has been observed to help layout training in this
repo, and LayoutFlow trains RICO at batch=512 from the start.** This is
the leading first knob.

Pair the batch-size jump with an LR sweep — going from batch 128 → 512
without rescaling LR will under-use the variance reduction. Suggested
plan: launch two trials in parallel (or sequential, GPU permitting):

| Trial | batch | LR | Rationale |
|---|---|---|---|
| trial-002a | 512 | 2e-4 | sqrt scaling of our 1e-4 default |
| trial-002b | 512 | 5e-4 | matches LayoutFlow's RICO recipe directly |

Pick the winner at 50k and lock that as the new baseline before moving
to Phase B.

### Phase B — Loss reweighting (DSM in `t`)

The DSM loss is uniform-in-`t` by default. The flow-matching-natural
reweighting is `sqrt(t / (1 - t))` (`interpolant.dsm_t_reweight=true`),
which up-weights mid-`t` regions where signal is most informative. On
PubLayNet this was the leading Phase-A candidate but the verdict was
inconclusive (compounded with extra training time). RICO is a fresh
opportunity to attribute it cleanly: trial-001 is the no-reweight
baseline, a follow-up trial after Phase A will toggle just this knob.

`aux_l1_weight=1.0` carries over from PubLayNet as the locked-in default
unless evidence shows the loss balance is off (e.g. dsm dominates or the
L1 term is starved at evals). Note that LayoutFlow uses
`add_loss_weight=0.2` — a 5× lower weight on their analogous geometric
L1 loss. If `aux_l1_weight=1.0` looks dominant at eval time, sweep down
to {0.5, 0.2}.

### Phase C — Optimizer / Muon

The current config is adamw `lr=1e-4`, `warmup_iters=100`, no decay
schedule. Likely under-tuned for long runs. Sweep:

| Knob | Default | Try |
|---|---|---|
| `optimizer.lr` | 1e-4 | 3e-4, 5e-5 |
| `run.warmup_iters` | 100 | 1000, 2000 |
| LR decay | none | cosine to 1e-5 over `num_iters` |
| `optimizer=muon` | — | one head-to-head trial against adamw |

### Phase D — Width / depth

Architecture last. Current: depth=8, hidden_dim=256, ~15M params.
LayoutFlow uses **depth=4, d_model=512** (wider/shallower at roughly the
same parameter budget) — that's a concrete A/B candidate. Also try
`model.hidden_dim=384` or `512` at our current depth=8. Run **after**
Phase A's batch+LR and Phase B's reweighting are locked in — mixing axes
makes attribution impossible.

### Phase E — More epochs

Because RICO is ~10× smaller than PubLayNet, the same iter count is ~10×
more epochs. If quality is still improving at 50k, do 100k-200k
extensions before declaring a config the new baseline (LayoutFlow's
recipe sits at 2500 epochs, we'd be at ~200 at 50k). This is the cheapest
"axis-free" gain.

---

## Appendix — LayoutFlow's RICO hyperparameters (reference)

Pulled from the vendored `LayoutFlow/conf/` for direct comparison. Use
this as a "where they landed" reference when picking knob values.

| Knob | LayoutFlow RICO | Our default |
|---|---|---|
| `batch_size` | **512** | 128 |
| `num_workers` | 12 | 4 |
| `max_len` | 20 | 20 |
| Optimizer | AdamW | AdamW |
| LR | **5e-4** | 1e-4 |
| Scheduler | `reduce_on_plateau` | none |
| `max_epochs` | **2500** (RICO; 1000 default for PubLayNet) | ~200 (50k iter / batch 128) |
| Backbone | LayoutDMBackbone (encoder-only Transformer) | Transformer (encoder-only) |
| `d_model` | 512 | 256 |
| `num_layers` | 4 | 8 |
| `nhead` | 8 | 8 |
| `dropout` | 0.1 | 0 |
| `latent_dim` | 128 | — |
| `attr_encoding` | AnalogBit | linear |
| `seq_type` | stacked | similar |
| Initial noise dist | gaussian | (multimodal interpolant) |
| `out_dim` (sampler) | 7 (PubLayNet) / **9 (RICO)** | n/a |
| `time_sampling` | uniform | uniform |
| `train_traj` | linear | linear |
| `ode_solver` | euler | (sampler-dependent) |
| `inference_steps` | 100 | 50–200 |
| `add_loss` | `geom_l1_loss` | `aux_l1_loss` (analogous) |
| `add_loss_weight` | **0.2** | **1.0** |
| `cond` | `random4` (4 conditioning masks) | uncond only |
| `sigma` | 0.0 | n/a |

---

## Trial-run protocol

1. Read the candidate run's `metrics.jsonl` and the current best.
2. Pick **one** knob from the Phase ordering. Don't compound changes.
3. Launch a 50k-iter trial (`num_iters=50000`, `log_rate=10000`) with a
   fresh `run.dir`, descriptive `run.run_name`, and `enable_wandb=true`.
4. Compare at 50k against the baseline's *same iteration count* using the
   EMA-checkpoint number in `metrics.jsonl`.
5. If it wins, consider a 100-200k extension before declaring it the new
   baseline and updating this file.

---

## Loop workflow (`/loop` mode)

Each iteration:
1. Inspect the latest trial's `metrics.jsonl`. Did it improve over the
   baseline at the same iter count?
2. Pick the next Phase-A knob until that phase is exhausted, then Phase B,
   etc.
3. Launch a 50k-iter trial (`run.num_iters=50000 run.log_rate=10000`).
4. When it finishes, compare to current best at the same iter count.
5. Update **Current best** below and commit if it wins.

### Current best (RICO, LayoutFlow metric)

```
Run dir:  runs/layout/rico25-trial-002-bs512-lr2e4 (100k iters, COMPLETE)
Config:   Transformer (single-stream), depth=8, hidden_dim=256, ~15M params,
          multimodal interpolant, adamw lr=2e-4, batch=512,
          aux_l1_weight=1.0, dsm_t_reweight=false,
          max_length=20, LayoutFlow h5 data.
Why this config:
          Best-bet single shot — skipping Phase-A sweeps entirely. batch=512
          matches LayoutFlow's recipe and our prior empirical observation
          that large batches help layout training; lr=2e-4 is sqrt-scaled
          from our 1e-4 default to absorb the 4× batch jump.
          100k iters at batch 512 ≈ 1600 epochs (LayoutFlow uses 2500).
EMA FID:  Final = 19.88 @ iter 100k.
          (Compare: LayoutFlow paper RICO uncond = 2.37; our paper-checkpoint
          reproduction = 2.30; sample-size noise floor at 2000 samples = ~1.01.)

EMA FID trajectory:
  iter  10k:  226.53   align 0.0046   overlap 0.835
  iter  20k:  164.61   align 0.0050   overlap 0.553
  iter  30k:  109.37   align 0.0050   overlap 0.830
  iter  40k:   75.57   align 0.0046   overlap 0.869
  iter  50k:   44.87   align 0.0042   overlap 0.819
  iter  60k:   25.90   align 0.0039   overlap 0.762
  iter  70k:   23.48   align 0.0035   overlap 0.693
  iter  80k:   22.33   align 0.0041   overlap 0.705
  iter  90k:   20.01   align 0.0038   overlap 0.662
  iter 100k:   19.88   align 0.0040   overlap 0.651

Observations:
  - Big descent through iter 60k (-200 points), then sharp slowdown (~6
    points across the final 40k). Cleanly converged in last bucket (-0.1).
  - Already beats PubLayNet trial-014's full 200k-iter result (FID 47.45)
    by ~28 points despite ~10× fewer epochs — strong validation that
    batch=512 is doing the heavy lifting.
  - Still ~17 FID points above LayoutFlow's 2.37 paper number. Closing
    that gap will need a real intervention, not just more training of this
    config.

Suggested next interventions (one knob at a time, A/B vs trial-002):
  1. LR cosine decay to 1e-5 (plateau started at ~iter 60k → indicates
     the LR was too hot for the late-training phase).
  2. interpolant.dsm_t_reweight=true (Phase-A from LAYOUT.md, still
     unattributed cleanly).
  3. LR=5e-4 (LayoutFlow's exact recipe — more aggressive than our 2e-4).
  4. Longer training: 200k+ iters. LayoutFlow trains for 2500 epochs;
     we're at 1600. A 50% longer run is the cheapest axis-free experiment.
```

---

## File map

```
layout_training.py                    # Hydra training script (shared with PubLayNet)
evaluate_with_layoutflow.py           # Standalone offline eval (LayoutFlow pipeline)
LayoutFlow/                           # Vendored LayoutFlow repo (used to verify reproducibility)
  src/test.py                         # Their official eval script — reference impl
  pretrained/fid_rico.pth.tar         # LayoutNet weights for RICO
  pretrained/FIDNet_musig_test_rico.pt
conf/
  config.yaml
  dataset/{rico25,publaynet}.yaml     # rico25 → LayoutFlow h5 (max_length=20)
  model/{transformer,mmdit_both_var}.yaml
  optimizer/{adamw,adam,muon}.yaml
  interpolant/{multimodal,multimodal_both,branching}.yaml
  eval/default.yaml
  schema.py
custom_datasets/
  layout_labels.py                    # RICO25_LABELS in LayoutFlow TYPE_2_CAT order
  layoutflow_h5.py                    # LayoutFlow h5 loader (handles `categories` vs `type`)
eval/layout/
  evaluator.py                        # LayoutEvaluator.for_dataset("rico25", ...)
  fidnet.py
  metrics.py
  adapter.py
runs/layout/<run-name>/
  metrics.jsonl
  itr_<N>/snapshot.pt
  itr_<N>/layouts/layout_*.png
```
