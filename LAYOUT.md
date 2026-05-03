# Variable-Length Diffusion — Layout Training Loop (PubLayNet, LayoutFlow setup)

## Status

We've switched to **LayoutFlow's evaluation pipeline** as our north star:
LayoutDiffusion's `LayoutNet` feature extractor + LayoutFlow's precomputed
PubLayNet test musig. Numbers are now directly comparable to LayoutFlow's
paper. The training data is also their h5 release (LayoutFormer++ /
LayoutDiffusion split, `max_length=20`).

The previous baseline (trial-008, trained on Inoue's split with
`max_length=25`) scored **FID 41.2** under this metric off-the-shelf, and
**FID 43** after a 10k-iter finetune on the new data — confirming a fresh
training run on the matched data is needed before iterating on knobs.
We're starting that baseline next.

---

## Goal

Close the gap to **LayoutFlow's reported PubLayNet uncond FID 8.87** (their
LayoutNet metric). Stretch: match it, modulo the structural advantage they
have from sampling lengths from an oracle distribution while we model
length naturally.

| Reference | FID |
|---|---|
| LayoutFlow (paper, uncond) | **8.87** |
| Real train data → test musig (sanity) | ~10.77 |
| Val → test musig (sanity) | 8.16 |
| Sample-size noise floor at 2000 samples | ~1.86 |
| Test → test musig (perfect) | ~0.00 |

LayoutFlow's 8.87 is essentially "as close to the test distribution as a
random real subset gets." That's the bar.

---

## Dataset

LayoutFlow's HF release at
`/workspace/LayoutFlow-data/dataset/publaynet/publaynet_{train,val,test}.h5`,
loaded via `custom_datasets.layoutflow_h5.LayoutFlowH5Dataset`. Counts:
311,397 train / 16,390 val / 10,998 test. Layouts are length 1–20; bboxes
are stored as `(x_topleft, y_topleft, w, h) ∈ [0,1]` and converted to our
internal `(cx, cy, w, h) ∈ [-1, 1]` at load time.

Populate (one-time):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/JulianGuerreiro/LayoutFlow /workspace/LayoutFlow-data
cd /workspace/LayoutFlow-data && git lfs pull --include="dataset/publaynet/publaynet_train.h5,dataset/publaynet/publaynet_val.h5,dataset/publaynet/publaynet_test.h5"
```

---

## Evaluation library

`eval/layout/` provides the LayoutFlow pipeline only:
- `LayoutEvaluator.for_dataset("publaynet", tokenizer, device)` loads
  LayoutNet weights from `<layoutflow_root>/pretrained/fid_publaynet.pth.tar`
  and the test musig from `FIDNet_musig_test_publaynet.pt`.
- Conversions handled internally: `(cx,cy,w,h) → (l,t,r,b)` for LayoutNet
  input, and label `+1` shift for valid bboxes (LayoutNet uses `0=pad,
  1..5=classes`).
- Wired into `layout_training.py`: every `log_rate` checkpoint draws
  `eval.num_samples` samples and appends to `{run.dir}/metrics.jsonl`
  (and wandb under `eval/*` if `run.enable_wandb`).
- **In-training eval uses the EMA model** (flipped 2026-05-03; previously
  used base, which biased the metric ~2 points worse).

Standalone offline eval:
```bash
uv run python evaluate_with_layoutflow.py \
  --checkpoint runs/layout/<run>/itr_<N>/snapshot.pt \
  --use_ema --num_steps 200 --num_samples 2000 --seed 0
```

---

## How to run training

Standard 50k-iter trial (eval every 10k):
```bash
uv run python layout_training.py \
  dataset=publaynet \
  run.dir=runs/layout/publaynet-trial-NNN-<knob> \
  run.run_name=publaynet-trial-NNN-<knob> \
  run.num_iters=50000 \
  run.batch_size=128 \
  run.aux_l1_weight=1.0 \
  run.log_rate=10000 \
  run.enable_wandb=true
```

Resume / extend:
```bash
uv run python layout_training.py \
  dataset=publaynet \
  run.dir=runs/layout/<existing-run> \
  run.load_checkpoint=runs/layout/<existing-run>/itr_<N>/snapshot.pt \
  run.num_iters=<N + extension>
```

`eval.num_steps=50`, `eval.num_samples=1000`, and the split sampler are
the defaults from prior sweeps; don't reduce without reason.

---

## Iteration plan

Once the fresh baseline lands, the productive directions are below in
priority order. **Change one knob at a time** and launch a 50k-iter trial
against the same iter count for comparison.

### Phase A — Loss reweighting

The DSM loss is currently uniform-in-`t`. The flow-matching-natural
reweighting is `sqrt(t / (1 - t))`, applied to the per-`t` DSM
contribution. This is the FM analogue of Min-SNR on diffusion; it tends
to up-weight the harder mid-`t` regime where the signal is most
informative.

`aux_l1_weight=1.0` carries over from the prior baseline as the locked-in
default. If after Phase A the loss balance looks off (e.g. dsm dominates
or the L1 term is starved), retune `aux_l1_weight` before moving to
Phase B — but only if there's clear evidence; otherwise keep it at 1.0.

### Phase B — Larger batch size

LayoutFlow trains at batch 512. Diffusion gradients are notoriously
noisy, and a 4× batch increase is one of the cleanest variance-reduction
levers. Try `run.batch_size=512` paired with appropriate LR scaling.

### Phase C — Optimizer / Muon

The current config is adamw `lr=1e-4`, `warmup_iters=100`, no decay
schedule. Likely under-tuned for long runs. Sweep:

| Knob | Default | Try |
|---|---|---|
| `optimizer.lr` | 1e-4 | 3e-4, 5e-5 |
| `run.warmup_iters` | 100 | 1000, 2000 |
| LR decay | none | cosine to 1e-5 over `num_iters` |
| `optimizer=muon` | — | one head-to-head trial against adamw |

### Phase D — Width

Architecture last. Current: depth=8, hidden_dim=256, ~15M params. Try
`model.hidden_dim=384` or `512`. Run **after** Phase A's reweighting and
Phase C's optimizer are locked in — mixing axes makes attribution
impossible.

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

### Current best (PubLayNet, LayoutFlow metric)

```
Run dir:  (not yet — fresh baseline pending)
Config:   Transformer (single-stream), depth=8, hidden_dim=256, ~15M params,
          multimodal interpolant, adamw lr=1e-4, batch=128,
          aux_l1_weight=1.0, max_length=20, LayoutFlow h5 data.
EMA FID:  unknown
```

### Stale reference (different metric / dataset, kept for context)

```
trial-008-l1w1.0 (Inoue split, max_length=25, layout-dm FIDNetV3 metric):
  EMA @ 350k: FID 10.51  Align 0.00316  Overlap 0.242
  Same checkpoint scored under LayoutFlow metric: FID 41.2 (at split/200, seed=0).
```

---

## File map

```
layout_training.py                    # Hydra training script
evaluate_with_layoutflow.py           # Standalone offline eval (LayoutFlow pipeline)
conf/
  config.yaml                         # Composes run/dataset/model/optimizer/interpolant/eval
  dataset/{rico25,publaynet}.yaml     # publaynet → LayoutFlow h5; rico25 still references download/
  model/{transformer,mmdit_both_var}.yaml
  optimizer/{adamw,adam,muon}.yaml
  interpolant/{multimodal,multimodal_both,branching}.yaml
  eval/default.yaml                   # eval cadence + sampling settings
  schema.py                           # Hydra dataclass schemas
custom_datasets/
  layout_labels.py                    # Shared label tables + tokenizer
  layoutflow_h5.py                    # LayoutFlow h5 dataset loader
eval/layout/
  evaluator.py                        # LayoutEvaluator.for_dataset(...)
  fidnet.py                           # FIDNetV3 architecture (hosts LayoutNet weights)
  metrics.py                          # compute_alignment / compute_overlap / frechet_*
  adapter.py                          # bbox/label conversion
runs/layout/<run-name>/
  metrics.jsonl                       # one JSON per eval (FID, Align, Overlap, ...)
  itr_<N>/snapshot.pt                 # model + ema + optimizer + scheduler
  itr_<N>/layouts/layout_*.png        # qualitative samples
```
