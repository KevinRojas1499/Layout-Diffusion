# Parenthesis Equation Generation

## Key Finding

Running Stage 1 (pure DSM, no constraint loss) for 100k+ iterations achieves the same accuracy (~55% at 50 steps, ~80%+ at 400 steps) as the full 3-stage constraint pipeline. No constraint loss is needed.

**Critical:** `--ema_beta 0.999` (faster EMA tracking). The default 0.9999 causes the EMA to lag at evaluation, producing worse samples.

**Dataset:** Use `train_l5.jsonl` (160k samples) for training, `test_l5.jsonl` (20k) for evaluation. Do not use `equations_l5.jsonl` (50k, no train/test split).

**Metrics (500 samples):**
- **accuracy** — fraction of valid samples with |L−R| < 0.5
- **mean_abs_error (MAE)** — mean |L−R| over valid samples
- **validity** — fraction of samples that parse successfully

**Note:** `eval_steps=50` used during training; `eval_steps=400` gives ~25pp higher accuracy.

---

## Current Best

| Checkpoint | 50-step acc | validity | Notes |
|---|---|---|---|
| stage1-depth8-100k/itr_135000 | **55.4%** | 78.5% | Trained on equations_l5.jsonl (50k) — superseded |
| stage1-depth8-160k (in progress) | — | — | Retraining on train_l5.jsonl (160k) |

Post-processing (`eval_with_scale_correction.py`): balance-project onto L=R, then scale ×3.51 → **~99.5% post-proc accuracy** on valid samples.

---

## Commands

### Start stage 1 from scratch (canonical)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/train_l5.jsonl \
  --dir runs/parenthesis/stage1-depth8-160k \
  --depth 8 \
  --lr 1e-4 --lr_min 1e-6 --lr_schedule cosine \
  --warmup_iters 500 \
  --num_iters 90000 \
  --batch_size 128 \
  --ema_beta 0.999 \
  --eval_samples 200 --eval_steps 50 \
  --log_rate 5000
```

### Resume (extend training)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/train_l5.jsonl \
  --dir runs/parenthesis/stage1-depth8-160k \
  --load_checkpoint runs/parenthesis/stage1-depth8-160k/itr_<N>/snapshot.pt \
  --depth 8 \
  --lr 1e-4 --lr_min 1e-6 --lr_schedule cosine \
  --warmup_iters 500 \
  --num_iters 300000 \
  --batch_size 128 \
  --ema_beta 0.999 \
  --eval_samples 200 --eval_steps 50 \
  --log_rate 5000
```

### Post-process best checkpoint

```bash
uv run python eval_with_scale_correction.py \
  --checkpoint runs/parenthesis/stage1-depth8-160k/itr_<N>/snapshot.pt \
  --eval_steps 400 --eval_samples 500 \
  --scale_factor 3.51 \
  --out corrected_samples.jsonl
```
