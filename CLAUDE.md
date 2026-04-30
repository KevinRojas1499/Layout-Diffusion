# Variable-Length Diffusion — Parenthesis Training Loop

## Goal

Improve the accuracy of the model trained by `parenthesis_training.py` on the parenthesized arithmetic equations dataset. Accuracy is measured by:

1. **`accuracy`** — fraction of valid generated equations where |left_value − right_value| < 0.5 (higher is better).
2. **`mean_abs_error`** — mean absolute difference between left and right sides of valid equations (lower is better).
3. **`invalid`** — number of samples that cannot be parsed as equations at all (lower is better).

These metrics are written after every evaluation to `{--dir}/metrics.jsonl` (one JSON object per line).

---

## How to run training

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/<run-name> \
  --num_iters 10000 \
  --lr 1e-4 \
  --batch_size 128 \
  --log_rate 500 \
  --eval_samples 200 \
  --eval_steps 50
```

To resume from a checkpoint:
```bash
python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/<run-name> \
  --load_checkpoint runs/parenthesis/<run-name>/itr_5000/snapshot.pt \
  --num_iters 20000 ...
```

To evaluate generated samples offline:
```bash
python evaluate_equations_parenthesis.py \
  --input runs/parenthesis/<run-name>/itr_<N>/samples.jsonl \
  --gt-file data/parenthesis/equations_l5.jsonl
```

---

## Dataset

- **Path**: `data/parenthesis/equations_l5.jsonl` (default; also `l1`, `l3`, `l8`, `l10` variants)
- **Format**: each line is `{"equation": "...", "numbers": [...], "symbols": [...], "result": ..., "length": ...}`
- **Symbols**: categorical tokens — `(`, `)`, `+`, `-`, `*`, `=`, `.`  (`.` terminates the equation)
- **Numbers**: continuous values aligned to operator/anchor positions in `symbols`
- The model must jointly generate a correct symbol sequence **and** correct numeric values

---

## Model and architecture

- **Model**: `MMDiTQM9` in `models/mmdit_qm9.py`
- **Key hyperparameters** (set in `parenthesis_training.py`):
  - `depth=4` — number of joint attention blocks
  - `dim_modalities=[256, 256]` — embedding dim for symbols and positions
  - `dim_joint_attn=256` — joint cross-attention dimension
  - `symbols_depth=4`, `positions_depth=4` — per-modality transformer depths
- The model receives noisy symbols + noisy positions at time `t` and predicts the clean versions

---

## Training loop knobs to tune

These are the most impactful levers, roughly in order:

| Parameter | Current default | Notes |
|-----------|----------------|-------|
| `--lr` | 1e-4 | Try 3e-4, 5e-5; use warmup |
| `--batch_size` | 128 | Larger batches stabilize gradient estimates |
| `--warmup_iters` | 100 | Try 500–1000 for stability |
| `--num_iters` | 5000 | More iters = more signal |
| `--interpolant` | multimodal | `autoregressive` may help symbol ordering |
| `--ema_beta` | 0.9999 | Try 0.999 for faster EMA tracking |
| `--eval_steps` | 20 | More steps at eval = better quality samples |

Architecture changes (in `models/mmdit_qm9.py`):
- Increase `depth` to 6 or 8 for more model capacity
- Increase `dim_modalities` to `[512, 512]` for wider network

---

## Loop workflow

When running in `/loop` mode, each iteration should:

1. **Read metrics**: Load `runs/parenthesis/<latest-run>/metrics.jsonl` and inspect the accuracy trend.
2. **Diagnose**: Look at the last few metric entries. Is accuracy still improving, plateaued, or degraded?
3. **Decide**: If accuracy is improving → continue or increase iters. If plateaued → change a hyperparameter (lr, depth, batch size). If invalid samples are high → the symbolic structure is wrong; try `--interpolant autoregressive`.
4. **Run a short trial**: Launch a 2000-iter run with the new config and compare metrics to the baseline.
5. **Commit the winner**: If the new config improves accuracy, update this file with the new best config.

### Current best known config

_(Update this section each loop iteration with the best run found so far.)_

```
Run dir: runs/parenthesis/baseline
Accuracy: unknown (not yet trained)
Mean abs error: unknown
Config: lr=1e-4, batch_size=128, depth=4, dim=256, interpolant=multimodal
```

---

## File structure

```
parenthesis_training.py     # Training script (hardcoded to parenthesis dataset + DiT)
evaluate_equations_parenthesis.py  # Standalone evaluation CLI
data/parenthesis/           # Dataset files (l1, l3, l5, l8, l10)
runs/parenthesis/           # Output directory for runs
  <run-name>/
    metrics.jsonl           # One JSON per eval checkpoint: accuracy, mean_abs_error, ...
    itr_<N>/
      snapshot.pt           # Model + optimizer checkpoint
      samples.jsonl         # Generated samples at this checkpoint
models/mmdit_qm9.py         # MMDiTQM9 architecture
multimodal_interpolant.py   # Forward/backward diffusion process
```
