# Parenthesis Equation Generation — Research Summary

## Problem Statement

Train a multimodal diffusion model (`MMDiTQM9`) to jointly generate valid, balanced parenthesized arithmetic equations. "Balanced" means the left-hand side evaluates to the same value as the right-hand side: |L − R| < 0.5.

**Dataset (`equations_l5.jsonl`):** 50,000 equations with numbers drawn from roughly [−5, 5]. Structure: `(a+b−c)=(d−e)+f.` — parentheses, binary +/−, and an `=` separator. Numbers and symbols are stored in parallel arrays (one numeric value per symbol position).

**Metrics (evaluated on 500 generated samples):**
- **accuracy** — fraction of valid samples with |L−R| < 0.5 (higher is better)
- **mean_abs_error (MAE)** — mean |L−R| over valid samples (lower is better)
- **validity** — fraction of samples that parse without structural errors (higher is better, target ≥ 75%)

**Evaluation note:** `eval_steps=50` is used during training for speed; `eval_steps=400` is used for honest reporting. At 400 steps, accuracy is systematically ~25 pp higher than at 50 steps due to inference scaling (more denoising steps produce more precise outputs). All "best" results quoted here use 400 steps unless explicitly noted.

---

## Architecture

`MMDiTQM9` in `models/mmdit_qm9.py`: a joint multimodal DiT operating on two streams — discrete symbol tokens and continuous numeric values — linked by cross-attention blocks.

**Final config:**
- `depth=8` joint attention blocks
- `hidden_dim=256` for both modalities
- `ema_beta=0.999` (faster EMA tracking vs default 0.9999)
- Multimodal stochastic interpolant (`--interpolant multimodal`)

**Training entry point:** `parenthesis_training.py`  
**Evaluation:** `evaluate_equations_parenthesis.py`

---

## The Training Pipeline That Works

Getting to the best result required a three-stage pipeline. Each stage builds on the previous checkpoint.

### Stage 1 — Symbolic foundation (≈ 10k iters)
Train from scratch to learn valid equation structure. No constraint on numeric balance.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-baseline \
  --num_iters 10000 --lr 1e-4 --lr_min 1e-6 --lr_schedule cosine \
  --warmup_iters 500 --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Result:** ~15% accuracy, 75% validity. The model learns to generate syntactically correct equation structures (matching parens, valid operator placement) but has no numeric precision.

### Stage 2 — Constraint training with w=1.0 (≈ 20k iters across sub-stages)
Finetune with a differentiable equation-balance loss: `(L−R)²`, weighted by `constraint_weight`.

```bash
# Sub-stage 2a: constraint-loss with cosine LR
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-constraint \
  --finetune_checkpoint runs/parenthesis/depth8-baseline/itr_10000/snapshot.pt \
  --num_iters 10000 --lr 3e-5 --lr_min 1e-5 --lr_schedule cosine \
  --warmup_iters 500 --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 1.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Result:** Peaks at ~49.7% then oscillates. The cosine LR weakens the constraint gradient late in training; the model partially forgets.

```bash
# Sub-stage 2b: stabilize with CONSTANT LR (critical!)
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-stabilize \
  --finetune_checkpoint runs/parenthesis/depth8-constraint/itr_5000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 1.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Result:** Plateaus at 50.0% / 76% validity / MAE ≈ 1.01. This is the w=1.0 ceiling — three independent stabilize runs all peaked at exactly 50.0% at iter 2000.

> **Key finding:** Constant LR ≥ 1e-5 is mandatory. The constraint requires continuous gradient pressure to stay converged. LR below 1e-5 causes MAE to revert toward 2.5 (baseline level).

### Stage 3 — Boost with w=2.0 (5k iters)
Double the constraint weight, keeping constant LR.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-constraint-w2 \
  --finetune_checkpoint runs/parenthesis/depth8-stabilize2/itr_2000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Result (50-step):** 47.3% → 52.8% → **55.6%** → 54.3% → 52.1%. Peak at iter 3000.  
**Result (400-step, best checkpoint):** **80.5% accuracy / 91.2% validity / MAE 0.494**

This is the current best honest result for l5-scale equation generation.

---

## Inference Scaling

More denoising steps at evaluation time improve quality dramatically — a key finding.

| eval_steps | accuracy | validity | MAE |
|-----------|---------|---------|-----|
| 50 | 55.6% | 73.0% | 0.955 |
| 100 | ~62% | ~83% | ~0.85 |
| 200 | ~69% | ~87% | ~0.72 |
| 400 | **80.5%** | **91.2%** | **0.494** |
| 800 | 81.4% | 92.8% | 0.483 |

Plateau is confirmed at 400–800 steps. Use `eval_steps=400` for reporting.  
**50-step accuracy is NOT a reliable proxy for 400-step performance** — some checkpoints with lower 50-step accuracy give higher 400-step accuracy.

---

## The Scale Shrinkage Problem

**This is the central unresolved issue.** The model generates valid, balanced equations — but at roughly 1/3 the numerical scale of the training data.

| | |x| > 1 | |x| > 2 | |x| > 3 | per-sample max_abs |
|---|---|---|---|---|
| **l5 training data** | 78.8% | 58.2% | 37.7% | mean ≈ 4.3 |
| **Best generated (400-step)** | 10.9% | 4.6% | 0.0% | mean ≈ 1.2 |

**Example generated (correct):**
```
-1.41+(1.58−0.37) = (0.05−0.31+0.25−0.28+0.18)    |L−R| = 0.097
 (0.11+0.34) = 0.23−0.27+(0.12+0.48)+0.05           |L−R| = 0.157
−0.22+(0.23+0.44+0.35) = 0.27+(0.51−0.19)+(0.25+0.18)  |L−R| = 0.206
```

**Example training data:**
```
(−2.25−2.77)+2.36+1.77+3.92 = 3.03
−2.9−2.33+(4.37+1.48) = (1.09−3.29+2.29+0.53)
4.78+0.33−3.74+(1.6+4.47) = (−3.35+0.28+1.07)+(4.64+4.29+0.51)
```

**Root cause:** The constraint loss `(L−R)²` has a trivial solution at **x=0** for all positions. Both L and R equal 0 when all numbers are zero, so the gradient pushes predictions toward small magnitudes. The DSM (score-matching) loss pulls back toward l5 scale, but at weight 2.0 the constraint wins for borderline numbers. The model reaches a compromise: generate small-but-nonzero equations that satisfy the |L−R| < 0.5 threshold easily.

The 80.5% accuracy is real — those equations genuinely balance — but they don't resemble l5-scale data.

---

## What Was Tried and Why It Failed

### Higher constraint weights (w=3.0, w=2.0 × repeat)
Starting from the w=2.0 peak (55.6%), applying w=3.0 or repeating w=2.0 gave 54.0–54.8% — worse than the starting point. The model has already converged to the local optimum for the given starting weights; more gradient force from the same position doesn't escape it.

### Cosine LR from baseline with w=2.0
Applying w=2.0 directly from the symbolic baseline with a cosine LR schedule (skipping the w=1.0 pipeline) caused MAE to oscillate between 1.0 and 1.2 without ever converging to 0.955. The constant LR in Stage 3 is essential — it provides steady constraint pressure without the weakening phase of cosine decay.

### l1→l5 curriculum (inflated metric, not l5-appropriate)
Training first on the l1 dataset (numbers in [−1, 1]) with w=2.0, then finetuning on l5, boosted the raw accuracy metric to 85–88% at 400 steps. However, the generated numbers remained at l1 scale (mean max_abs ≈ 0.3–0.4), not l5 scale. At comparable number scale (phase 2 iter 2000, mean max_abs ≈ 1.0), 400-step accuracy was 78.1% — **worse** than the 80.5% honest result. The curriculum's accuracy gain is an artifact of scale shrinkage, not genuine l5 improvement.

### Mixed l1+l5 training
Training phase 2 on a 50/50 mixed dataset caused mode collapse: the model generated only l1-scale equations (the easier distribution to satisfy the constraint). Accuracy metric inflated to 93%, but all generated numbers had max_abs < 1.

### Scale-normalized constraint
Modifying the constraint loss to divide by prediction magnitude (`residual / x0_orig.abs().sum(dim=1)`) caused accuracy to drop to 19–31%. The bug: the denominator included all positions (including padding), making the effective constraint 25× weaker than needed for l5-scale numbers.

### Proxy constraint (detach residual)
`(residual.detach() * residual).mean()` removes the x_i self-coupling from the Hessian. However, Adam uses only first-order gradients, and the first-order gradient is numerically identical to the standard `residual.pow(2).mean()`. Accuracy declined 47.3%→47.4%→45.9% — no improvement. The proxy idea is sound in theory but irrelevant to Adam-based training.

### Hinge constraint loss — TRIED AND FAILED (2026-05)
`max(0, |L-R| - 0.5)²` (zero gradient when |L-R| < 0.5). Starting from depth8-constraint-w2/itr_3000 (55.6%, scale avg_max_abs 1.575):

- ~55% of samples are already balanced at the starting checkpoint → hinge gives zero gradient for them immediately.
- The unbalanced 45% still get gradient, but this creates an asymmetry: balanced samples receive no constraint signal and are free to drift in scale driven purely by DSM.
- 50-step trajectory (2500 iters): 50.3%→42.6%→44.4%→44.5%→43.8%→38.5%
- Scale trend: 1.575→1.595→1.689→1.776→1.789→1.826 (slow growth, ~+0.1/1000 iters)
- **400-step eval at itr_2500: 63.8% accuracy / 89.4% validity / MAE 0.586 / avg_max_abs 1.757**

**Verdict:** Scale grew by only 0.18 units (1.575→1.757) over 2500 iters while accuracy fell 16.7pp at 400 steps. The hinge weakens the constraint (no gradient for 55% of samples), causing the model to forget balance over time. Not a viable path.

### Scale hinge (explicit scale reward as hinge) — sw=1.0 too strong
Added `scale_hinge_target=2.5, scale_weight=1.0` to the constraint loss. Scale jumped 1.575→2.9 in 1000 iters but accuracy collapsed 55.6%→33.9%. The scale hinge was as strong as the entire balance loss, dominating the gradient and destroying the constraint equilibrium. Would need sw≈0.05–0.1 for balance to remain dominant.

### Threshold curriculum — TRIED AND FAILED (2026-05)

Hinge threshold decaying from 2.0→0.5 over 5k iters, starting from depth8-constraint-w2/itr_3000 (55.6%, scale 1.575). At threshold=2.0, the hinge gives zero gradient for all samples (since MAE ~0.955 < 2.0), so DSM dominates and scale is expected to grow. As threshold tightens, balance precision is gradually re-enforced.

- Scale grew 1.575 → 2.029 (peak at iter 2000, threshold=1.40)
- As threshold tightened below 1.0, constraint reasserted and scale declined to 1.854 at iter 5000
- 50-step accuracy: 43.5%→31.5%→34.8%→32.7%→36.7%→32.2%→38.5%→37.7%→40.2%→39.3% (floored ~32–44%)
- **400-step: 50.3% (itr_2000/scale 2.145) → 61.1% (itr_5000/scale 1.854)**

**Verdict:** The best 400-step result (61.1% at scale 1.854) is **dominated by scale hinge sw=0.05** (70.4% at scale 1.895): nearly identical scale but 9.3pp worse accuracy. The loose-threshold phase does grow scale temporarily, but the subsequent tightening phase pulls it back. The curriculum cannot maintain the scale advantage gained during the loose phase.

### batch_size=256
GPU OOM. The 7.6 GiB GPU is fully utilized at batch=128. Gradient accumulation (2× steps, batch=128) is feasible but was not tested.

---

## Scale-Accuracy Tradeoff: Fundamental Conclusion

Every attempt to improve the generated number scale while maintaining ≥80% accuracy has failed. The table below summarizes the Pareto frontier as of 2026-05-01:

| Approach | 400-step accuracy | avg_max_abs (400-step) | Notes |
|---|---|---|---|
| Standard (L-R)², w=2.0 | **80.5%** | ~1.3 | Best accuracy; scale ≈ 1/3 of l5 data |
| Scale hinge sw=0.05 | 70.4% | 1.895 | -10.1pp accuracy for +0.6 scale; new equilibrium at 2.056 (50-step) |
| Threshold curriculum itr_5000 | 61.1% | 1.854 | Dominated by sw=0.05: -9.3pp accuracy at same scale |
| Balance hinge, w=2.0 | 63.8% | 1.76 | Dominated by sw=0.05: less scale AND less accuracy |
| Scale hinge sw=1.0 | ~34% | 2.9 | Catastrophic accuracy collapse |
| Constraint head (per-step) | 50.3% | 1.03 | Head collapses scale worse than standard; 30pp below baseline |
| l5 training data | — | **~4.3** | Target scale |

**Root cause (fundamental):** `(L-R)²` has its global minimum at x=0 for all positions (L=R=0 trivially). Any constraint of the form f(L-R) with f minimized at 0 shares this pathology. The model equilibrates at small-magnitude numbers where |L-R| < 0.5 is easy. Removing the gradient inside the success zone (hinge) weakens the constraint signal overall, causing forgetting. Adding an explicit scale reward (scale hinge) at sufficient strength to move scale meaningfully also destroys the balance constraint.

The 80.5% result is honest and genuinely correct, but the equations don't resemble l5-scale data.

---

## What Should Be Tried Next (Scale-Preserving Approaches)

All direct loss-function approaches have now been tried. Remaining options are more complex implementations.

### 1. Scale-reward hinge (very weak, sw ≈ 0.05–0.1) — DONE (sw=0.05 is the best Pareto point)

Already implemented and tested. Scale hinge sw=0.05 is the best Pareto alternative to the standard constraint: 70.4% accuracy at scale 1.895, better than all other scale-accuracy tradeoff attempts.

### 2. Relative constraint — `((L-R) / sigma_batch)²`

Replace the absolute (L-R)² with a residual normalized by the batch mean absolute value of active numbers. This makes the constraint penalty higher for small-scale equations (floor kicks in when mean|x|<0.5) and lower for large-scale equations. The gradient no longer inherently favors x=0 over l5-scale solutions.

**Key distinction from `--use_normalized` (already tried):** The prior attempt used `.sum()` over active positions divided by a floor of `normalized_scale_target=5.0`, equivalent to a per-mean floor of ~0.83. This version uses `.mean()` directly with floor=0.5 — stronger anti-shrinkage at small scales.

**Implementation:** Modify `make_constraint_loss_fn` to compute `sigma = (x0_orig.abs() * active_mask).sum(dim=1) / num_active.clamp(min=1)`, detach it, floor at 0.5, and divide residual by sigma. Use `--constraint_weight 2.0` starting from depth8-constraint-w2/itr_3000.

**Risk:** May converge to same equilibrium as prior normalized attempts (~1.26 scale). The fundamental issue (trivial minimum at x=0) is unchanged; only the gradient magnitude near x=0 is affected.

### 3. Curriculum on the constraint threshold — TRIED AND FAILED (2026-05)

Scale grew 1.575→2.029 during loose phase, but tightening to threshold=0.5 pulled scale back to 1.854. Best 400-step result: 61.1% at scale 1.854 — dominated by scale hinge sw=0.05 (70.4% at scale 1.895). Not a viable path.

### 4. Separate numeric prediction head — TRIED AND FAILED (2026-05)

Implemented as `ConstraintHead` (3→64→64→1 MLP) + `HeadWrappedModel` (applies head correction at each denoising step). Backbone trains only on DSM (x0_orig detached before head); head trains only on constraint loss (`(sum coeff_i*(x_i+delta_i))²`).

**Results (depth8-constraint-head, 5k iters from depth8-baseline):**

| Checkpoint | Mode | Steps | Accuracy | MAE | Scale avg_max |
|---|---|---|---|---|---|
| itr_4500 | Backbone only | 400 | 18.1% | 2.187 | **2.902** |
| itr_4500 | Per-step head | 400 | 44.7% | 0.829 | 1.127 |
| itr_5000 | Backbone only | 400 | 15.6% | 2.180 | **2.890** |
| itr_5000 | Per-step head | 400 | **50.3%** | 0.768 | 1.032 |

**Why it failed:** The head is trained to minimize `(sum coeff_i*(x_i+delta_i))²`, which has the same global minimum at x=0 as the main constraint. Backbone scale IS preserved (~2.90) but the head, applied 400× during inference, collapses scale from 2.9 → 1.03 — worse than standard w=2.0 (~1.3). Best accuracy 50.3%, 30pp below the 80.5% baseline. The head learns to satisfy the constraint by reducing magnitudes, compounding across 400 correction steps.

### 5. Two-optimizer setup

Run two Adam optimizers — one for the full model at LR=1e-5 (DSM), one for constraint gradient only at LR=1e-3. Disproportionately amplifies constraint convergence without destabilizing symbolic structure.

---

## Recommended Next Experiment

All straightforward scale-preserving approaches have been tried (threshold curriculum, scale hinge, balance hinge, constraint head, l1→l5 curriculum). The next untested implementation is the **relative constraint** `((L-R) / mean|x_active|)²` with a floor at 0.5. This is distinct from `--use_normalized` (which used `.sum()` and a higher floor) and may prevent scale shrinkage by making the penalty stronger at small scales.

**Implementation required:** Modify `make_constraint_loss_fn` to divide residual by per-batch mean absolute value of active number positions (active = positions with nonzero coefficients), with floor=0.5 and detach on normalization. Then:

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-relative-constraint \
  --finetune_checkpoint runs/parenthesis/depth8-constraint-w2/itr_3000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 \
  --use_relative  \
  --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Stopping criterion:** If scale collapses below 1.3 by iter 1000, the relative normalization is not preventing shrinkage — stop. If 400-step accuracy at iter 3000+ is below 65%, this is no better than the threshold curriculum failure — stop.

**Realistic expectation:** The trivial minimum at x=0 still has loss=0 (since L=R=0 when x=0). The floor only makes the path to x=0 slower/harder, not impossible. Success requires the floor being strong enough that DSM dominates in the small-scale regime, but not so strong that balance training is disrupted.

---

## Checkpoint Reference

| Run | 50-step acc | 400-step acc | validity | MAE (400) | avg_max_abs | Notes |
|-----|-------------|--------------|---------|-----------|-------------|-------|
| depth8-stabilize2/itr_2000 | 50.0% | ~74% | 75.6% | ~0.54 | ~1.0 | Best w=1.0 starting point |
| **depth8-constraint-w2/itr_3000** | **55.6%** | **80.5%** | **91.2%** | **0.494** | **1.2–1.6** | **Honest best (l5 scale)** |
| depth8-scale-hinge-sw005/itr_500 | 52.5% | 70.4% | — | 0.563 | 1.895 | sw=0.05: -10.1pp accuracy for +0.6 scale; better Pareto than hinge |
| depth8-balance-hinge/itr_2500 | 43.8% | 63.8% | 89.4% | 0.586 | 1.757 | Dominated by sw=0.05 |
| depth8-l1l5-w2-phase2/itr_1000 | 57.0% | 85.6% | 88.6% | 0.295 | ~0.44 | Inflated (l1 scale artifact) |
| depth8-constraint-head/itr_5000 | 23.1% (head) | 50.3% (head) | 91.0% | 0.768 | **2.89** backbone / 1.03 head-corrected | FAILED: backbone scale excellent but head collapses it; 30pp below baseline |
| depth8-threshold-curriculum/itr_5000 | 39.3% | 61.1% | 90.0% | 0.640 | 1.854 | FAILED: dominated by sw=0.05 (70.4% at scale 1.895); curriculum grows scale to 2.03 but tightening pulls it back |

All checkpoints are under `runs/parenthesis/`.
