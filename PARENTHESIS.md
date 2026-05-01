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

### batch_size=256
GPU OOM. The 7.6 GiB GPU is fully utilized at batch=128. Gradient accumulation (2× steps, batch=128) is feasible but was not tested.

---

## Scale-Accuracy Tradeoff: Fundamental Conclusion

Every attempt to improve the generated number scale while maintaining ≥80% accuracy has failed. The table below summarizes the Pareto frontier as of 2026-05-01:

| Approach | 400-step accuracy | avg_max_abs (400-step) | Notes |
|---|---|---|---|
| Standard (L-R)², w=2.0 | **80.5%** | ~1.3 | Best accuracy; scale ≈ 1/3 of l5 data |
| Scale hinge sw=0.05 | 70.4% | 1.895 | -10.1pp accuracy for +0.6 scale; new equilibrium at 2.056 (50-step) |
| Balance hinge, w=2.0 | 63.8% | 1.76 | Dominated by sw=0.05: less scale AND less accuracy |
| Scale hinge sw=1.0 | ~34% | 2.9 | Catastrophic accuracy collapse |
| l5 training data | — | **~4.3** | Target scale |

**Root cause (fundamental):** `(L-R)²` has its global minimum at x=0 for all positions (L=R=0 trivially). Any constraint of the form f(L-R) with f minimized at 0 shares this pathology. The model equilibrates at small-magnitude numbers where |L-R| < 0.5 is easy. Removing the gradient inside the success zone (hinge) weakens the constraint signal overall, causing forgetting. Adding an explicit scale reward (scale hinge) at sufficient strength to move scale meaningfully also destroys the balance constraint.

The 80.5% result is honest and genuinely correct, but the equations don't resemble l5-scale data.

---

## What Should Be Tried Next (Scale-Preserving Approaches)

These approaches have NOT been tried and may have better prospects. Roughly in order of expected payoff vs implementation cost.

### 1. Scale-reward hinge (very weak, sw ≈ 0.05–0.1)

The scale hinge at sw=1.0 was catastrophic. A much weaker version might nudge scale up without disrupting balance. The scale hinge adds `scale_weight * max(0, S - mean|x_active|)²` to the loss.

**Recommended trial:** `--scale_hinge_target 2.5 --scale_weight 0.05 --constraint_weight 2.0` starting from depth8-constraint-w2/itr_3000. The balance gradient should still dominate; scale might drift up 0.1-0.2 per 1000 iters. If 5k iters at sw=0.05 gives ~2.1 scale with 70%+ accuracy at 400 steps, it's a better Pareto point.

### 2. Relative constraint — `((L-R) / sigma_batch)²`

Replace the absolute (L-R)² with a relative residual normalized by the running batch standard deviation of L values. This makes the constraint gradient scale-invariant: a residual of 0.5 is equally penalized whether the numbers are at l1 or l5 scale. The gradient no longer inherently favors small numbers.

**Risk:** Implementation must be careful to use `sigma.detach()` to avoid pathological gradients through the normalization; also need a floor on sigma to prevent division by near-zero values.

### 3. Curriculum on the constraint threshold

Start with threshold = 2.0 (loose: `max(0, |L-R| - 2.0)²`) and tighten it gradually to 0.5 over training. The loose threshold lets the DSM objective maintain l5 scale while the model first learns approximate balance; tightening then forces precision without introducing scale pressure at initialization.

**Schedule:** threshold = max(0.5, 2.0 - 1.5 * iter/5000). Implement via `threshold_getter` callable in `make_constraint_loss_fn` (already supported in the code).

### 4. Separate numeric prediction head

Add a small 2-layer MLP "constraint head" that takes final per-position embeddings and predicts constraint-adjusted numbers. The constraint loss trains only this head; the backbone is trained only by DSM. This separates the competing objectives architecturally, reducing DSM-constraint oscillation.

### 5. Two-optimizer setup

Run two Adam optimizers — one for the full model at LR=1e-5 (DSM), one for constraint gradient only at LR=1e-3. Disproportionately amplifies constraint convergence without destabilizing symbolic structure.

---

## Recommended Next Experiment

The next untested idea is **scale-reward hinge with sw=0.05** (approach 1 above). It is the most conservative extension of what's been tried and has a clear stopping criterion: if 5k iters don't produce scale ≥ 2.0 AND 400-step accuracy ≥ 70%, the scale-accuracy tradeoff is confirmed as intractable with this loss family.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-scale-hinge-sw005 \
  --finetune_checkpoint runs/parenthesis/depth8-constraint-w2/itr_3000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 500 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 --scale_hinge_target 2.5 --scale_weight 0.05 \
  --depth 8 --hidden_dim 256 --ema_beta 0.999
```

Check avg_max_abs in generated samples every 1000 iters. If scale stays below 1.8 after 3000 iters, sw is still too weak — try 0.1. If accuracy drops below 45% (50-step), sw is too strong — try 0.02.

---

## Checkpoint Reference

| Run | 50-step acc | 400-step acc | validity | MAE (400) | avg_max_abs | Notes |
|-----|-------------|--------------|---------|-----------|-------------|-------|
| depth8-stabilize2/itr_2000 | 50.0% | ~74% | 75.6% | ~0.54 | ~1.0 | Best w=1.0 starting point |
| **depth8-constraint-w2/itr_3000** | **55.6%** | **80.5%** | **91.2%** | **0.494** | **1.2–1.6** | **Honest best (l5 scale)** |
| depth8-scale-hinge-sw005/itr_500 | 52.5% | 70.4% | — | 0.563 | 1.895 | sw=0.05: -10.1pp accuracy for +0.6 scale; better Pareto than hinge |
| depth8-balance-hinge/itr_2500 | 43.8% | 63.8% | 89.4% | 0.586 | 1.757 | Dominated by sw=0.05 |
| depth8-l1l5-w2-phase2/itr_1000 | 57.0% | 85.6% | 88.6% | 0.295 | ~0.44 | Inflated (l1 scale artifact) |

All checkpoints are under `runs/parenthesis/`.
