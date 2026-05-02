# Parenthesis Equation Generation — Replication & Longer-Run Roadmap

## What We Have and How We Got There

### Best checkpoint

```
runs/parenthesis/depth8-constraint-w2/itr_3000/snapshot.pt
  80.5% accuracy / 91.2% validity / MAE 0.494  (eval_steps=400, 500 samples)
  50-step proxy: 55.6% / 73.0% / MAE 0.955
  Generated scale: avg_max_abs ~1.3  (l5 training data: ~4.3)
```

### The four-stage pipeline

Getting here required four sequential training stages. Each stage is necessary for a specific reason.

---

#### Stage 1 — Symbolic foundation (10k iters)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-baseline \
  --num_iters 10000 --lr 1e-4 --lr_min 1e-6 --lr_schedule cosine \
  --warmup_iters 500 --batch_size 128 \
  --eval_samples 500 --eval_steps 50 \
  --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Take:** `itr_10000`  
**Result:** ~15% accuracy, ~75% validity, MAE ~2.3

**Why this is required:** Before the numeric constraint is turned on, the model needs to learn that equations have a specific token structure — matching parentheses, correct operator placement, the `=` separator, and a terminal `.`. Without this foundation the constraint has nothing to work with: the model generates syntactically broken sequences and the constraint gradient fires on garbage.

---

#### Stage 2 — Constraint with cosine LR (10k iters)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-constraint \
  --finetune_checkpoint runs/parenthesis/depth8-baseline/itr_10000/snapshot.pt \
  --num_iters 10000 --lr 3e-5 --lr_min 1e-5 --lr_schedule cosine \
  --warmup_iters 500 --batch_size 128 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 1.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Take:** `itr_5000` (peak, ~49.7%; accuracy oscillates after this)  
**Result at itr_5000:** ~49.7% accuracy, ~72% validity, MAE ~1.06

**Why cosine LR here:** The cosine schedule starts at 3e-5 and decays toward 1e-5. The high initial LR drives fast convergence of the constraint (`(L−R)²`) in the first 2–3k iterations. The constraint goes from MAE ~2.3 to ~1.0 within 5k steps. Using constant lr=1e-5 from the start is too slow — the model barely converges in the same number of iterations. The downside is that the cosine schedule weakens the constraint gradient in the second half of training, which is why accuracy oscillates and why we stop at the peak (itr_5000) rather than the end.

---

#### Stage 3 — Stabilize with constant LR (5k iters)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-stabilize \
  --finetune_checkpoint runs/parenthesis/depth8-constraint/itr_5000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 1.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Take:** `itr_2000` (first peak, 50.0% / 76% validity)  
**Result:** Plateaus at 50.0%, MAE ~1.01. This is the hard ceiling for w=1.0.

**Why constant LR and why stop at itr_2000:** The constraint requires LR ≥ 1e-5 continuously. If LR falls below this threshold, MAE reverts toward 2.5 — the model "forgets" the constraint. Constant lr=1e-5 keeps the constraint converged while allowing the DSM (denoising score matching) loss to refine the symbolic structure. The 50% ceiling is structural: at constraint_weight=1.0, the DSM and constraint gradients reach an equilibrium that prevents further accuracy gains. We take itr_2000 because this is the first time the model reaches 50% and has the best validity (76%); continuing further yields the same 50% peak but at lower validity.

---

#### Stage 4 — Boost with w=2.0 (5k iters)

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-constraint-w2 \
  --finetune_checkpoint runs/parenthesis/depth8-stabilize/itr_2000/snapshot.pt \
  --num_iters 5000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Take:** `itr_3000` (55.6% at 50 steps = 80.5% at 400 steps)  
**Result:** 50-step trajectory 47.3% → 52.8% → **55.6%** → 54.3% → 52.1%

**Why w=2.0 and why now:** Doubling the constraint weight is what breaks through the 50% ceiling. The w=1.0 equilibrium is a local optimum that a stronger constraint disrupts. Crucially, applying w=2.0 directly from Stage 1 or Stage 2 causes MAE oscillation that never converges — the model needs the w=1.0 warmup to settle the optimization landscape first. The peak at itr_3000 (vs itr_2000 in prior stages) reflects that the stronger constraint takes 1k extra iterations to establish the new equilibrium. Continuing past itr_3000 shows declining accuracy as the model oscillates around the new equilibrium.

---

### Key findings from all ablations

- **Constant LR ≥ 1e-5 is mandatory** once the constraint is active. LR below 1e-5 lets the MAE revert.
- **eval_steps=400 is required for honest reporting.** 50-step accuracy is ~25pp lower than 400-step and is not a reliable proxy. All "best" numbers above use 400 steps.
- **Scale shrinkage is fundamental.** The generated numbers (avg_max_abs ~1.3) are ~1/3 the scale of training data (avg_max_abs ~4.3). Every approach to fix this — scale hinge, hinge constraint, threshold curriculum, constraint head — trades away accuracy without meaningfully improving scale. The `(L−R)²` constraint has a trivial minimum at x=0 (predict all zeros). The model exploits this by generating small-magnitude balanced equations. This is an open problem.

---

## Longer Runs: What to Try Next

The current pipeline needed 4 stages because short runs require precise handoffs between regimes. With 50k-iter budgets, some stages can be merged or extended, and the key question is whether longer training in any phase can push accuracy above 80.5%.

### Hypothesis 1: Extended w=2.0 phase

**Motivation:** The current Stage 4 is only 5k iters. The accuracy peak at itr_3000 then declines to 52.1% at itr_5000 — this looks like an oscillation, not a hard ceiling. A very long constant-LR run might find a more stable higher-accuracy equilibrium.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-w2-long \
  --finetune_checkpoint runs/parenthesis/depth8-stabilize/itr_2000/snapshot.pt \
  --num_iters 50000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 2000 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**What to watch:** Does the accuracy envelope (the peaks of the oscillation) trend upward over 50k iters, or is 55.6% a true ceiling? Plot the rolling max every 5k iters.

**Success criterion:** 57%+ at 50 steps consistently across iters 30k–50k, or 83%+ at 400 steps on best checkpoint.

---

### Hypothesis 2: Merged Stage 2+3 with very slow LR decay

**Motivation:** Stages 2 and 3 are doing the same job (constraint w=1.0) but with different LR schedules. If we use a single very long cosine decay that ends at lr=1e-5 over 30k iters (instead of 10k), the LR stays high enough to drive fast convergence early while smoothly transitioning to the stable low-LR regime, eliminating the need for a checkpoint handoff.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-constraint-merged \
  --finetune_checkpoint runs/parenthesis/depth8-baseline/itr_10000/snapshot.pt \
  --num_iters 30000 --lr 3e-5 --lr_min 1e-5 --lr_schedule cosine \
  --warmup_iters 1000 --batch_size 128 --log_rate 2000 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 1.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

Then follow with the w=2.0 stage from the best checkpoint in this run (take whichever iter has the highest 50-step accuracy):

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth8-w2-from-merged \
  --finetune_checkpoint runs/parenthesis/depth8-constraint-merged/itr_<BEST>/snapshot.pt \
  --num_iters 20000 --lr 1e-5 --lr_schedule warmup_only --warmup_iters 100 \
  --batch_size 128 --log_rate 2000 \
  --eval_samples 500 --eval_steps 50 \
  --constraint_weight 2.0 --depth 8 --hidden_dim 256 --ema_beta 0.999
```

**Success criterion:** Reaches the same 80.5% at 400 steps with 2 stages instead of 4, saving ~15k iters total.

---

### Hypothesis 3: Constraint weight curriculum (single long run)

**Motivation:** Instead of separate stages with hard w=1.0 → w=2.0 transitions, ramp the constraint weight linearly from 1.0 to 2.0 over 20k iters. This requires a code change to support `--constraint_weight_end` + `--constraint_curriculum_iters`, but eliminates the stage boundary entirely.

**Rough schedule:** weight(t) = 1.0 + 1.0 * min(1, t / 20000)

The intuition is that a gradual ramp gives the model time to adjust to the stronger constraint continuously rather than experiencing a sudden doubling that temporarily disrupts the equilibrium (which is why accuracy dips at the start of the current Stage 4: 47.3% at itr_1000, below the 50.0% starting point).

This could be implemented as a `--constraint_weight_start 1.0 --constraint_weight_end 2.0 --constraint_curriculum_iters 20000` option similar to the existing `--hinge_threshold_start` / `--hinge_curriculum_iters` support.

**Expected benefit:** Smoother trajectory, potentially reaching a better equilibrium because the model is never disrupted as badly as in the hard transition.

---

### Hypothesis 4: Longer Stage 1 at higher capacity

**Motivation:** The symbolic ceiling (validity ~75%) may be model-capacity-limited. A longer Stage 1 with more depth or wider hidden dim might give the constraint a better starting point.

```bash
uv run python parenthesis_training.py \
  --data_path data/parenthesis/equations_l5.jsonl \
  --dir runs/parenthesis/depth12-baseline \
  --num_iters 20000 --lr 1e-4 --lr_min 1e-6 --lr_schedule cosine \
  --warmup_iters 1000 --batch_size 128 --log_rate 2000 \
  --eval_samples 500 --eval_steps 50 \
  --depth 12 --hidden_dim 256 --ema_beta 0.999
```

**Risk:** May OOM on 7.6 GiB GPU. Check memory before committing to 20k iters. If it fits, this is a different starting point for the full pipeline and could improve the symbolic ceiling from 75% to 80%+.

---

## Suggested order of experiments

| Priority | Experiment | Iters | Rationale |
|---|---|---|---|
| 1 | Hypothesis 1 (long w=2.0) | 50k | Uses existing checkpoint; directly tests whether 80.5% is the true ceiling |
| 2 | Hypothesis 2 (merged Stage 2+3) | 50k total | Tests whether the 4-stage pipeline can be simplified to 2 |
| 3 | Hypothesis 3 (weight curriculum) | Needs code | Cleanest long-run design; worth implementing if H1/H2 show promise |
| 4 | Hypothesis 4 (larger model) | 20k + full pipeline | Higher cost; only try if H1 confirms the ceiling is model-capacity-limited |

For all runs: log every 2000 iters (not 500) to reduce eval overhead on long runs, and always do a 400-step eval on the best checkpoint before reporting results.
