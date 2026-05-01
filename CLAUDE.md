# Variable-Length Diffusion — Parenthesis Training Loop

## Goal

Improve the accuracy of the model trained by `parenthesis_training.py` on the parenthesized arithmetic equations dataset. Accuracy is measured by:

1. **`accuracy`** — fraction of valid generated equations where |left_value − right_value| < 0.5 (higher is better).
2. **`mean_abs_error`** — mean absolute difference between left and right sides of valid equations (lower is better).
3. **`invalid`** — number of samples that cannot be parsed as equations at all (lower is better).
4. **`validity`** — fraction of samples that parse successfully, i.e. `(total - invalid) / total` (higher is better). **Target: ≥ 75%.** We reached 75.7% (depth6-baseline iter 8k) and ~78–88% (post-curriculum runs). Configs that drop validity below ~70% should be considered regressions even if accuracy improves, because accuracy is computed only over valid samples and can be artificially inflated by a low-validity model that gets lucky on the few samples it does parse.

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
| `--eval_steps` | 50 | More steps at eval = better quality samples |

Architecture changes (in `models/mmdit_qm9.py`):
- Increase `depth` to 6 or 8 for more model capacity
- Increase `dim_modalities` to `[512, 512]` for wider network

---

## Loop workflow

When running in `/loop` mode, each iteration should:

1. **Read metrics**: Load `runs/parenthesis/<latest-run>/metrics.jsonl` and inspect the accuracy trend.
2. **Diagnose**: Look at the last few metric entries. Is accuracy still improving, plateaued, or degraded? Also check validity (`(total - invalid) / total`) — flag if it drops below 70%.
3. **Decide**: If accuracy is improving AND validity ≥ 70% → continue or increase iters. If plateaued → change a hyperparameter (lr, depth, batch size). If invalids are high (validity < 70%) → symbolic structure is breaking down; reduce constraint weight or try `--interpolant autoregressive`. Never accept an accuracy gain that comes at the cost of validity dropping below 70%.
4. **Run a short trial**: Launch a 2000-iter run with the new config and compare metrics to the baseline.
5. **Commit the winner**: If the new config improves accuracy, update this file with the new best config.

### Current best known config

```
Run dir: runs/parenthesis/l1-baseline
Best accuracy: 61.8% (iter 3000), avg ~48%  ← l1 dataset (numbers in [-1,1])
Mean abs error: 0.46 (best)
Config: data=l1, lr=1e-4, batch_size=128, warmup=500, depth=4, dim=256, interpolant=multimodal

l5 best (eval_steps=50): 55.6% (depth8-constraint-w2, iter 3000) — validity 73.0% ← HONEST BEST
l5 best (eval_steps=400): 80.5% / 91.2% validity (depth8-constraint-w2/itr_3000) ← HONEST BEST (see scale caveat)
  Inference scaling (depth8-stabilize2/itr_2000):
    steps  50: acc 50.0%, valid 75.6%, MAE 1.012
    steps 100: acc 57.6%, valid 83.4%, MAE 0.900 (+7.6pp)
    steps 200: acc 62.2%, valid 86.2%, MAE 0.759 (+4.6pp)
    steps 400: acc 73.6%, valid 91.8%, MAE 0.544 (+11.4pp)
    steps 800: acc 74.7%, valid 93.2%, MAE 0.502 (+1.1pp)
  Inference scaling (depth8-constraint-w2/itr_3000) ← HONEST BEST CHECKPOINT (l5-scale, avg_max_abs ~1.3):
    steps  50: acc 55.6%, valid 73.0%, MAE 0.955
    steps 400: acc 80.5%, valid 91.2%, MAE 0.494 (+24.9pp)
  Inference scaling (depth8-l1l5-w2-phase2/itr_500) ← scale artifact (avg_max_abs 0.57; see caveat):
    steps  50: acc 62.8%, valid 79.6%, MAE 0.478
    steps 400: acc 84.7%, valid 90.4%, MAE 0.219 (+22.0pp — MAE 56% lower than w2 record!)
  KEY INSIGHTS:
    - l1→l5 curriculum: train on l1 data first (easier), then finetune on l5; constraint converges to MAE 0.267 on l1 in 2500 iters
    - At l5 phase 2 iter 500, model applies l1-learned constraint to l5 → 62.8%/84.7% (new records)
    - Forgetting effect: by iter 2000 on l5, accuracy decays back to 55.6% as l5 scale overwrites l1 precision
    - The "sweet spot" is very early in phase 2 (iter 500) — need anti-forgetting strategy to extend it
    - constraint_weight=2.0 gave +6.9pp at 400 steps (73.6%→80.5%) vs constraint_weight=1.0
    - validity at 400 steps recovers from training validity — training-step validity NOT predictive of eval validity
    - use eval_steps=400 going forward; plateau confirmed at 400-800 steps
constraint-loss (done): peaked 50.7% iter 4k, final 47.9% iter 10k; invalids oscillated 37–64
constraint-light (killed iter 4k): weight=0.1 never converged Cstr; same invalids as c-loss, 20pp lower acc
constraint-medium (killed iter 2k): weight=0.5 worse than c-loss across all metrics despite faster Cstr convergence
constraint-stabilize (killed iter 3k): MAE rose 1.05→1.32 as LR decayed below 1e-5; confirmed constraint needs LR≥1e-5
depth6-baseline (done): depth=6, no constraint, 10k iters; best acc 16.9% (iter 9000), best validity 75.7% (iter 8000), MAE 2.2–3.7; same ceiling as depth=4 without constraint
depth6-constraint (done): finetune depth6-baseline/itr_10000, constraint_weight=1.0, lr=3e-5 cosine→1e-5; peak 55.5% (iter 5k), final 43.8% (iter 10k); post-peak avg 42.7%
depth6-stabilize (done): finetune depth6-constraint/itr_5000, constant lr=1e-5; peak 50.0% (iter 2k), avg 46.5%; +3.8pp avg vs depth6-constraint post-peak; oscillation persists but at higher floor
depth6-stabilize2 (done): finetune depth6-stabilize/itr_2000, const lr=1e-5, eval_samples=500; trajectory 41.6%→43.3%→43.9%→44.5%→49.4%, avg 44.5%; validity 77.4%→70.4% (declining); strong upward acc trend but validity hit floor at itr_5000
depth6-stabilize3 (done): finetune depth6-stabilize2/itr_5000, const lr=1e-5, eval_samples=500; trajectory 41.5%→43.2%→46.0%→47.3%→46.8%, avg 45.0%; best: 47.3%/76.6% @ itr_4000; first run with 3 consecutive valid ≥75% checkpoints
depth6-stabilize4 (done): finetune depth6-stabilize3/itr_4000, const lr=1e-5; trajectory 43.4%→46.9%→46.2%→43.7%→43.5%, avg 44.7%; best 46.9%/74.2%; 3/5 checkpoints valid≥75%; warm-start strategy conclusively plateaued
depth8-baseline (done): depth=8, lr=1e-4 cosine→1e-6, 10k iters; final acc 15.3%, validity 75.0%, MAE 2.30; validity plateaued 74-75% from iter 6k onward; learned symbolic structure comparable to depth6-baseline
depth8-constraint (done): finetune depth8-baseline/itr_10000, constraint_weight=1.0, lr=3e-5 cosine→1e-5, 10k iters; trajectory 39.4%→47.7%→48.3%→48.6%→49.7%→49.3%→43.5%→48.0%→46.9%→48.2%; peak 49.7%/72.4% @ itr_5000; best validity 75.0% @ itr_9000; avg 46.96%; did NOT break 55.5% barrier; smoother trajectory than depth6 (no spike), better avg, validity more consistently near 74-75%; MAE bottomed at 1.043 (vs depth6's 0.93) — gradient dilution hypothesis: more params → weaker constraint force per param
depth8-stabilize (done): finetune depth8-constraint/itr_5000, const lr=1e-5, 5k iters; trajectory 42.0%→50.0%→47.3%→48.9%→46.9%, avg 47.0% (BEST AVG of any stabilize run); best: 50.0%/76.0% @ itr_2000; more stable post-peak than depth6 (-2.7pp drop vs -6.8pp)
depth8-stabilize2 (done): finetune depth8-stabilize/itr_2000 (50.0%/76.0%), const lr=1e-5, 5k iters; trajectory 44.0%→50.0%→48.1%→48.2%→46.9%, avg 47.4%; best: 50.0%/75.6% @ itr_2000; confirms 50.0% ceiling — identical peaks to depth8-stabilize; warm-start chain w=1.0 conclusively plateaued at 50%
depth8-constraint-w2 (done): finetune depth8-stabilize2/itr_2000, constraint_weight=2.0, const lr=1e-5, 5k iters; trajectory 47.3%→52.8%→55.6%→54.3%→52.1%, avg 52.4%; best: 55.6%/73.0%/MAE 0.955 @ itr_3000 ← 80.5%/91.2% at eval_steps=400 (NEW ALL-TIME BEST)
```

### Loop history

| Run | Key change | Best acc% | Avg acc% | Validity range | Notes |
|-----|-----------|-----------|----------|---------------|-------|
| test | defaults, 1k iters | 6.4% | ~5% | — | Too few iters |
| baseline | warmup=500, 10k iters | 15.4% | ~11% | 57–78% | Noisy; constant LR oscillates |
| cosine-lr | +cosine decay 1e-4→1e-6 | 14.2% | 9.7% | ~77% | Oscillation persists; marginal improvement at 10k |
| l1-baseline | l1 dataset (numbers in [-1,1]) | 61.8% | ~48% | ~80% | Confirmed: architecture works; bottleneck is l5 number scale |
| l5-normalized | l5 + --num_scale 5.0 | 14.5% | 9.5% | ~77% | Normalization didn't help — delta becomes 5× harder in model space |
| curriculum-l1-to-l5 | finetune l1-best on l5, lr=3e-5 cosine | 14.8% | ~12% | 76–88% | Validity improved; same numeric bottleneck |
| ema-fix | beta=0.999 + eval with EMA + curriculum warmstart | 15.1% | ~12% | 78–89% | EMA fix helped validity; numeric ceiling unchanged |
| constraint-loss | fixed evaluator + (L-R)² loss weight=1.0 | 50.7% (iter 4000) | ~45% | 62–81% | Peaked iter 4k; invalids 37→64 then settling ~52-56; final 47.9% at iter 10k |
| constraint-light | same but constraint_weight=0.1 | 31.3% (iter 3000) | ~27% | ~72% | Killed: Cstr never converged (<1.5 at iter 4000 vs <0.1 for c-loss); same invalids, 20pp lower acc |
| constraint-medium | same but constraint_weight=0.5 | 35.7% (iter 2000) | ~32% | ~71% | Killed: higher DSM disruption than c-loss, invalids jumped faster (34→57 vs 37→52), worse across all metrics |
| constraint-stabilize | finetune c-loss/itr_4000, lr=1e-5 cosine→1e-7, w=1.0 | 51.4% (iter 500) | ~46% | ~74% | MAE rose 1.05→1.32 as LR fell below 1e-5; constraint fades without active LR pressure |
| depth6-baseline | depth=6, no constraint, from scratch, lr=1e-4 cosine, 10k iters | 16.9% (iter 9k) | ~13% | **75.7% peak** | Same ceiling as depth=4 without constraint; best raw validity seen |
| depth6-constraint | finetune depth6-baseline/10k, constraint_weight=1.0, lr=3e-5 cosine→1e-5, 10k iters | 55.5% (iter 5k) | ~44% | ~72–78% | Broke 50.7% ceiling; peak iter 5k then oscillated 40-47%; final 43.8% |
| depth6-stabilize | finetune depth6-constraint/itr_5000, const lr=1e-5, constraint_weight=1.0, 5k iters | 50.0% (iter 2k) | 46.5% | ~73–77% | +3.8pp avg over d6-constraint post-peak; oscillation persists at higher floor |
| depth6-stabilize2 | finetune depth6-stabilize/itr_2000, const lr=1e-5, eval_samples=500, 5k iters | 49.4% (iter 5k) | 44.5% | 70–77% | Upward acc trend (+7.8pp over 5k); validity declined to floor at iter 5k |
| depth6-stabilize3 | finetune depth6-stabilize2/itr_5000, const lr=1e-5, eval_samples=500, 5k iters | 47.3% (iter 4k) | 45.0% | 71–78% | First run with 3 consecutive valid ≥75% checkpoints; best validity-constrained combo yet |
| depth6-stabilize4 | finetune depth6-stabilize3/itr_4000, const lr=1e-5, 5k iters | 46.9% (iter 2k) | 44.7% | 72–76% | 3/5 valid≥75%; warm-start strategy conclusively plateaued — peaks declining run-over-run |
| depth8-baseline | depth=8 from scratch, lr=1e-4 cosine→1e-6, warmup=500, 10k iters | 15.3% | ~13% | 18→75% | Validity 75% final; symbolic learning matches depth6-baseline |
| depth8-constraint | finetune depth8-baseline/itr_10000, constraint_weight=1.0, lr=3e-5 cosine→1e-5, 10k iters | 49.7% (iter 5k) | 46.96% | 70–75% | Smooth plateau (no spike); better avg than d6-constraint; did NOT break 55.5%; MAE floor 1.04 vs d6's 0.93 |
| depth8-stabilize | finetune depth8-constraint/itr_5000, const lr=1e-5, 5k iters | 50.0% (iter 2k) | 47.0% | 73–78% | BEST AVG of any stabilize run; best valid≥75% checkpoint ever (50.0%/76.0%) |
| depth8-stabilize2 | finetune depth8-stabilize/itr_2000, const lr=1e-5, 5k iters | 50.0% (iter 2k) | 47.4% | 74–76% | Peak identical to depth8-stabilize; w=1.0 warm-start chain conclusively plateaued at 50% |
| depth8-constraint-w2 | finetune depth8-stabilize2/itr_2000, constraint_weight=2.0, const lr=1e-5, 5k iters | **55.6%** (iter 3k) | 52.4% | 72–75% | NEW RECORD: **80.5%/91.2% at 400 steps**; peak iter 3k; MAE broke below 1.0 at training |
| depth8-constraint-w2-s2 | finetune depth8-constraint-w2/itr_3000, w=2.0, const lr=1e-5, killed @ iter 3k | 54.0% (iter 2k) | — | 72–77% | Peaked BELOW starting point (54.0% < 55.6%); w=2.0 gain is one-time, doesn't compound |
| depth8-constraint-w3 | finetune depth8-constraint-w2/itr_3000, constraint_weight=3.0, const lr=1e-5, killed @ iter 3k | 54.8% (iter 2k) | — | 72–77% | Peak at iter 2k (same as w2-s2), MAE floor ~1.03; escalating w doesn't break ceiling from same start |
| depth8-constraint-w2-lr3e5 | finetune depth8-constraint-w2/itr_3000, w=2.0, const lr=3e-5, killed @ iter 2k | 52.0% (iter 2k) | — | ~76% | LR=3e-5 from peak worse than both w2-s2 and w3; more disruption than benefit |
| depth8-constraint-w2-from-baseline | finetune depth8-baseline/itr_10000, w=2.0, lr=3e-5 cosine→1e-5, stopped @ iter 6500 | 55.9% (iter 6500) | ~51% | 69–77% | Cosine LR caused MAE oscillation 1.0–1.2, never converging; 55.9% was oscillation peak; 400-step eval: 77.2%/86.0% (below 80.5% record) |
| depth8-l1-w2-phase1 | finetune depth8-baseline/itr_10000 on l1 data, w=2.0, const lr=1e-5, stopped @ iter 2500 | **78.9%** (l1 acc, iter 2500) | — | 71–76% | l1 MAE converged to 0.267 in 2500 iters (vs l5 needing 3000+ iters to reach 0.955); acc plateaued ~78%; took itr_2500 for phase 2 |
| depth8-l1l5-w2-phase2 | finetune depth8-l1-w2-phase1/itr_2500 on l5 data, w=2.0, const lr=1e-5 | **62.8%** (iter 500) | ~57% | 73–80% | 400-step: 84.7%→85.6%→82.6%→78.1%; scale shrinks (0.57→1.04 avg_max_abs) as l5 adapted; 85.6% at itr_1000 (scale 0.44); 80.5% original is better for l5-scale |
| depth8-l1l5-phase2-lr5e6 | finetune depth8-l1-w2-phase1/itr_2500 on l5 data, w=2.0, const lr=5e-6 | 63.5% (iter 500) | ~59% | 73–79% | Slower forgetting; 400-step: 86.6%→84.0%→84.4%; but avg_max_abs ~0.29 at 400 steps (scale artifact) |
| depth8-l1-phase1-eval400 | phase1/itr_2500 evaluated directly on l5 | — | — | ~88% | 400-step = 88.5% but avg_max_abs=0.24 — pure l1 scale, metric inflation; not a genuine l5 result |
| depth8-proxy-constraint | finetune depth8-constraint-w2/itr_3000, proxy constraint (residual.detach() * residual), w=2.0, lr=1e-5 | ~47% | — | ~73% | Proxy = standard for Adam (identical first-order gradients); acc declining: 47.3%→47.4%→45.9%; killed after 3 declines |
| depth8-scale-hinge-sw1 | finetune depth8-constraint-w2/itr_3000, scale_hinge_target=2.5, scale_weight=1.0, w=2.0, lr=1e-5 | ~34% | — | ~72% | Scale jumped 1.575→2.9 but accuracy collapsed 55.6%→33.9% in 1k iters; sw=1.0 was ~10× too strong; killed |
| depth8-balance-hinge | finetune depth8-constraint-w2/itr_3000, use_hinge=True (max(0,\|L-R\|-0.5)²), w=2.0, lr=1e-5, 3k iters | 50.3% (iter 500) | ~44% | 73–80% | Scale grew slowly: 1.575→1.826 in 2500 iters; 50-step acc: 50.3%→42.6%→44.4%→44.5%→43.8%→38.5%; 400-step/itr_2500: **63.8%**/89.4%/MAE 0.586/scale 1.757 — **16.7pp below baseline** |
| depth8-scale-hinge-sw005 | finetune depth8-constraint-w2/itr_3000, scale_hinge_target=2.5, scale_weight=0.05, w=2.0, lr=1e-5, 5k iters | 52.5% (iter 500) | ~47.4% | 69–79% | Scale jumped to 2.056 at itr_500 and STABILIZED there (all 10 checkpoints identical!); new equilibrium; 50-step traj: 52.5%→44.4%→46.7%→46.7%→46.6%→43.7%→46.1%→49.0%→50.1%→48.2%; 400-step/itr_500: **70.4%**/MAE 0.563/scale 1.895 — -10.1pp vs baseline for +0.3 scale |

### Diagnosis log

**baseline (10k iters):** `WarmUpScheduler` keeps LR at 1e-4 forever after warmup. Accuracy peaks at 15% at
iter 3k then oscillates 9–15% — classic constant-LR instability. Symbol validity improves steadily (57%→78%)
but EucUnmask loss stays 2.4–3.0 throughout; numeric prediction never settles.

**cosine-lr (done):** Cosine LR didn't fix the oscillation. Accuracy still noisy (3.5–14.2%, avg 9.7%).
Failure modes at itr_10000: 22.7% invalid (mismatched parens), 67.3% valid-but-wrong-numbers, 10% correct.
The dominant failure is numeric precision: l5 numbers span [-5,5] so delta=0.5 requires ~5% relative precision.

**l1-baseline (done):** l1 accuracy peaks at 61.8% (iter 3000), avg ~48%. Architecture works.
l1 mean_abs_error 0.46–0.82 — right at the delta=0.5 threshold, explaining why accuracy oscillates with
sampling noise. Confirms bottleneck for l5 is purely the scale: delta=0.5 is 50% of l1 range but only 5% of l5.

**l5-normalized (done, failed):** Normalization doesn't help. When model works in [-1,1] space, the delta=0.5
threshold becomes 0.1 in normalized space — 5× harder. Model achieves same relative precision either way.
Sampling step sweep on baseline/itr_3000: steps=100 gives best accuracy (15.4%) but mean_err stays at ~2.4.

**Root cause confirmed:** Model achieves ~46% of data std as mean_abs_error regardless of scale. For l1 that's
~0.46 (passes delta=0.5); for l5 that's ~2.3 (fails). More steps only help validity, not numeric precision.

**curriculum-l1-to-l5 (done):** Validity improved dramatically (76-88% from iter 1) but numeric accuracy
still 7-15%. Confirmed: symbolic transfer works but numeric component relearns from scratch.

**EMA diagnosis:** With beta=0.9999, the EMA at 10k steps still has 37% weight on the random initial model.
Tested model vs EMA at baseline/3k: EMA produces only 7-21/300 valid samples but those have NEAR-PERFECT
accuracy (81-100%, mean_err=0.06-0.23). The continuous denoising IS capable of precision — the EMA just
can't generate valid structures because it's ~74% random init at 3k steps.

**ema-fix (done):** beta=0.999 + EMA eval. Validity improved (only 11-22% invalid vs 30-50% before).
But accuracy ceiling is the same ~12-15%. mean_abs_error stays at ~2.5-2.8.

**Evaluator bug discovered and fixed:** `parse_equation` in `evaluate_equations_parenthesis.py` had a
critical bug: `parse_factor` treated binary `+`/`-` operators as UNARY signs, misevaluating 95% of
training data (only 1/20 samples gave L≈R with the buggy evaluator, while all 20/20 pass with the fix).
The data format stores the first term's sign embedded in the number value; subsequent terms' signs come
from binary operators (+/-) in the symbol sequence. Parentheses are structural only and don't change sign.
The fix: remove the unary-sign case from `parse_factor`. Prior accuracy metrics (10-15%) were coincidental
— the buggy evaluator gave similar numbers on generated samples, so all prior comparisons are still valid.

**Constraint weight sweep (done):** Tested w=0.1, 0.5, 1.0. w=0.1: Cstr never converged (<1.5 vs <0.1 for w=1.0),
same invalids (52-56) but 20pp lower accuracy. w=0.5: Cstr converged fast but DSM degraded MORE than w=1.0
(invalids jumped 34→57 in 1k iters vs 37→52 for w=1.0). Optimal constraint_weight=1.0 confirmed.

**Constraint requires active LR pressure:** constraint-stabilize (lr=1e-5 cosine→1e-7) showed MAE rising
1.05→1.32 as LR fell below ~1e-5. The constraint doesn't get permanently encoded — it requires gradient
pressure to maintain. LR must stay ≥1e-5 to prevent MAE from reverting toward 2.5.

**True accuracy ceiling:** All constraint_weight=1.0 runs: true mean accuracy ~47-48% (not 51%!). Observed
range 44-51% is sampling variance (±3.5pp for 200-sample eval, binomial std=sqrt(0.48*0.52/200)).
The 51% peaks were sampling noise. Need larger eval or more capacity to improve.

**Constraint loss (done):** Key insight from the bug analysis: since all equations are purely additive
(no * in the dataset), each number's contribution to L-R is a linear combination with coefficients ±1
determined by: (1) first term always +1 (sign in number), (2) subsequent terms: +1 if preceding binary
symbol is "+", -1 if "-", (3) right-side terms: all negated for L-R=0.

Implementation: `compute_constraint_coefficients_batch` strips parens, reads binary ops to build coefficient
tensors, then the loss = mean((coeffs @ x0_pred)²). The model's x0 prediction is un-reordered using the
interpolant's permutation to align with original number positions. Loss added to compute_loss via callback
`extra_loss_fn`. Config: finetune from ema-fix/itr_9000, lr=3e-5 cosine, constraint_weight=1.0.

**depth6-constraint (done):** Depth=6 model + constraint_weight=1.0 finetuned from depth6-baseline.
Cstr converged in <100 iters. Full trajectory: 36.4%→43.2%→45.0%→51.9%→55.5%→41.2%→47.3%→40.4%→40.8%→43.8%.
MAE bottomed at 0.93 (iter 5k) then rose to 1.11-1.23. Post-peak oscillation (40-47%) caused by LR decay:
as LR fell from ~2e-5 (iter 5k) to ~1.1e-5 (iter 10k), constraint gradient weakened. Confirms LR decay
is the driver of post-peak accuracy decay even with lr_min=1e-5.

**depth6-stabilize (done):** Finetune from depth6-constraint/itr_5000 (55.5% peak), constant lr=1e-5.
Trajectory: 46.2%→50.0%→43.2%→44.8%→48.1%. Avg 46.5% vs depth6-constraint post-peak avg 42.7%.
Constant LR confirmed to add ~+3.8pp vs cosine decay. Oscillation still present (~±3.5pp around 46.5%
mean) — NOT caused by LR decay alone. The DSM-Cstr oscillation is inherent to joint training dynamics.

**depth6-stabilize2 (done):** Finetune from depth6-stabilize/itr_2000 (50.0% peak), constant lr=1e-5,
eval_samples=500. Trajectory: 41.6%→43.3%→43.9%→44.5%→49.4%, avg 44.5%. Validity: 77.4%→74.4%→75.6%→74.6%→70.4%.
Despite starting from a 50% peak, dropped to 41.6% at itr_1000 then climbed steadily — classic oscillation then recovery.
Strong upward acc trend (+7.8pp over 5k iters) but validity hit the 70% floor exactly at the acc peak (itr_5000).
Avg (44.5%) is below depth6-stabilize avg (46.5%), but the endpoint (49.4%) is the second-highest ever recorded.
Key tension: accuracy and validity move in opposite directions — the constraint drives acc up but invalids up too.

**depth6-stabilize3 (done):** Finetune from depth6-stabilize2/itr_5000. Trajectory: 41.5%→43.2%→46.0%→47.3%→46.8%, avg 45.0%.
Best checkpoint: itr_4000 at 47.3% acc / 76.6% validity — the best validity-constrained accuracy ever recorded.
First run with 3 consecutive checkpoints (itr 2k, 3k, 4k) above 75% validity. Iter 5000 dropped back (71.0%) as usual.
The warm-start-from-peaks strategy has not elevated the ceiling: peaks across stabilize runs are 50.0%→49.4%→47.3%.
Oscillation is structural; warm-starting from the best validity checkpoint (itr_4000) is the next test.

**depth6-stabilize4 (done):** Finetune from depth6-stabilize3/itr_4000 (47.3%/76.6%). Trajectory:
43.4%→46.9%→46.2%→43.7%→43.5%, avg 44.7%. Best: 46.9% @ iter 2000 (valid 74.2%). 3/5 checkpoints valid≥75%.
Peaks across all 4 stabilize runs: 50.0→49.4→47.3→46.9% — declining every run. Warm-start approach is
conclusively plateaued. Depth=6 with constraint has hit its ceiling (~47% sustained, 50% occasional peaks).

**depth8-baseline (done):** Train depth=8 MMDiTQM9 from scratch. 10k iters, final acc 15.3%, validity 75.0%, MAE 2.30.

**depth8-constraint (done):** Finetuned depth8-baseline with constraint_weight=1.0, lr=3e-5 cosine→1e-5.
Full trajectory: 39.4%→47.7%→48.3%→48.6%→49.7%→49.3%→43.5%→48.0%→46.9%→48.2%. Peak: 49.7% @ iter 5k (MAE 1.064).
Depth8 did NOT break the 55.5% barrier. Key finding: depth8 constraint MAE never dropped below 1.04, while
depth6 hit MAE 0.93 at its peak. Hypothesis: constraint gradient is diluted across more parameters in depth8.

**depth8-stabilize (done):** Finetune from depth8-constraint/itr_5000, const lr=1e-5. Trajectory: 42.0%→50.0%→47.3%→48.9%→46.9%, avg 47.0%. Best: 50.0%/76.0% @ itr_2000.

**depth8-stabilize2 (done):** Finetune from depth8-stabilize/itr_2000 (50.0%/76.0%), const lr=1e-5. Trajectory: 44.0%→50.0%→48.1%→48.2%→46.9%, avg 47.4%. Best: 50.0%/75.6% @ itr_2000. Third consecutive run peaking at exactly 50% at iter 2000— w=1.0 warm-start chain is conclusively plateaued.

**constraint_weight=2.0 breakthrough (depth8-constraint-w2, done):** Finetune from depth8-stabilize2/itr_2000, constraint_weight=2.0, const lr=1e-5. Full trajectory: 47.3%→52.8%→55.6%→54.3%→52.1%, avg 52.4%. Peak: 55.6%/73.0%/MAE 0.955 @ itr_3000 — new all-time record at eval_steps=50, first depth8 run to match depth6's 55.5% ceiling. MAE broke below 1.0 for the first time in any depth8 run. Peak at iter 3000 (vs iter 2000 for all w=1.0 runs) — the doubled constraint pushes the peak 1000 iters later. Eval at 400 steps: 80.5%/91.2% — new all-time best.

**Ceiling at 55.6%/80.5% — confirmed for direct approaches:** Three approaches starting from depth8-constraint-w2/itr_3000 (the 55.6% checkpoint) all failed to exceed it:
- w2-s2 (w=2.0, lr=1e-5): peaked 54.0% @ iter 2k — below start
- w3 (w=3.0, lr=1e-5): peaked 54.8% @ iter 2k — below start  
- lr3e5 (w=2.0, lr=3e-5): iter 1k 47.5%, MAE 1.145 (more disruption than benefit)
Pattern: escalating w or LR from an already-w=2.0-trained checkpoint doesn't help; the MAE floor of 0.955 cannot be improved by gradient force alone from the same starting weights.

**from-baseline run (done):** finetune depth8-baseline/itr_10000 with w=2.0, cosine lr 3e-5→1e-5, stopped @ iter 6500. Peak: 55.9% @ iter 6500 (oscillation peak — same value as before but context: MAE 1.04 never broke 1.0). 400-step eval of itr_6500: 77.2%/86.0% — below 80.5% record. Confirmed: cosine LR causes MAE oscillation (1.0-1.2 range), preventing steady convergence. The constant lr=1e-5 in the original recipe was essential.

**l1→l5 curriculum findings (done):** Two-phase curriculum explored extensively.
- Phase 1: finetune depth8-baseline/itr_10000 on l1 data (numbers in [-1,1]), w=2.0, const lr=1e-5.
  Trajectory: 35.4%→58.3%→70.9%→72.9%→78.9%, MAE 1.04→0.584→0.357→0.290→0.267 in just 2500 iters.
  l1 MAE converges 4× faster than l5 because threshold (0.5) is 50% of l1 range vs 5% of l5 range.
  Best checkpoint: itr_2500 (78.9% l1 acc, MAE 0.267). Used as phase 2 starting point.
- Phase 2 (l5, lr=1e-5): 62.8%→57.0%→55.8%→55.6% at 50 steps; 400-step: 84.7%→85.6%→82.6%→78.1%.
  400-step peak at itr_1000: 85.6%/88.6%/MAE 0.295.
- Phase 2 (l5, lr=5e-6 — slower forgetting): 63.5%→59.3%→59.5% at 50 steps; 400-step: 86.6%→84.0%→84.4%.
  400-step peak at itr_500: 86.6%/91.2%/MAE 0.188. Lower LR → peak moves earlier.
- Pure l1 checkpoint (phase1/itr_2500, NO l5 at all): 400-step eval = 88.5%/88.4%/MAE 0.149.

CRITICAL SCALE CAVEAT — scale-accuracy tradeoff:
  All models exhibit scale shrinkage vs. l5 training data (avg_max_abs ~4.3):
    - depth8-constraint-w2/itr_3000 (best honest l5): avg_max_abs ~1.3–1.6 at 50/400 steps
    - phase2/itr_2000 (most l5-adapted curriculum): avg_max_abs ~1.04 at 50 steps
    - phase2/itr_1000 (best 400-step): avg_max_abs ~0.44 at 400 steps
    - phase1/itr_2500 (pure l1): avg_max_abs ~0.24 at 400 steps
  The curriculum models generate SMALLER numbers that are easier to satisfy |L-R|<0.5.
  At comparable scale (itr_2000, avg_max_abs ~1.0), 400-step accuracy = 78.1% < 80.5% (original).
  CONCLUSION: curriculum accuracy gains (85.6%→88.5%) are artifacts of scale shrinkage.
  TRUE BEST for l5-scale equations remains depth8-constraint-w2/itr_3000 at 80.5%.
  Mixed l1+l5 training: model collapsed to generating only l1-scale equations (mode collapse); not useful.
  
ROOT CAUSE: the constraint loss (L-R)² pulls numeric predictions toward 0 (all L=R=0 satisfies the constraint exactly). The diffusion model satisfies this by generating small-magnitude numbers where |L-R| < 0.5 trivially. This is a fundamental conflict between the constraint objective and the data distribution objective.

**Scale-normalized constraint attempt (done, FAILED):** Modified `make_constraint_loss_fn` to divide residual by `x0_orig.abs().sum(dim=1).clamp(min=1.0)`. Result: 19-31% accuracy — much worse than original 47%. Root cause: the normalization includes ALL positions (including pad), making the constraint 25× weaker than needed for l5 numbers. Reverted to original (unnormalized) constraint.

**Proxy constraint (done, FAILED):** Modified constraint to `(residual.detach() * residual).mean()` — this removes the x_i self-coupling from the Hessian (no ∂²L/∂xi² from the constraint), but the first-order gradient is numerically identical to standard `residual.pow(2).mean()` because `residual_sg = residual` in value. Adam uses only first-order gradients, so proxy and standard are functionally equivalent. Accuracy declined: 47.3%→47.4%→45.9% over 1500 iters; killed after 3 consecutive declines.

**Scale hinge sw=1.0 (done, FAILED):** Added `scale_hinge_target=2.5, scale_weight=1.0` to push avg_max_abs toward 2.5. Scale jumped from 1.575→2.9 in just 1000 iters, but accuracy collapsed 55.6%→33.9%. Root cause: `scale_weight=1.0` made the scale hinge as strong as the entire balance loss — dominated the gradient, destroyed the constraint equilibrium. Would need sw≈0.05–0.1 for balance to remain dominant.

**Balance hinge (done):** Switched to hinge constraint `max(0, |L-R| - 0.5)²` (zero gradient when |L-R|<0.5, matching the evaluation criterion). Starting from depth8-constraint-w2/itr_3000 (55.6%, scale 1.575): ~55% of samples already balanced → hinge gives 0 gradient for them immediately.
- 50-step trajectory (2500 iters): 50.3%→42.6%→44.4%→44.5%→43.8% (then 38.5% at 3000 before kill)
- Scale trend: 1.575→1.595→1.689→1.776→1.789→1.826 (slow linear growth, ~+0.1/1000 iters)
- 400-step eval of itr_2500: 63.8% accuracy / 89.4% validity / MAE 0.586 / avg_max_abs 1.757
- Verdict: scale grew by only 0.18 units while accuracy dropped 16.7pp at 400 steps vs baseline (80.5%). Hinge removes gradient for already-balanced samples, weakening the constraint and allowing L-R precision to erode. Not a viable approach.

**Scale-accuracy tradeoff — FUNDAMENTAL CONCLUSION (as of 2026-05-01):** Every attempt to grow scale has traded accuracy:
- Standard (L-R)² constraint → scale ~1.2-1.6, accuracy 80.5% ← best Pareto point for accuracy
- Scale hinge sw=0.05 → scale 1.895, accuracy 70.4% (-10.1pp for +0.3 scale); scale stabilized at 2.056 (50-step) immediately and held; better Pareto than balance hinge
- Balance hinge → scale 1.76, accuracy 63.8% (-16.7pp for +0.18 scale) ← dominated by sw=0.05
- Scale hinge sw=1.0 → scale 2.9, accuracy 33.9% (catastrophic)
The (L-R)² constraint has a global minimum at x=0 (all L=R=0 trivially). The diffusion model finds this equilibrium by generating small-magnitude numbers where |L-R|<0.5 is easy to satisfy. Any loss that rewards accuracy without penalizing small numbers converges to this equilibrium. A fundamentally different constraint formulation is needed to escape it.

**Future work directions:**
1. **Scale-normalized constraint (active terms only)**: `(residual / active_scale)²` where active_scale = sum(|x_active|).detach().clamp(min=1). Removes the pull toward 0 by making gradient scale-invariant. Requires constraint_weight ≈ 50–100 to compensate for smaller loss magnitude. Previously failed version used ALL positions (including padding) which diluted by ~25×; correct version uses only (coeffs.abs()>0.5) positions.
2. **Relative constraint**: `((L-R) / sigma_batch)²` where sigma_batch = running std of L values in the batch. Scale-invariant gradient, no preference for small numbers. Requires careful implementation to avoid mode collapse.
3. **Separate numeric heads**: give constraint its own smaller subnet that doesn't affect the denoising backbone, reducing DSM-Cstr competition.
4. **Contrastive scale loss**: require generated numbers to match the scale distribution of training data (Wasserstein/MMD on marginals).

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
