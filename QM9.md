# Variable-Length Diffusion — QM9 Training Loop

## Goal

Raise **`avg_ks`** on QM9 across the late checkpoints of a training run, trained by `qm9_training.py`. The metric is the average of 10 Kolmogorov–Smirnov-derived scores — 4 over molecular properties (`mw`, `logp`, `hbd`, `hba`) and 6 over atom counts (`total`, `C`, `H`, `N`, `O`, `F`) — between generated and real QM9 molecules. **Higher is better; 1.0 means perfect distributional match, 0.0 means no overlap.** (`compute_ks_statistic` in `eval/evaluate_distribution.py` returns `1 - KSD`.) Secondary metric: **`valid_frac`** (higher is better) — it both indicates sample quality and controls how much sampling noise contaminates `avg_ks`.

Metrics are written after every eval to `{--dir}/metrics.jsonl` (one JSON object per line) with fields: `ks_mw`, `ks_logp`, `ks_hbd`, `ks_hba`, `ks_atoms`, `ks_C`, `ks_H`, `ks_N`, `ks_O`, `ks_F`, `avg_ks`, `n_valid`, `n_total`, `valid_frac`, `iteration`, `lr`.

### Caveat on what `avg_ks` is computed over

`avg_ks` is currently computed over **only the chemically valid subset** of generated molecules (the `n_valid` field). This is a known limitation — ideally the metric would be over all generated samples — but we are not changing the code for now. Practical consequence: the sample size for `avg_ks` shifts checkpoint-to-checkpoint as `valid_frac` moves, which adds structural variance on top of pure sampling noise. **Always read `avg_ks` together with `valid_frac` and `n_valid`.** A jump in `avg_ks` while `n_valid` is small (< 200) is mostly noise.

### Judgment metric (what the loop optimizes)

Use the **mean of `avg_ks` over the last K=3 eval checkpoints** of a run, not a single best point. A single checkpoint can swing 0.05–0.10 from sample-size noise alone (see "Why variance is high" below); a window average is the only honest signal that a config change actually helped.

A new config is promoted only if its late-window mean is **higher** than the current best by **> 0.01**. Differences within ±0.01 are ties.

---

## How to run training

The loop has two stages:

**Stage 1 (concluded): 15k-iter rapid-iteration trials.** Used to discover which knobs help: validated wins are Muon optimizer and `--warmup_iters 500`. See "Tried and rejected" for what didn't work and why.

**Stage 2 (in progress): 100k-iter production-scale runs.** Stage 1 showed the model was monotonically improving at iter 15000 — length is the next lever. We're using the validated stage-1 config and only extending `--num_iters`. From here on, "current best" is established at the 100k scale, and further trials should be at 100k unless we discover something cheap to test that justifies dropping back to 15k.

Stage-2 launch command (the one in progress at `runs/qm9/transformer-muon-w500-100k`):

```bash
uv run torchrun --nproc_per_node=1 qm9_training.py \
  --model Transformer \
  --optimizer muon \
  --dir runs/qm9/<run-name> \
  --num_iters 100000 \
  --log_rate 10000 \
  --lr 1e-4 \
  --batch_size 128 \
  --warmup_iters 500 \
  --lr_schedule cosine \
  --ema_beta 0.999 \
  --eval_samples 1000 \
  --eval_steps 100 \
  --eval_sampler staggered
```

`uv run torchrun` is required on this machine — `torchrun` is in the project venv at `.venv/bin/torchrun`. Approx wall time on the available RTX PRO 4000 Blackwell: ~5 hours for 100k. Note the **reverted recommendations** vs the old stage-1 example: `--eval_use_ema` is off (it hurt under cosine in our tests), `--ema_beta` stays at 0.999 (0.9999 crashes early evals when n_valid hits 0), and `--eval_samples` stays at 1000 (n_valid is reliably > 200 at our quality level).

To resume from a checkpoint (eval-side tuning only — model unchanged):

```bash
uv run torchrun --nproc_per_node=1 qm9_training.py \
  --dir runs/qm9/<run-name> \
  --load_checkpoint runs/qm9/<base-run>/itr_<N>/snapshot.pt \
  --num_iters <N+...> ...
```

A 15k stage-1 trial takes ~40 min on this GPU; a 100k stage-2 run takes ~5h. Full QM9 production runs in the original paper were ~250k iters.

---

## Dataset

- **Source**: `custom_datasets/qm9.py` (`QM9Dataset`) — already cached on disk.
- **Vocab**: `{H, C, N, O, F}` plus mask/pad/bos tokens.
- **Real-side eval cache**: `eval/cache/qm9_real_eval_cache.pkl.gz` is built once on first run and reused — do not delete unless the dataset itself changes.
- The model jointly generates atom-symbol sequences and 3D positions.

---

## Model and architecture

Two architectures are wired up in `qm9_training.py`. **For loop trials, default to `--model Transformer`** — MMDiT runs were too slow on available hardware to iterate on. The MMDiT baseline below is kept for reference only; it is not directly comparable to Transformer numbers, so the Transformer rebaseline is the new reference for promotion decisions.

**Transformer** (preferred for trials, `--model Transformer`) — `models/transformer.py`. Current config:
- `dim=384`, `depth=12`, `num_heads=12`, `head_dim=64`

**MMDiTQM9** (`--model DiT`) — `models/mmdit_qm9.py`. Current config:
- `depth=6` joint blocks, `symbols_depth=6`, `positions_depth=6`
- `dim_modalities=[384, 384]`, `dim_joint_attn=384`, `dim_conds=[384, 384]`

Architecture changes are fair game in the loop. To change them, edit the `MMDiTQM9(...)` / `Transformer(...)` construction in `qm9_training.py` lines ~219–238. Suggested sweeps:
- MMDiT: `depth ∈ {6, 8}`, `dim_modalities ∈ {[384,384], [512,512]}`, `dim_joint_attn` matches.
- Transformer: `depth ∈ {12, 16}`, `dim ∈ {384, 512}`. Smaller and faster than MMDiT — useful for quick architecture probes.

---

## Why variance is high (and how to attack it)

Three structural sources, in order of impact:

1. **`avg_ks` is computed on the *valid subset only*.** At baseline, `n_valid` was 128–323 out of 1000. The KS-derived score has standard deviation ~0.05 per sub-metric on n=128 — alone enough to explain the 0.77 → 0.86 → 0.80 swing in the baseline. **Mitigation**: raise `--eval_samples` (3000+), and improve model quality so `valid_frac` rises (which raises `n_valid` proportionally).
2. **Eval uses raw model weights, not EMA, by default.** `--eval_use_ema` is off in the script. Raw weights move with optimizer noise; EMA averages them. **Mitigation**: always pass `--eval_use_ema` and bump `--ema_beta` from 0.999 → 0.9999 so the EMA actually smooths something on a 15k-iter horizon.
3. **Single eval seed.** All sampler stochasticity collapses to one realization per checkpoint. **Mitigation (optional)**: run two trials with different `--eval_seed` and average their late-window means, or modify the eval block to loop over a few seeds.

**Order of operations**: lock down (1)–(3) first on the baseline config, see what the residual variance looks like, *then* compare architecture changes. Comparing architectures while metrics are noisy at ±0.05 is wasted compute.

---

## Training loop knobs to tune

Roughly in order of expected payoff:

| Parameter | Current default | Recommended starting point | Notes |
|-----------|----------------|----------------------------|-------|
| `--eval_use_ema` | False | **True** | Cheapest variance reduction. Always on. |
| `--ema_beta` | 0.999 | **0.9999** | Slow EMA needed for late-window smoothing. |
| `--eval_samples` | 1000 | **3000** | KS noise drops as ~1/√N on the valid subset. |
| `--log_rate` | 5000 | **2500** | More eval points → more stable late-window mean. |
| `--warmup_iters` | 100 | **500–1000** | Cosine + short warmup is unstable at lr=1e-4. |
| `--lr_schedule` | cosine | cosine | Already good; keep. |
| `--lr` | 1e-4 | 1e-4, try 3e-4 | With Muon try 3e-4 (Muon's adam group). |
| `--optimizer` | adam | try **muon** | Has been a clear win in similar setups; uses `model.get_muon_adam_params()`. |
| `--num_iters` | 15000 | 15000 (trial) | Don't change for trials — keeps comparisons honest. |
| `--batch_size` | 128 | 128 | Larger if memory allows; gradient noise drops. |
| `--train_only_dsm` | False | False | Only flip if discrete losses are clearly hurting. |

**Knobs to *not* tune in the loop** (per project decision — improvements are predictable, not informative):
- `--eval_steps` — keep at 100.
- `--eval_sampler` — keep at `staggered`.

**Architecture levers** (edit `qm9_training.py` model construction):
- MMDiT depth: 6 → 8
- MMDiT widths: [384, 384] → [512, 512] with matching `dim_joint_attn=512`
- Transformer as a faster proxy when iterating on training knobs

---

## Loop workflow

When running in `/loop` mode, each iteration should:

1. **Read metrics**: `cat runs/qm9/<latest-run>/metrics.jsonl` and compute the late-window mean (last 3 entries) of `avg_ks` and `valid_frac`. Also note `n_valid` for those checkpoints — if it's small (< 200) discount the signal.
2. **Diagnose**:
   - Is the late-window mean **higher** than the current best by > 0.01? If yes → promote.
   - Is per-checkpoint variance still high (range > 0.05 within the late window)? If yes → variance fix first, not architecture.
   - Is `valid_frac` < 0.3 in the late window? Then `avg_ks` is mostly noise; prioritize quality knobs (longer warmup, Muon, larger model) over variance knobs.
3. **Decide on one change**: change exactly one thing per trial — variance knob, training knob, *or* architecture knob. Bundling changes makes attribution impossible.
4. **Run a trial** with the new config — at the current stage's iter count (100k for stage 2; 15k only if cheaply revisiting a known-cheap eval-side knob). Resume-from-checkpoint is allowed only for eval-side tuning (the model itself is unchanged); for any architecture or optimizer change, start fresh.
5. **Promote if better**: if the new config's late-window mean is **higher than** the current best by > 0.01, update the "Current best known config" section below.
6. **Log negative results**: keep a one-line note of what you tried and why it didn't help, so the loop doesn't re-try it.

Stage-2 candidate experiments worth running once the 100k baseline lands (in rough priority order):
- **Transformer depth=16 at 100k**: rejected at 15k for overfitting, but the failure pattern was peak-then-regress, which longer training may resolve. Best test of "more capacity" at production scale.
- **Transformer dim=512** (with depth=12): different capacity axis from depth.
- **`--batch_size 256`**: less gradient noise. We have headroom (~800 MB of 24 GB used).
- **`--eval_samples 3000`**: tighter per-checkpoint signal. Cheap given how many evals a 100k run already has.

---

## Current best known config

_(Update this section each loop iteration with the best run found so far.)_

**Architecture experiment result: distance-bias is a tie on `avg_ks`, a +0.036 win on `valid_frac`.** A 250k from-scratch run with `RBFDistanceBias` (`models/transformer.py:90`) — Gaussian RBF features over pairwise ‖r_i − r_j‖ mapped to per-head additive attention-logit bias, zero-init final MLP so the model starts as the plain Transformer.

```
[Distance-bias 250k — tied with plain]
Run dir: runs/qm9/transformer-muon-w500-distbias-250k
Trial length: 250000 iters from scratch (~10.5h on RTX PRO 4000 Blackwell)
Late-window mean avg_ks (iters 230k, 240k, 250k): 0.8947
Late-window mean valid_frac: 0.828
Per-checkpoint avg_ks: [0.8918, 0.8931, 0.8993]
Per-checkpoint n_valid: [828, 820, 837]
Single-best checkpoint: iter 130000 at avg_ks=0.9176, valid_frac=0.800
                      (vs plain best single 0.8967 at iter 200000)
Notes: −0.0009 vs plain late-window 0.8956 (well within tie band).
       valid_frac is consistently +0.03–0.04 ahead of plain across the
       run — the geometric prior IS helping the model produce more
       chemically valid molecules, but the KS distributional match on the
       valid subset is unchanged. Net effect on the metric is neutral.
       The single iter-130k checkpoint is unusually good (0.918 vs the
       plain max of 0.897), worth probing with a full-sampling grid to
       see if it carries through to richer eval.
```

**Stage-2 best (current comparison reference): 250k = 100k baseline + 150k continuation.** Transformer + Muon + warmup=500.

```
[Stage-2 250k extension — current comparison reference]
Run dir: runs/qm9/transformer-muon-w500-250k-v2 (resumed from 100k)
Trial length: 250000 iters total (~6.6h for the +150k extension on
              RTX PRO 4000 Blackwell)
Late-window mean avg_ks (iters 230k, 240k, 250k): 0.8956   (higher = better)
Late-window mean valid_frac: 0.792
Per-checkpoint avg_ks: [0.8964, 0.8948, 0.8955]   range = 0.002 ← fully converged
Per-checkpoint n_valid: [786, 794, 797]
Single-best checkpoint: iter 200000 at avg_ks=0.8967, valid_frac=0.748
Config: same as the 100k baseline below (architecture and training-time
        knobs unchanged) — only --num_iters extended from 100k to 250k.
Notes: +0.012 avg_ks vs 100k (above the 0.01 promotion bar); range
       collapsed from 0.012 → 0.002 (model has converged). The last 5 evals
       (210k–250k) span [0.8948, 0.8967], a 0.002-wide band. Further
       training is unlikely to give meaningful gains at this config.
       valid_frac plateau ~0.79; n_valid ~790 means the metric signal
       itself is very tight.

       *How the resume worked.* `--load_checkpoint
       runs/qm9/transformer-muon-w500-100k/itr_100000/snapshot.pt
       --num_iters 250000` re-enters training at iter 100001 with the
       cosine scheduler reconfigured for the new horizon (T_max → 249500,
       _step_count = 1, optimizer lrs reseeded from closed form). See the
       resume-with-extension gotcha note in "Tried and rejected" — the
       first attempt at this resume produced two evals with lr=1e-6 and
       no learning, which is the reason that note exists.

[Stage-2 100k baseline — reference only, superseded]
Run dir: runs/qm9/transformer-muon-w500-100k
Late-window mean avg_ks: 0.884, valid_frac: 0.749, range: 0.012
Per-checkpoint avg_ks: [0.8787, 0.8835, 0.8910]
Same config as 250k extension; trained from scratch for 100k iters.
```

**Stage-1 final (15k trials; reference only): Transformer + Muon + warmup=500.**

```
[Transformer + Muon + warmup=500 — current comparison reference]
Run dir: runs/qm9/transformer-muon-warmup500
Trial length: 15000 iters (~40 min on RTX PRO 4000 Blackwell)
Late-window mean avg_ks (iters 5000, 10000, 15000): 0.843   (higher = better)
Late-window mean valid_frac: 0.390
Per-checkpoint avg_ks: [0.8382, 0.8441, 0.8455]   range = 0.007  ← excellent stability
Per-checkpoint n_valid: [227, 430, 512]
Config: model=Transformer (dim=384, depth=12, heads=12, head_dim=64),
        optimizer=muon, lr=1e-4 (adam group only; muon group hardcoded 5e-4
        in qm9_training.py:245), warmup=500, schedule=cosine,
        ema_beta=0.999, batch=128,
        eval_use_ema=False, eval_samples=1000, log_rate=5000,
        eval_steps=100, eval_sampler=staggered
Notes: warmup 100→500 added +0.014 avg_ks and cut per-checkpoint range from
       0.021→0.007. avg_ks is now monotonically improving across the run
       (no late-iter plateau like Muon-only had), so the model still has
       room and a larger model or more iters are the natural next levers.

       *Muon lr clarification:* qm9_training.py:245 hardcodes the Muon
       matrix-param lr to 5e-4; --lr only affects the AdamW group. So the
       QM9.md table's "try lr 3e-4 with Muon" only adjusts a minority of
       params and is unlikely to be the gain it suggests.

[Earlier Transformer + Muon — reference only, superseded]
Run dir: runs/qm9/transformer-muon
Late-window mean avg_ks: 0.829, valid_frac: 0.410, range: 0.021
Same config except warmup=100.

[Transformer baseline — superseded]
Run dir: runs/qm9/transformer-baseline
Late-window mean avg_ks: 0.809, valid_frac: 0.269, range: 0.044
Same config except optimizer=adam, warmup=100.
```

```
[MMDiT baseline — reference only, not the comparison point anymore]
Run dir: runs/baseline
Trial length: 15000 iters
Late-window mean avg_ks (iters 5000, 10000, 15000): 0.813   (higher = better)
Late-window mean valid_frac: 0.228
Per-checkpoint avg_ks: [0.7745, 0.8618, 0.8019]  ← peak at 10k, regression at 15k
Per-checkpoint n_valid: [128, 234, 323]          ← small subset → noisy avg_ks
Config: model=DiT (depth=6, dims=[384,384], joint=384),
        optimizer=adam, lr=1e-4, warmup=100, schedule=cosine,
        ema_beta=0.999, batch=128,
        eval_use_ema=False, eval_samples=1000, eval_steps=100, eval_sampler=staggered
```

### Tried and rejected

_(Append one-liners here so the loop doesn't repeat experiments.)_

- **Variance bundle as one trial** (`--eval_use_ema` + `--ema_beta 0.9999` + `--eval_samples 3000` + `--log_rate 2500`, fresh, 15k, MMDiT): **crashed at iter-5000 eval** with `ValueError: No valid fingerprints computed (failed for all 3000 molecules)` from `eval/evaluate_distribution.py`. Root cause: with `ema_beta=0.9999` (half-life ~7k iters) the EMA at iter 5000 is still essentially init noise, so 0/3000 molecules are valid and the fingerprint code throws. Iter-2500 squeaked through with n_valid=5; iter-5000 hit zero. **Don't bundle `eval_use_ema=True` with `ema_beta=0.9999` on a 15k-iter trial without first making the eval robust to 0 valid samples.**
- **MMDiT for trial runs**: aborted (not rejected). On the available GPU (RTX PRO 4000 Blackwell) MMDiT 15k-iter trials take ~2.5h, too slow to iterate. Switched the loop to `--model Transformer`. MMDiT could be revisited for final/production runs.
- **Transformer depth 12→16** (Muon, warmup=500, otherwise current best): late-window mean avg_ks=0.830 (−0.013 vs current best 0.843), valid_frac=0.439 (+0.049), range=0.018. Per-checkpoint pattern [0.822, 0.840, 0.828] — peaks at iter 10000 then *regresses* at iter 15000, unlike depth=12 which is monotonic. Larger model produces more chemically valid molecules but they're distributionally narrower; looks like overfitting at this 15k-iter horizon. Don't bump depth without also extending num_iters.
- **`--warmup_iters 1000`** (Muon, otherwise current best): late-window mean avg_ks=0.8445 (+0.0019 vs current best 0.8426 — within ±0.01 tie band → no promotion), valid_frac=0.387, range=0.021. Per-checkpoint [0.832, 0.852, 0.850] — iter-10000 peak is higher than warmup=500's, but iter-5000 is worse and iter-15000 plateaus, so the late-window mean is a wash. warmup=500 stays the sweet spot.
- **`--eval_use_ema`** with `ema_beta=0.999` (otherwise current best): late-window mean avg_ks=0.8364 (−0.006 vs current best — within tie band but trending down), valid_frac=0.387, range=0.015 (*worse* than the 0.007 with raw eval). Counter to the doc's claim that EMA reduces variance: under cosine-decay schedule the EMA chases the model and the EMA-vs-raw deviation grows over the run, so range actually inflates. Don't enable on cosine-decay short-horizon trials.
- **Resume-with-extension gotcha (fixed):** When resuming a cosine-schedule run with a larger `--num_iters`, three things must be reset together: (a) `_schedulers[1].T_max`, (b) `_step_count = 1` (force closed-form `get_lr` instead of the recurrent formula), and (c) the optimizer's `param_groups[i]['lr']` re-seeded from the closed-form computation. Without (b)+(c), the recurrent formula `new_lr = ratio * (group['lr'] - eta_min) + eta_min` keeps lr stuck at `eta_min` forever because the saved `group['lr']` already equals `eta_min` at end of the original cosine. First attempt at the 100k→250k extension only did (a) and produced two evals with lr=1e-6 and no learning. Second attempt with all three working. Patch is in `qm9_training.py:load_checkpoint`.
- **Transformer depth=16 at 100k**: aborted, not run to completion. Replaced by the 100k→250k extension after concluding (per user input) that more iters of the proven config is more likely to help than re-litigating capacity changes that had already failed at short horizons.

---

## File structure

```
qm9_training.py                  # Training script (this file's target)
custom_datasets/qm9.py           # QM9Dataset
models/mmdit_qm9.py              # MMDiTQM9 architecture (default)
models/transformer.py            # Transformer architecture (faster proxy)
multimodal_interpolant.py        # Forward/backward diffusion process
eval/evaluate_distribution.py    # KS metric implementation + cache helpers
eval/cache/                      # Real-side QM9 eval cache (do not delete)
runs/baseline/                   # Baseline reference run
  metrics.jsonl                  # KS metrics, one JSON per eval
  itr_<N>/snapshot.pt            # Model + EMA + optimizer + scheduler state
runs/qm9/<run-name>/             # New trial runs go here
```
