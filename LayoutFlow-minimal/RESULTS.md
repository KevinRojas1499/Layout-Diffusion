# Results log: whole-element masking and variable-length generation

Raw material for the paper. Everything here was measured on 2026-09-17 from the runs under
`$LAB/runs/layoutflow-minimal/` (wandb project `kevinrojas1499/LayoutFlow-minimal`, same run names).
Code: branch `varlen-element-masking` (results up to commit `fd11590`; sampler sweep at `4ee600e`).

## Evaluation protocol

Two different FIDs appear below; do not mix them.

- **Paper protocol** (what LayoutFlow / LayoutDM / LayoutDiffusion report; `test.py`): real statistics
  from the **test split** (RICO: the continuous boxes in `pretrained/rico_test.pt`; PubLayNet: the h5
  test split); **2,000 generated layouts**, first 2,000 of a pass over the test loader; number of
  elements per layout drawn from the training-set length table in `conf/test.yaml` (our variable-length
  model ignores it and generates its own); LayoutGAN++ FID network with the LayoutDM weights
  (`fid_{rico,publaynet}.pth.tar`); **single run**, no averaging (the LayoutFlow paper reports no std
  and no averaging; `multirun` is only an option of its code release). Alignment is printed x100.
  Sanity check: real validation layouts scored against the test statistics give **2.10 (RICO)** and
  **8.10 (PubLayNet)**, identical to the "validation data" rows of LayoutFlow's Table 1.
- **In-training validation FID** (`FID_Layout` in wandb / checkpoint names): real statistics from the
  **validation split**, all validation layouts generated. Much lower on PubLayNet (val and test differ)
  and optimistic everywhere when quoted as the best of many validations. Only for model selection.

Noise: repeating generation gives a std of about 0.08-0.2 FID on RICO (3-5 repeats, see the sampler sweep) and ~0.2 on PubLayNet (3 repeats). Single-run numbers, ours and the paper's, carry that uncertainty.

Published numbers (LayoutFlow, ECCV 2024, Table 1, unconditional): LayoutFlow RICO **2.37**,
PubLayNet **8.87**; LayoutDiffusion 2.49 / 8.63; LayoutDM (retrained) 4.43 / 36.85. The baselines are
given the number of elements; ours (VarLen) generates it.

## RICO, unconditional, paper protocol

Checkpoint = best in-training validation FID of each run (2000 epochs, validation every 25).

| Model | FID | Alignment x100 | Overlap | run |
|---|---|---|---|---|
| Validation data (reference) | 2.10 | | | |
| LayoutFlow, paper | 2.37 | 0.150 | 0.498 | |
| LayoutFlow, upstream checkpoint, upstream `test.py` | 2.24 | 0.143 | 0.538 | `$LAB/repos/LayoutFlow/checkpoints/wgdgmm5x`, epoch 2349 |
| LayoutDiffusion, paper | 2.49 | | | |
| **Unmasking (ElemMask), GMM, t_max=0.5** | **2.30** | 0.205 | 0.512 | `rico-elemmask-gmm-t050`, ep 1549 |
| Unmasking, mean head, t_max=0.5 | 2.39 | 0.235 | 0.509 | `rico-elemmask-mean-t050`, ep 1874 |
| Unmasking, single Gaussian, t_max=0.5 | 2.39 | 0.223 | 0.532 | `rico-elemmask-gauss1-t050`, ep 1749 |
| Unmasking, GMM / mean / single Gaussian, t_max=1.0 | 3.72 / 5.66 / 12.79 | 0.391 / 0.291 / 0.366 | 0.547 / 0.548 / 0.555 | `rico-elemmask-*-t100` |
| **Variable length (VarLen), GMM, t_max=0.5** | **2.43** | 0.233 | 0.509 | `rico-varlen-gmm-t050`, ep 1474 |
| Variable length, single Gaussian, t_max=0.5 | 2.51 | 0.277 | 0.522 | `rico-varlen-gauss1-t050`, ep 1974 |
| Variable length, GMM, t_max=0.25 | 2.55 | 0.240 | 0.497 | `rico-varlen-gmm-t025`, ep 1599 |
| Variable length, GMM, t_max=1.0 | 11.94 | 0.269 | 0.646 | `rico-varlen-gmm-t100`, ep 1949 |
| Reimplemented LayoutFlow (this repo's `LayoutFlow`) | 2.80 | 0.166 | 0.522 | `rico-sota-check` (weak run; use the upstream checkpoint as the baseline) |

Takeaways: fixed-length unmasking is at LayoutFlow's level (2.30 vs 2.24-2.37); generating the
length costs 0.13 FID (2.43); alignment is consistently worse than LayoutFlow (0.20-0.23 vs 0.14-0.15)
while overlap is on par; t_max=1.0 (reveals spread over the whole trajectory) is 3-5x worse.

### In-training validation FID (RICO), for the ablation and convergence story

"Best" = best of 80 validations; "late mean" = mean of the last 10 (epochs 1750-1999), the fairer
number (best-of-80 is optimistic by 0.3-0.4 for every run). "first <= x" = epoch of the first
validation at or below x. LR: ReduceLROnPlateau (patience 10 validations, factor 0.1) from 5e-4.

| Head | Best (epoch) | Late mean | first <= 3.0 | first <= 2.5 | LR drops (epoch) |
|---|---|---|---|---|---|
| **Fixed length, t_max=0.5** | | | | | |
| GMM (16 comp.) | **2.02** (1549) | **2.47** | **699** | **749** | 676, 1026, 1301, 1826 |
| mean (MSE) | 2.22 (1874) | 2.50 | 974 | 1299 | 1251, 1751 |
| single Gaussian | 2.34 (1749) | 2.62 | 1374 | 1524 | 1351 |
| **Fixed length, t_max=1.0** | | | | | |
| GMM | 3.27 | 3.68 | never | never | 851, 1176, 1451, 1726 |
| mean | 4.76 | 5.42 | never | never | 1251, 1576, 1851 |
| single Gaussian | 11.12 | 12.45 | never | never | 1376, 1726 |
| **Variable length, t_max=0.5** | | | | | |
| GMM | 2.46 (1474) | **2.71** | **999** | **1474** | 976, 1451, 1751 |
| single Gaussian | 2.45 (1974) | 2.89 | 1174 | 1924 | 1001, 1901 |
| **Variable length, t_max=0.25**, GMM | 2.44 (1599) | 2.78 | 1074 | 1599 | 1051, 1876 |
| **Variable length, t_max=1.0**, GMM | 10.38 | 12.36 | never | never | 926 |

Ablation conclusions: GMM > mean >= single Gaussian in final quality; the GMM converges ~2x faster
(FID <= 2.5 at epoch 749 vs 1299 / 1524, i.e. ~46k vs 81k / 94k steps) and is the only head the
scheduler fully annealed. Sampling from a unimodal Gaussian is worse than not sampling (mean), and
collapses when reveals are late (t_max=1.0: 11-13): what the mixture buys is commitment to one mode,
not stochasticity as such. Epochs are 62 steps (batch 512, 31,694 layouts).

Reference (earlier work, same validation protocol, 5 seeds): continuous LayoutFlow 2.32 +/- 0.12,
LayoutFlowDiscrete (flow geometry + masked-diffusion category, cat_loss_weight 0.25) 2.07 +/- 0.11.

## PubLayNet, unconditional, paper protocol (runs stopped early)

Runs used the same grid (1000-epoch cap, validation every 3 epochs = 1,827 steps) and were **stopped
at epoch 236-299 of 1000** with validation FID still falling. Numbers are therefore lower bounds on
what the recipe does.

| Model | FID | Alignment x100 | Overlap | run |
|---|---|---|---|---|
| Validation data (reference) | 8.10 | | | |
| LayoutDiffusion, paper | 8.63 | | | |
| LayoutFlow, paper | 8.87 | | | |
| **Variable length, GMM, t_max=0.5** | **9.05** | 0.145 | 0.013 | `publaynet-varlen-gmm-t050`, ep 299 |
| Unmasking, GMM, t_max=0.5 | 9.81 | 0.072 | 0.018 | `publaynet-elemmask-gmm-t050`, ep 257 |
| Unmasking, mean, t_max=0.5 | 10.08 | 0.074 | 0.015 | `publaynet-elemmask-mean-t050`, ep 239 |
| Variable length, GMM, t_max=0.25 | 10.30 | 0.142 | 0.014 | `publaynet-varlen-gmm-t025`, ep 236 |

In-training validation FID at the stop (best): ElemMask gmm 0.54, mean 0.57, VarLen t050 0.65,
t025 0.84; t_max=1.0 runs 2.8 (ElemMask gmm), 6.5 (ElemMask mean), 25.6 (VarLen). Epochs are 609
steps (311,397 layouts). Earlier snapshot for the curve: VarLen t025 at epoch 128 had val 1.10 and
test 10.82 +/- 0.18.

## Generated length and class statistics (best VarLen checkpoints, 2,048 samples, seed 0)

| | mean elements, real val | generated | empty layouts | class-share max deviation |
|---|---|---|---|---|
| RICO, t_max=0.25 (ep 1599) | 9.72 | 9.31 | 3 / 2048 | 0.8 pt (class 4: 2.9 -> 3.7%) |
| PubLayNet, t_max=0.25 (ep 128) | 9.44 | 9.39 | 0 / 2048 | 1.4 pt (title: 19.0 -> 20.4%) |

RICO comes out 4% short because the reveal step on which kappa reaches 1 skips its insertions
(`p < 1.0` guard in `inference`); about `1 / (t_max * inference_steps)` of the elements are never
inserted, and 19-20 element layouts are under-produced on both datasets. Poisson insertion itself
is kept as the numerical method. Qualitative samples (real vs generated grids, length histograms,
insert -> unmask -> denoise filmstrips): https://claude.ai/artifact/A73RTGxs3EcKQWigdJgCxX ;
regenerate with `scripts/sample_layouts.py`. Observations: page/screen structure is right
(column flows, titles on their blocks, top toolbars, full-width rows); flaws are local overlaps and
slightly ragged edges; with t_max=0.25 the element count and classes are fixed while boxes are
still 75% noise, and the recognisable layout forms in the denoise-only phase.

## Caveats to carry into the paper

- RICO class names for `ldm_rico_*.h5` are unknown: upstream's `TYPE_2_CAT` table does not match
  the ids (its "Image" is the top toolbar). PubLayNet names (1 text, 2 title, 3 list, 4 table,
  5 figure) match the box shapes.
- The discrete-category baseline checkpoints (`rico-fmdisc-*`) predate the squash of their branch
  and cannot be evaluated with today's configs (rebuilt backbone gives FID 116).
- Layouts are treated as sets (no positional encoding; hidden slots are exactly equivalent to a
  compacted sequence, verified numerically). Insertion is therefore one rate per layout, not per gap.

## Sampler experiments (in progress)

Inference-only changes on the committed checkpoints, evaluated on the paper protocol; see the
section appended below as they are run.

### Sampler sweep on the RICO t_max=0.5 checkpoints (paper protocol, mean +/- std over repeats)

Inference-only knobs of `LayoutFlowVarLen.inference` (`+sampling.<key>=<value>` in `test.py`; defaults
reproduce the committed sampler bit-for-bit). Insertion stays Poisson. gmm_temp scales the mixture
component's sigma at reveal (0 = component mean); reveal_eps scales the fresh noise mixed into the
revealed box; heun = second-order steps once everything is revealed; cfg_w = classifier-free
guidance on the categories using the mask token as the null condition (**untrained** here: the
checkpoints never saw visible boxes with hidden categories); snap_grid rounds final ltrb edges.

3 repeats:

| checkpoint | sampler | FID | Alignment x100 | Overlap |
|---|---|---|---|---|
| varlen | baseline | 2.6189 +/- 0.1327 | 0.2553 +/- 0.0348 | 0.5232 +/- 0.0140 |
| varlen | gmm_temp=0.7 | 2.6808 +/- 0.2208 | 0.2512 +/- 0.0402 | 0.5073 +/- 0.0023 |
| varlen | gmm_temp=0.5 | 2.5984 +/- 0.1071 | 0.2333 +/- 0.0216 | 0.5147 +/- 0.0065 |
| varlen | gmm_temp=0.3 | 2.5389 +/- 0.0560 | 0.2645 +/- 0.0377 | 0.5078 +/- 0.0055 |
| varlen | gmm_temp=0.0 | 2.5577 +/- 0.1007 | 0.2420 +/- 0.0600 | 0.5129 +/- 0.0125 |
| varlen | reveal_eps=0.7 | 5.1187 +/- 0.2417 | 0.2416 +/- 0.0681 | 0.5041 +/- 0.0129 |
| varlen | reveal_eps=0.5 | 8.3352 +/- 0.5819 | 0.2500 +/- 0.0585 | 0.5302 +/- 0.0024 |
| varlen | solver=heun | 2.4407 +/- 0.0719 | 0.2788 +/- 0.0685 | 0.5136 +/- 0.0038 |
| varlen | cfg_w=1.5 | 4.6118 +/- 0.2014 | 0.1865 +/- 0.0307 | 0.4103 +/- 0.0081 |
| varlen | cfg_w=2.0 | 12.9036 +/- 0.1674 | 0.2159 +/- 0.1067 | 0.1750 +/- 0.0051 |
| varlen | snap_grid=32 | 7.0427 +/- 0.1318 | 0.1952 +/- 0.0189 | 0.4960 +/- 0.0035 |
| varlen | snap_grid=64 | 3.1738 +/- 0.0310 | 0.1875 +/- 0.0190 | 0.5115 +/- 0.0032 |
| varlen | snap_grid=128 | 2.5945 +/- 0.1362 | 0.1897 +/- 0.0192 | 0.5098 +/- 0.0147 |
| elemmask | baseline | 2.4333 +/- 0.0176 | 0.2362 +/- 0.0483 | 0.5166 +/- 0.0043 |
| elemmask | gmm_temp=0.7 | 2.3857 +/- 0.0296 | 0.2502 +/- 0.0315 | 0.5106 +/- 0.0011 |
| elemmask | gmm_temp=0.5 | 2.4978 +/- 0.0845 | 0.2801 +/- 0.0183 | 0.5154 +/- 0.0078 |
| elemmask | gmm_temp=0.3 | 2.4714 +/- 0.0792 | 0.2278 +/- 0.0334 | 0.5152 +/- 0.0023 |
| elemmask | gmm_temp=0.0 | 2.2623 +/- 0.0353 | 0.2172 +/- 0.0156 | 0.5334 +/- 0.0070 |
| elemmask | reveal_eps=0.7 | 4.8297 +/- 0.3732 | 0.2122 +/- 0.0666 | 0.5278 +/- 0.0086 |
| elemmask | reveal_eps=0.5 | 8.4703 +/- 0.2117 | 0.1927 +/- 0.0155 | 0.5492 +/- 0.0044 |
| elemmask | solver=heun | 2.3857 +/- 0.1976 | 0.2403 +/- 0.0264 | 0.5060 +/- 0.0125 |
| elemmask | cfg_w=1.5 | 3.4517 +/- 0.0065 | 0.1894 +/- 0.0262 | 0.4900 +/- 0.0091 |
| elemmask | cfg_w=2.0 | 4.9629 +/- 0.1469 | 0.1875 +/- 0.0626 | 0.4926 +/- 0.0113 |
| elemmask | snap_grid=32 | 6.8527 +/- 0.0593 | 0.1985 +/- 0.0042 | 0.5028 +/- 0.0071 |
| elemmask | snap_grid=64 | 3.2509 +/- 0.1636 | 0.2100 +/- 0.0518 | 0.5100 +/- 0.0079 |
| elemmask | snap_grid=128 | 2.6341 +/- 0.1408 | 0.2241 +/- 0.0304 | 0.5060 +/- 0.0037 |

5 repeats, combinations:

| checkpoint | sampler | FID | Alignment x100 | Overlap |
|---|---|---|---|---|
| varlen | baseline5 | 2.5522 +/- 0.0814 | 0.2407 +/- 0.0574 | 0.5131 +/- 0.0097 |
| elemmask | baseline5 | 2.4138 +/- 0.0776 | 0.2713 +/- 0.0845 | 0.5151 +/- 0.0072 |
| varlen | heun+temp0 | 2.5764 +/- 0.2085 | 0.2647 +/- 0.0476 | 0.5039 +/- 0.0077 |
| elemmask | heun+temp0 | 2.3823 +/- 0.1525 | 0.2194 +/- 0.0330 | 0.5066 +/- 0.0060 |
| varlen | heun+temp0.3 | 2.6187 +/- 0.1539 | 0.2489 +/- 0.0158 | 0.5124 +/- 0.0053 |
| elemmask | heun+temp0.3 | 2.3986 +/- 0.1181 | 0.2039 +/- 0.0137 | 0.5175 +/- 0.0118 |
| varlen | heun+temp0+snap128 | 2.7460 +/- 0.1574 | 0.2404 +/- 0.0594 | 0.5119 +/- 0.0147 |
| elemmask | heun+temp0+snap128 | 2.5871 +/- 0.0702 | 0.2443 +/- 0.0497 | 0.5091 +/- 0.0108 |

Conclusions: (1) the repeat-to-repeat std of the paper protocol is 0.08-0.2 FID on RICO, larger than
the +/-0.05 quoted above, so the single-run numbers in the tables above (2.30 / 2.43) sit on the
lucky side of their 5-run means (2.41 / 2.55) and the paper's own single-run 2.37 carries the same
uncertainty; (2) no inference-only knob beats the baseline beyond that noise: Heun and gmm_temp=0
looked like -0.15 at 3 repeats and vanished at 5; (3) shrinking the reveal noise (reveal_eps < 1)
and coarse snapping (grid 32/64) are clearly harmful, i.e. the revealed box must sit on the
training path and the real RICO test boxes are continuous; (4) untrained CFG lowers alignment and
overlap (the layouts get "cleaner") but moves the distribution away from the data (FID 3.5-13);
the trained version (`model.cat_drop=0.1`, then `+sampling.cfg_w`) is the open question.

## Classifier-free guidance on the categories (RICO, paper protocol, 5 repeats)

Retrained with `model.cat_drop=0.1` (every category of a layout hidden w.p. 0.1 during training), t_max=0.5, then
`+sampling.cfg_w=w` at test time: `v = v_u + w (v_c - v_u)`, the null condition being the mask token on every category.

| w | Unmasking (`rico-elemmask-gmm-t050-catdrop10`, ep 1924) | Variable length (`rico-varlen-gmm-t050-catdrop10`, ep 1099) |
|---|---|---|
| 0.75 | 5.97 +/- 0.24 | 5.42 +/- 0.35 |
| **1.0 (off)** | **2.46 +/- 0.15** | **2.46 +/- 0.09** |
| 1.25 | 2.39 +/- 0.04 | 2.52 +/- 0.12 |
| 1.5 | 2.60 +/- 0.06 | 2.74 +/- 0.15 |
| 2.0 | 3.41 +/- 0.17 | 3.52 +/- 0.16 |
| 3.0 | 4.32 +/- 0.23 | 4.50 +/- 0.21 |

Conclusions: category dropout costs nothing (w = 1 matches the no-dropout runs: 2.41 / 2.55 five-run means), but
guidance does not help: w = 1.25 is within noise (-0.07 / +0.06) and anything stronger hurts monotonically.
Overlap drifts up and alignment does not improve either. The category signal is already fully used by the
conditional velocity; sharpening it moves the samples off the data distribution. Not worth keeping on.

## Conditional tasks with `model.cond=random4` (RICO, paper protocol, full test set, max-IoU)

LayoutFlow paper: C->S+P 1.48 / mIoU 0.322, C+S->P 1.03 / 0.470, completion 1.51 / 0.741 (unconditional 2.37).

| Model | uncond (5 runs) | C->S+P (`cat_cond`) | C+S->P (`size_cond`) | completion (`elem_compl`) |
|---|---|---|---|---|
| Unmasking, uncond-only training (`rico-elemmask-gmm-t050`) | 2.41 +/- 0.08 | **1.26** / 0.348 | 2.92 / 0.389 | 3.77 / 0.586 |
| Unmasking, random4 (`rico-elemmask-gmm-t050-random4`, ep 1699) | 2.63 +/- 0.09 | 1.59 / 0.344 | **1.45** / 0.434 | **2.50** / 0.604 |
| Variable length, uncond-only (`rico-varlen-gmm-t050`) | 2.55 +/- 0.08 | 1.58 / 0.335 | 3.46 / 0.385 | 7.05 / 0.460 (length given) |
| Variable length, random4 (`rico-varlen-gmm-t050-random4`, ep 1224) | 4.68 +/- 0.21 | 2.21 / 0.302 | 3.29 / 0.386 | 6.97 / 0.517 (length given); 5.86 / 0.488 (length generated) |
| Variable length, random4 + `cond_input` (`rico-varlen-gmm-t050-random4-condin`, ep 1449) | 3.10 +/- 0.27 | **1.37** / 0.337 | **1.15** / 0.442 | 3.73 / 0.540 (length given); 2.70 / 0.527 (length generated) |

Conclusions: (1) category conditioning is native, the uncond-only unmasking model already beats LayoutFlow on
C->S+P (1.26 vs 1.48) with no conditional training; (2) the random4 mix makes size-conditioning and completion
work for the fixed-length model (2.92 -> 1.45, 3.77 -> 2.50) at a 0.2 unconditional cost, but still short of
LayoutFlow (1.03 / 1.51); (3) **the variable-length random4 run is broken**: its unconditional FID collapses to
4.68 and its validation FID never went below 4.8. Diagnosis: the backbone never sees *which* elements are
given. In the cat_cond / size_cond quarters every element is visible from t = 0 with nothing missing, in the
unconditional quarter a state with the same number of visible elements has elements still to insert, and the
insertion rate (and the unmask heads) cannot tell the two apart. LayoutFlow feeds its cond_mask to the network
as an input embedding; we must do the same (a per-element given / per-coordinate held indicator) before
random4 training can be trusted for the insertion model. The fixed-length model is less exposed because it has
no insertion rate to confuse, which is why it degrades only mildly.

**Fix (commit 121340d, `model.backbone_model.cond_input=true`):** the given / held mask is embedded into the
element tokens, as LayoutFlow does. Retrained: unconditional 4.68 -> 3.10, C->S+P 2.21 -> 1.37 (paper 1.48),
C+S->P 3.29 -> 1.15 (paper 1.03), completion with its own length 5.86 -> 2.70 (paper 1.51, length given).
One variable-length model now serves all tasks; the remaining unconditional cost of the mix (3.10 vs 2.55) is
larger than the fixed-length model's (2.63 vs 2.41) and is the open item.

### Refinement (inference only, no training): `test.py task=refinement`

Upstream's protocol: perturb the test layouts with N(0, 0.01) in [0,1] box space, keep the categories, integrate
the flow from t = 0.97 to 1. In our model that start state is an ordinary point of the unconditional training
path (everything is visible after t_max), so any checkpoint does it. The noisy input itself scores FID ~64
against the test split (the FID net is extremely sensitive to sub-grid jitter).

| Checkpoint | FID | mIoU |
|---|---|---|
| LayoutFlow paper | **0.77** | **0.700** |
| Unmasking, uncond-only (`rico-elemmask-gmm-t050`) | 1.10 | 0.679 |
| Variable length, uncond-only (`rico-varlen-gmm-t050`) | 1.15 | 0.679 |
| Unmasking, random4 | 1.38 | 0.676 |
| Variable length, random4 + cond_input | 1.30 | 0.665 |

Start-time sweep on the unmasking checkpoint (test split, so *not* used for selection): t_start 0.90 -> 1.25 / 0.546,
0.95 -> 0.98 / 0.632, 0.97 -> 1.10 / 0.679, 0.985 -> 2.47 / 0.708, 0.995 -> 27.3 / 0.703. FID and fidelity trade
off; no setting dominates LayoutFlow's point.

## Content-aware generation on CGL (RALF release, RALF's evaluator, test split)

Setup: `src.data.CGL` + frozen DINOv2-small canvas tokens (8x8 grid + saliency, `scripts/precompute_canvas_feats.py`)
prepended to the element set; scored by exporting samples in RALF's format (`scripts/export_ralf_samples.py`) and
running their untouched `eval.py` (reproduces their published scores to 4 decimals). LayoutGD (ICMR 2026) Table 1
reports exactly these five columns. References on CGL: LayoutGD 0.120 / 0.0140 / 0.999 / 0.994 / 0.0004,
LayoutDiT 0.124 / 0.0157 / 0.995 / 0.988 / 0.0016, RALF 0.126 / 0.0180 / 0.992 / 0.978 / 0.0042 (Occ / Rea / Und_l /
Und_s / Ove); RALF FID 1.32 (real val vs test floor 0.80); count MAE from RALF's released outputs: RALF 1.08,
autoregressive 1.27, canvas-blind histogram 2.13.

### v2 runs (1000 epochs, val FID = ported FIDNetV3 vs RALF's val features, EMA on; the v1 runs had no FID signal)

| checkpoint | FID | Occ | Rea | Und_s | Ove | count MAE |
|---|---|---|---|---|---|---|
| variable length, canvas-blind, ep 649 | **1.63** | 0.318 | 0.041 | **0.877** | **0.006** | 2.13 |
| variable length + canvas, ep 199 / 419 / 999 | 5.55 / 4.45 / 5.08 | 0.136 | 0.021 | 0.57 / 0.71 / 0.72 | 0.021 / 0.013 / 0.011 | 1.42 / 1.34 / 1.33 |
| padding baseline + canvas, ep 489 / 999 | 4.61 / 5.20 | 0.136 | 0.021 | 0.74 / 0.76 | 0.011 / 0.012 | 1.29 / 1.28 |

**Diagnosis: the canvas pathway memorises the training canvases.** Generated on 6,000 *training* canvases the canvas
model scores FID 1.15 (vs train features) / 1.53 (vs test), better than the blind model (1.58 / 1.89); on unseen test
canvases it collapses to 4.45 while occlusion and count stay good. The blind model learns an excellent layout prior
(FID 1.6, underlay 0.88, overlap 0.006) but ignores the canvas (occlusion 0.32, chance-level count). The padding
baseline shows the same collapse, so it is the conditioning, not insertion. EMA is not the cause (v1 without EMA:
3.5-4.3). The validation FID was right all along; the v1 "blind 3.99" was an epoch-129 checkpoint.

v3 (launched 2026-09-18 16:40, EMA off): `model.ctx_token_drop=0.5 model.ctx_drop=0.1` (hide half the canvas tokens per
sample, the whole canvas for 10%), `dataset.dataset.hflip=true` (mirror canvas grid + layout), and both.
