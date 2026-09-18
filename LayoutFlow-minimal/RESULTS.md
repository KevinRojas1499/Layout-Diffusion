# Results log: whole-element masking and variable-length generation

Raw material for the paper. Everything here was measured on 2026-09-17 from the runs under
`$LAB/runs/layoutflow-minimal/` (wandb project `kevinrojas1499/LayoutFlow-minimal`, same run names).
Code: branch `varlen-element-masking`, commit `fd11590`.

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

Noise: repeating generation gives about +/-0.05 FID on RICO and +/-0.2 on PubLayNet (3 repeats).

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
