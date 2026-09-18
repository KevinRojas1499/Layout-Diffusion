# LayoutFlow, minimal

A small, readable, hackable reimplementation of [LayoutFlow](https://julianguerreiro.github.io/layoutflow/)'s
unconditional layout generation recipe (RICO / PubLayNet), meant to be a base for
experimentation: swap the backbone, change the loss, try a different flow trajectory,
etc., and see what moves the needle. It is **not** a replacement for upstream:

- `../LayoutFlow/` is the pristine, unmodified upstream clone (read-only reference).
- `$LAB/repos/LayoutFlow` is a separately-running, unmodified working copy that
  already reproduces the paper's numbers (RICO FID ≈ 2.06-2.26 vs. paper's 2.37) —
  that's the ground truth to diff against, don't touch it.
- This directory is neither of those. It's LayoutFlow's method, restated in ~950
  lines instead of ~2,500, with the parts specific to reproducing our own project's
  gap (`layout_training.py`) stripped out: only unconditional generation is exposed
  as an eval task (training still uses the real `random4` conditioning mix — that's
  core to the method, not something we dropped), only the `AnalogBit` category
  encoding and `stacked` sequence layout are supported (the only combination
  LayoutFlow's own config ever selects), and the wandb/tensorboard-dual-path
  trajectory visualization was replaced with a plain PNG dump.

See each file's module docstring for exactly what was cut and why. Nothing here
silently changes behavior relative to upstream's default recipe — every trim either
removes a code path upstream's own config never exercises, or is called out explicitly.

## Layout

```
train.py, test.py          # entry points (Hydra + PyTorch Lightning, same as upstream)
conf/                       # train.yaml, test.yaml, dataset/{RICO,PubLayNet}.yaml, model/LayoutFlow.yaml
src/
  data.py                   # RICO / PubLayNet h5 datasets + shared collate_fn
  sampler.py                # flow-matching prior (gaussian only)
  backbone.py                # the swappable per-element transformer -- start here to try a new NN
  analog_bit.py              # categorical <-> continuous analog-bit encoding
  models/base.py              # training-loop plumbing (optimizer, cond masks, val/FID loop)
  models/layout_flow.py       # the flow-matching model itself -- start here to try a new loss/trajectory
  models/layout_flow_discrete.py   # flow-matching geometry + masked-diffusion category
  models/layout_flow_varlen.py, backbone_varlen.py   # whole-element masking and insertion (variable length)
  fid.py, metrics.py, visualize.py   # eval: layout FID net, alignment/overlap/mIoU, PNG rendering
scripts/
  launch_varlen_grid.sh       # the 7-run grid behind the variable-length results
  sample_layouts.py           # real vs generated layouts, length/class histograms, trajectory snapshots -> JSON
```

## Running

Needs the same deps as upstream LayoutFlow (torch, lightning, hydra, torchcfm,
torchdyn, h5pickle, einops, pytorch_fid, seaborn). The already-working venv at
`$LAB/repos/LayoutFlow/.venv` has all of them; no need to build a new one.

**Home has a 13GB quota — point `run_dir` and any real data at `$LAB`, never home.**

```bash
LAB=/network/rit/lab/Yelab/kevin-back/kevin_rojas
VENV=$LAB/repos/LayoutFlow/.venv

# RICO (26 categories -> 5 analog bits -> sampler.out_dim must be 9, not the
# model config's PubLayNet-shaped default of 7 -- upstream's own RICO run needs
# the same override, this isn't a bug we introduced)
$VENV/bin/python train.py \
  dataset=RICO dataset_name=RICO \
  dataset.dataset.data_path=$LAB/datasets/LayoutFlow-data/dataset/rico \
  model.sampler.out_dim=9 \
  model.pretrained_dir=$LAB/repos/LayoutFlow/pretrained \
  run_dir=$LAB/runs/layoutflow-minimal/rico-run1

# PubLayNet (6 categories -> out_dim=7, the config default)
$VENV/bin/python train.py \
  dataset=PubLayNet dataset_name=PubLayNet \
  dataset.dataset.data_path=$LAB/datasets/LayoutFlow-data/dataset/publaynet \
  model.pretrained_dir=$LAB/repos/LayoutFlow/pretrained \
  run_dir=$LAB/runs/layoutflow-minimal/publaynet-run1
```

Evaluate a checkpoint (`test.py`, unconditional generation only):

```bash
$VENV/bin/python test.py \
  dataset=RICO dataset_name=RICO \
  dataset.dataset.data_path=$LAB/datasets/LayoutFlow-data/dataset/rico \
  model.sampler.out_dim=9 \
  pretrained_dir=$LAB/repos/LayoutFlow/pretrained \
  checkpoint=$LAB/runs/layoutflow-minimal/rico-run1/checkpoints/<ckpt>.ckpt
```

`train.py run_dir=... trainer.max_epochs=1 +trainer.limit_train_batches=3
+trainer.limit_val_batches=2 dataset.batch_size=16 dataset.num_workers=0
dataset.persistent_workers=false enable_wandb=false` is a good few-second smoke test
before launching a real run.

## What's genuinely different from upstream (beyond dropped code paths)

- `test.py`'s conditioning tasks are reduced to `uncond`; the other four
  (cat_cond/size_cond/elem_compl/refinement) aren't wired up as standalone eval
  targets here (training still sees all of them via `random4`).
- Validation only computes the unconditional FID (upstream also computes a
  conditional-FID side metric during validation when training uses `random4`); the
  LR-plateau scheduler was already only watching the unconditional one, so this
  doesn't change what gets optimized.
- No trajectory/wandb-image visualization; `save_example_layouts` writes plain PNGs
  to `{run_dir}/vis/` instead, gated behind `visualize=true`.

## Discrete-category variant (`model=LayoutFlowDiscrete`)

Geometry keeps LayoutFlow's Euclidean flow matching at every element; the category is
modelled with masked discrete diffusion (masked w.p. `1-t`, cross-entropy on masked
elements, unmasked at sampling with prob `dt/(1-t)`), both noised jointly at the same `t`.
See `src/models/layout_flow_discrete.py`. RICO, truly unconditional val FID, 5 seeds:
continuous LayoutFlow 2.32 +/- 0.12, LayoutFlowDiscrete (`cat_loss_weight=0.25`) 2.07 +/- 0.11.

```bash
python train.py dataset=RICO dataset_name=RICO model=LayoutFlowDiscrete \
  dataset.dataset.data_path=... model.pretrained_dir=... run_dir=... trainer.max_epochs=2000 trainer.check_val_every_n_epoch=25
```

Note: validation/test now evaluate with `cond='uncond'` (as upstream does). Earlier
in-training FIDs of random4-trained models (e.g. "1.53") were measured on the leaky
random4 mix and read ~0.5 too low.

## Whole-element masking and variable length (`model=LayoutFlowElemMask`, `model=LayoutFlowVarLen`)

One class (`src/models/layout_flow_varlen.py`), two stages selected by `model.insertion`:

- **`LayoutFlowElemMask`** (fixed length): every element, box and category together, starts as one
  mask token, is unmasked at its own random time and is flow-matched from then on.
- **`LayoutFlowVarLen`** (variable length): the layout starts empty. Masked elements are inserted,
  then unmasked, then denoised, so the number of elements is generated too.

How it works:

- Reveal schedule `kappa(t) = min(t / t_max, 1)`: all inserting and unmasking happens in `[0, t_max]`,
  the rest of the trajectory is ordinary joint denoising with every element visible.
- Unmasking reveals the category clean and the box **at the current noise level**,
  `t x_1 + (1-t) eps`, where the clean box `x_1` is sampled from a 16-component diagonal Gaussian
  mixture head conditioned on the sampled class (`unmask_geom=gmm`, NLL-trained). `unmask_geom=mean`
  collapses the same head to its mean (MSE-trained, no sampling); `backbone_model.gmm_components=1`
  gives a single Gaussian.
- Insertion is one Poisson rate per layout (expected number of elements still missing), read off a
  global token. There is no per-gap rate because **layouts are treated as sets**: `VarLenBackbone` has
  no positional encoding and hides not-yet-existing slots with a key-padding mask, so keeping missing
  elements as hidden slots is exactly equivalent to a compacted sequence (checked numerically to
  1e-15). This stops being true the moment slot index carries meaning (positional/rotary encodings,
  an index-bound text stream): compact the sequence, or the gaps leak the final length.

RICO, unconditional, single seed per cell. *val* = best in-training validation FID (best of 80
validations on 1,865 layouts, so differences of 0.1-0.2 are noise); *test* = `test.py`, test split,
2,000 samples, 3 repeats:

| | `t_max=0.25` | `t_max=0.5` | `t_max=1.0` |
|---|---|---|---|
| ElemMask, gmm | | val **2.02**, test **2.41 +/- 0.05** | val 3.27 |
| ElemMask, mean | | val 2.22 | val 4.76 |
| VarLen, gmm | val 2.44, test **2.61 +/- 0.01** | val 2.46 | val 10.38 |

`t_max` is the dominant knob (hence the 0.5 default); gmm beats mean at both settings. For scale,
`test.py`'s own sanity check, *real* validation layouts scored against the test split, gives 2.10 on
RICO, so everything between 2.1 and 2.6 is close to what the metric can resolve. Single-Gaussian
ablation (`gmm_components=1`), not finished when this was written: val 2.54 (ElemMask, 0.5) and
11.3 (ElemMask, 1.0), i.e. between or behind the other two heads.

PubLayNet, same grid, **still training when this was written** (epoch ~145 of 1,000): in-training
val FID 0.78 (ElemMask gmm, 0.5), 0.87 (ElemMask mean, 0.5), 0.94 (VarLen, 0.25), with `t_max=1.0`
again far behind (6-8, and 36 for VarLen). **These val numbers are not comparable to published
PubLayNet FIDs**, which use the test split: real validation layouts score **8.10** against the test
split, and the VarLen `t_max=0.25` checkpoint at epoch 128 (val 1.10) scores **test FID
10.82 +/- 0.18** (alignment 0.148, overlap 0.019). Re-run `test.py` on the final checkpoints before
quoting a PubLayNet number.

Known issues:

- At inference the step on which `kappa` reaches 1 skips its insertions (`p < 1.0` guard), so about
  `1 / (t_max * inference_steps)` of the elements are never inserted: generated RICO layouts are 4%
  short at `t_max=0.25` (9.31 vs 9.72 elements) and 19-20 element layouts are under-produced on
  both datasets. One-line fix, but it changes the sampler the numbers above were produced with.
- The class ids in `ldm_rico_*.h5` do **not** follow upstream's `TYPE_2_CAT` name table (its id 6,
  "Image", is geometrically the top toolbar), so RICO class names are unknown; PubLayNet's
  (1 text, 2 title, 3 list, 4 table, 5 figure) fit the box shapes.

Conditional tasks and guidance (both models):

- `model.cond=random4` trains LayoutFlow's mix: per batch quarter, element completion (a random
  ~20% of the elements given entirely), category-conditioned, category+size-conditioned, and
  unconditional. A given element is one that is visible from t = 0 with its given coordinates held
  clean; given coordinates get no velocity target. `test.py task=cat_cond|size_cond|elem_compl
  calc_miou=true` evaluates on the full test set as upstream (given values from the test layouts).
  With insertion, `elem_compl` generates the number of extra elements unless `given_length=true`.
- `model.cat_drop=0.1` hides every category of a layout w.p. 0.1 during training, so the
  category-unconditional velocity exists; `test.py +sampling.cfg_w=<w>` then applies
  classifier-free guidance `v_u + w (v_c - v_u)` (w = 1 is off; on RICO no w beats 1, see RESULTS.md). Other `+sampling.*` knobs
  (`gmm_temp`, `reveal_eps`, `solver=heun`, `snap_grid`) are documented in `RESULTS.md`; none of
  them beat the default sampler.

```bash
scripts/launch_varlen_grid.sh RICO          # or PubLayNet; 7 detached runs over 4 GPUs
python test.py dataset=RICO dataset_name=RICO model=LayoutFlowVarLen checkpoint=... \
  dataset.dataset.data_path=... pretrained_dir=... model.pretrained_dir=... multirun_n=3
python scripts/sample_layouts.py out.json rico:RICO:0.25:path/to.ckpt
```

The LR scheduler steps once per validation, so set `trainer.check_val_every_n_epoch` in gradient
steps, not epochs: 25 on RICO (62 it/epoch), 3 on PubLayNet (609 it/epoch). Use
`dataset.dataset.in_memory=true`; without it training is dataloader-bound.
