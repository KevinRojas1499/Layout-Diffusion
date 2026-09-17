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
  fid.py, metrics.py, visualize.py   # eval: layout FID net, alignment/overlap/mIoU, PNG rendering
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
