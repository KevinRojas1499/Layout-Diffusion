# Layout Diffusion

Content-aware design generation: canvas → variable number of layout elements +
per-element text. **Read `CONTENT_AWARE.md` first** for the actual project goal;
`LAYOUT.md` and `RICO.md` track the current best configs and experiment log for
the two datasets (PubLayNet, RICO) this is validated against.

This branch has been trimmed to the layout-only subset of the codebase (the
parent repo also contains earlier QM9 and parenthesized-arithmetic-equation
projects, stripped out here for readability — see `CLAUDE.md` in the full repo
if you need that history).

## Running

```bash
uv run python layout_training.py \
  dataset=publaynet model=transformer optimizer=adamw interpolant=multimodal \
  dataset.dataset.data_path=<path-to-h5-data> \
  run.dir=runs/layout/<run-name> run.num_iters=1000000
```

See `conf/config.yaml` for the full config group structure (`run`, `dataset`,
`model`, `optimizer`, `interpolant`, `eval`) and `conf/schema.py` for every
field's meaning and default.

## Layout

```
layout_training.py          # training entry point (Hydra config, PyTorch Lightning-free loop)
multimodal_interpolant.py   # the insert/unmask/denoise forward+reverse process
models/transformer.py       # the model (joint geometry + category transformer)
models/mmdit.py              # shared building blocks models/transformer.py depends on
model/rotary.py              # RoPE, used by models/transformer.py
custom_datasets/layoutflow_h5.py, layout_labels.py   # dataset loading + tokenizer vocab
eval/layout/                 # FID + alignment/overlap evaluator (LayoutFlow's FIDNet)
utils/{tokenizer,optimizers}.py
visualize_dataset.py         # plot_layout_sample: renders a generated layout to PNG
conf/                        # Hydra config groups
LayoutFlow-minimal/          # separate, standalone reimplementation of upstream LayoutFlow's
                              # own recipe, used as ground truth to diff this pipeline against
LayoutFlow/                  # pristine upstream LayoutFlow clone (untouched reference)
```

## Docs

- `CONTENT_AWARE.md` — the actual project goal and staged plan.
- `LAYOUTFLOW_REPLICATION.md` — the Stage A effort (match LayoutFlow's published
  numbers with our own pipeline) and its full experiment log.
- `LAYOUT.md`, `RICO.md` — best-known configs per dataset.
