# Layout Diffusion

Content-aware design generation: canvas → variable number of layout elements +
per-element text. See `CONTENT_AWARE.md` for the project goal and staged plan.

All code lives in `LayoutFlow-minimal/` (see its README): a minimal Lightning +
Hydra reimplementation of LayoutFlow's recipe, plus `LayoutFlowDiscrete`, which
models the category with masked discrete diffusion while the geometry keeps
Euclidean flow matching.

`LayoutFlow/` is a gitlink to the pristine upstream LayoutFlow repo (reference only).

The earlier custom training pipeline (`layout_training.py`,
`multimodal_interpolant.py`, ...) was retired; it remains in git history, and its
insertion/deletion mechanism is ported (with tests) on the
`worktree-layoutflow-minimal` branch.
