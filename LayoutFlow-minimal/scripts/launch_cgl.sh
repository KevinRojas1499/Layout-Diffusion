#!/bin/bash
# First content-aware runs on RALF's CGL release (canvas -> variable-length layout), one per GPU:
#   varlen-canvas     LayoutFlowVarLen + canvas tokens   (the model: p(N | canvas) via the insertion rate)
#   varlen-blind      LayoutFlowVarLen, no canvas         (canvas-blind reference: N from the data prior)
#   padding-canvas    LayoutFlowElemMask, 10 slots + 'empty' class, canvas tokens (the padding baseline)
#   oracle-canvas     LayoutFlowElemMask with the true N given, canvas tokens (length-oracle upper bound)
#   varlen-canvas-wide (WIDE=1) the headline model with d_model 768 / 6 layers (~40M)
# Validation FID = RALF's FIDNetV3 vs their val-split features (src/fid_ralf.py), which drives the plateau
# LR schedule and checkpoint selection; paper numbers come from RALF's eval.py (scripts/export_ralf_samples.py).
#   usage: [SKIP="oracle-canvas ..."] [WIDE=1 WIDE_GPU=3] scripts/launch_cgl.sh [extra hydra overrides for every run...]
LAB=${LAB:-/network/rit/lab/Yelab/kevin-back/kevin_rojas}
RUNS=${RUNS:-$LAB/runs/layoutflow-minimal}
PY=${PY:-$LAB/repos/LayoutFlow/.venv/bin/python}
SRC=$(cd "$(dirname "$0")/.." && pwd)
DATA=$LAB/datasets/ralf/cache/dataset/cgl
FEATS=$LAB/datasets/ralf/canvas_feats/cgl
COMMON=("$@")

launch() {  # launch <gpu> <name> <model> <overrides...>
    local gpu=$1 name=cgl-$2 model=$3; shift 3
    [[ " $SKIP " == *" $2 "* ]] && { echo "skip $2 (SKIP)" >&2; return; }
    if [ -e "$RUNS/$name" ]; then echo "skip $name: $RUNS/$name exists" >&2; return; fi
    cd "$SRC" || exit 1
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$PY" train.py \
        dataset=CGL dataset_name=CGL model=$model data.max_len=10 \
        dataset.dataset.data_path=$DATA dataset.dataset.feats_path=$FEATS \
        model.pretrained_dir=$LAB/repos/LayoutFlow/pretrained model.ralf_cache=$LAB/datasets/ralf/cache ckpt_every_n_epochs=100 \
        run_dir=$RUNS/$name wandb_dir=$RUNS/wandb expname=$name enable_wandb=true dataset.num_workers=2 \
        trainer.devices=1 trainer.max_epochs=1000 trainer.check_val_every_n_epoch=10 model.t_max=0.5 \
        "$@" "${COMMON[@]}" > "$RUNS/$name.log" 2>&1 < /dev/null &
    echo "gpu $gpu  $name"
}

launch 0 varlen-canvas  LayoutFlowVarLen   model.backbone_model.ctx_dim=385
launch 1 varlen-blind   LayoutFlowVarLen   model.backbone_model.ctx_dim=0
launch 2 padding-canvas LayoutFlowElemMask model.backbone_model.ctx_dim=385 +dataset.dataset.pad_empty=true dataset.dataset.num_cat=6 model.num_cat=6 model.fid_empty_id=5
launch 3 oracle-canvas  LayoutFlowElemMask model.backbone_model.ctx_dim=385
# wider layout transformer (d_model 768, 6 layers, ~40M) on the headline configuration; pass WIDE=1 to include it
[ -n "$WIDE" ] && launch ${WIDE_GPU:-3} varlen-canvas-wide LayoutFlowVarLen model.backbone_model.ctx_dim=385 \
    model.backbone_model.d_model=768 model.backbone_model.num_layers=6 model.backbone_model.dim_feedforward=3072
