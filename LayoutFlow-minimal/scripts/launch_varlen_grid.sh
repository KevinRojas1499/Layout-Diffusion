#!/bin/bash
# The 7-run grid behind the README's whole-element-masking results: unmasking (LayoutFlowElemMask)
# over {gmm, mean} x t_max {1.0, 0.5}, and insertion (LayoutFlowVarLen) over t_max {1.0, 0.5, 0.25}.
# Two runs per GPU on GPUs 0-3, detached; run it from inside a 4-GPU allocation.
#   usage: scripts/launch_varlen_grid.sh RICO|PubLayNet [extra hydra overrides for every run...]
#
# - ReduceLROnPlateau steps once per validation, so the validation cadence is set in gradient steps:
#   RICO 25 epochs x 62 it, PubLayNet 3 epochs x 609 it.
# - These models are CPU-bound. 7 runs fit a 32-core node; ~13 thrash and halve total throughput,
#   so do not start a second grid on the same node until the first one is done.
DS=${1:?usage: $0 RICO|PubLayNet [overrides...]}; shift
case $DS in
    RICO)      DATA=rico;      EPOCHS=2000; VAL_EVERY=25 ;;
    PubLayNet) DATA=publaynet; EPOCHS=1000; VAL_EVERY=3 ;;
    *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac
LAB=${LAB:-/network/rit/lab/Yelab/kevin-back/kevin_rojas}
RUNS=${RUNS:-$LAB/runs/layoutflow-minimal}
PY=${PY:-$LAB/repos/LayoutFlow/.venv/bin/python}
SRC=$(cd "$(dirname "$0")/.." && pwd)
COMMON=("$@")

launch() {  # launch <gpu> <name> <model> <overrides...>
    local gpu=$1 name=${DS,,}-$2 model=$3; shift 3
    if [ -e "$RUNS/$name" ]; then echo "skip $name: $RUNS/$name exists" >&2; return; fi
    cd "$SRC" || exit 1
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$PY" train.py \
        dataset=$DS dataset_name=$DS model=$model \
        dataset.dataset.data_path=$LAB/datasets/LayoutFlow-data/dataset/$DATA \
        model.pretrained_dir=$LAB/repos/LayoutFlow/pretrained \
        run_dir=$RUNS/$name wandb_dir=$RUNS/wandb expname=$name enable_wandb=true \
        dataset.num_workers=2 dataset.dataset.in_memory=true \
        trainer.devices=1 trainer.max_epochs=$EPOCHS trainer.check_val_every_n_epoch=$VAL_EVERY \
        "$@" "${COMMON[@]}" > "$RUNS/$name.log" 2>&1 < /dev/null &
    echo "gpu $gpu  pid $!  $name"
}

MEAN="model.unmask_geom=mean model.geom_unmask_weight=1.0"
launch 0 elemmask-gmm-t100  LayoutFlowElemMask model.t_max=1.0
launch 0 elemmask-mean-t100 LayoutFlowElemMask model.t_max=1.0 $MEAN
launch 1 elemmask-gmm-t050  LayoutFlowElemMask model.t_max=0.5
launch 1 elemmask-mean-t050 LayoutFlowElemMask model.t_max=0.5 $MEAN
launch 2 varlen-gmm-t100    LayoutFlowVarLen   model.t_max=1.0
launch 2 varlen-gmm-t025    LayoutFlowVarLen   model.t_max=0.25
launch 3 varlen-gmm-t050    LayoutFlowVarLen   model.t_max=0.5
