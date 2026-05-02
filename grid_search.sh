#!/usr/bin/env bash
# Grid over --num_steps and --sampler for sampling.py (torchrun).
# Edit the arrays below to change the ablation.

set -euo pipefail

MASTER_PORT="${MASTER_PORT:-29502}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-experiments/experiments/muon-single-heavy/itr_165000/snapshot.pt}"
OUT_ROOT="${OUT_ROOT:-num_steps_ablation}"

# num_steps values to try
NUM_STEPS=(75 100 125 250 500 750 1000)

# Must match click.Choice in sampling.py: euler, split, staggered
SAMPLERS=(euler split staggered)

for steps in "${NUM_STEPS[@]}"; do
  for sampler in "${SAMPLERS[@]}"; do
    out_dir="${OUT_ROOT}/steps-${steps}_${sampler}"
    echo "========================================"
    echo "num_steps=${steps} sampler=${sampler} -> ${out_dir}"
    echo "========================================"
    uv run torchrun --master-port "${MASTER_PORT}" sampling.py \
      --model Transformer \
      --num_steps "${steps}" \
      --num_samples "${NUM_SAMPLES}" \
      --load_checkpoint "${LOAD_CHECKPOINT}" \
      --dir "${out_dir}" \
      --sampler "${sampler}"
  done
done

echo "All runs finished."
