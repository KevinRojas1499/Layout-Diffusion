#!/usr/bin/env bash
# Grid over --num_steps for sampling.py with snapshot.pt (MMDiTQM9 / DiT).
# Usage: ./run_snapshot_grid.sh <sampler>
#   <sampler> ∈ {euler, split, staggered}
#
# Fixed:
#   model        = DiT
#   num_samples  = 10000
#   batch_size   = 500
#   no EMA
#   checkpoint   = snapshot.pt
#   out root     = num_steps_ablation_snapshot/

set -euo pipefail

SAMPLER="${1:?usage: $0 <euler|split|staggered>}"
case "$SAMPLER" in
  euler|split|staggered) ;;
  *) echo "sampler must be one of: euler, split, staggered" >&2; exit 2 ;;
esac

MASTER_PORT="${MASTER_PORT:-29504}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-500}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-snapshot.pt}"
OUT_ROOT="${OUT_ROOT:-num_steps_ablation_snapshot}"
AMP_DTYPE="${AMP_DTYPE:-fp32}"

NUM_STEPS=(75 125 250 500 750 1000)

for steps in "${NUM_STEPS[@]}"; do
  out_dir="${OUT_ROOT}/steps-${steps}_${SAMPLER}"
  echo "========================================"
  echo "[snapshot grid] num_steps=${steps} sampler=${SAMPLER} -> ${out_dir}"
  echo "[snapshot grid] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "========================================"
  uv run torchrun --nproc_per_node=1 --master-port "${MASTER_PORT}" sampling.py \
    --model DiT \
    --num_steps "${steps}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --load_checkpoint "${LOAD_CHECKPOINT}" \
    --dir "${out_dir}" \
    --sampler "${SAMPLER}" \
    --amp_dtype "${AMP_DTYPE}"
  echo "[snapshot grid] done steps=${steps} sampler=${SAMPLER} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[snapshot grid] sampler=${SAMPLER} grid finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
