#!/usr/bin/env bash
# Wait for the euler grid (PID arg) to exit, then run split + staggered grids in bf16.

set -euo pipefail

EULER_PID="${1:?usage: $0 <euler-runner-pid>}"
LOG_DIR="num_steps_ablation_snapshot/_logs"
mkdir -p "${LOG_DIR}"
AUTO_LOG="${LOG_DIR}/auto_continue.log"

echo "[auto] waiting for euler runner PID=${EULER_PID} to exit..." | tee -a "${AUTO_LOG}"
while kill -0 "${EULER_PID}" 2>/dev/null; do
  sleep 30
done
echo "[auto] euler runner exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${AUTO_LOG}"
sleep 5  # let GPU settle

export AMP_DTYPE=bf16

echo "[auto] launching split grid with AMP_DTYPE=bf16 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${AUTO_LOG}"
./run_snapshot_grid.sh split >"${LOG_DIR}/split.log" 2>&1
echo "[auto] split grid finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${AUTO_LOG}"

echo "[auto] launching staggered grid with AMP_DTYPE=bf16 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${AUTO_LOG}"
./run_snapshot_grid.sh staggered >"${LOG_DIR}/staggered.log" 2>&1
echo "[auto] staggered grid finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${AUTO_LOG}"

echo "[auto] all grids done." | tee -a "${AUTO_LOG}"
