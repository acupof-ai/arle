#!/usr/bin/env bash
# Baseline CP parity run (cp=2, default seq 16, devices 0,1). Detached +
# RUN_EXIT marker so it is re-attachable from the log alone. Fast (seconds).
set -uo pipefail
BIN="/host/arle-build/target/release/examples/nd_parallel_parity"
LOG="/host/nd-parity-run.log"
: > "$LOG"
exec >"$LOG" 2>&1
export ARLE_ND_CUDA_DEVICES=0,1
# no ARLE_ND_SEQ -> default 16 (divisible by 2*CP_SIZE=4)
export ARLE_ND_DIR="/host/arle_nd_parity_baseline_$$"
"$BIN"
rc=$?
echo "RUN_EXIT=$rc"
exit "$rc"
