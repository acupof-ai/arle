#!/usr/bin/env bash
# Detached launcher for the nd_parallel_parity CP gate — pod.sh can't drive an
# --example (its build validator requires exactly one --bin), so this mirrors
# pod-build-env.sh and adds a re-attachable marker.
#   pod_parity.sh build                       # compile the example only
#   pod_parity.sh run <label> <seq|-> <devs>  # run; seq '-' = default (14)
set -uo pipefail
TREE=/host/arle-build
MODE="${1:?usage: pod_parity.sh build|run ...}"
cd "$TREE" || exit 1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"

if [ "$MODE" = build ]; then
  STATE="$TREE/parity-runs/build"
  mkdir -p "$STATE"
  LOG="$STATE/log"; MARKER="$STATE/marker"
  rm -f "$MARKER"
  {
    echo "=== example build START $(date -u +%FT%TZ) ==="
    cargo build -p train --release --no-default-features --features cuda,nccl --example nd_parallel_parity
    rc=$?
    echo "BUILD_EXIT=$rc"
    echo "BUILD_EXIT=$rc" > "$MARKER"
  } >"$LOG" 2>&1
  exit 0
fi

if [ "$MODE" = run ]; then
  LABEL="${2:?missing label}"; SEQ="${3:?missing seq}"; DEVICES="${4:?missing devices}"
  STATE="$TREE/parity-runs/$LABEL"
  mkdir -p "$STATE/nd"
  LOG="$STATE/log"; MARKER="$STATE/marker"
  rm -f "$MARKER"
  {
    echo "=== parity run $LABEL seq=$SEQ devices=$DEVICES START $(date -u +%FT%TZ) ==="
    [ "$SEQ" != "-" ] && export ARLE_ND_SEQ="$SEQ"
    ARLE_ND_CUDA_DEVICES="$DEVICES" ARLE_ND_DIR="$STATE/nd" \
      cargo run -p train --release --no-default-features --features cuda,nccl --example nd_parallel_parity
    rc=$?
    echo "RUN_EXIT=$rc"
    echo "RUN_EXIT=$rc" > "$MARKER"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
  } >"$LOG" 2>&1
  exit 0
fi

echo "usage: pod_parity.sh build|run ..." >&2
exit 2
