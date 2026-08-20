#!/usr/bin/env bash
# DSv4 TP4 serve for throughput bench.
# Usage: serve_dsv4_tp4.sh [baseline|dspark]
set -uo pipefail
MODE="${1:-baseline}"
TREE="${POD_TREE:-/host/arle-build}"
source "$TREE/scripts/pod-build-env.sh"
cd "$TREE"

DSV4_FLAGS=()
if [[ "$MODE" == "dspark" ]]; then
  DSV4_FLAGS+=(
    --spec-type dspark
    --mtp-draft-model /host/DeepSeek-V4-Flash-DSpark-draft-fp8
    --mtp-draft-tokens 5
    --comm-backend nccl
  )
fi

CUDA_VISIBLE_DEVICES="0,1,2,3" \
  ARLE_DSV4_MOE_BACKEND=allreduce \
  DG_JIT_CACHE_DIR=/host/deepgemm-warm \
  ./target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 \
    --tensor-parallel-size 4 \
    "${DSV4_FLAGS[@]}" \
    --max-running-requests 32 \
    --port 8000
