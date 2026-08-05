#!/usr/bin/env bash
# DSv4 TP4 no-spec baseline serve for throughput bench.
set -uo pipefail
TREE="${POD_TREE:-/host/arle-build}"
source "$TREE/scripts/pod-build-env.sh"
cd "$TREE"
CUDA_VISIBLE_DEVICES="0,1,2,3" INFER_TP_SIZE=4 INFER_CUDA_DEVICES=0,1,2,3 \
  ARLE_DSV4_MOE_BACKEND=allreduce \
  DG_JIT_CACHE_DIR=/host/deepgemm-warm \
  ./target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 \
    --max-running-requests 32 \
    --port 8000
