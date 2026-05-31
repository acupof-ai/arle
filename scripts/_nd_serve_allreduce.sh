#!/usr/bin/env bash
# native-deepep serve only (stays alive for manual probing). Pod tmux session.
set -u
ROOT=/data01/build/arle
BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash
PORT=18200
LOG=/tmp/nd_serve.log
: >"$LOG"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
sleep 3
cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_MULTIPROC_SERVE=1 \
ARLE_DSV4_MOE_BACKEND=allreduce \
ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1 \
ARLE_DEEPEP_DIR=/data01/build/DeepEP \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 \
ARLE_DSV4_GPU_FULL_LAYERS=43 \
ARLE_DSV4_INCREMENTAL_KV=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
exec "$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 \
  --max-seq-len 4096 --mem-fraction-static 0.10 \
  --kv-cache-dtype bf16 --deepseek-distributed-layers 43 >>"$LOG" 2>&1
