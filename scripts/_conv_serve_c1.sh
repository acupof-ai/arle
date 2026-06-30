#!/usr/bin/env bash
# C1 validation serve: new convergence binary, SAME default config as the #60
# reference run (8-slot, max_seq_len 16384). Confirms budget log byte-identity
# (per_slot 924MB ...) + clean boot. setsid so it survives kubectl-exec reaping.
set -uo pipefail
ROOT=/data01/build/arle
LOG=/data01/build/conv_c1_serve.log
pkill -f "release-fast/arle serve" 2>/dev/null || true
sleep 4
rm -f "$LOG"
cd "$ROOT" || exit 2
setsid bash -lc '
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7
  export INFER_TP_SIZE=8
  export INFER_DSV4_MAX_SEQ_LEN=16384
  export RUST_LOG=info
  export NCCL_DEBUG=WARN
  export ARLE_DSV4_MOE_BACKEND=allreduce
  export ARLE_DSV4_INCREMENTAL_KV=1
  export ARLE_DSV4_EXPERT_BACKEND=deepgemm
  export ARLE_DSV4_LOCAL_GROUPED_EXPERTS=0
  export ARLE_DEEPGEMM_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm
  export ARLE_DEEPGEMM_LIBRARY_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm/deep_gemm
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
  exec /data01/build/arle/target/release-fast/arle serve --backend cuda \
    --model-path /data01/models/DeepSeek-V4-Flash --port 18188 \
    --num-slots 8 --kv-cache-dtype auto
' < /dev/null > "$LOG" 2>&1 &
echo "c1-serve launched pid=$!"
