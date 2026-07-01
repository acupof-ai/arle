#!/usr/bin/env bash
# C4 reject test: synthetic huge max_seq_len (2,000,000) drives the per-slot KV
# estimate past free×0.9 → affordable=0 → C4 must fail closed with
# "DSv4 KV budget rejected startup: ... affords 0 slots" (NOT a device OOM),
# uniformly across all 8 TP ranks (lockstep-safe post-NCCL-reduce bail).
set -uo pipefail
ROOT=/data01/build/arle
LOG=/data01/build/conv_c4_reject.log
pkill -f "release-fast/arle serve" 2>/dev/null || true
sleep 4
rm -f "$LOG"
cd "$ROOT" || exit 2
setsid bash -lc '
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7
  export INFER_TP_SIZE=8
  export INFER_DSV4_MAX_SEQ_LEN=2000000
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
    --max-running-requests 8 --kv-cache-dtype auto
' < /dev/null > "$LOG" 2>&1 &
echo "c4-reject serve launched pid=$!"
