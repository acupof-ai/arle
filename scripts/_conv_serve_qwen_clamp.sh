#!/usr/bin/env bash
# C2 clamp test (Qwen3.5/3.6 MoE lane, TP=2): oversized --max-running-requests at the
# default 128K context (internal total_pages 8192 × page_size 16) drives requested ≫ affordable,
# so Qwen35Model::kv_budget_num_slots must (a) log the real per_slot/free budget
# on every rank, (b) emit the clamp warn, (c) boot with the clamped slot count
# — uniformly across both TP ranks (lockstep-safe post-NCCL-min-reduce). This
# exercises the EXACT Qwen budget code (qwen35.rs:810-861), distinct from the
# DSv4 path. The per_slot/free it prints sizes the C4 reject serve.
# TP=2 (not 8): Qwen3.6 has num_kv_heads=2, so the full-attn head shard requires
# world_size | 2 → TP=8 fails the divisibility check before the budget code runs.
set -uo pipefail
ROOT=/data01/build/arle
LOG=/data01/build/conv_qwen_clamp.log
pkill -f "release-fast/arle serve" 2>/dev/null || true
sleep 4
rm -f "$LOG"
cd "$ROOT" || exit 2
setsid bash -lc '
  export CUDA_VISIBLE_DEVICES=0,1
  export INFER_CUDA_DEVICES=0,1
  export INFER_TP_SIZE=2
  export RUST_LOG=info
  export NCCL_DEBUG=WARN
  export ARLE_DEEPGEMM_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm
  export ARLE_DEEPGEMM_LIBRARY_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm/deep_gemm
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
  exec /data01/build/arle/target/release-fast/arle serve --backend cuda \
    --model-path /data01/models/Qwen3.6-35B-A3B --port 18188 \
    --max-running-requests 1024 --kv-cache-dtype auto
' < /dev/null > "$LOG" 2>&1 &
echo "qwen-clamp serve launched pid=$!"
