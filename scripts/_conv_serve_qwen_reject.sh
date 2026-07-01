#!/usr/bin/env bash
# C4 reject test (Qwen3.5/3.6 MoE lane, TP=1 on GPU 0 + a VRAM filler): the
# Qwen RoPE cache caps max_seq_len at 262144, so per-slot KV tops out at ~5 GB
# — far below a 96 GB H20's free VRAM. The affordable=0 reject is a tight-VRAM /
# small-GPU guard, unreachable here via context length alone. So a ~25 GiB filler
# (scripts/_conv_gpu_filler.py, launched separately on GPU 0) constrains free so
# that POST-weights free (~3.6 GB) < one slot's KV at 262144 tokens (~5.1 GB) →
# affordable=0 → Qwen35Model::kv_budget_num_slots must fail closed with
# "Qwen3.5 KV budget rejected startup: ... affords 0 slots" (NOT a device OOM).
# total_pages 16384 × page_size 16 = 262144 = the RoPE cap (largest valid → max
# per_slot → most robust reject).
set -uo pipefail
ROOT=/data01/build/arle
LOG=/data01/build/conv_qwen_reject.log
pkill -f "release-fast/arle serve" 2>/dev/null || true
sleep 5
rm -f "$LOG"
cd "$ROOT" || exit 2
setsid bash -lc '
  export CUDA_VISIBLE_DEVICES=0
  export INFER_CUDA_DEVICES=0
  export INFER_TP_SIZE=1
  export RUST_LOG=info
  export NCCL_DEBUG=WARN
  export ARLE_DEEPGEMM_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm
  export ARLE_DEEPGEMM_LIBRARY_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm/deep_gemm
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
  exec /data01/build/arle/target/release-fast/arle serve --backend cuda \
    --model-path /data01/models/Qwen3.6-35B-A3B --port 18188 \
    --max-running-requests 8 --kv-cache-dtype auto
' < /dev/null > "$LOG" 2>&1 &
echo "qwen-reject serve launched pid=$!"
