#!/usr/bin/env bash
# Event pool fix verification + baseline re-run on pod.
# Usage: bash /host/arle-runs/run_event_pool_bench.sh
set -euo pipefail

ARLE=/host/arle-build
RUNS=/host/arle-runs/event-pool-$(date +%Y%m%d-%H%M)
mkdir -p "$RUNS"

echo "=== BUILD ==="
cd "$ARLE"
RUSTC_WRAPPER="" CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,nccl -j 32 2>&1 | tail -5
BIN="$ARLE/target/release/arle"
echo "binary: $(sha256sum "$BIN" | cut -d' ' -f1)"

echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo "=== CGROUP MEMORY ==="
cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "no cgroup v2"
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
free -h | head -3

# --- TP=8 decode probe (event pool verification) ---
echo "=== TP=8 DECODE PROBE (128K) ==="
TP8_PORT=18200
TP8_LOG="$RUNS/tp8_decode.log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"$BIN" serve \
  --model /host/nvme0/ThinkingCap-Qwen3.6-27B-FP8 \
  --port "$TP8_PORT" \
  --max-running-requests 16 \
  --spec-type dspark \
  --mtp-draft-model /host/nvme0/Qwen3.6-27B-DFlash \
  --dspark-block-size 6 \
  > "$TP8_LOG" 2>&1 &
TP8_PID=$!
echo "tp8 serve pid=$TP8_PID"

# Wait for engine ready (max 120s)
for i in $(seq 1 60); do
  if curl -s "http://localhost:$TP8_PORT/v1/models" | grep -q "ThinkingCap"; then
    echo "tp8 engine-ready after ~$((i*2))s"
    break
  fi
  sleep 2
done

python3 /host/arle-build/scripts/decode_rate_probe.py \
  --url "http://localhost:$TP8_PORT" \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --target-tokens 128000 \
  --max-tokens 128 \
  2>&1 | tee "$RUNS/tp8_decode_probe.log"

kill $TP8_PID 2>/dev/null || true
wait $TP8_PID 2>/dev/null || true
echo "=== TP=8 DONE ==="

# --- Single-GPU 27B-FP8 throughput bench ---
echo "=== 27B-FP8 THROUGHPUT (single-GPU, 32K, DSpark) ==="
GPU_ID=6
SG_PORT=18190
SG_LOG="$RUNS/sg_throughput.log"
CUDA_VISIBLE_DEVICES=$GPU_ID \
"$BIN" serve \
  --model /host/nvme0/ThinkingCap-Qwen3.6-27B-FP8 \
  --port "$SG_PORT" \
  --max-running-requests 16 \
  --spec-type dspark \
  --mtp-draft-model /host/nvme0/Qwen3.6-27B-DFlash \
  --dspark-block-size 6 \
  > "$SG_LOG" 2>&1 &
SG_PID=$!
echo "sg serve pid=$SG_PID (gpu=$GPU_ID)"

for i in $(seq 1 60); do
  if curl -s "http://localhost:$SG_PORT/v1/models" | grep -q "ThinkingCap"; then
    echo "sg engine-ready after ~$((i*2))s"
    break
  fi
  sleep 2
done

python3 /host/arle-build/scripts/bench_throughput.py \
  --url "http://localhost:$SG_PORT" \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --prompts-jsonl /host/arle-runs/bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,2,4,8,16 \
  --requests-per-concurrency 128 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output "$RUNS/bench-27b-fp8" \
  2>&1 | tee "$RUNS/sg_throughput_bench.log"

# Needle gate on same serve
echo "=== NEEDLE GATE ==="
RAW=1 TEMPLATE=qwen3_nonthink \
python3 /host/arle-build/scripts/needle_gate.py \
  --url "http://localhost:$SG_PORT" \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --lengths 512,4096,16384,32768 \
  --runs 3 \
  --output "$RUNS/needle_gate.log" \
  2>&1 | tee -a "$RUNS/needle_gate.log"

kill $SG_PID 2>/dev/null || true
wait $SG_PID 2>/dev/null || true
echo "=== ALL DONE ==="
