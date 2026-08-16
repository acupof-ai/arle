#!/usr/bin/env bash
# One-shot cp=2 128K cold-prefill TTFT: launch serve, wait for ready, probe, kill.
# Designed to fit inside a transient GPU-free window on the shared H20 box.
set -uo pipefail
TREE=/host/arle-build
RUNS=/host/arle-runs
PORT=18189
LOG="$RUNS/refactor-cp2-ttft-serve.log"
OUT="$RUNS/refactor-cp2-ttft128k.log"
mkdir -p "$RUNS"

cd "$TREE"
INFER_TP_SIZE=8 INFER_ATTN_CP_SIZE=2 setsid nohup ./target/release/arle serve \
  --model-path /data00/ThinkingCap-Qwen3.6-27B-FP8 \
  --port "$PORT" --kv-cache-dtype bf16 --max-total-tokens 160000 \
  > "$LOG" 2>&1 < /dev/null &
SERVE_PID=$!
echo "serve pid=$SERVE_PID"

# Wait up to 300s for engine-ready.
for i in $(seq 1 60); do
  if grep -q 'serving OpenAI v1' "$LOG" 2>/dev/null; then
    echo "engine-ready after ~$((i*5))s"
    break
  fi
  if grep -qE 'setup failed|exited Some|panic|OOM|out of memory' "$LOG" 2>/dev/null; then
    echo "SERVE FAILED:"
    grep -E 'setup failed|exited Some|panic|OOM|out of memory|ERROR' "$LOG" | tail -5
    kill -9 "$SERVE_PID" 2>/dev/null
    exit 1
  fi
  sleep 5
done

if ! grep -q 'serving OpenAI v1' "$LOG" 2>/dev/null; then
  echo "TIMEOUT waiting for engine-ready"
  tail -5 "$LOG"
  kill -9 "$SERVE_PID" 2>/dev/null
  exit 1
fi

# Run the TTFT probe (128K cold prefill, max_tokens=1, stream for TTFT).
PORT="$PORT" python3 scripts/ttft_probe.py --target-tokens 128000 --runs 1 > "$OUT" 2>&1
echo "=== TTFT RESULT ==="
cat "$OUT"

# Kill the serve (bracket pattern avoids pkill self-match).
kill -9 "$SERVE_PID" 2>/dev/null
pkill -9 -f '[t]arget/release/arle serve' 2>/dev/null
echo "serve killed"
