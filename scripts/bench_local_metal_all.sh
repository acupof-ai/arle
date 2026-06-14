#!/usr/bin/env bash
# Sequential single-user (c=1) latency/throughput sweep across all local MLX
# models on the Metal backend. STRICTLY serial — one model loaded at a time
# (two concurrent Metal loads swap->SSD-thrash->freeze on a 48 GB box).
set -uo pipefail
cd /path/to/code/agent-infer
BIN=target/release/arle
PORT=8000
PROMPT=512
NOUT=128
K=6
RESULTS=/tmp/metal_bench_results.jsonl
: > "$RESULTS"

# label : model-path : ready-timeout-seconds
MODELS=(
  "qwen35-0.8b-4bit:mlx-community/Qwen3.5-0.8B-MLX-4bit:90"
  "qwen35-4b-4bit:mlx-community/Qwen3.5-4B-MLX-4bit:120"
  "qwen35-9b-4bit:mlx-community/Qwen3.5-9B-MLX-4bit:180"
  "qwen36-35b-a3b-4bit:mlx-community/Qwen3.6-35B-A3B-4bit:360"
)

pkill -f "arle serve" 2>/dev/null; sleep 2

for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"; rest="${entry#*:}"; model="${rest%:*}"; timeout="${rest##*:}"
  mid="$(basename "$model")"
  echo "================ $label ($model) ================"
  pkill -f "arle serve" 2>/dev/null; sleep 2
  "$BIN" serve --backend metal --model-path "$model" --port $PORT \
      --max-prompt-tokens 6144 --max-total-tokens 8192 > "/tmp/serve-$label.log" 2>&1 &
  SP=$!
  ready=0
  for i in $(seq 1 "$timeout"); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; echo "  READY after ${i}s"; break; fi
    if ! kill -0 "$SP" 2>/dev/null; then echo "  SERVE DIED — tail:"; tail -3 "/tmp/serve-$label.log"; break; fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "  NOT READY ($label) — skipping"; tail -4 "/tmp/serve-$label.log"
    kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; sleep 2; continue
  fi
  served_id="$(curl -sf "http://localhost:$PORT/v1/models" | jq -r '.data[0].id' 2>/dev/null)"
  echo "  served id: $served_id"
  out="$(python3 scripts/bench_local_metal.py "http://localhost:$PORT" "$served_id" "$PROMPT" "$NOUT" "$K" 2>>/tmp/serve-$label.log)"
  echo "  RESULT: $out"
  echo "{\"label\":\"$label\",\"path\":\"$model\",$(echo "$out" | sed 's/^{//')" >> "$RESULTS"
  kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; sleep 4
done

pkill -f "arle serve" 2>/dev/null
echo "================ SUMMARY ================"
cat "$RESULTS"
