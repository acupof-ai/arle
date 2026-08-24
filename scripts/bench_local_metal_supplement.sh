#!/usr/bin/env bash
# Supplement run: attempt every *remaining* locally-cached model on the Metal
# serve path (the main ladder covered Qwen3.5 0.8/4/9B + Qwen3.6-35B-A3B).
# Records a measurement for the ones that serve, and the fail-closed reason for
# the ones that don't — so the snapshot documents real coverage, not guesses.
# STRICTLY serial (no concurrent Metal loads).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=target/release/arle
PORT=8000
PROMPT=512; NOUT=128; K=6
RESULTS=/tmp/metal_bench_supplement.jsonl
: > "$RESULTS"

# label : model-path : ready-timeout : extra-env
MODELS=(
  "qwen36-35b-a3b-mtp-4bit:mlx-community/Qwen3.6-35B-A3B-MTP-4bit:360:"
  "zlab-qwen35-4b-dflash:z-lab/Qwen3.5-4B-DFlash:180:"
  "zlab-qwen36-35b-a3b-dflash:z-lab/Qwen3.6-35B-A3B-DFlash:360:"
  "qwen25-0.5b-bf16:mlx-community/Qwen2.5-0.5B-Instruct-bf16:90:"
  "qwen25-1.5b-bf16:mlx-community/Qwen2.5-1.5B-Instruct-bf16:120:"
  "llama32-1b-bf16:mlx-community/Llama-3.2-1B-Instruct-bf16:90:"
  "qwen3-0.6b:Qwen/Qwen3-0.6B:120:"
)

pkill -f "arle serve" 2>/dev/null; sleep 2

for entry in "${MODELS[@]}"; do
  IFS=':' read -r label model timeout extraenv <<< "$entry"
  echo "================ $label ($model) ================"
  pkill -f "arle serve" 2>/dev/null; sleep 2
  log="/tmp/serve-sup-$label.log"
  env ${extraenv:+$extraenv} "$BIN" serve --backend metal --model-path "$model" --port $PORT \
      --max-prompt-tokens 6144 --max-total-tokens 8192 > "$log" 2>&1 &
  SP=$!
  ready=0
  for i in $(seq 1 "$timeout"); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; echo "  READY after ${i}s"; break; fi
    if ! kill -0 "$SP" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    reason="$(grep -iE 'error:|rejected|not (a )?local|unsupported|fail|panic|unknown|classif' "$log" | head -1 | tr -d '"' | cut -c1-220)"
    [ -z "$reason" ] && reason="serve did not become ready within ${timeout}s"
    echo "  FAILED: $reason"
    echo "{\"label\":\"$label\",\"path\":\"$model\",\"status\":\"failed\",\"reason\":\"$reason\"}" >> "$RESULTS"
    kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; sleep 2; continue
  fi
  served_id="$(curl -sf "http://localhost:$PORT/v1/models" | jq -r '.data[0].id' 2>/dev/null)"
  echo "  served id: $served_id"
  out="$(python3 scripts/bench_local_metal.py "http://localhost:$PORT" "$served_id" "$PROMPT" "$NOUT" "$K" 2>>"$log")"
  if [ -n "$out" ]; then
    echo "  RESULT: $out"
    echo "{\"label\":\"$label\",\"path\":\"$model\",\"status\":\"ok\",$(echo "$out" | sed 's/^{//')" >> "$RESULTS"
  else
    echo "  MEASURE FAILED (served but probe errored) — tail:"; tail -2 "$log"
    echo "{\"label\":\"$label\",\"path\":\"$model\",\"status\":\"served_but_probe_failed\"}" >> "$RESULTS"
  fi
  kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; sleep 4
done

pkill -f "arle serve" 2>/dev/null
echo "================ SUPPLEMENT SUMMARY ================"
cat "$RESULTS"
