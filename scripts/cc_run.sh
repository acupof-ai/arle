#!/bin/bash
# One-key CC-as-harness OPD collection: serve Qwen -> cc_swe_baseline (serial) ->
# cc-convert -> training records. Owns the serve so it controls the dump dir the
# time-windows attribute against. Edit CONFIG or override via env.
set -u

# ===== CONFIG =====
GPU=${GPU:-0}
PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-/host/Qwen3.6-27B-FP8}
DATASET=${DATASET:-/host/swe_run2/train.jsonl}
STAGED=${STAGED:-/host/swe_run2/staged}
PYTHONPATH_TASK=${PYTHONPATH_TASK:-lib}                 # scoring PYTHONPATH (ansible=lib)
SAMPLES=${SAMPLES:-1}                                   # attempts/task (SAO needs >=2)
CC_TIMEOUT=${CC_TIMEOUT:-600}                           # per-task cc cap; 1800 wastes 30min on stuck tasks
DECODE_GRAPH=${DECODE_GRAPH:-1}                         # CUDA whole-step decode graph (rollout decode speedup)
LORA=${LORA:-}                                          # optional adapter
OUT_DIR=${OUT_DIR:-/host/cc_run/$(date -u +%Y%m%d_%H%M%S)}
ARLE=${ARLE:-/host/arle-build/target/release/arle}
# CC offline env — IS_SANDBOX=1 mandatory on a root container. Rarely edited:
export ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT ANTHROPIC_API_KEY=dummy IS_SANDBOX=1
export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export PATH="/host/npm-global/bin:/host/node-v22.14.0-linux-x64/bin:$PATH"
# ===== end CONFIG =====

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DUMP="$OUT_DIR/dumps"; mkdir -p "$OUT_DIR"
command -v claude >/dev/null || { echo "install: npm i -g @anthropic-ai/claude-code (node>=18)"; exit 1; }
curl -s --max-time 3 "$ANTHROPIC_BASE_URL/v1/models" >/dev/null && { echo "port $PORT busy"; exit 1; }

# Own the serve; kill only our PID on exit (never pkill).
CUDA_VISIBLE_DEVICES=$GPU nohup "$ARLE" serve --model-path "$MODEL_PATH" \
  --bind 0.0.0.0 --port "$PORT" --max-running-requests 4 --dump-messages-dir "$DUMP" \
  ${DECODE_GRAPH:+--qwen35-decode-graph true} \
  ${LORA:+--lora-adapters "$LORA" --lora-alpha 32} > "$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!; trap 'kill "$SERVE_PID" 2>/dev/null' EXIT

echo "[cc-run] serve pid=$SERVE_PID gpu=$GPU port=$PORT — waiting for load…"
for _ in $(seq 1 90); do
  curl -s --max-time 3 "$ANTHROPIC_BASE_URL/v1/models" >/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve died — see $OUT_DIR/serve.log"; exit 1; }
  sleep 10
done
MODEL_ID=$(curl -s "$ANTHROPIC_BASE_URL/v1/models" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
export ANTHROPIC_MODEL=$MODEL_ID ANTHROPIC_SMALL_FAST_MODEL=$MODEL_ID
echo "[cc-run] serve up, model=$ANTHROPIC_MODEL"

python3 "$ROOT/scripts/cc_swe_baseline.py" \
  --dataset "$DATASET" --staged-root "$STAGED" --model "$MODEL_ID" \
  --pythonpath "$PYTHONPATH_TASK" --work-root "$OUT_DIR/work" --samples "$SAMPLES" \
  --cc-timeout "$CC_TIMEOUT" \
  --windows-out "$OUT_DIR/windows.jsonl" --out "$OUT_DIR/results.jsonl"

if [ -s "$OUT_DIR/windows.jsonl" ]; then
  "$ARLE" train cc-convert --dump-dir "$DUMP" --tokenizer "$MODEL_PATH/tokenizer.json" \
    --windows "$OUT_DIR/windows.jsonl" --out "$OUT_DIR/records.jsonl"
  echo "[cc-run] records -> $OUT_DIR/records.jsonl"
else
  echo "[cc-run] no attempt windows — no records written"
fi
echo "[cc-run] done. results=$OUT_DIR/results.jsonl serve.log=$OUT_DIR/serve.log"
