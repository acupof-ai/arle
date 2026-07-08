#!/bin/bash
# One-key CC-as-harness OPD collection: serve Qwen -> cc_swe_baseline (serial,
# emits time-window rows) -> cc-convert -> masked-CE records. Shell orchestrates;
# the per-task logic lives in Python (cc_swe_baseline.py). Owns the serve so it
# controls the --dump-messages-dir the windows attribute against.
#
#   GPU=0 DATASET=/host/swe_run2/train.jsonl STAGED=/host/swe_run2/staged \
#   PYTHONPATH_TASK=lib bash scripts/cc_run.sh
set -u

GPU=${GPU:-0}; PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-/host/Qwen3.6-27B-FP8}
DATASET=${DATASET:-/host/swe_run2/train.jsonl}
STAGED=${STAGED:-/host/swe_run2/staged}
PYTHONPATH_TASK=${PYTHONPATH_TASK:-lib}           # scoring PYTHONPATH (ansible=lib)
LORA=${LORA:-}                                     # optional adapter
ARLE=${ARLE:-/host/arle-build/target/release/arle}
PY=${PY:-python3}
OUT_DIR=${OUT_DIR:-/host/cc_run/$(date -u +%Y%m%d_%H%M%S)}
BASE=http://127.0.0.1:$PORT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/host/npm-global/bin:/host/node-v22.14.0-linux-x64/bin:$PATH"

command -v claude >/dev/null || { echo "claude not on PATH — install once:
  npm i -g @anthropic-ai/claude-code  (node >=18)"; exit 1; }
mkdir -p "$OUT_DIR"; DUMP="$OUT_DIR/dumps"

# CC self-hosted, offline. IS_SANDBOX=1 is mandatory (root container blocks
# --dangerously-skip-permissions without it).
export ANTHROPIC_BASE_URL=$BASE ANTHROPIC_API_KEY=dummy IS_SANDBOX=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 DISABLE_ERROR_REPORTING=1

curl -s --max-time 3 "$BASE/v1/models" >/dev/null 2>&1 && \
  { echo "port $PORT already serving — pick another PORT"; exit 1; }

# Own the serve so we own its dump dir. Kill only our PID on exit (never pkill).
CUDA_VISIBLE_DEVICES=$GPU nohup "$ARLE" serve --model-path "$MODEL_PATH" \
  --bind 0.0.0.0 --port "$PORT" --max-running-requests 4 --dump-messages-dir "$DUMP" \
  ${LORA:+--lora-adapters "$LORA" --lora-alpha 32} > "$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
trap '[ -n "${SERVE_PID:-}" ] && kill "$SERVE_PID" 2>/dev/null' EXIT

echo "[cc-run] serve pid=$SERVE_PID gpu=$GPU port=$PORT — waiting for model load…"
for _ in $(seq 1 90); do
  curl -s --max-time 3 "$BASE/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve died — see $OUT_DIR/serve.log"; exit 1; }
  sleep 10
done
MODEL_ID=$(curl -s "$BASE/v1/models" | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
export ANTHROPIC_MODEL=$MODEL_ID ANTHROPIC_SMALL_FAST_MODEL=$MODEL_ID
echo "[cc-run] serve up, model=$MODEL_ID"

# Serial collection: cc_swe_baseline emits one time-window row per PASSING attempt.
ANTHROPIC_BASE_URL=$BASE "$PY" "$ROOT/scripts/cc_swe_baseline.py" \
  --dataset "$DATASET" --staged-root "$STAGED" --model "$MODEL_ID" \
  --pythonpath "$PYTHONPATH_TASK" --work-root "$OUT_DIR/work" \
  --windows-out "$OUT_DIR/windows.jsonl" --out "$OUT_DIR/results.jsonl"

# Windows -> training records (skip if nothing passed).
if [ -s "$OUT_DIR/windows.jsonl" ]; then
  "$ARLE" train cc-convert --dump-dir "$DUMP" \
    --tokenizer "$MODEL_PATH/tokenizer.json" \
    --windows "$OUT_DIR/windows.jsonl" --out "$OUT_DIR/records.jsonl"
  echo "[cc-run] records -> $OUT_DIR/records.jsonl"
else
  echo "[cc-run] no passing windows — no records written"
fi
echo "[cc-run] done. results=$OUT_DIR/results.jsonl serve.log=$OUT_DIR/serve.log"
