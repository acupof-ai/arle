#!/bin/bash
# One-key CC-as-harness OPD collection: serve Qwen -> cc_swe_baseline (serial,
# emits time-window rows) -> cc-convert -> masked-CE records. Shell orchestrates;
# the per-task logic lives in Python (cc_swe_baseline.py). Owns the serve so it
# controls the --dump-messages-dir the windows attribute against.
#
#   bash scripts/cc_run.sh                 # edit the CONFIG block below, or:
#   GPU=1 DATASET=/host/x.jsonl bash scripts/cc_run.sh   # override per run
set -u

# ============================ CONFIG — edit here =============================
# Every knob defaults below and is overridable from the environment.
GPU=${GPU:-0}                                        # CUDA device for the serve
PORT=${PORT:-8000}                                   # serve port (must be free)
MODEL_PATH=${MODEL_PATH:-/host/Qwen3.6-27B-FP8}      # served checkpoint dir
DATASET=${DATASET:-/host/swe_run2/train.jsonl}       # SWE-Pro task JSONL
STAGED=${STAGED:-/host/swe_run2/staged}              # staged task trees
PYTHONPATH_TASK=${PYTHONPATH_TASK:-lib}              # scoring PYTHONPATH (ansible=lib)
LORA=${LORA:-}                                       # optional adapter (empty = none)
MAX_RUNNING=${MAX_RUNNING:-4}                        # serve --max-running-requests (>=2)
CC_TIMEOUT=${CC_TIMEOUT:-1800}                       # per-task wall-clock cap (no --max-turns exists)
OUT_DIR=${OUT_DIR:-/host/cc_run/$(date -u +%Y%m%d_%H%M%S)}
ARLE=${ARLE:-/host/arle-build/target/release/arle}
PY=${PY:-python3}
CLAUDE_BIN_DIRS=${CLAUDE_BIN_DIRS:-/host/npm-global/bin:/host/node-v22.14.0-linux-x64/bin}

# CC / Anthropic client env — self-hosted, offline. IS_SANDBOX=1 is MANDATORY on
# a root container (else --dangerously-skip-permissions refuses). ANTHROPIC_MODEL
# is auto-filled from /v1/models after the serve boots (leave unset).
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-dummy}
export IS_SANDBOX=${IS_SANDBOX:-1}
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}
export DISABLE_TELEMETRY=${DISABLE_TELEMETRY:-1}
export DISABLE_AUTOUPDATER=${DISABLE_AUTOUPDATER:-1}
export DISABLE_ERROR_REPORTING=${DISABLE_ERROR_REPORTING:-1}
# ========================== end CONFIG ======================================

BASE=http://127.0.0.1:$PORT
export ANTHROPIC_BASE_URL=$BASE
export PATH="$CLAUDE_BIN_DIRS:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

command -v claude >/dev/null || { echo "claude not on PATH — install once:
  npm i -g @anthropic-ai/claude-code  (node >=18)"; exit 1; }
mkdir -p "$OUT_DIR"; DUMP="$OUT_DIR/dumps"

curl -s --max-time 3 "$BASE/v1/models" >/dev/null 2>&1 && \
  { echo "port $PORT already serving — pick another PORT"; exit 1; }

# Own the serve so we own its dump dir. Kill only our PID on exit (never pkill).
CUDA_VISIBLE_DEVICES=$GPU nohup "$ARLE" serve --model-path "$MODEL_PATH" \
  --bind 0.0.0.0 --port "$PORT" --max-running-requests "$MAX_RUNNING" \
  --dump-messages-dir "$DUMP" \
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
"$PY" "$ROOT/scripts/cc_swe_baseline.py" \
  --dataset "$DATASET" --staged-root "$STAGED" --model "$MODEL_ID" \
  --cc-timeout "$CC_TIMEOUT" --pythonpath "$PYTHONPATH_TASK" \
  --work-root "$OUT_DIR/work" \
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
