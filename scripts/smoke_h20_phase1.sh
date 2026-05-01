#!/usr/bin/env bash
# H20 single-card Phase 1 long-context smoke.
#
# Expected bundle layout:
#   ./bin/infer
#   ./scripts/bench_guidellm.sh
#   ./infer/models/Qwen3-4B/
#
# The default shape matches the Phase 1 W1/H1 longctx c=4 envelope:
#   32768 input tokens, 256 output tokens, concurrency=4, 300s.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    ROOT="$SCRIPT_DIR"
fi
BIN="${INFER_BIN:-$ROOT/bin/infer}"
MODEL_PATH="${MODEL_PATH:-$ROOT/infer/models/Qwen3-4B}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TARGET="${TARGET:-http://${HOST}:${PORT}}"
LABEL="${LABEL:-h20-phase1-longctx-c4}"
SERVER_LOG="${SERVER_LOG:-$ROOT/h20_phase1_server.log}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
NUM_SLOTS="${NUM_SLOTS:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required tool not found: $1" >&2
        exit 2
    fi
}

require_tool nvidia-smi
require_tool curl
require_tool awk
require_tool jq
require_tool guidellm

if [[ ! -x "$BIN" ]]; then
    echo "error: infer binary is missing or not executable: $BIN" >&2
    exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "error: model path missing: $MODEL_PATH" >&2
    echo "       place Qwen3-4B weights there or set MODEL_PATH=/path/to/Qwen3-4B" >&2
    exit 2
fi

compute_cap="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
        head -n1 |
        tr -d '[:space:]'
)"
if [[ -z "$compute_cap" ]]; then
    echo "error: unable to read GPU compute capability from nvidia-smi" >&2
    exit 2
fi
awk -v cc="$compute_cap" 'BEGIN { exit !(cc + 0 >= 9.0) }' || {
    echo "error: H20/SM90-class GPU required; nvidia-smi compute_cap=$compute_cap" >&2
    exit 2
}

echo "H20 smoke: compute_cap=$compute_cap target=$TARGET model=$MODEL_PATH"

if curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
    echo "error: server already responding at $TARGET; set PORT/TARGET or stop it first" >&2
    exit 2
fi

"$BIN" \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --kv-cache-dtype fp8 \
    --num-slots "$NUM_SLOTS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --schedule-policy fcfs \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

deadline=$((SECONDS + WAIT_SECONDS))
until curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "error: infer server exited during startup; log: $SERVER_LOG" >&2
        tail -120 "$SERVER_LOG" >&2 || true
        exit 3
    fi
    if (( SECONDS >= deadline )); then
        echo "error: infer server did not become ready within ${WAIT_SECONDS}s; log: $SERVER_LOG" >&2
        tail -120 "$SERVER_LOG" >&2 || true
        exit 3
    fi
    sleep 2
done

WORKLOAD=longctx-32k \
LONGCTX_CONCURRENCIES=4 \
LONGCTX_MAX_SECONDS=300 \
LONGCTX_SECONDARY_C1_ONLY=0 \
"$ROOT/scripts/bench_guidellm.sh" \
    "$LABEL" \
    --target "$TARGET" \
    --model Qwen/Qwen3-4B \
    --processor "$MODEL_PATH"

echo "H20 Phase 1 smoke complete. Inspect bench-output/ and generated wins entry for success(W1,H2) candidate data."
