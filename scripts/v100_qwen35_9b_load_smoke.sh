#!/usr/bin/env bash
# V100 Qwen3.5-9B load smoke.
#
# Intended to run on the configured `v100` host from the ARLE repo root after
# the 4B capability eval has released port/GPU memory. This script deliberately
# does not run an eval sweep: it only starts `arle serve`, verifies one short
# `/v1/completions` request returns HTTP 200 with output, then stops the server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8123}"
HOST="${HOST:-127.0.0.1}"
TARGET="${TARGET:-http://${HOST}:${PORT}}"
ARLE_BIN="${ARLE_BIN:-target/release/arle}"
PYTHON="${PYTHON:-python3}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
SERVER_LOG="${SERVER_LOG:-/tmp/arle_v100_qwen35_9b_load_smoke.log}"
SMOKE_PROMPT="${SMOKE_PROMPT:-Hello}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-4}"

MODEL_PATH="${MODEL_PATH:-/home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B}"
if [[ ! -d "$MODEL_PATH" && -d /home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B ]]; then
    MODEL_PATH="/home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B"
fi

if [[ -d /usr/local/cuda-12.4 ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
else
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export RUST_LOG="${RUST_LOG:-info}"

server_pid=""
models_body=""
response_body=""

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    [[ -n "$models_body" ]] && rm -f "$models_body"
    [[ -n "$response_body" ]] && rm -f "$response_body"
}
trap cleanup EXIT

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required tool not found: $1" >&2
        exit 2
    fi
}

port_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
        return
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$PORT" 2>/dev/null |
            sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
            sort -u
        return
    fi
    pgrep -f "(target/release/(infer|arle).*--port[ =]${PORT}|arle serve .*--port[ =]${PORT})([[:space:]]|$)" 2>/dev/null || true
}

require_tool curl
require_tool "$PYTHON"

if [[ ! -x "$ARLE_BIN" ]]; then
    echo "error: arle binary is missing or not executable: $ARLE_BIN" >&2
    echo "       build first, or set ARLE_BIN=/path/to/arle" >&2
    exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "error: Qwen3.5-9B model path missing: $MODEL_PATH" >&2
    echo "       set MODEL_PATH=/path/to/Qwen3.5-9B" >&2
    exit 2
fi

mapfile -t existing_pids < <(port_pids)
if ((${#existing_pids[@]} > 0)); then
    echo "stopping existing listener(s) on ${HOST}:${PORT}: ${existing_pids[*]}"
    kill "${existing_pids[@]}" >/dev/null 2>&1 || true
    for _ in {1..20}; do
        mapfile -t existing_pids < <(port_pids)
        ((${#existing_pids[@]} == 0)) && break
        sleep 0.5
    done
    if ((${#existing_pids[@]} > 0)); then
        echo "forcing existing listener(s) on ${HOST}:${PORT}: ${existing_pids[*]}"
        kill -9 "${existing_pids[@]}" >/dev/null 2>&1 || true
    fi
fi

echo "starting Qwen3.5-9B smoke server: target=$TARGET model=$MODEL_PATH log=$SERVER_LOG"
"$ARLE_BIN" serve \
    --backend cuda \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    -- \
    --num-slots 1 \
    --max-seq-len 2048 \
    --kv-cache-dtype bf16 \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

deadline=$((SECONDS + WAIT_SECONDS))
until grep -q "Server listening" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "error: arle serve exited before readiness; log: $SERVER_LOG" >&2
        tail -120 "$SERVER_LOG" >&2 || true
        exit 3
    fi
    if ((SECONDS >= deadline)); then
        echo "error: arle serve did not log readiness within ${WAIT_SECONDS}s; log: $SERVER_LOG" >&2
        tail -120 "$SERVER_LOG" >&2 || true
        exit 3
    fi
    sleep 2
done

models_body="$(mktemp)"
curl -fsS "$TARGET/v1/models" -o "$models_body"
served_model="$("$PYTHON" - "$models_body" "$MODEL_PATH" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
items = payload.get("data") or []
if items and isinstance(items[0], dict) and items[0].get("id"):
    print(items[0]["id"])
else:
    print(os.path.basename(sys.argv[2].rstrip("/")))
PY
)"

response_body="$(mktemp)"
http_code="$(
    curl -sS -o "$response_body" -w "%{http_code}" \
        -X POST "$TARGET/v1/completions" \
        -H "content-type: application/json" \
        --data-binary "$(
            "$PYTHON" - "$served_model" "$SMOKE_PROMPT" "$SMOKE_MAX_TOKENS" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "prompt": sys.argv[2],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0.0,
}))
PY
        )"
)"

if [[ "$http_code" != "200" ]]; then
    echo "error: smoke request returned HTTP $http_code" >&2
    cat "$response_body" >&2 || true
    echo >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 4
fi

"$PYTHON" - "$response_body" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

choices = payload.get("choices") or []
text = ""
if choices and isinstance(choices[0], dict):
    text = choices[0].get("text") or choices[0].get("message", {}).get("content") or ""
usage = payload.get("usage") or {}
completion_tokens = int(usage.get("completion_tokens") or 0)
if completion_tokens < 1 and not text:
    raise SystemExit(f"no completion token/text in response: {json.dumps(payload)[:500]}")
print(f"smoke ok: status=200 completion_tokens={completion_tokens} text={text!r}")
PY

echo "Qwen3.5-9B V100 load smoke complete; stopping server pid=$server_pid"
