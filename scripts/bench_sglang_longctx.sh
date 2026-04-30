#!/usr/bin/env bash
# First-class SGLang baseline runner for the ARLE 32k long-context benchmark.
#
# Pinned against the audited SGLang revision from:
#   docs/plans/2026-04-23-cuda-decode-sglang-alignment.md
#   docs/projects/2026-04-30-longctx-32k-128k-leadership.md Phase 1 S4
#
# Usage:
#   scripts/bench_sglang_longctx.sh <label>
#
# Smoke-friendly overrides:
#   SMOKE=1                         512-in/128-out, c=1, max-seconds=30,
#                                   no c=1 secondary rerun.
#   SGLANG_NO_LAUNCH=1              reuse an existing server at TARGET.
#   PROMPT_TOKENS=... OUTPUT_TOKENS=...
#   CONCURRENCIES=... MAX_SECONDS=...
#   RUN_SECONDARY_C1=0|1 C1_SECONDS=...

set -euo pipefail

SGLANG_COMMIT="214c35b03184c354acf1f86f99746799e1c9b3a9"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-infer/models/Qwen3-4B}"
PROCESSOR="${PROCESSOR:-$MODEL_PATH}"
SGLANG_DIR="${SGLANG_DIR:-/tmp/sglang-arle-${SGLANG_COMMIT}}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
TARGET="${TARGET:-http://localhost:${PORT}}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
CONCURRENCIES="${CONCURRENCIES:-1,4}"
MAX_SECONDS="${MAX_SECONDS:-300}"
C1_SECONDS="${C1_SECONDS:-360}"
RUN_SECONDARY_C1="${RUN_SECONDARY_C1:-1}"
RANDOM_SEED="${RANDOM_SEED:-20260416}"
BACKEND="${BACKEND:-openai_http}"
BACKEND_KWARGS="${BACKEND_KWARGS:-{\"validate_backend\":\"/v1/models\",\"request_format\":\"/v1/completions\"}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-140000}"  # 4 * (32768 + 256) + margin
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
OUTPUTS=(json csv html)
LABEL=""

if [[ "${SMOKE:-0}" == "1" ]]; then
    PROMPT_TOKENS="${SMOKE_PROMPT_TOKENS:-512}"
    OUTPUT_TOKENS="${SMOKE_OUTPUT_TOKENS:-128}"
    CONCURRENCIES="${SMOKE_CONCURRENCIES:-1}"
    MAX_SECONDS="${SMOKE_MAX_SECONDS:-30}"
    RUN_SECONDARY_C1="${SMOKE_RUN_SECONDARY_C1:-0}"
fi

usage() {
    cat <<EOF
usage: $(basename "$0") <label>

Runs SGLang at commit $SGLANG_COMMIT with the longctx-32k GuideLLM shape.

Environment:
  TARGET=$TARGET
  MODEL=$MODEL
  MODEL_PATH=$MODEL_PATH
  SGLANG_DIR=$SGLANG_DIR
  PROMPT_TOKENS=$PROMPT_TOKENS OUTPUT_TOKENS=$OUTPUT_TOKENS
  CONCURRENCIES=$CONCURRENCIES MAX_SECONDS=$MAX_SECONDS
  RUN_SECONDARY_C1=$RUN_SECONDARY_C1 C1_SECONDS=$C1_SECONDS
  SMOKE=1 for a short non-32k CUDA-light validation shape
  SGLANG_NO_LAUNCH=1 to reuse an already-running SGLang server
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -z "$LABEL" ]]; then
                LABEL="$1"
                shift
            else
                echo "error: unexpected positional arg: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "$LABEL" ]]; then
    echo "error: <label> is required" >&2
    usage >&2
    exit 2
fi

for tool in git guidellm curl python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not on PATH: $tool" >&2
        exit 2
    fi
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="$(date +%Y-%m-%d)"
base_dir="$REPO_ROOT/bench-output/${DATE}-sglang-${LABEL}"
OUTPUT_DIR="$base_dir"
run=1
while [[ -e "$OUTPUT_DIR" ]]; do
    run=$((run + 1))
    OUTPUT_DIR="${base_dir}-run${run}"
done
mkdir -p "$OUTPUT_DIR"

SERVER_LOG="$OUTPUT_DIR/sglang_server.log"
PRIMARY_DIR="$OUTPUT_DIR/guidellm-primary"
SECONDARY_DIR="$OUTPUT_DIR/guidellm-c1-secondary"
HEADLINE_JSON="$OUTPUT_DIR/sglang_headline.json"
HEADLINE_TABLE="$OUTPUT_DIR/headline_table.md"
mkdir -p "$PRIMARY_DIR"

if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$REPO_ROOT/$MODEL_PATH"
fi
if [[ "$PROCESSOR" != /* && -d "$REPO_ROOT/$PROCESSOR" ]]; then
    PROCESSOR="$REPO_ROOT/$PROCESSOR"
fi

ensure_sglang_checkout() {
    if [[ ! -d "$SGLANG_DIR/.git" ]]; then
        git clone "$SGLANG_REPO" "$SGLANG_DIR"
    fi
    git -C "$SGLANG_DIR" fetch origin main
    git -C "$SGLANG_DIR" checkout --detach "$SGLANG_COMMIT"
    local actual
    actual="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
    if [[ "$actual" != "$SGLANG_COMMIT" ]]; then
        echo "error: SGLang checkout mismatch: expected $SGLANG_COMMIT got $actual" >&2
        exit 2
    fi
}

wait_for_server() {
    local deadline=$((SECONDS + WAIT_SECONDS))
    until curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "error: SGLang server did not become ready within ${WAIT_SECONDS}s" >&2
            echo "       log: $SERVER_LOG" >&2
            exit 3
        fi
        # Verify SGLang process is still alive (if we launched it)
        if [[ "${SGLANG_NO_LAUNCH:-0}" != "1" ]] && ! kill -0 "$SGLANG_PID" 2>/dev/null; then
            echo "error: SGLang process $SGLANG_PID exited during startup" >&2
            echo "       log: $SERVER_LOG" >&2
            exit 3
        fi
        sleep 2
    done
}

run_guidellm() {
    local out_dir="$1"
    local conc="$2"
    local seconds="$3"
    local log_file="$out_dir/guidellm.log"
    local cmd_file="$out_dir/command.txt"
    mkdir -p "$out_dir"

    local data
    data="prompt_tokens=${PROMPT_TOKENS},prompt_tokens_stdev=1,prompt_tokens_min=${PROMPT_TOKENS},prompt_tokens_max=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS},output_tokens_stdev=1,output_tokens_min=${OUTPUT_TOKENS},output_tokens_max=${OUTPUT_TOKENS}"

    local args=(
        --target "$TARGET"
        --model "$MODEL"
        --processor "$PROCESSOR"
        --profile concurrent
        --data "$data"
        --max-seconds "$seconds"
        --random-seed "$RANDOM_SEED"
        --output-dir "$out_dir"
        --backend "$BACKEND"
        --backend-kwargs "$BACKEND_KWARGS"
        --disable-console-interactive
        --rate "$conc"
    )
    local output
    for output in "${OUTPUTS[@]}"; do
        args+=(--outputs "$output")
    done

    {
        echo "SGLANG_COMMIT=$SGLANG_COMMIT"
        echo "SGLANG_LAUNCH_PARAMS=--model-path $MODEL_PATH --kv-cache-dtype $KV_CACHE_DTYPE --max-running-requests $MAX_RUNNING_REQUESTS --mem-fraction-static $MEM_FRACTION_STATIC --disable-radix-cache --max-total-tokens $MAX_TOTAL_TOKENS"
        printf 'guidellm benchmark run'
        local arg
        for arg in "${args[@]}"; do
            printf ' %q' "$arg"
        done
        printf '\n'
    } > "$cmd_file"

    guidellm benchmark run "${args[@]}" 2>&1 | tee "$log_file"
}

write_headline() {
    python3 - "$HEADLINE_JSON" "$HEADLINE_TABLE" "$SGLANG_COMMIT" "$PRIMARY_DIR/benchmarks.json" "$SECONDARY_DIR/benchmarks.json" <<'PY'
import json
import pathlib
import sys

out_json, out_table, commit, *inputs = sys.argv[1:]
rows = []

def rate_name(strategy):
    if strategy.get("type_") == "concurrent":
        return f"conc{strategy.get('max_concurrency', '?')}"
    return strategy.get("type_", "unknown")

for input_path in inputs:
    path = pathlib.Path(input_path)
    if not path.exists():
        continue
    obj = json.loads(path.read_text())
    for bm in obj.get("benchmarks") or []:
        metrics = bm.get("metrics", {})
        ttft = metrics.get("time_to_first_token_ms", {}).get("successful", {})
        itl = metrics.get("inter_token_latency_ms", {}).get("successful", {})
        rows.append({
            "source": str(path),
            "rate": rate_name(bm.get("config", {}).get("strategy", {})),
            "output_tokens_per_second": metrics.get("output_tokens_per_second", {}).get("successful", {}).get("mean"),
            "tokens_per_second": metrics.get("tokens_per_second", {}).get("successful", {}).get("mean"),
            "requests_per_second": metrics.get("requests_per_second", {}).get("successful", {}).get("mean"),
            "ttft_p50_ms": (ttft.get("percentiles") or {}).get("p50"),
            "ttft_p99_ms": (ttft.get("percentiles") or {}).get("p99"),
            "itl_p50_ms": (itl.get("percentiles") or {}).get("p50"),
            "itl_p99_ms": (itl.get("percentiles") or {}).get("p99"),
        })

payload = {"sglang_commit": commit, "benchmarks": rows}
pathlib.Path(out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)

lines = [
    "| rate | out tok/s | total tok/s | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['rate']} | {fmt(row['output_tokens_per_second'])} | "
        f"{fmt(row['tokens_per_second'])} | {fmt(row['requests_per_second'])} | "
        f"{fmt(row['ttft_p50_ms'])} | {fmt(row['ttft_p99_ms'])} | "
        f"{fmt(row['itl_p50_ms'])} | {fmt(row['itl_p99_ms'])} |"
    )
pathlib.Path(out_table).write_text("\n".join(lines) + "\n")
PY
}

SGLANG_PID=""
cleanup() {
    if [[ -n "$SGLANG_PID" ]] && kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
        kill "$SGLANG_PID" >/dev/null 2>&1 || true
        wait "$SGLANG_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo ">>> SGLang longctx baseline"
echo "    commit : $SGLANG_COMMIT"
echo "    target : $TARGET"
echo "    model  : $MODEL"
echo "    weights: $MODEL_PATH"
echo "    data   : ${PROMPT_TOKENS}-in/${OUTPUT_TOKENS}-out"
echo "    conc   : $CONCURRENCIES"
echo "    seconds: $MAX_SECONDS"
echo "    output : $OUTPUT_DIR"

if [[ "${SGLANG_NO_LAUNCH:-0}" == "1" ]]; then
    echo "    server : reusing existing target"
else
    ensure_sglang_checkout
    echo "    server : launching SGLang"
    (
        cd "$SGLANG_DIR"
        export PYTHONPATH="$SGLANG_DIR/python:$PYTHONPATH"
        python3 -m sglang.launch_server \
            --host "$HOST" \
            --port "$PORT" \
            --model-path "$MODEL_PATH" \
            --kv-cache-dtype "$KV_CACHE_DTYPE" \
            --max-running-requests "$MAX_RUNNING_REQUESTS" \
            --mem-fraction-static "$MEM_FRACTION_STATIC" \
            --disable-radix-cache \
            --max-total-tokens "$MAX_TOTAL_TOKENS"
    ) > "$SERVER_LOG" 2>&1 &
    SGLANG_PID=$!
fi

wait_for_server
run_guidellm "$PRIMARY_DIR" "$CONCURRENCIES" "$MAX_SECONDS"

if [[ "$RUN_SECONDARY_C1" == "1" && "$CONCURRENCIES" == *"1"* && "${SMOKE:-0}" != "1" ]]; then
    mkdir -p "$SECONDARY_DIR"
    run_guidellm "$SECONDARY_DIR" "1" "$C1_SECONDS"
fi

write_headline

echo
echo ">>> headline table"
cat "$HEADLINE_TABLE"
echo
echo ">>> artefacts"
echo "    raw     : $OUTPUT_DIR"
echo "    primary : $PRIMARY_DIR/benchmarks.json"
echo "    c1 sec  : $SECONDARY_DIR/benchmarks.json"
echo "    headline: $HEADLINE_JSON"
echo "    server  : $SERVER_LOG"
