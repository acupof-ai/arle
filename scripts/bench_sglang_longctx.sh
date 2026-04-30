#!/usr/bin/env bash
# First-class SGLang baseline runner for the ARLE 32k long-context benchmark.
#
# Pinned against the audited SGLang revision from:
#   docs/plans/2026-04-23-cuda-decode-sglang-alignment.md
#   docs/projects/2026-04-30-longctx-32k-128k-leadership.md Phase 1 S4
#
# Usage:
#   scripts/bench_sglang_longctx.sh <label>
#   scripts/bench_sglang_longctx.sh <label> --smoke
#
# Smoke-friendly overrides:
#   SMOKE=1                         512-in/128-out, c=1, max-seconds=30,
#                                   no c=1 secondary rerun.
#   --smoke                         first-class 5s harness validation path.
#   SGLANG_NO_LAUNCH=1              reuse an existing server at TARGET.
#   SGLANG_SERVER_COMMIT=<sha>       required to claim a pinned commit when
#                                   SGLANG_NO_LAUNCH=1 is used.
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
SGLANG_COMMIT_FOR_ARTIFACTS="$SGLANG_COMMIT"

PRESET_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=1
            SMOKE_MAX_SECONDS="${SMOKE_MAX_SECONDS:-5}"
            shift
            ;;
        --smoke-seconds)
            [[ $# -ge 2 ]] || { echo "error: --smoke-seconds requires a value" >&2; exit 2; }
            SMOKE=1
            SMOKE_MAX_SECONDS="$2"
            shift 2
            ;;
        *)
            PRESET_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PRESET_ARGS[@]}"

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
  --smoke for a first-class 5s validation shape
  SGLANG_NO_LAUNCH=1 to reuse an already-running SGLang server
  SGLANG_SERVER_COMMIT=$SGLANG_COMMIT to mark reused server as pin-verified
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --smoke|--smoke-seconds)
            echo "error: internal parser error: preset flag was not consumed: $1" >&2
            exit 2
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
        echo "SGLANG_COMMIT=$SGLANG_COMMIT_FOR_ARTIFACTS"
        if [[ "${SGLANG_NO_LAUNCH:-0}" == "1" ]]; then
            echo "SGLANG_PIN_EXPECTED=$SGLANG_COMMIT"
            echo "SGLANG_NO_LAUNCH=1"
        fi
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

validate_guidellm_results() {
    python3 - "$1" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
obj = json.loads(json_path.read_text())
benchmarks = obj.get("benchmarks") or []
if not benchmarks:
    print(f"guidellm validation failed: no benchmarks in {json_path}", file=sys.stderr)
    sys.exit(1)

errors = []
for bm in benchmarks:
    strategy = bm.get("config", {}).get("strategy", {})
    rate = strategy.get("type_", "unknown")
    if rate == "concurrent":
        rate = f"conc{strategy.get('max_concurrency', '?')}"
    metrics = bm.get("metrics", {})
    request_totals = metrics.get("request_totals", {})
    successful = int(request_totals.get("successful") or 0)
    output_mean = (metrics.get("output_token_count", {}).get("successful", {}) or {}).get("mean") or 0.0
    outputs = bm.get("requests", {}).get("successful", []) or []
    nonempty_outputs = sum(1 for req in outputs if (req.get("output") or "") != "")

    if successful <= 0:
        errors.append(f"{rate}: no successful requests recorded")
    elif output_mean > 0.0 and nonempty_outputs == 0:
        errors.append(
            f"{rate}: successful requests reported {output_mean:.1f} output tokens on average but every sampled output was empty"
        )

if errors:
    print(f"guidellm validation failed for {json_path}:", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    sys.exit(4)
PY
}

write_headline() {
    python3 - "$HEADLINE_JSON" "$HEADLINE_TABLE" "$SGLANG_COMMIT_FOR_ARTIFACTS" "$PRIMARY_DIR/benchmarks.json" "$SECONDARY_DIR/benchmarks.json" <<'PY'
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

write_wins_entry() {
    if [[ "${SMOKE:-0}" == "1" ]]; then
        echo ">>> smoke mode — skipping SGLang wins entry seed"
        return 0
    fi
    if [[ "${SGLANG_NO_LAUNCH:-0}" == "1" && "${SGLANG_SERVER_COMMIT:-}" != "$SGLANG_COMMIT" ]]; then
        echo ">>> reused SGLang server commit is unverified — skipping wins entry seed"
        echo "    set SGLANG_SERVER_COMMIT=$SGLANG_COMMIT only when the running server is pinned"
        return 0
    fi

    local wins_dir="$REPO_ROOT/docs/experience/wins"
    local wins_base="$wins_dir/${DATE}-bench-sglang-longctx-${LABEL}"
    local wins_file="${wins_base}.md"
    local wrun=1
    while [[ -e "$wins_file" ]]; do
        wrun=$((wrun + 1))
        wins_file="${wins_base}-run${wrun}.md"
    done

    python3 - "$wins_file" "$LABEL" "$DATE" "$SGLANG_COMMIT_FOR_ARTIFACTS" \
        "$SGLANG_COMMIT" "$MODEL" "$MODEL_PATH" "$TARGET" "$OUTPUT_DIR" \
        "$PROMPT_TOKENS" "$OUTPUT_TOKENS" "$CONCURRENCIES" "$MAX_SECONDS" \
        "$RUN_SECONDARY_C1" "$C1_SECONDS" "$HEADLINE_TABLE" \
        "$KV_CACHE_DTYPE" "$MAX_RUNNING_REQUESTS" "$MEM_FRACTION_STATIC" \
        "$MAX_TOTAL_TOKENS" "$RANDOM_SEED" <<'PY'
import pathlib
import sys

(
    wins_file,
    label,
    date,
    actual_commit,
    expected_commit,
    model,
    model_path,
    target,
    output_dir,
    prompt_tokens,
    output_tokens,
    concurrencies,
    max_seconds,
    run_secondary_c1,
    c1_seconds,
    table_file,
    kv_cache_dtype,
    max_running_requests,
    mem_fraction_static,
    max_total_tokens,
    random_seed,
) = sys.argv[1:]
table = pathlib.Path(table_file).read_text().rstrip()
data_spec = (
    f"prompt_tokens={prompt_tokens},prompt_tokens_stdev=1,"
    f"prompt_tokens_min={prompt_tokens},prompt_tokens_max={prompt_tokens},"
    f"output_tokens={output_tokens},output_tokens_stdev=1,"
    f"output_tokens_min={output_tokens},output_tokens_max={output_tokens}"
)
launch = (
    f"python3 -m sglang.launch_server --model-path {model_path} "
    f"--kv-cache-dtype {kv_cache_dtype} "
    f"--max-running-requests {max_running_requests} "
    f"--mem-fraction-static {mem_fraction_static} --disable-radix-cache "
    f"--max-total-tokens {max_total_tokens}"
)
body = f"""# SGLang longctx baseline — {label}, {date}

## Goal

- Baseline: capture the pinned SGLang longctx-32k reference row for ARLE
  Phase 1 S4.

## Hypothesis

- SGLang at the project pin provides the reproducible competitor baseline for
  prompt={prompt_tokens}, output={output_tokens}, c={concurrencies}.

## Command

```bash
scripts/bench_sglang_longctx.sh {label}
```

## Environment

- **Backend:** SGLang
- **Model:** {model}
- **Weights:** `{model_path}`
- **Target:** `{target}`
- **Commit:** `{actual_commit}`
- **Expected pin:** `{expected_commit}`
- **Launch:** `{launch}`

## Canonical params

- `--profile concurrent`
- `--data {data_spec}`
- `--rate {concurrencies}`
- `--max-seconds {max_seconds}`
- `--random-seed {random_seed}`
- Secondary c=1 run: `{run_secondary_c1}` (`{c1_seconds}s`)

## Results

{table}

## Problems

- Pending reviewer fill-in after the remote run if startup, CUDA memory, or
  GuideLLM validation deviated from the watch-list.

## Learnings

- Pending remote run.

## Delta vs baseline

- First pinned SGLang longctx-32k baseline for this mission slice.

## Artefacts

- Raw: `{output_dir}/`
- Primary: `{output_dir}/guidellm-primary/benchmarks.json`
- Secondary c=1: `{output_dir}/guidellm-c1-secondary/benchmarks.json`
- Headline: `{output_dir}/headline_table.md`
- Server log: `{output_dir}/sglang_server.log`
"""
pathlib.Path(wins_file).write_text(body)
PY

    echo "    wins    : $wins_file"
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
    if [[ "${SGLANG_SERVER_COMMIT:-}" == "$SGLANG_COMMIT" ]]; then
        SGLANG_COMMIT_FOR_ARTIFACTS="$SGLANG_SERVER_COMMIT"
        echo "    pin    : verified by SGLANG_SERVER_COMMIT"
    else
        SGLANG_COMMIT_FOR_ARTIFACTS="unverified-existing-server(expected:${SGLANG_COMMIT})"
        echo "    pin    : unverified existing server; wins entry will not be seeded"
    fi
else
    ensure_sglang_checkout
    echo "    server : launching SGLang"
    (
        cd "$SGLANG_DIR"
        export PYTHONPATH="$SGLANG_DIR/python:${PYTHONPATH:-}"
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
validate_guidellm_results "$PRIMARY_DIR/benchmarks.json"

if [[ "$RUN_SECONDARY_C1" == "1" && "$CONCURRENCIES" == *"1"* && "${SMOKE:-0}" != "1" ]]; then
    mkdir -p "$SECONDARY_DIR"
    run_guidellm "$SECONDARY_DIR" "1" "$C1_SECONDS"
    validate_guidellm_results "$SECONDARY_DIR/benchmarks.json"
fi

write_headline
write_wins_entry

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
