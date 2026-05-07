#!/usr/bin/env bash
# M_nsys P1 — Spawn ARLE under nsys with --capture-range=cudaProfilerApi,
# fire SIGUSR1/SIGUSR2 around a guidellm load to delimit the capture window
# precisely. Bypasses the nsys 2025.6 kernel-density drop seen on long-ctx
# 4k/c=4 (`5.8 MB, 0 kernel data` in profile_nsys_guidellm.sh runs).
#
# Pre-req: ARLE main.rs has install_cuda_profiler_signal_handlers (M_nsys
# P0, commit 9b1fb8c). Without it, the SIGUSR1/USR2 sends become no-ops
# and you'll get an empty capture window.
#
# Flow:
#   1. Spawn `nsys profile --capture-range=cudaProfilerApi <ARLE>` in bg.
#   2. Wait for ARLE /v1/stats ready.
#   3. Fire SIGUSR1 → cuProfilerStart on ARLE.
#   4. Run bench (defaults to bench_guidellm.sh --fast).
#   5. Fire SIGUSR2 → cuProfilerStop.
#   6. nsys captures only between USR1 and USR2 → no kernel-density drop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LABEL=""
TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
BENCH_DIR=""
BENCH_PRESET="fast"
CONCURRENCIES=""
MAX_SECONDS=""
WARMUP=""
SERVER_BIN="${REPO_ROOT}/target/release/infer"
SERVER_ARGS=""
TRACE_SET="cuda,nvtx,osrt"
CUDA_GRAPH_TRACE="node"
READY_TIMEOUT=120

usage() {
    cat <<EOF
M_nsys P1 — capture-range=cudaProfilerApi nsys wrapper.

Usage:
  $(basename "$0") <label> --server-args "<args>" [options]

Required:
  --server-args ARGS      args passed to ${SERVER_BIN}
                          e.g. "--model models/Qwen3-4B --max-seq-len 4096
                          --port 8000 --num-slots 8 ..."

Bench:
  --bench DIR             reuse existing bench-output dir + replay command.txt
  --fast                  short load via bench_guidellm.sh --fast (default)
  --quick                 1,2,4,8 concurrency quick sweep
  --concurrencies LIST    forwarded to bench_guidellm.sh
  --max-seconds N         forwarded to bench_guidellm.sh
  --warmup N              forwarded to bench_guidellm.sh
  --target URL            default: ${TARGET}
  --model NAME            default: ${MODEL}

nsys:
  --trace LIST            default: ${TRACE_SET}
  --cuda-graph-trace MODE default: ${CUDA_GRAPH_TRACE}
  --ready-timeout N       default: ${READY_TIMEOUT} (sec to wait for /v1/stats)

Examples:
  $(basename "$0") longctx-4k-c4 --server-args \\
    "--model models/Qwen3-4B --max-seq-len 4097 --port 8000 --num-slots 8" \\
    --concurrencies 4 --max-seconds 60
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-args) SERVER_ARGS="$2"; shift 2 ;;
        --bench)       BENCH_DIR="$2"; shift 2 ;;
        --fast)        BENCH_PRESET="fast"; shift ;;
        --quick)       BENCH_PRESET="quick"; shift ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
        --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
        --warmup)      WARMUP="$2"; shift 2 ;;
        --target)      TARGET="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --trace)       TRACE_SET="$2"; shift 2 ;;
        --cuda-graph-trace) CUDA_GRAPH_TRACE="$2"; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        --*)           echo "error: unknown flag: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -z "$LABEL" ]]; then LABEL="$1"; shift
            else echo "error: unexpected positional arg: $1" >&2; exit 2; fi ;;
    esac
done

if [[ -z "$LABEL" || -z "$SERVER_ARGS" ]]; then
    echo "error: <label> and --server-args are required" >&2
    usage >&2
    exit 2
fi

OUTPUT_DIR="${REPO_ROOT}/bench-output/$(date +%Y-%m-%d)-${LABEL}-profile-nsys-signal"
mkdir -p "$OUTPUT_DIR"
PROFILE_BASE="${OUTPUT_DIR}/trace"
SERVER_LOG="${OUTPUT_DIR}/server.log"
BENCH_LOG="${OUTPUT_DIR}/bench-anchor.log"
ENV_FILE="${OUTPUT_DIR}/env.txt"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"

command -v nsys >/dev/null || { echo "error: nsys not on PATH"; exit 5; }
[[ -x "$SERVER_BIN" ]] || { echo "error: missing server binary: $SERVER_BIN (run cargo build --release first)"; exit 5; }

NSYS_CMD=(
    nsys profile
    --output "$PROFILE_BASE"
    --force-overwrite=true
    --trace "$TRACE_SET"
    --cuda-graph-trace "$CUDA_GRAPH_TRACE"
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --export=sqlite
    --kill none
    -- "$SERVER_BIN"
)
# shellcheck disable=SC2206
SERVER_ARGS_ARRAY=($SERVER_ARGS)
NSYS_CMD+=("${SERVER_ARGS_ARRAY[@]}")

{
    echo "label=${LABEL}"
    echo "target=${TARGET}"
    echo "trace=${TRACE_SET}"
    echo "cuda_graph_trace=${CUDA_GRAPH_TRACE}"
    echo "commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "nsys_version=$(nsys --version 2>/dev/null | head -n1 || echo unavailable)"
    echo "server_args=${SERVER_ARGS}"
} > "$ENV_FILE"

{
    printf 'nsys'
    for arg in "${NSYS_CMD[@]:1}"; do printf ' %q' "$arg"; done
    printf '\n'
} > "$COMMAND_FILE"

echo ">>> spawn nsys + ARLE under cudaProfilerApi capture range"
echo "    output : ${OUTPUT_DIR}"
echo

"${NSYS_CMD[@]}" >"$SERVER_LOG" 2>&1 &
NSYS_PID=$!

cleanup() {
    if kill -0 "$NSYS_PID" 2>/dev/null; then
        echo ">>> cleanup: stop ARLE (TERM)"
        # Find ARLE PID under nsys; nsys spawns the target as child
        ARLE_PID="$(pgrep -P "$NSYS_PID" -f infer 2>/dev/null || true)"
        [[ -n "$ARLE_PID" ]] && kill -TERM "$ARLE_PID" 2>/dev/null || true
        sleep 5
        kill -0 "$NSYS_PID" 2>/dev/null && kill -TERM "$NSYS_PID" 2>/dev/null || true
        wait "$NSYS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for /v1/stats ready
echo ">>> waiting for ARLE ready at ${TARGET}/v1/stats (timeout ${READY_TIMEOUT}s)"
WAIT_START=$(date +%s)
until curl -sm 1 "${TARGET}/v1/stats" >/dev/null 2>&1; do
    sleep 2
    NOW=$(date +%s)
    if (( NOW - WAIT_START > READY_TIMEOUT )); then
        echo "error: ARLE did not become ready in ${READY_TIMEOUT}s" >&2
        echo "       server log: ${SERVER_LOG}" >&2
        exit 4
    fi
    if ! kill -0 "$NSYS_PID" 2>/dev/null; then
        echo "error: nsys/ARLE exited before ready" >&2
        echo "       server log: ${SERVER_LOG}" >&2
        exit 4
    fi
done
echo "    ready"

ARLE_PID="$(pgrep -P "$NSYS_PID" -f infer 2>/dev/null | head -1)"
if [[ -z "$ARLE_PID" ]]; then
    echo "error: cannot resolve ARLE child PID under nsys $NSYS_PID" >&2
    exit 4
fi
echo "    ARLE PID: ${ARLE_PID}"

# Build bench load command
LOAD_CMD=()
if [[ -n "$BENCH_DIR" ]]; then
    [[ -f "${BENCH_DIR}/command.txt" ]] || { echo "error: missing ${BENCH_DIR}/command.txt"; exit 2; }
    LOAD_CMD=(bash -lc "cat ${BENCH_DIR}/command.txt | bash")
else
    LOAD_CMD=("${REPO_ROOT}/scripts/bench_guidellm.sh" "$LABEL"
        --target "$TARGET" --model "$MODEL")
    case "$BENCH_PRESET" in
        fast)  LOAD_CMD+=(--fast) ;;
        quick) LOAD_CMD+=(--quick) ;;
    esac
    [[ -n "$CONCURRENCIES" ]] && LOAD_CMD+=(--concurrencies "$CONCURRENCIES")
    [[ -n "$MAX_SECONDS" ]]   && LOAD_CMD+=(--max-seconds "$MAX_SECONDS")
    [[ -n "$WARMUP" ]]        && LOAD_CMD+=(--warmup "$WARMUP")
fi

echo ">>> SIGUSR1 → cuProfilerStart"
kill -USR1 "$ARLE_PID"
sleep 1
echo ">>> bench: ${LOAD_CMD[*]}"
set +e
"${LOAD_CMD[@]}" 2>&1 | tee "$BENCH_LOG"
LOAD_RC=${PIPESTATUS[0]}
set -e
echo ">>> SIGUSR2 → cuProfilerStop"
kill -USR2 "$ARLE_PID"
sleep 3

echo ">>> stop ARLE → nsys flushes report"
kill -TERM "$ARLE_PID" 2>/dev/null || true
wait "$NSYS_PID" 2>/dev/null || true
trap - EXIT INT TERM

if [[ ! -f "${PROFILE_BASE}.nsys-rep" ]]; then
    echo "error: nsys did not produce ${PROFILE_BASE}.nsys-rep" >&2
    echo "       server log: ${SERVER_LOG}" >&2
    exit 4
fi

KERNEL_REPORT="${OUTPUT_DIR}/cuda_gpu_kern_sum.txt"
API_REPORT="${OUTPUT_DIR}/cuda_api_sum.txt"
nsys stats --report cuda_gpu_kern_sum "${PROFILE_BASE}.nsys-rep" >"$KERNEL_REPORT" 2>&1 || true
nsys stats --report cuda_api_sum     "${PROFILE_BASE}.nsys-rep" >"$API_REPORT"    2>&1 || true

REP_BYTES="$(stat -c%s "${PROFILE_BASE}.nsys-rep" 2>/dev/null || echo 0)"
KERN_LINES="$(wc -l <"$KERNEL_REPORT" 2>/dev/null || echo 0)"

echo
echo ">>> done"
echo "    rep    : ${PROFILE_BASE}.nsys-rep ($REP_BYTES bytes)"
echo "    kernels: ${KERNEL_REPORT} ($KERN_LINES lines)"
echo "    api    : ${API_REPORT}"
echo "    bench  : ${BENCH_LOG}"
echo "    server : ${SERVER_LOG}"

if (( REP_BYTES < 10485760 )); then
    echo "    WARN   : .nsys-rep < 10MB — verify kernel data is present" >&2
fi

exit "$LOAD_RC"
