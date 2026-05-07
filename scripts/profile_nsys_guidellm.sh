#!/usr/bin/env bash
# Bench-anchored Nsight Systems wrapper for the infer service path.
#
# Default flow:
#   1. Attach `nsys` to an already-running infer server (PID resolved from --target).
#   2. Drive a short `bench_guidellm.sh --fast` load against the same server.
#   3. Export `.nsys-rep` + `.sqlite` + stats + a short markdown summary.
#
# Reuse flow:
#   scripts/profile_nsys_guidellm.sh <label> --bench bench-output/<date>-<label>
#   Replays the exact guidellm command recorded in `command.txt` and links the
#   new profile capture back to the existing bench anchor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./profile_guidellm_common.sh
source "${SCRIPT_DIR}/profile_guidellm_common.sh"

TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
PROCESSOR=""
TRACE_INTERVAL_MS=1000
SERVER_PID=""
BENCH_DIR=""
BENCH_PRESET="fast"
CONCURRENCIES=""
BENCH_PROFILE=""
MAX_SECONDS=""
WARMUP=""
LABEL=""
DELAY_SECONDS=5
DURATION_SECONDS=10
TRACE_SET="cuda,nvtx,osrt"
CUDA_GRAPH_TRACE="node"
SAMPLE_MODE="none"
CPUCTXSW_MODE="none"
DRY_RUN=false

usage() {
    cat <<EOF
Bench-anchored Nsight Systems wrapper for infer.

Usage:
  $(basename "$0") <label> [options]

Bench anchor:
  --bench DIR             reuse an existing bench-output dir as the anchor
                          and replay DIR/command.txt for the profiling load
  --fast                  short load anchor via bench_guidellm.sh --fast
                          (default when --bench is not provided)
  --quick                 1,2,4,8 concurrency quick sweep anchor
  --target URL            default: ${TARGET}
  --model NAME            default: ${MODEL}
  --processor PATH        forwarded to bench_guidellm.sh
  --trace-interval-ms N   forwarded to bench_guidellm.sh (default: ${TRACE_INTERVAL_MS})
  --concurrencies LIST    forwarded to bench_guidellm.sh
  --profile TYPE          forwarded to bench_guidellm.sh
  --max-seconds N         forwarded to bench_guidellm.sh
  --warmup N              forwarded to bench_guidellm.sh

Profiler:
  --server-pid PID        explicit infer server PID (else resolve from --target via lsof)
  --delay-seconds N       nsys delay before capture (default: ${DELAY_SECONDS})
  --duration-seconds N    nsys capture duration (default: ${DURATION_SECONDS})
  --trace LIST            default: ${TRACE_SET}
  --cuda-graph-trace MODE default: ${CUDA_GRAPH_TRACE}
  --sample MODE           default: ${SAMPLE_MODE}
  --cpuctxsw MODE         default: ${CPUCTXSW_MODE}
  --dry-run               print resolved commands without executing them

Examples:
  scripts/profile_nsys_guidellm.sh cuda-qwen3 --target http://127.0.0.1:8000
  scripts/profile_nsys_guidellm.sh cuda-qwen3 --quick --delay-seconds 8 --duration-seconds 12
  scripts/profile_nsys_guidellm.sh cuda-qwen3 --bench bench-output/2026-04-22-cuda-qwen3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)
            [[ $# -ge 2 ]] || { echo "error: --bench requires a value" >&2; exit 2; }
            BENCH_DIR="$2"; shift 2 ;;
        --fast)
            BENCH_PRESET="fast"; shift ;;
        --quick)
            BENCH_PRESET="quick"; shift ;;
        --target)
            [[ $# -ge 2 ]] || { echo "error: --target requires a value" >&2; exit 2; }
            TARGET="$2"; shift 2 ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 2; }
            MODEL="$2"; shift 2 ;;
        --processor)
            [[ $# -ge 2 ]] || { echo "error: --processor requires a value" >&2; exit 2; }
            PROCESSOR="$2"; shift 2 ;;
        --trace-interval-ms)
            [[ $# -ge 2 ]] || { echo "error: --trace-interval-ms requires a value" >&2; exit 2; }
            TRACE_INTERVAL_MS="$2"; shift 2 ;;
        --concurrencies)
            [[ $# -ge 2 ]] || { echo "error: --concurrencies requires a value" >&2; exit 2; }
            CONCURRENCIES="$2"; shift 2 ;;
        --profile)
            [[ $# -ge 2 ]] || { echo "error: --profile requires a value" >&2; exit 2; }
            BENCH_PROFILE="$2"; shift 2 ;;
        --max-seconds)
            [[ $# -ge 2 ]] || { echo "error: --max-seconds requires a value" >&2; exit 2; }
            MAX_SECONDS="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "error: --warmup requires a value" >&2; exit 2; }
            WARMUP="$2"; shift 2 ;;
        --server-pid)
            [[ $# -ge 2 ]] || { echo "error: --server-pid requires a value" >&2; exit 2; }
            SERVER_PID="$2"; shift 2 ;;
        --delay-seconds)
            [[ $# -ge 2 ]] || { echo "error: --delay-seconds requires a value" >&2; exit 2; }
            DELAY_SECONDS="$2"; shift 2 ;;
        --duration-seconds)
            [[ $# -ge 2 ]] || { echo "error: --duration-seconds requires a value" >&2; exit 2; }
            DURATION_SECONDS="$2"; shift 2 ;;
        --trace)
            [[ $# -ge 2 ]] || { echo "error: --trace requires a value" >&2; exit 2; }
            TRACE_SET="$2"; shift 2 ;;
        --cuda-graph-trace)
            [[ $# -ge 2 ]] || { echo "error: --cuda-graph-trace requires a value" >&2; exit 2; }
            CUDA_GRAPH_TRACE="$2"; shift 2 ;;
        --sample)
            [[ $# -ge 2 ]] || { echo "error: --sample requires a value" >&2; exit 2; }
            SAMPLE_MODE="$2"; shift 2 ;;
        --cpuctxsw)
            [[ $# -ge 2 ]] || { echo "error: --cpuctxsw requires a value" >&2; exit 2; }
            CPUCTXSW_MODE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --*)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2 ;;
        *)
            if [[ -z "$LABEL" ]]; then
                LABEL="$1"; shift
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

REPO_ROOT="$(profile_repo_root "$SCRIPT_DIR")"
OUTPUT_DIR="$(profile_unique_dir "${REPO_ROOT}/bench-output/$(date +%Y-%m-%d)-${LABEL}-profile-nsys")"
PROFILE_BASE="${OUTPUT_DIR}/trace"
PROFILER_LOG="${OUTPUT_DIR}/nsys.log"
BENCH_LOG="${OUTPUT_DIR}/bench-anchor.log"
SUMMARY_FILE="${OUTPUT_DIR}/summary.md"
ENV_FILE="${OUTPUT_DIR}/env.txt"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"
SHA_FILE="${OUTPUT_DIR}/sha256.txt"
KERNEL_REPORT="${OUTPUT_DIR}/cuda_gpu_kern_sum.txt"
API_REPORT="${OUTPUT_DIR}/cuda_api_sum.txt"
REPLAY_SCRIPT="${OUTPUT_DIR}/replay-guidellm.sh"
REPLAY_OUTPUT_DIR="${OUTPUT_DIR}/replay-bench"
mkdir -p "$OUTPUT_DIR"

profile_require_command curl
profile_require_command jq
profile_require_command python3
if [[ "$DRY_RUN" != true ]]; then
    profile_require_command nsys
fi

if [[ -n "$BENCH_DIR" ]]; then
    if [[ ! -f "${BENCH_DIR}/command.txt" ]]; then
        echo "error: bench anchor is missing ${BENCH_DIR}/command.txt" >&2
        exit 2
    fi
else
    if [[ ! -x "${REPO_ROOT}/scripts/bench_guidellm.sh" ]]; then
        echo "error: missing executable bench wrapper: ${REPO_ROOT}/scripts/bench_guidellm.sh" >&2
        exit 2
    fi
fi

# Nsight Systems 2024+ removed `--attach-pid`. The attach-to-running-PID
# pattern this wrapper was originally built around no longer works against
# modern nsys; only spawn-the-target-under-nsys mode is supported.
# Detect BEFORE resolving the server PID so the user sees the real cause
# first (otherwise they'd hit the unrelated "no listening server" error).
NSYS_VERSION_LINE="$(nsys --version 2>/dev/null | head -n1 || echo '')"
NSYS_MAJOR="$(awk '{
    for (i=1; i<=NF; i++) {
        if (tolower($i) == "version" && (i+1) <= NF) {
            split($(i+1), parts, ".")
            print parts[1]
            exit
        }
    }
    print 0
}' <<<"$NSYS_VERSION_LINE")"
NSYS_MAJOR="${NSYS_MAJOR:-0}"
if [[ "$DRY_RUN" != true ]] && (( NSYS_MAJOR >= 2024 )); then
    cat >&2 <<EOF
error: nsys ${NSYS_VERSION_LINE#NVIDIA Nsight Systems } removed --attach-pid
       (the running-server-attach pattern this wrapper used). To run the
       same Phase 1 / Phase 2 traces against ARLE or vLLM today, spawn the
       server directly under \`nsys profile\` instead. Recipe:

         # 1. (skip if already running) terminate the server you wanted to attach to
         # 2. spawn it under nsys with delay+duration covering the bench window:

         nsys profile \\
           --output bench-output/<date>-<label>/<label> \\
           --force-overwrite=true \\
           --trace ${TRACE_SET} \\
           --cuda-graph-trace ${CUDA_GRAPH_TRACE} \\
           --delay <secs_to_skip_warmup> \\
           --duration <bench_secs> \\
           --kill none \\
           <server-binary-with-args>

         # 3. drive guidellm from a separate terminal during the duration window
         # 4. nsys writes \`*.nsys-rep\` when --duration elapses; \`nsys stats\` reads it

       Fully worked-out examples for ARLE and vLLM live in:
         docs/experience/wins/2026-05-07-m3.6-phase1-nsys-arle-s48-highconc.md
         docs/experience/wins/2026-05-07-m3.6-phase2-vllm-s14-bench.md
       (the latter notes the deferred trace + the corrected --delay value
       once vLLM nsys-startup overhead is accounted for).

       This wrapper is preserved against older nsys (<2024) where
       --attach-pid still exists. Migration to a session-API workflow is
       tracked as a follow-up.
EOF
    exit 5
fi

SERVER_PID="$(profile_resolve_server_pid "$TARGET" "$SERVER_PID")"

NSYS_CMD=(
    nsys profile
    --force-overwrite=true
    --trace "$TRACE_SET"
    --sample "$SAMPLE_MODE"
    --cpuctxsw "$CPUCTXSW_MODE"
    --cuda-graph-trace "$CUDA_GRAPH_TRACE"
    --delay "$DELAY_SECONDS"
    --duration "$DURATION_SECONDS"
    --export=sqlite
    -o "$PROFILE_BASE"
    --attach-pid "$SERVER_PID"
)

if [[ -n "$BENCH_DIR" ]]; then
    profile_build_guidellm_replay_script "${BENCH_DIR}/command.txt" "$REPLAY_OUTPUT_DIR" "$REPLAY_SCRIPT"
    LOAD_CMD=("$REPLAY_SCRIPT")
else
    LOAD_CMD=(
        "${REPO_ROOT}/scripts/bench_guidellm.sh"
        "$LABEL"
        --target "$TARGET"
        --model "$MODEL"
        --trace-interval-ms "$TRACE_INTERVAL_MS"
    )
    if [[ -n "$PROCESSOR" ]]; then
        LOAD_CMD+=(--processor "$PROCESSOR")
    fi
    case "$BENCH_PRESET" in
        fast) LOAD_CMD+=(--fast) ;;
        quick) LOAD_CMD+=(--quick) ;;
        *) echo "error: unsupported bench preset: $BENCH_PRESET" >&2; exit 2 ;;
    esac
    if [[ -n "$CONCURRENCIES" ]]; then
        LOAD_CMD+=(--concurrencies "$CONCURRENCIES")
    fi
    if [[ -n "$BENCH_PROFILE" ]]; then
        LOAD_CMD+=(--profile "$BENCH_PROFILE")
    fi
    if [[ -n "$MAX_SECONDS" ]]; then
        LOAD_CMD+=(--max-seconds "$MAX_SECONDS")
    fi
    if [[ -n "$WARMUP" ]]; then
        LOAD_CMD+=(--warmup "$WARMUP")
    fi
fi

{
    echo "target=${TARGET}"
    echo "server_pid=${SERVER_PID}"
    echo "bench_anchor=${BENCH_DIR:-auto:${LABEL}}"
    echo "delay_seconds=${DELAY_SECONDS}"
    echo "duration_seconds=${DURATION_SECONDS}"
    echo "trace=${TRACE_SET}"
    echo "cuda_graph_trace=${CUDA_GRAPH_TRACE}"
    echo "sample=${SAMPLE_MODE}"
    echo "cpuctxsw=${CPUCTXSW_MODE}"
    echo "commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "nsys_version=$(command -v nsys >/dev/null 2>&1 && nsys --version 2>/dev/null | head -n1 || echo unavailable)"
} > "$ENV_FILE"

{
    printf 'nsys'
    for arg in "${NSYS_CMD[@]:1}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
    printf 'load'
    for arg in "${LOAD_CMD[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
} > "$COMMAND_FILE"

if [[ "$DRY_RUN" == true ]]; then
    cat <<EOF
nsys output dir: ${OUTPUT_DIR}
server pid     : ${SERVER_PID}
profiler cmd   : $(printf '%q ' "${NSYS_CMD[@]}")
load cmd       : $(printf '%q ' "${LOAD_CMD[@]}")
EOF
    exit 0
fi

if ! curl -sS -f "${TARGET}/v1/models" >/dev/null 2>&1; then
    echo "error: server not reachable at ${TARGET}/v1/models" >&2
    exit 2
fi

echo ">>> nsys attach"
echo "    output : ${OUTPUT_DIR}"
echo "    target : ${TARGET}"
echo "    pid    : ${SERVER_PID}"
if [[ -n "$BENCH_DIR" ]]; then
    echo "    mode   : replay existing bench (${BENCH_DIR})"
else
    echo "    mode   : bench_guidellm.sh --${BENCH_PRESET}"
fi
echo

set +e
"${NSYS_CMD[@]}" >"$PROFILER_LOG" 2>&1 &
profiler_pid=$!
sleep 1
"${LOAD_CMD[@]}" 2>&1 | tee "$BENCH_LOG"
load_rc=${PIPESTATUS[0]}
wait "$profiler_pid"
nsys_rc=$?
set -e

if [[ $load_rc -ne 0 ]]; then
    echo "error: load generation failed with status $load_rc" >&2
    echo "       log: $BENCH_LOG" >&2
    exit 3
fi
if [[ $nsys_rc -ne 0 ]]; then
    echo "error: nsys failed with status $nsys_rc" >&2
    echo "       log: $PROFILER_LOG" >&2
    exit 4
fi
if [[ ! -f "${PROFILE_BASE}.sqlite" ]]; then
    echo "error: expected sqlite export missing: ${PROFILE_BASE}.sqlite" >&2
    exit 4
fi

set +e
nsys stats --report cuda_gpu_kern_sum "${PROFILE_BASE}.sqlite" >"$KERNEL_REPORT" 2>&1
kern_rc=$?
nsys stats --report cuda_api_sum "${PROFILE_BASE}.sqlite" >"$API_REPORT" 2>&1
api_rc=$?
set -e

ANCHOR_DIR="${BENCH_DIR}"
if [[ -z "$ANCHOR_DIR" ]]; then
    ANCHOR_DIR="$(profile_extract_output_dir_from_log "$BENCH_LOG")"
fi

{
    [[ -f "${PROFILE_BASE}.nsys-rep" ]] && profile_sha256 "${PROFILE_BASE}.nsys-rep"
    profile_sha256 "${PROFILE_BASE}.sqlite"
    [[ -f "$KERNEL_REPORT" ]] && profile_sha256 "$KERNEL_REPORT"
    [[ -f "$API_REPORT" ]] && profile_sha256 "$API_REPORT"
} > "$SHA_FILE"

python3 - "$SUMMARY_FILE" "$ANCHOR_DIR" "${PROFILE_BASE}.nsys-rep" "${PROFILE_BASE}.sqlite" \
    "$KERNEL_REPORT" "$API_REPORT" "$PROFILER_LOG" "$BENCH_LOG" "$SHA_FILE" "$LABEL" \
    "$TARGET" "$SERVER_PID" "$DELAY_SECONDS" "$DURATION_SECONDS" "$TRACE_SET" \
    "$CUDA_GRAPH_TRACE" "$kern_rc" "$api_rc" <<'PY'
import json
import pathlib
import re
import sys

(
    summary_path,
    anchor_dir,
    rep_path,
    sqlite_path,
    kernel_report,
    api_report,
    profiler_log,
    bench_log,
    sha_file,
    label,
    target,
    server_pid,
    delay_s,
    duration_s,
    trace_set,
    cuda_graph_trace,
    kern_rc,
    api_rc,
) = sys.argv[1:]


def fenced_excerpt(path_str: str, limit: int = 12) -> str:
    path = pathlib.Path(path_str)
    if not path.exists():
        return "_missing_"
    lines = [line.rstrip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not lines:
        return "_empty_"
    return "```text\n" + "\n".join(lines[:limit]) + "\n```"


def parse_total_output_tokens(anchor: pathlib.Path):
    bench_json = anchor / "benchmarks.json"
    if not bench_json.exists():
        return None
    try:
        payload = json.loads(bench_json.read_text())
    except json.JSONDecodeError:
        return None
    total = 0.0
    for bench in payload.get("benchmarks") or []:
        metrics = bench.get("metrics") or {}
        successful = int((metrics.get("request_totals") or {}).get("successful") or 0)
        mean_tokens = float(((metrics.get("output_token_count") or {}).get("successful") or {}).get("mean") or 0.0)
        total += successful * mean_tokens
    return total if total > 0 else None


def parse_api_counts(report_path: pathlib.Path):
    launches = 0
    graph_launches = 0
    copy_lines = []
    if not report_path.exists():
        return launches, graph_launches, copy_lines
    for raw_line in report_path.read_text(errors="replace").splitlines():
        cols = [col for col in re.split(r"\s{2,}", raw_line.strip()) if col]
        if len(cols) < 2:
            continue
        name = cols[-1]
        maybe_calls = cols[-2].replace(",", "")
        if not maybe_calls.isdigit():
            continue
        calls = int(maybe_calls)
        if name == "cuLaunchKernel":
            launches += calls
        elif name == "cuGraphLaunch":
            graph_launches += calls
        if "Memcpy" in name or "memcpy" in name:
            copy_lines.append(raw_line.strip())
    return launches, graph_launches, copy_lines[:6]


anchor = pathlib.Path(anchor_dir)
output_tokens = parse_total_output_tokens(anchor)
launches, graph_launches, copy_lines = parse_api_counts(pathlib.Path(api_report))
launches_per_token = None
if output_tokens:
    launches_per_token = (launches + graph_launches) / output_tokens

lines = [
    f"# Nsight Systems Summary — {label}",
    "",
    "## Capture",
    "",
    f"- Target: `{target}`",
    f"- Server PID: `{server_pid}`",
    f"- Bench anchor: `{anchor_dir}`",
    f"- Trace: `{trace_set}`",
    f"- Delay / duration: `{delay_s}s / {duration_s}s`",
    f"- CUDA graph trace: `{cuda_graph_trace}`",
    f"- Raw report: `{rep_path}`",
    f"- SQLite export: `{sqlite_path}`",
    f"- Bench log: `{bench_log}`",
    f"- Profiler log: `{profiler_log}`",
    "",
    "## Top Kernels",
    "",
    fenced_excerpt(kernel_report, limit=14),
    "",
    "## Top CUDA APIs",
    "",
    fenced_excerpt(api_report, limit=14),
    "",
    "## Copy Calls",
    "",
]

if copy_lines:
    lines.extend(f"- `{line}`" for line in copy_lines)
else:
    lines.append("- `n/a`")

lines.extend([
    "",
    "## Launches Per Token",
    "",
    f"- Approx output tokens from bench anchor: `{output_tokens if output_tokens is not None else 'n/a'}`",
    f"- `cuLaunchKernel` calls: `{launches}`",
    f"- `cuGraphLaunch` calls: `{graph_launches}`",
    f"- Approx launches/token: `{f'{launches_per_token:.3f}' if launches_per_token is not None else 'n/a'}`",
    "",
    "## Status",
    "",
    f"- `nsys stats --report cuda_gpu_kern_sum`: `{'ok' if kern_rc == '0' else 'failed'}`",
    f"- `nsys stats --report cuda_api_sum`: `{'ok' if api_rc == '0' else 'failed'}`",
    "",
    "## Sha256",
    "",
    "```text",
    pathlib.Path(sha_file).read_text().rstrip(),
    "```",
])

pathlib.Path(summary_path).write_text("\n".join(lines) + "\n")
PY

echo ">>> nsys artefacts"
echo "    anchor : ${ANCHOR_DIR}"
echo "    raw    : ${PROFILE_BASE}.nsys-rep"
echo "    sqlite : ${PROFILE_BASE}.sqlite"
echo "    stats  : ${KERNEL_REPORT}"
echo "    stats  : ${API_REPORT}"
echo "    note   : ${SUMMARY_FILE}"
