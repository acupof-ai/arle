#!/usr/bin/env bash
# Native-bench-anchored Nsight Compute wrapper for hotspot infer kernels.
#
# Default flow:
#   1. Attach `ncu` to an already-running infer server (PID resolved from --url).
#   2. Drive `bench_throughput.py` to hit the selected kernel family.
#   3. Export `.ncu-rep` + profiler log + a short markdown summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./profile_bench_common.sh
source "${SCRIPT_DIR}/profile_bench_common.sh"

TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
PROMPTS_JSONL="${BENCH_PROMPTS_JSONL:-}"
SERVER_PID=""
BENCH_DIR=""
CONCURRENCY_GRID="16"
BENCH_DURATION=10
LABEL=""
KERNEL_FAMILY=""
KERNEL_REGEX=""
SECTION_SET="full"
LAUNCH_SKIP=5
LAUNCH_COUNT=1
TARGET_PROCESSES="all"
DRY_RUN=false

kernel_family_regex() {
    case "$1" in
        attention)
            printf '%s\n' 'regex:(attention|tilelang|fmha|paged_attention|decode_attention|nonpaged_prefill)'
            ;;
        sampling)
            printf '%s\n' 'regex:(sample|sampling|topk|top_p|temperature|argmax)'
            ;;
        paged-kv)
            printf '%s\n' 'regex:(paged|kv|cache|page|readmission)'
            ;;
        dequant)
            printf '%s\n' 'regex:(dequant|quant)'
            ;;
        fused-op|fused-ops)
            printf '%s\n' 'regex:(fused|rmsnorm|layernorm|residual|mlp)'
            ;;
        *)
            echo "error: unsupported kernel family: $1" >&2
            echo "       expected one of: attention, sampling, paged-kv, dequant, fused-op" >&2
            exit 2
            ;;
    esac
}

usage() {
    cat <<EOF
Native-bench-anchored Nsight Compute wrapper for infer hotspot kernels.

Usage:
  $(basename "$0") <label> (--family NAME | --kernel REGEX) [options]

Kernel focus:
  --family NAME          one of: attention, sampling, paged-kv, dequant, fused-op
  --kernel REGEX         explicit NCU kernel selector, e.g. regex:decode_attention_.*
  --set NAME             NCU section set (default: ${SECTION_SET})
  --launch-skip N        skip N matching launches before profiling (default: ${LAUNCH_SKIP})
  --launch-count N       profile N matching launches (default: ${LAUNCH_COUNT})
  --target-processes M   default: ${TARGET_PROCESSES}

Bench anchor:
  --bench DIR             reuse an existing bench-output dir as the anchor
                          and replay DIR/command.txt for the profiling load
  --url URL               default: ${TARGET}
  --model NAME            default: ${MODEL}
  --prompts-jsonl PATH    optional JSONL prompt dataset
  --concurrency-grid LIST fixed concurrency list (default: ${CONCURRENCY_GRID})
  --seconds-per-concurrency N
                          measurement duration (default: ${BENCH_DURATION})

Profiler:
  --server-pid PID        explicit infer server PID (else resolve from --url via lsof)
  --dry-run               print resolved commands without executing them

Examples:
  scripts/profile_ncu_bench.sh cuda-qwen3 --family attention --prompts-jsonl data/prompts.jsonl
  scripts/profile_ncu_bench.sh cuda-qwen3 --family paged-kv --launch-skip 8 --launch-count 2
  scripts/profile_ncu_bench.sh cuda-qwen3 --kernel 'regex:decode_attention_int8_.*_kernel'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)
            [[ $# -ge 2 ]] || { echo "error: --family requires a value" >&2; exit 2; }
            KERNEL_FAMILY="$2"; shift 2 ;;
        --kernel)
            [[ $# -ge 2 ]] || { echo "error: --kernel requires a value" >&2; exit 2; }
            KERNEL_REGEX="$2"; shift 2 ;;
        --set)
            [[ $# -ge 2 ]] || { echo "error: --set requires a value" >&2; exit 2; }
            SECTION_SET="$2"; shift 2 ;;
        --launch-skip)
            [[ $# -ge 2 ]] || { echo "error: --launch-skip requires a value" >&2; exit 2; }
            LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count)
            [[ $# -ge 2 ]] || { echo "error: --launch-count requires a value" >&2; exit 2; }
            LAUNCH_COUNT="$2"; shift 2 ;;
        --target-processes)
            [[ $# -ge 2 ]] || { echo "error: --target-processes requires a value" >&2; exit 2; }
            TARGET_PROCESSES="$2"; shift 2 ;;
        --bench)
            [[ $# -ge 2 ]] || { echo "error: --bench requires a value" >&2; exit 2; }
            BENCH_DIR="$2"; shift 2 ;;
        --url)
            [[ $# -ge 2 ]] || { echo "error: --url requires a value" >&2; exit 2; }
            TARGET="$2"; shift 2 ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 2; }
            MODEL="$2"; shift 2 ;;
        --prompts-jsonl)
            [[ $# -ge 2 ]] || { echo "error: --prompts-jsonl requires a value" >&2; exit 2; }
            PROMPTS_JSONL="$2"; shift 2 ;;
        --concurrency-grid)
            [[ $# -ge 2 ]] || { echo "error: --concurrency-grid requires a value" >&2; exit 2; }
            CONCURRENCY_GRID="$2"; shift 2 ;;
        --seconds-per-concurrency)
            [[ $# -ge 2 ]] || { echo "error: --seconds-per-concurrency requires a value" >&2; exit 2; }
            BENCH_DURATION="$2"; shift 2 ;;
        --server-pid)
            [[ $# -ge 2 ]] || { echo "error: --server-pid requires a value" >&2; exit 2; }
            SERVER_PID="$2"; shift 2 ;;
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
if [[ -z "$KERNEL_FAMILY" && -z "$KERNEL_REGEX" ]]; then
    echo "error: pass either --family or --kernel" >&2
    usage >&2
    exit 2
fi
if [[ -n "$KERNEL_FAMILY" && -n "$KERNEL_REGEX" ]]; then
    echo "error: use either --family or --kernel, not both" >&2
    exit 2
fi
if [[ -n "$KERNEL_FAMILY" ]]; then
    KERNEL_REGEX="$(kernel_family_regex "$KERNEL_FAMILY")"
fi

REPO_ROOT="$(profile_repo_root "$SCRIPT_DIR")"
OUTPUT_DIR="$(profile_unique_dir "${REPO_ROOT}/bench-output/$(date +%Y-%m-%d)-${LABEL}-profile-ncu")"
PROFILE_BASE="${OUTPUT_DIR}/trace"
PROFILER_LOG="${OUTPUT_DIR}/ncu.log"
BENCH_LOG="${OUTPUT_DIR}/bench-anchor.log"
SUMMARY_FILE="${OUTPUT_DIR}/summary.md"
ENV_FILE="${OUTPUT_DIR}/env.txt"
COMMAND_FILE="${OUTPUT_DIR}/command.txt"
SHA_FILE="${OUTPUT_DIR}/sha256.txt"
REPLAY_SCRIPT="${OUTPUT_DIR}/replay-bench.sh"
REPLAY_OUTPUT_DIR="${OUTPUT_DIR}/replay-bench"
REPLAY_OUTPUT="${REPLAY_OUTPUT_DIR}/bench_throughput"
mkdir -p "$OUTPUT_DIR"

profile_require_command curl
profile_require_command python3
if [[ "$DRY_RUN" != true ]]; then
    profile_require_command ncu
fi

if [[ -n "$BENCH_DIR" ]]; then
    if [[ ! -f "${BENCH_DIR}/command.txt" ]]; then
        echo "error: bench anchor is missing ${BENCH_DIR}/command.txt" >&2
        exit 2
    fi
else
    if [[ ! -f "${REPO_ROOT}/scripts/bench_throughput.py" ]]; then
        echo "error: missing native bench runner: ${REPO_ROOT}/scripts/bench_throughput.py" >&2
        exit 2
    fi
fi

SERVER_PID="$(profile_resolve_server_pid "$TARGET" "$SERVER_PID")"

NCU_CMD=(
    ncu
    --mode attach
    --attach-pid "$SERVER_PID"
    --target-processes "$TARGET_PROCESSES"
    --kernel-name-base demangled
    --kernel-name "$KERNEL_REGEX"
    --launch-skip "$LAUNCH_SKIP"
    --launch-count "$LAUNCH_COUNT"
    --set "$SECTION_SET"
    --clock-control none
    --cache-control none
    --force-overwrite
    --log-file "$PROFILER_LOG"
    -o "$PROFILE_BASE"
)

if [[ -n "$BENCH_DIR" ]]; then
    profile_build_bench_replay_script "${BENCH_DIR}/command.txt" "$REPLAY_OUTPUT" "$REPLAY_SCRIPT"
    LOAD_CMD=("$REPLAY_SCRIPT")
else
    LOAD_CMD=(
        python3 "${REPO_ROOT}/scripts/bench_throughput.py"
        --url "$TARGET"
        --model "$MODEL"
        --concurrency-grid "$CONCURRENCY_GRID"
        --seconds-per-concurrency "$BENCH_DURATION"
        --output "$REPLAY_OUTPUT"
    )
    if [[ -n "$PROMPTS_JSONL" ]]; then
        LOAD_CMD+=(--prompts-jsonl "$PROMPTS_JSONL")
    fi
    mkdir -p "$REPLAY_OUTPUT_DIR"
    printf '%q ' "${LOAD_CMD[@]}" > "${REPLAY_OUTPUT_DIR}/command.txt"
    printf '\n' >> "${REPLAY_OUTPUT_DIR}/command.txt"
fi

{
    echo "target=${TARGET}"
    echo "server_pid=${SERVER_PID}"
    echo "bench_anchor=${BENCH_DIR:-auto:${LABEL}}"
    echo "kernel_regex=${KERNEL_REGEX}"
    echo "section_set=${SECTION_SET}"
    echo "launch_skip=${LAUNCH_SKIP}"
    echo "launch_count=${LAUNCH_COUNT}"
    echo "target_processes=${TARGET_PROCESSES}"
    echo "commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "ncu_version=$(command -v ncu >/dev/null 2>&1 && ncu --version 2>/dev/null | head -n1 || echo unavailable)"
} > "$ENV_FILE"

{
    printf 'ncu'
    for arg in "${NCU_CMD[@]:1}"; do
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
ncu output dir : ${OUTPUT_DIR}
server pid     : ${SERVER_PID}
kernel regex   : ${KERNEL_REGEX}
profiler cmd   : $(printf '%q ' "${NCU_CMD[@]}")
load cmd       : $(printf '%q ' "${LOAD_CMD[@]}")
EOF
    exit 0
fi

if ! curl -sS -f "${TARGET}/v1/models" >/dev/null 2>&1; then
    echo "error: server not reachable at ${TARGET}/v1/models" >&2
    exit 2
fi

echo ">>> ncu attach"
echo "    output : ${OUTPUT_DIR}"
echo "    target : ${TARGET}"
echo "    pid    : ${SERVER_PID}"
echo "    kernel : ${KERNEL_REGEX}"
echo

set +e
"${NCU_CMD[@]}" &
profiler_pid=$!
sleep 1
"${LOAD_CMD[@]}" 2>&1 | tee "$BENCH_LOG"
load_rc=${PIPESTATUS[0]}
wait "$profiler_pid"
ncu_rc=$?
set -e

if [[ $load_rc -ne 0 ]]; then
    echo "error: load generation failed with status $load_rc" >&2
    echo "       log: $BENCH_LOG" >&2
    exit 3
fi
if [[ $ncu_rc -ne 0 ]]; then
    echo "error: ncu failed with status $ncu_rc" >&2
    echo "       log: $PROFILER_LOG" >&2
    exit 4
fi

ANCHOR_DIR="${BENCH_DIR}"
if [[ -z "$ANCHOR_DIR" ]]; then
    ANCHOR_DIR="$REPLAY_OUTPUT_DIR"
fi

{
    [[ -f "${PROFILE_BASE}.ncu-rep" ]] && profile_sha256 "${PROFILE_BASE}.ncu-rep"
    [[ -f "$PROFILER_LOG" ]] && profile_sha256 "$PROFILER_LOG"
} > "$SHA_FILE"

python3 - "$SUMMARY_FILE" "$ANCHOR_DIR" "${PROFILE_BASE}.ncu-rep" "$PROFILER_LOG" "$SHA_FILE" \
    "$LABEL" "$TARGET" "$SERVER_PID" "$KERNEL_FAMILY" "$KERNEL_REGEX" "$SECTION_SET" \
    "$LAUNCH_SKIP" "$LAUNCH_COUNT" "$TARGET_PROCESSES" <<'PY'
import pathlib
import re
import sys

(
    summary_path,
    anchor_dir,
    rep_path,
    log_path,
    sha_file,
    label,
    target,
    server_pid,
    kernel_family,
    kernel_regex,
    section_set,
    launch_skip,
    launch_count,
    target_processes,
) = sys.argv[1:]

log_lines = pathlib.Path(log_path).read_text(errors="replace").splitlines() if pathlib.Path(log_path).exists() else []

highlights = []
patterns = [
    r"Kernel Name",
    r"Achieved Occupancy",
    r"Memory Throughput",
    r"DRAM Throughput",
    r"SM .*Throughput",
    r"Roofline",
    r"Scheduler Statistics",
    r"Warp State Statistics",
    r"No kernels were profiled",
]
for line in log_lines:
    if any(re.search(pattern, line) for pattern in patterns):
        highlights.append(line.rstrip())
highlights = highlights[:30]

if not highlights:
    highlights = [line.rstrip() for line in log_lines[:40] if line.strip()]

lines = [
    f"# Nsight Compute Summary — {label}",
    "",
    "## Capture",
    "",
    f"- Target: `{target}`",
    f"- Server PID: `{server_pid}`",
    f"- Bench anchor: `{anchor_dir}`",
    f"- Kernel family: `{kernel_family or 'custom'}`",
    f"- Kernel selector: `{kernel_regex}`",
    f"- Section set: `{section_set}`",
    f"- Launch skip / count: `{launch_skip} / {launch_count}`",
    f"- Target processes: `{target_processes}`",
    f"- Raw report: `{rep_path}`",
    f"- Profiler log: `{log_path}`",
    "",
    "## Requested Contract",
    "",
    "- Occupancy",
    "- Memory throughput",
    "- Stall reasons",
    "- Roofline / achieved-vs-peak signal",
    "",
    "## Highlights",
    "",
    "```text",
    "\n".join(highlights) if highlights else "<no highlights captured>",
    "```",
    "",
    "## Sha256",
    "",
    "```text",
    pathlib.Path(sha_file).read_text().rstrip(),
    "```",
]

pathlib.Path(summary_path).write_text("\n".join(lines) + "\n")
PY

echo ">>> ncu artefacts"
echo "    anchor : ${ANCHOR_DIR}"
echo "    raw    : ${PROFILE_BASE}.ncu-rep"
echo "    log    : ${PROFILER_LOG}"
echo "    note   : ${SUMMARY_FILE}"
