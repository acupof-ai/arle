#!/usr/bin/env bash
# Focused DSv4 CUDA/DeepGEMM helper: preflight, build, smoke, and nsys.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBCOMMAND="${1:-}"
[[ -n "$SUBCOMMAND" ]] && shift || true

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/docs/trace-artifacts/dsv4-toolchain-local}"
SERVER_BIN="${SERVER_BIN:-$ROOT/target/release/arle}"
PORT="${PORT:-18188}"
HOST="${HOST:-127.0.0.1}"
TARGET="${TARGET:-http://${HOST}:${PORT}}"
MODEL_PATH="${ARLE_DSV4_MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-Compute 137 + 269. Answer with the number only.}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
NSYS_DELAY_SECONDS="${NSYS_DELAY_SECONDS:-5}"
NSYS_DURATION_SECONDS="${NSYS_DURATION_SECONDS:-10}"
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
# The runtime reads ARLE_DSV4_MOE_TRANSPORT only.
MOE_BACKEND="${ARLE_DSV4_MOE_TRANSPORT:-allreduce}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
SPEC_TYPE="${SPEC_TYPE:-none}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-}"
MTP_DRAFT_TOPK="${MTP_DRAFT_TOPK:-}"
DEEPGEMM_ROOT="${ARLE_DEEPGEMM_ROOT:-$ROOT/crates/cuda-kernels/vendor/deepgemm}"
DEEPGEMM_LIBRARY_ROOT="${ARLE_DEEPGEMM_LIBRARY_ROOT:-$DEEPGEMM_ROOT/deep_gemm}"
DEEPGEMM_CUTLASS_INCLUDE="${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-}"
# DeepEP source tree (deepseek-ai/DeepEP) required for native-deepep MoE
# backend; left unset → deepep-sys stub mode (NativeDeepEp::boot will bail).
DEEPEP_DIR="${ARLE_DEEPEP_DIR:-}"
CUDA_HOME_DETECTED="${CUDA_HOME:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <env-check|build|smoke|nsys> [options]

Options:
  --model-path DIR   DSv4 model path; overrides ARLE_DSV4_MODEL_PATH
  --artifact-root DIR
                    artifact directory (default: $ARTIFACT_ROOT)
  --server-bin PATH  arle binary for smoke/nsys (default: $SERVER_BIN)
  --port PORT        HTTP port (default: $PORT)
  --max-tokens N     smoke max_tokens; default 32, must be >=32
  --devices LIST     CUDA device list (default: $DEVICES)
  --max-running-requests N      logical scheduler slots passed to arle serve
  --spec-type NAME   speculative decode route passed to arle serve
                    (default: $SPEC_TYPE)
  --mtp-draft-tokens N
                    MTP draft depth passed to arle serve
  --mtp-draft-topk K MTP top-k width passed to arle serve
  --moe-backend NAME DSv4 MoE backend (default: $MOE_BACKEND).
                    Accepts: deepep | native-deepep | allreduce.
                    native-deepep requires --deepep-dir + nvcc at build time.
  --expert-backend NAME
                    DSv4 expert backend (default: $EXPERT_BACKEND)
  --deepep-dir DIR   path to deepseek-ai/DeepEP source tree; required when
                    --moe-backend=native-deepep unless found under standard
                    pod paths. Supports both csrc/kernels/api.cuh and
                    csrc/kernels/legacy/api.cuh layouts. Overrides
                    ARLE_DEEPEP_DIR.
  --prompt TEXT      prompt for smoke
  --nsys-delay-seconds N
                    delay before nsys capture (default: $NSYS_DELAY_SECONDS)
  --nsys-duration-seconds N
                    nsys capture duration (default: $NSYS_DURATION_SECONDS)
  -h, --help         show this help

Environment:
  CUDA_HOME, ARLE_DEEPGEMM_ROOT, ARLE_DEEPGEMM_LIBRARY_ROOT,
  ARLE_DEEPGEMM_CUTLASS_INCLUDE,
  ARLE_DSV4_MODEL_PATH, ARLE_DSV4_MOE_TRANSPORT,
  ARLE_DEEPEP_DIR, ARTIFACT_ROOT, PORT, SERVER_BIN, MAX_TOKENS, PROMPT.
  NSYS_DELAY_SECONDS, NSYS_DURATION_SECONDS.
  MAX_RUNNING_REQUESTS, SPEC_TYPE, MTP_DRAFT_TOKENS, MTP_DRAFT_TOPK.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found on PATH: $1"
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$ROOT/$path"
    fi
}

need_value() {
    [[ $# -ge 2 && -n "${2:-}" && "${2:0:1}" != "-" ]] || die "$1 requires a value"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-path) need_value "$@"; MODEL_PATH="$2"; shift 2 ;;
            --artifact-root|--out) need_value "$@"; ARTIFACT_ROOT="$(abs_path "$2")"; shift 2 ;;
            --server-bin) need_value "$@"; SERVER_BIN="$(abs_path "$2")"; shift 2 ;;
            --port) need_value "$@"; PORT="$2"; TARGET="http://${HOST}:${PORT}"; shift 2 ;;
            --max-tokens) need_value "$@"; MAX_TOKENS="$2"; shift 2 ;;
            --devices) need_value "$@"; DEVICES="$2"; shift 2 ;;
            --max-running-requests) need_value "$@"; MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
            --spec-type) need_value "$@"; SPEC_TYPE="$2"; shift 2 ;;
            --mtp-draft-tokens) need_value "$@"; MTP_DRAFT_TOKENS="$2"; shift 2 ;;
            --mtp-draft-topk) need_value "$@"; MTP_DRAFT_TOPK="$2"; shift 2 ;;
            --moe-backend) need_value "$@"; MOE_BACKEND="$2"; shift 2 ;;
            --expert-backend) need_value "$@"; EXPERT_BACKEND="$2"; shift 2 ;;
            --deepep-dir) need_value "$@"; DEEPEP_DIR="$(abs_path "$2")"; shift 2 ;;
            --prompt) need_value "$@"; PROMPT="$2"; shift 2 ;;
            --nsys-delay-seconds) need_value "$@"; NSYS_DELAY_SECONDS="$2"; shift 2 ;;
            --nsys-duration-seconds) need_value "$@"; NSYS_DURATION_SECONDS="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
}

detect_cuda() {
    if [[ -n "$CUDA_HOME_DETECTED" ]]; then
        [[ -x "$CUDA_HOME_DETECTED/bin/nvcc" ]] ||
            die "CUDA_HOME is set but nvcc is not executable: $CUDA_HOME_DETECTED/bin/nvcc"
    else
        local nvcc_path
        nvcc_path="$(command -v nvcc || true)"
        [[ -n "$nvcc_path" ]] || die "CUDA_HOME is unset and nvcc was not found on PATH"
        CUDA_HOME_DETECTED="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
    fi
    export CUDA_HOME="$CUDA_HOME_DETECTED"
}

detect_nccl() {
    local -a dirs=()
    IFS=':' read -r -a dirs <<<"${LD_LIBRARY_PATH:-}"
    dirs+=("$CUDA_HOME/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/local/cuda/lib64" "/usr/local/lib" "/usr/lib64")
    for dir in "${dirs[@]}"; do
        [[ -n "$dir" ]] || continue
        compgen -G "$dir/libnccl.so*" >/dev/null && return 0
        compgen -G "$dir/libnccl.dylib*" >/dev/null && return 0
    done
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl\.so'; then
        return 0
    fi
    die "NCCL library not found; set LD_LIBRARY_PATH to a directory containing libnccl.so"
}

detect_deepgemm() {
    DEEPGEMM_ROOT="$(abs_path "$DEEPGEMM_ROOT")"
    DEEPGEMM_LIBRARY_ROOT="$(abs_path "$DEEPGEMM_LIBRARY_ROOT")"
    [[ -d "$DEEPGEMM_LIBRARY_ROOT/include" ]] ||
        die "ARLE_DEEPGEMM_LIBRARY_ROOT is unusable; missing include/: $DEEPGEMM_LIBRARY_ROOT"
    if [[ -n "$DEEPGEMM_CUTLASS_INCLUDE" ]]; then
        DEEPGEMM_CUTLASS_INCLUDE="$(abs_path "$DEEPGEMM_CUTLASS_INCLUDE")"
    elif [[ -f "$DEEPGEMM_ROOT/third-party/cutlass/include/cutlass/arch/barrier.h" ]]; then
        DEEPGEMM_CUTLASS_INCLUDE="$DEEPGEMM_ROOT/third-party/cutlass/include"
    elif [[ -f "$ROOT/crates/cuda-kernels/vendor/flashmla/csrc/cutlass/include/cutlass/arch/barrier.h" ]]; then
        DEEPGEMM_CUTLASS_INCLUDE="$ROOT/crates/cuda-kernels/vendor/flashmla/csrc/cutlass/include"
    else
        die "DeepGEMM CUTLASS include dir missing; checked $DEEPGEMM_ROOT/third-party/cutlass/include and FlashMLA vendor CUTLASS"
    fi
    [[ -f "$DEEPGEMM_CUTLASS_INCLUDE/cutlass/arch/barrier.h" ]] ||
        die "DeepGEMM CUTLASS barrier header missing: $DEEPGEMM_CUTLASS_INCLUDE/cutlass/arch/barrier.h"
    export ARLE_DEEPGEMM_ROOT="$DEEPGEMM_ROOT"
    export ARLE_DEEPGEMM_LIBRARY_ROOT="$DEEPGEMM_LIBRARY_ROOT"
    export ARLE_DEEPGEMM_CUTLASS_INCLUDE="$DEEPGEMM_CUTLASS_INCLUDE"
}

deepep_dir_valid() {
    local dir="$1"
    [[ -d "$dir/csrc/kernels" ]] || return 1
    [[ -f "$dir/csrc/kernels/api.cuh" || -f "$dir/csrc/kernels/legacy/api.cuh" ]]
}

deepep_layout_label() {
    local dir="$1"
    if [[ -f "$dir/csrc/kernels/api.cuh" ]]; then
        printf 'flat\n'
    elif [[ -f "$dir/csrc/kernels/legacy/api.cuh" ]]; then
        printf 'legacy\n'
    else
        printf 'unsupported\n'
    fi
}

# Validate ARLE_DEEPEP_DIR for any DeepEP MoE backend (intranode native-deepep OR
# the NVSHMEM low-latency deepep_ll). Other backends (allreduce/deepgemm) don't
# need it — deepep-sys falls back to stub mode without it. The _ll variants also
# need NVSHMEM (auto-detected below).
detect_deepep_dir() {
    # Full DeepEP accept-set, matching dsv4.rs dsv4_use_deepep_transport.
    case "$MOE_BACKEND" in
        native-deepep | native_deepep | deepep | deepep_ll | deepep-ll | deepep_low_latency | native_deepep_ll) ;;
        *) return 0 ;;
    esac

    if [[ -n "$DEEPEP_DIR" ]]; then
        DEEPEP_DIR="$(abs_path "$DEEPEP_DIR")"
        deepep_dir_valid "$DEEPEP_DIR" ||
            die "unsupported DeepEP source tree: $DEEPEP_DIR (expected csrc/kernels/api.cuh or csrc/kernels/legacy/api.cuh)"
    else
        local candidate
        for candidate in \
            "$ROOT/../DeepEP" \
            "$ROOT/../deepep" \
            "${DEEPEP_BUILD:-/data00/build/DeepEP}" \
            "/workspace/DeepEP" \
            "/workspace/deepep"; do
            if deepep_dir_valid "$candidate"; then
                DEEPEP_DIR="$candidate"
                echo "using DeepEP source tree from $DEEPEP_DIR"
                break
            fi
        done
    fi

    [[ -n "$DEEPEP_DIR" ]] ||
        die "ARLE_DSV4_MOE_TRANSPORT=$MOE_BACKEND requires --deepep-dir DIR or ARLE_DEEPEP_DIR"
    export ARLE_DEEPEP_DIR="$DEEPEP_DIR"

    # deepep_ll (low-latency) needs NVSHMEM. Auto-detect the pip nvidia-nvshmem
    # package (ships alongside deep_ep) unless overridden; deepep-sys build.rs
    # compiles internode_ll against ARLE_DEEPEP_NVSHMEM_DIR, and libnvshmem_host.so
    # must be on LD_LIBRARY_PATH at runtime.
    case "$MOE_BACKEND" in
    deepep_ll | deepep-ll | deepep_low_latency | native_deepep_ll)
        if [[ -z "${ARLE_DEEPEP_NVSHMEM_DIR:-}" ]]; then
            local nv
            nv="$(python3 -c 'import os,nvidia.nvshmem as n; print(os.path.dirname(n.__file__))' 2>/dev/null || true)"
            [[ -n "$nv" && -f "$nv/include/nvshmem.h" ]] && ARLE_DEEPEP_NVSHMEM_DIR="$nv"
        fi
        [[ -n "${ARLE_DEEPEP_NVSHMEM_DIR:-}" && -f "$ARLE_DEEPEP_NVSHMEM_DIR/include/nvshmem.h" ]] ||
            die "deepep_ll needs NVSHMEM; set ARLE_DEEPEP_NVSHMEM_DIR (pip nvidia-nvshmem: <site-packages>/nvidia/nvshmem)"
        export ARLE_DEEPEP_NVSHMEM_DIR
        export LD_LIBRARY_PATH="$ARLE_DEEPEP_NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
        echo "using NVSHMEM from $ARLE_DEEPEP_NVSHMEM_DIR (deepep_ll)"
        ;;
    esac
}

export_runtime_env() {
    export CUDA_VISIBLE_DEVICES="$DEVICES"
    export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-$DEVICES}"
    export RUST_LOG="${RUST_LOG:-info}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export ARLE_DSV4_MOE_TRANSPORT="$MOE_BACKEND"
    export ARLE_DEEPGEMM_LIBRARY_ROOT="$DEEPGEMM_LIBRARY_ROOT"
    export ARLE_DEEPGEMM_CUTLASS_INCLUDE="$DEEPGEMM_CUTLASS_INCLUDE"
    # native-deepep needs the source tree available at runtime too (the
    # Buffer lifecycle is driven by the static archive linked at build,
    # but the env var is logged for traceability and consumed by smoke
    # diagnostics). Export only when set so we don't shadow stub mode.
    if [[ -n "${ARLE_DEEPEP_DIR:-}" ]]; then
        export ARLE_DEEPEP_DIR
    fi
}

detect_model() {
    [[ -n "$MODEL_PATH" ]] ||
        die "model path missing; pass --model-path DIR or set ARLE_DSV4_MODEL_PATH"
    MODEL_PATH="$(abs_path "$MODEL_PATH")"
    [[ -d "$MODEL_PATH" ]] || die "model path is not a directory: $MODEL_PATH"
}

require_max_tokens_decode() {
    [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || die "--max-tokens must be an integer, got: $MAX_TOKENS"
    (( MAX_TOKENS >= 32 )) ||
        die "--max-tokens must be >=32 by default; max_tokens=1 does not run decode"
}

preflight() {
    detect_cuda
    detect_nccl
    detect_deepgemm
    detect_deepep_dir
    detect_model
}

env_check() {
    preflight
    echo "CUDA_HOME=$CUDA_HOME"
    echo "nvcc=$CUDA_HOME/bin/nvcc"
    echo "NCCL=found"
    echo "ARLE_DEEPGEMM_ROOT=$ARLE_DEEPGEMM_ROOT"
    echo "ARLE_DEEPGEMM_LIBRARY_ROOT=$ARLE_DEEPGEMM_LIBRARY_ROOT"
    echo "ARLE_DEEPGEMM_CUTLASS_INCLUDE=$ARLE_DEEPGEMM_CUTLASS_INCLUDE"
    echo "ARLE_DSV4_MODEL_PATH=$MODEL_PATH"
    echo "CUDA_VISIBLE_DEVICES=$DEVICES"
    echo "ARLE_DSV4_MOE_TRANSPORT=$MOE_BACKEND"
    echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-auto}"
    echo "SPEC_TYPE=$SPEC_TYPE"
    echo "MTP_DRAFT_TOKENS=${MTP_DRAFT_TOKENS:-unset}"
    echo "MTP_DRAFT_TOPK=${MTP_DRAFT_TOPK:-unset}"
    if [[ -n "${ARLE_DEEPEP_DIR:-}" ]]; then
        echo "ARLE_DEEPEP_DIR=$ARLE_DEEPEP_DIR"
        echo "ARLE_DEEPEP_LAYOUT=$(deepep_layout_label "$ARLE_DEEPEP_DIR")"
        [[ -n "${ARLE_DEEPEP_NVSHMEM_DIR:-}" ]] && echo "ARLE_DEEPEP_NVSHMEM_DIR=$ARLE_DEEPEP_NVSHMEM_DIR"
    else
        echo "ARLE_DEEPEP_DIR=(unset — non-DeepEP backend)"
    fi
}

build_infer() {
    detect_cuda
    detect_nccl
    detect_deepgemm
    detect_deepep_dir
    need_cmd cargo
    cd "$ROOT"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    # A native-DeepEP MoE backend needs the `deepep` cargo feature compiled in
    # (it transitively pulls nccl+cuda); cuda,nccl alone yields a stub DeepEP.
    local features="cuda,nccl"
    case "$MOE_BACKEND" in
        native-deepep|native_deepep|deepep|deepep_ll|deepep-ll|deepep_low_latency|native_deepep_ll)
            features="deepep" ;;
    esac
    \
        cargo build --release --features "$features" --bin arle
}

wait_ready() {
    local log="$1"
    local deadline=$((SECONDS + WAIT_SECONDS))
    until curl -sS -f "$TARGET/v1/models" >"$ARTIFACT_ROOT/models.json" 2>"$ARTIFACT_ROOT/curl-ready.err"; do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "error: infer server exited during startup; log: $log" >&2
            tail -160 "$log" >&2 || true
            exit 3
        fi
        if (( SECONDS >= deadline )); then
            echo "error: infer server did not become ready within ${WAIT_SECONDS}s; log: $log" >&2
            tail -160 "$log" >&2 || true
            exit 3
        fi
        sleep 2
    done
}

serve_args() {
    local -n out="$1"
    out=(
        serve
        --backend cuda
        --model-path "$MODEL_PATH"
        --port "$PORT"
        --spec-type "$SPEC_TYPE"
        --max-total-tokens "$MAX_SEQ_LEN"
    )
    [[ -z "$MAX_RUNNING_REQUESTS" ]] || out+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
    [[ -z "$MTP_DRAFT_TOKENS" ]] || out+=(--mtp-draft-tokens "$MTP_DRAFT_TOKENS")
    [[ -z "$MTP_DRAFT_TOPK" ]] || out+=(--mtp-draft-topk "$MTP_DRAFT_TOPK")
}

smoke() {
    require_max_tokens_decode
    preflight
    export_runtime_env
    need_cmd curl
    need_cmd python3
    [[ -x "$SERVER_BIN" ]] || die "arle binary missing or not executable: $SERVER_BIN; run build first"
    mkdir -p "$ARTIFACT_ROOT"
    if curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
        die "server already responding at $TARGET; set PORT or stop it first"
    fi

    local server_log="$ARTIFACT_ROOT/server.log"
    # ARLE_SERVER_WRAP — optional launcher prefix for the serve process,
    # e.g. `compute-sanitizer --tool memcheck --target-processes all` to
    # trace device-side OOB across all 8 self-spawned worker ranks (the
    # coordinator forks them via current_exe; --target-processes all
    # instruments each, and children inherit the coordinator stderr so all
    # reports land in this server.log). Unset by default → zero change to
    # production runs. Intentionally unquoted so the multi-word command
    # word-splits into argv.
    # Rewrite-stack serve: DSv4 runtime knobs ride ARLE_DSV4_* env vars;
    # request-shape knobs (including max-seq-len) stay explicit CLI args for
    # reproducible smoke/nsys runs.
    local -a args
    serve_args args
    (
        cd "$ROOT"
        exec ${ARLE_SERVER_WRAP:-} "$SERVER_BIN" "${args[@]}"
    ) >"$server_log" 2>&1 &
    server_pid=$!

    cleanup() {
        set +e
        if kill -0 "$server_pid" >/dev/null 2>&1; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT INT TERM

    wait_ready "$server_log"
    python3 - "$TARGET" "$MODEL_NAME" "$MAX_TOKENS" "$PROMPT" "$ARTIFACT_ROOT/smoke-response.json" <<'PY'
import json
import sys
import time
import urllib.request

target, model, max_tokens, prompt, out = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "ignore_eos": True,
    "stream": False,
    "max_tokens": int(max_tokens),
}
req = urllib.request.Request(
    f"{target}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    body = resp.read()
elapsed = time.perf_counter() - t0
parsed = json.loads(body)
result = {
    "elapsed_s": elapsed,
    "usage": parsed.get("usage"),
    "text": parsed["choices"][0]["message"]["content"],
}
with open(out, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
PY
    echo "smoke artifacts: $ARTIFACT_ROOT"
}

nsys_profile() {
    preflight
    export_runtime_env
    need_cmd nsys
    need_cmd curl
    [[ -x "$SERVER_BIN" ]] || die "arle binary missing or not executable: $SERVER_BIN; run build first"
    [[ -x "$ROOT/scripts/profile_nsys_bench.sh" ]] ||
        die "missing nsys wrapper: $ROOT/scripts/profile_nsys_bench.sh"

    mkdir -p "$ARTIFACT_ROOT"
    if curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
        die "server already responding at $TARGET; set PORT or stop it first"
    fi

    local server_log="$ARTIFACT_ROOT/nsys-server.log"
    local -a args
    serve_args args
    (
        cd "$ROOT"
        exec ${ARLE_SERVER_WRAP:-} "$SERVER_BIN" "${args[@]}"
    ) >"$server_log" 2>&1 &
    server_pid=$!

    cleanup() {
        set +e
        if kill -0 "$server_pid" >/dev/null 2>&1; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT INT TERM

    wait_ready "$server_log"
    ARLE_DSV4_MOE_TRANSPORT="$ARLE_DSV4_MOE_TRANSPORT" \
    ARLE_DEEPGEMM_LIBRARY_ROOT="$ARLE_DEEPGEMM_LIBRARY_ROOT" \
        "$ROOT/scripts/profile_nsys_bench.sh" \
        dsv4-toolchain \
        --url "$TARGET" \
        --model "$MODEL_NAME" \
        --server-pid "$server_pid" \
        --concurrency-grid 1 \
        --seconds-per-concurrency "$NSYS_DURATION_SECONDS" \
        --delay-seconds "$NSYS_DELAY_SECONDS" \
        --duration-seconds "$NSYS_DURATION_SECONDS"
}

case "$SUBCOMMAND" in
    env-check) parse_args "$@"; env_check ;;
    build) parse_args "$@"; build_infer ;;
    smoke) parse_args "$@"; smoke ;;
    nsys) parse_args "$@"; nsys_profile ;;
    -h|--help) usage ;;
    "") usage; exit 1 ;;
    *) usage >&2; die "unknown subcommand: $SUBCOMMAND" ;;
esac
