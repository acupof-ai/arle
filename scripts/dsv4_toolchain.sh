#!/usr/bin/env bash
# Focused DSv4 CUDA/DeepGEMM helper: preflight, build, smoke, and nsys.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBCOMMAND="${1:-}"
[[ -n "$SUBCOMMAND" ]] && shift || true

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/docs/trace-artifacts/dsv4-toolchain-local}"
SERVER_BIN="${SERVER_BIN:-$ROOT/target/release/infer}"
PORT="${PORT:-18188}"
HOST="${HOST:-127.0.0.1}"
TARGET="${TARGET:-http://${HOST}:${PORT}}"
MODEL_PATH="${ARLE_DSV4_MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-Compute 137 + 269. Answer with the number only.}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_SLOTS="${NUM_SLOTS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.10}"
MOE_BACKEND="${ARLE_DSV4_MOE_BACKEND:-deepep_unsafe}"
EXPERT_BACKEND="${ARLE_DSV4_EXPERT_BACKEND:-deepgemm-auto}"
DEEPGEMM_ROOT="${ARLE_DEEPGEMM_ROOT:-$ROOT/crates/cuda-kernels/vendor/deepgemm}"
DEEPGEMM_LIBRARY_ROOT="${ARLE_DEEPGEMM_LIBRARY_ROOT:-$DEEPGEMM_ROOT/deep_gemm}"
CUDA_HOME_DETECTED="${CUDA_HOME:-}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <env-check|build|smoke|nsys> [options]

Options:
  --model-path DIR   DSv4 model path; overrides ARLE_DSV4_MODEL_PATH
  --artifact-root DIR
                    artifact directory (default: $ARTIFACT_ROOT)
  --server-bin PATH  infer binary for smoke/nsys (default: $SERVER_BIN)
  --port PORT        HTTP port (default: $PORT)
  --max-tokens N     smoke/nsys max_tokens; default 32, must be >=32
  --devices LIST     CUDA device list (default: $DEVICES)
  --moe-backend NAME DSv4 MoE backend (default: $MOE_BACKEND)
  --expert-backend NAME
                    DSv4 expert backend (default: $EXPERT_BACKEND)
  --prompt TEXT      prompt for smoke/nsys
  -h, --help         show this help

Environment:
  CUDA_HOME, ARLE_DEEPGEMM_ROOT, ARLE_DEEPGEMM_LIBRARY_ROOT,
  ARLE_DSV4_MODEL_PATH, ARLE_DSV4_MOE_BACKEND, ARLE_DSV4_EXPERT_BACKEND,
  ARTIFACT_ROOT, PORT, SERVER_BIN, MAX_TOKENS, PROMPT.
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
            --moe-backend) need_value "$@"; MOE_BACKEND="$2"; shift 2 ;;
            --expert-backend) need_value "$@"; EXPERT_BACKEND="$2"; shift 2 ;;
            --prompt) need_value "$@"; PROMPT="$2"; shift 2 ;;
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
    [[ -d "$DEEPGEMM_ROOT/third-party/cutlass/include" ]] ||
        die "DeepGEMM CUTLASS include dir missing: $DEEPGEMM_ROOT/third-party/cutlass/include"
    export ARLE_DEEPGEMM_ROOT="$DEEPGEMM_ROOT"
    export ARLE_DEEPGEMM_LIBRARY_ROOT="$DEEPGEMM_LIBRARY_ROOT"
}

export_runtime_env() {
    export CUDA_VISIBLE_DEVICES="$DEVICES"
    export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-$DEVICES}"
    export RUST_LOG="${RUST_LOG:-info}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export ARLE_DSV4_MOE_BACKEND="$MOE_BACKEND"
    export ARLE_DSV4_INCREMENTAL_KV="${ARLE_DSV4_INCREMENTAL_KV:-1}"
    export ARLE_DSV4_FUSED_DISPATCH_PAYLOAD="${ARLE_DSV4_FUSED_DISPATCH_PAYLOAD:-1}"
    export ARLE_DSV4_EXPERT_BACKEND="$EXPERT_BACKEND"
    export ARLE_DEEPGEMM_LIBRARY_ROOT="$DEEPGEMM_LIBRARY_ROOT"
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
    detect_model
}

env_check() {
    preflight
    echo "CUDA_HOME=$CUDA_HOME"
    echo "nvcc=$CUDA_HOME/bin/nvcc"
    echo "NCCL=found"
    echo "ARLE_DEEPGEMM_ROOT=$ARLE_DEEPGEMM_ROOT"
    echo "ARLE_DEEPGEMM_LIBRARY_ROOT=$ARLE_DEEPGEMM_LIBRARY_ROOT"
    echo "ARLE_DSV4_MODEL_PATH=$MODEL_PATH"
    echo "CUDA_VISIBLE_DEVICES=$DEVICES"
    echo "ARLE_DSV4_MOE_BACKEND=$MOE_BACKEND"
    echo "ARLE_DSV4_EXPERT_BACKEND=$EXPERT_BACKEND"
}

build_infer() {
    detect_cuda
    detect_nccl
    detect_deepgemm
    need_cmd cargo
    cd "$ROOT"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
        cargo build --release -p infer --features cuda,nccl --bin infer
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

smoke() {
    require_max_tokens_decode
    preflight
    export_runtime_env
    need_cmd curl
    need_cmd python3
    [[ -x "$SERVER_BIN" ]] || die "infer binary missing or not executable: $SERVER_BIN; run build first"
    mkdir -p "$ARTIFACT_ROOT"
    if curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
        die "server already responding at $TARGET; set PORT or stop it first"
    fi

    local server_log="$ARTIFACT_ROOT/server.log"
    (
        cd "$ROOT"
        exec "$SERVER_BIN" \
            --model-path "$MODEL_PATH" \
            --port "$PORT" \
            --num-slots "$NUM_SLOTS" \
            --max-seq-len "$MAX_SEQ_LEN" \
            --mem-fraction-static "$MEM_FRACTION_STATIC" \
            --kv-cache-dtype fp8 \
            --deepseek-distributed-layers 43
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
    require_max_tokens_decode
    preflight
    export_runtime_env
    need_cmd nsys
    [[ -x "$ROOT/scripts/profile_dsv4_single_decode_nsys.sh" ]] ||
        die "missing nsys wrapper: $ROOT/scripts/profile_dsv4_single_decode_nsys.sh"
    ARLE_DSV4_MOE_BACKEND="$ARLE_DSV4_MOE_BACKEND" \
    ARLE_DSV4_EXPERT_BACKEND="$ARLE_DSV4_EXPERT_BACKEND" \
    ARLE_DEEPGEMM_LIBRARY_ROOT="$ARLE_DEEPGEMM_LIBRARY_ROOT" \
        "$ROOT/scripts/profile_dsv4_single_decode_nsys.sh" \
        --out "$ARTIFACT_ROOT/nsys" \
        --port "$PORT" \
        --model-path "$MODEL_PATH" \
        --server-bin "$SERVER_BIN" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS"
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
