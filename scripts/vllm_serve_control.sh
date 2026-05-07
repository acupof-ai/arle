#!/usr/bin/env bash
# vLLM equal-width control-group server for M3.6 Phase 2 nsys traces.
#
# Codifies the env / flag combination that codex first assembled by hand
# for the M6 high-conc gap investigation
# (`m6-cuda-vllm-gap-followups.md` §"Phase 1 verify"). Subsequent control
# runs go through this script so the comparison shape stays bit-stable
# across captures.
#
# Usage:
#   scripts/vllm_serve_control.sh [--max-num-seqs N] [--max-model-len M]
#                                 [--port P] [--model PATH]
#
# Defaults match the M3.6 Phase-2 control config:
#   --max-num-seqs 14 --max-model-len 2048 --port 8000 --model infer/models/Qwen3-4B
#
# Run in foreground; Ctrl-C cleanly shuts down. To background, prefix
# with `setsid env nohup ... < /dev/null & disown` (per the M4 lifecycle
# notes — `nohup` alone does NOT survive cron-bash exit).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${VLLM_VENV:-/tmp/arle-vllm-venv}"

MODEL="${REPO_ROOT}/infer/models/Qwen3-4B"
SERVED_NAME="Qwen/Qwen3-4B"
PORT=8000
MAX_NUM_SEQS=14
MAX_MODEL_LEN=2048
DTYPE=bfloat16
KV_CACHE_DTYPE=fp8
ATTN_BACKEND=TRITON_ATTN
GPU_MEM_UTIL=0.85
HOST=0.0.0.0

usage() {
    cat <<EOF
vLLM control-group server for M3.6 Phase 2.

Usage:
  $(basename "$0") [options]

Options:
  --model PATH            local model dir (default: ${MODEL})
  --served-name NAME      served-model-name (default: ${SERVED_NAME})
  --port N                port (default: ${PORT})
  --host H                bind host (default: ${HOST})
  --max-num-seqs N        admission width (default: ${MAX_NUM_SEQS})
  --max-model-len N       max sequence length (default: ${MAX_MODEL_LEN})
  --dtype T               weight dtype (default: ${DTYPE})
  --kv-cache-dtype T      KV cache dtype (default: ${KV_CACHE_DTYPE})
  --attention-backend B   vLLM attn backend (default: ${ATTN_BACKEND})
  --gpu-mem-util F        GPU mem fraction (default: ${GPU_MEM_UTIL})
  --venv DIR              vLLM python venv (default: ${VENV}; or VLLM_VENV env)
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --served-name) SERVED_NAME="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --attention-backend) ATTN_BACKEND="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

# Preflight: venv exists and has vllm
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: vllm venv not found at ${VENV}/bin/python" >&2
    echo "       set VLLM_VENV=<path> or pass --venv" >&2
    exit 1
fi
if ! "${VENV}/bin/python" -c 'import vllm' 2>/dev/null; then
    echo "error: vllm not importable from ${VENV}" >&2
    echo "       check: ${VENV}/bin/pip show vllm" >&2
    exit 1
fi

# Preflight: model exists
if [[ ! -d "${MODEL}" ]]; then
    echo "error: model dir not found: ${MODEL}" >&2
    exit 1
fi

# Preflight: port not already taken (do NOT silently take over)
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "error: port ${PORT} already in use" >&2
    echo "       check: ss -tlnp | grep :${PORT}" >&2
    exit 1
fi

# Preflight: GPU has reasonable headroom (warn, don't block)
GPU_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
if [[ "${GPU_USED_MIB}" -gt 2000 ]]; then
    echo "warn: GPU already using ${GPU_USED_MIB} MiB — vLLM may OOM at gpu_mem_util=${GPU_MEM_UTIL}" >&2
fi

VLLM_VERSION=$("${VENV}/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)

echo "==> vLLM control server"
echo "    venv:           ${VENV} (vllm ${VLLM_VERSION})"
echo "    model:          ${MODEL}"
echo "    served as:      ${SERVED_NAME}"
echo "    bind:           ${HOST}:${PORT}"
echo "    width:          --max-num-seqs ${MAX_NUM_SEQS} --max-model-len ${MAX_MODEL_LEN}"
echo "    dtype:          ${DTYPE} (kv ${KV_CACHE_DTYPE})"
echo "    attn-backend:   ${ATTN_BACKEND}"
echo "    gpu-mem-util:   ${GPU_MEM_UTIL}"
echo

# The env block matches codex's working config from m6-cuda-vllm-gap-followups.md.
# NVCC_PREPEND_FLAGS and CC/CXX overrides are required because vLLM's at-import
# JIT compile path uses the CUDA toolchain; without the gcc14 override it fails
# under CUDA 13.2.1 + GCC 16 (CUDA does not yet support GCC 16 hosts).
exec env \
    PATH="${VENV}/bin:${PATH}" \
    NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-14' \
    CC=/usr/bin/gcc-14 \
    CXX=/usr/bin/g++-14 \
    CUDAHOSTCXX=/usr/bin/g++-14 \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --served-model-name "${SERVED_NAME}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --dtype "${DTYPE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --attention-backend "${ATTN_BACKEND}" \
        --trust-remote-code \
        --no-enable-log-requests \
        --uvicorn-log-level warning
