#!/usr/bin/env bash
# Canonical one-command runner for Metal DFlash on Apple Silicon.
#
# Subcommands:
#   serve    (default) Launch metal_serve with DFlash on the default Qwen3.5 pair
#   bench    Run metal_bench baseline + DFlash, print throughput delta
#   request  One-shot POST /v1/chat/completions to a running server
#   help     Show this help
#
# Env overrides:
#   AGENT_INFER_TARGET       default: mlx-community/Qwen3.5-4B-MLX-4bit
#   AGENT_INFER_DFLASH_DRAFT default: z-lab/Qwen3.5-4B-DFlash
#   AGENT_INFER_PORT         default: 8000
#   AGENT_INFER_PROMPT_TOK   default: 32 (bench)
#   AGENT_INFER_GEN_TOK      default: 256 (bench)
#
# Examples:
#   ./scripts/run_dflash.sh
#   ./scripts/run_dflash.sh bench
#   ./scripts/run_dflash.sh request "write quicksort in python"
#   AGENT_INFER_TARGET=mlx-community/Qwen3-4B-bf16 \
#     AGENT_INFER_DFLASH_DRAFT=z-lab/Qwen3-4B-DFlash-b16 \
#     ./scripts/run_dflash.sh

set -euo pipefail

case "$(uname -s)" in
    Darwin) ;;
    *)
        echo "run_dflash.sh is only intended for macOS Apple Silicon." >&2
        exit 1
        ;;
esac

case "$(uname -m)" in
    arm64) ;;
    *)
        echo "run_dflash.sh expects Apple Silicon (arm64)." >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${AGENT_INFER_TARGET:-mlx-community/Qwen3.5-4B-MLX-4bit}"
DRAFT="${AGENT_INFER_DFLASH_DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${AGENT_INFER_PORT:-8000}"
PROMPT_TOK="${AGENT_INFER_PROMPT_TOK:-32}"
GEN_TOK="${AGENT_INFER_GEN_TOK:-256}"

CARGO_COMMON=(--release -p infer --no-default-features --features metal,no-cuda)

usage() {
    cat <<EOF
Metal DFlash runner.

Subcommands:
  serve    Launch metal_serve with DFlash (default)
  bench    Run metal_bench baseline + DFlash, print throughput
  request  One-shot /v1/chat/completions POST against a running server
  help     Show this help

Defaults:
  target:   ${TARGET}
  draft:    ${DRAFT}
  port:     ${PORT}
  bench:    prompt_tokens=${PROMPT_TOK}, generation_tokens=${GEN_TOK}

Override via AGENT_INFER_TARGET / AGENT_INFER_DFLASH_DRAFT / AGENT_INFER_PORT.
EOF
}

run_serve() {
    cd "${REPO_ROOT}"
    echo "=== DFlash serve ==="
    echo "  target: ${TARGET}"
    echo "  draft:  ${DRAFT}"
    echo "  port:   ${PORT}"
    echo ""
    exec cargo run "${CARGO_COMMON[@]}" --bin metal_serve -- \
        --model-path "${TARGET}" \
        --dflash-draft-model "${DRAFT}" \
        --port "${PORT}" \
        --bind 127.0.0.1 \
        "$@"
}

run_bench() {
    cd "${REPO_ROOT}"
    echo "=== DFlash bench: baseline ==="
    cargo run "${CARGO_COMMON[@]}" --bin metal_bench -- \
        --model "${TARGET}" \
        --prompt-tokens "${PROMPT_TOK}" \
        --generation-tokens "${GEN_TOK}" \
        --warmup 1 --runs 3 "$@"
    echo ""
    echo "=== DFlash bench: DFlash on ==="
    cargo run "${CARGO_COMMON[@]}" --bin metal_bench -- \
        --model "${TARGET}" \
        --dflash-draft-model "${DRAFT}" \
        --prompt-tokens "${PROMPT_TOK}" \
        --generation-tokens "${GEN_TOK}" \
        --warmup 1 --runs 3 "$@"
}

run_request() {
    local prompt="${1:-write a quicksort in python}"
    shift || true
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl not found; install it or use metal_request directly." >&2
        exit 1
    fi
    echo "=== DFlash request → http://127.0.0.1:${PORT}/v1/chat/completions ==="
    curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(cat <<JSON
{
  "model": "${TARGET}",
  "messages": [{"role": "user", "content": "${prompt//\"/\\\"}"}],
  "max_tokens": 128
}
JSON
)"
    echo ""
}

cmd="${1:-serve}"
case "${cmd}" in
    serve) shift; run_serve "$@" ;;
    bench) shift; run_bench "$@" ;;
    request) shift; run_request "$@" ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
