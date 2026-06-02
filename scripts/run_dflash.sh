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
#   ARLE_TARGET              default: mlx-community/Qwen3.5-4B-MLX-4bit
#   ARLE_DFLASH_DRAFT        default: z-lab/Qwen3.5-4B-DFlash
#   ARLE_PORT                default: 8000
#   ARLE_BENCH_PROMPT        real inline bench prompt
#   ARLE_BENCH_PROMPT_FILE   real bench prompt file; overrides ARLE_BENCH_PROMPT
#   ARLE_GEN_TOK             default: 256 (bench)
#   Legacy AGENT_INFER_* names still work.
#
# Examples:
#   ./scripts/run_dflash.sh
#   ./scripts/run_dflash.sh bench
#   ./scripts/run_dflash.sh request "write quicksort in python"
#   ARLE_TARGET=mlx-community/Qwen3-4B-bf16 \
#     ARLE_DFLASH_DRAFT=z-lab/Qwen3-4B-DFlash-b16 \
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

TARGET="${ARLE_TARGET:-${AGENT_INFER_TARGET:-mlx-community/Qwen3.5-4B-MLX-4bit}}"
DRAFT="${ARLE_DFLASH_DRAFT:-${AGENT_INFER_DFLASH_DRAFT:-z-lab/Qwen3.5-4B-DFlash}}"
PORT="${ARLE_PORT:-${AGENT_INFER_PORT:-8000}}"
BENCH_PROMPT="${ARLE_BENCH_PROMPT:-${AGENT_INFER_BENCH_PROMPT:-write a concise Python quicksort and explain the pivot choice}}"
BENCH_PROMPT_FILE="${ARLE_BENCH_PROMPT_FILE:-${AGENT_INFER_BENCH_PROMPT_FILE:-}}"
GEN_TOK="${ARLE_GEN_TOK:-${AGENT_INFER_GEN_TOK:-256}}"

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
  bench:    prompt=${BENCH_PROMPT_FILE:-${BENCH_PROMPT}}, generation_tokens=${GEN_TOK}

Override via ARLE_TARGET / ARLE_DFLASH_DRAFT / ARLE_PORT.
Legacy AGENT_INFER_TARGET / AGENT_INFER_DFLASH_DRAFT / AGENT_INFER_PORT also work.
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
    local prompt_args=()
    if [ -n "${BENCH_PROMPT_FILE}" ]; then
        prompt_args=(--prompt-file "${BENCH_PROMPT_FILE}")
    else
        prompt_args=(--prompt "${BENCH_PROMPT}")
    fi
    echo "=== DFlash bench: baseline ==="
    cargo run "${CARGO_COMMON[@]}" --bin metal_bench -- \
        --model "${TARGET}" \
        "${prompt_args[@]}" \
        --generation-tokens "${GEN_TOK}" \
        --warmup 1 --runs 3 "$@"
    echo ""
    echo "=== DFlash bench: DFlash on ==="
    cargo run "${CARGO_COMMON[@]}" --bin metal_bench -- \
        --model "${TARGET}" \
        --dflash-draft-model "${DRAFT}" \
        "${prompt_args[@]}" \
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
