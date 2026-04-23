#!/usr/bin/env bash
# Start the local ARLE REPL against a model.
#
# Usage:
#   ./scripts/start_agent.sh [model-path-or-hf-id] [extra-agent-args...]
#
# Examples:
#   ./scripts/start_agent.sh
#   ./scripts/start_agent.sh mlx-community/Qwen3-0.6B-4bit --max-turns 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${ARLE_BIN:-${AGENT_INFER_BIN:-${REPO_ROOT}/target/release/arle}}"

MODEL_ARGS=()
if [[ $# -gt 0 && "${1}" != --* ]]; then
    MODEL_ARGS=(--model-path "${1}")
    shift
fi

if [[ ! -x "${BIN}" ]]; then
    echo "ARLE binary not found at ${BIN}" >&2
    echo "Build one of these first:" >&2
    echo "  cargo build --release --features cli -p agent-infer --bin arle" >&2
    echo "  cargo build --release --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle" >&2
    exit 1
fi

echo "=== ARLE REPL ==="
echo "  Binary: ${BIN}"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    echo "  Model:  ${MODEL_ARGS[1]}"
else
    echo "  Model:  auto-detect"
fi
echo ""

exec "${BIN}" "${MODEL_ARGS[@]}" "$@"
