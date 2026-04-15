#!/usr/bin/env bash
# Start the canonical local Metal server on Apple Silicon.
#
# Usage:
#   ./scripts/start_metal_serve.sh [model-path-or-hf-id] [port] [-- extra metal_serve args]
#
# Defaults:
#   model: ${AGENT_INFER_MODEL:-mlx-community/Qwen3-0.6B-4bit}
#   port:  8000
#   bind:  127.0.0.1
#
# Examples:
#   ./scripts/start_metal_serve.sh
#   ./scripts/start_metal_serve.sh mlx-community/Qwen3-4B-bf16 8012 -- --warmup 0

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Start the canonical local Metal server on Apple Silicon.

Usage:
  ./scripts/start_metal_serve.sh [model-path-or-hf-id] [port] [-- extra metal_serve args]

Defaults:
  model: ${AGENT_INFER_MODEL:-mlx-community/Qwen3-0.6B-4bit}
  port:  8000
  bind:  127.0.0.1

Examples:
  ./scripts/start_metal_serve.sh
  ./scripts/start_metal_serve.sh mlx-community/Qwen3-4B-bf16 8012 -- --warmup 0

The wrapper hides Cargo feature flags and runs:
  cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve -- ...
EOF
    exit 0
fi

case "$(uname -s)" in
    Darwin) ;;
    *)
        echo "start_metal_serve.sh is only intended for macOS Apple Silicon." >&2
        exit 1
        ;;
esac

case "$(uname -m)" in
    arm64) ;;
    *)
        echo "start_metal_serve.sh expects Apple Silicon (arm64)." >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${AGENT_INFER_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
PORT="8000"

if [[ $# -gt 0 && "${1}" != --* ]]; then
    MODEL_PATH="${1}"
    shift
fi

if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
    PORT="${1}"
    shift
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

cd "${REPO_ROOT}"

echo "=== agent-infer Metal bring-up ==="
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${PORT}"
echo "  Bind:  127.0.0.1"
echo "  Cargo: cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve"
echo ""

exec cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve -- \
    --model-path "${MODEL_PATH}" \
    --port "${PORT}" \
    --bind 127.0.0.1 \
    "$@"
