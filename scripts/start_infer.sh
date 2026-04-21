#!/bin/bash
# Start infer inference server
#
# Usage:
#   ./scripts/start_infer.sh [model_path] [port]
#
# Examples:
#   ./scripts/start_infer.sh                                    # defaults
#   ./scripts/start_infer.sh models/Qwen3-4B 8000

set -euo pipefail

MODEL_PATH="${1:-models/Qwen3-4B}"
PORT="${2:-8000}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/bench-output/server-logs"
TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S)"
LOG_FILE="${INFER_LOG_FILE:-${LOG_DIR}/${TIMESTAMP}-port${PORT}.log}"

mkdir -p "$LOG_DIR"
export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
export RUST_LOG="${RUST_LOG:-info}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting Pegainfer ==="
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${PORT}"
echo "  Log:   ${LOG_FILE}"
echo ""

cd "$(dirname "$0")/../infer"

cargo run --release -- \
    --model-path "../${MODEL_PATH}" \
    --port "${PORT}"
