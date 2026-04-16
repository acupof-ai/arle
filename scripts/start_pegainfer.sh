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

echo "=== Starting Pegainfer ==="
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${PORT}"
echo ""

cd "$(dirname "$0")/../infer"

cargo run --release -- \
    --model-path "../${MODEL_PATH}" \
    --port "${PORT}"
