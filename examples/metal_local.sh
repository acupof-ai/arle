#!/usr/bin/env bash
set -euo pipefail

MODEL="${ARLE_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
PORT="${ARLE_PORT:-8000}"

cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle

./target/release/arle --doctor
exec ./target/release/arle serve --backend metal --model-path "$MODEL" --port "$PORT"
