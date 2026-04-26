#!/usr/bin/env bash
set -euo pipefail

BACKEND="${ARLE_TRAIN_BACKEND:-cpu}"
OUT_DIR="${ARLE_TINY_OUT:-$(mktemp -d)}"

cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
./target/release/arle train test --backend "$BACKEND" --out-dir "$OUT_DIR" --json
./target/release/arle --model-path "$OUT_DIR/sft/latest" run \
  --no-tools \
  --prompt "Say hello in one word." \
  --json
