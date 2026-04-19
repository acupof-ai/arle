#!/usr/bin/env bash
# scripts/train_and_chat.sh — 2-step SFT smoke test that exercises the full
# train → save → load → generate loop on Metal. If this ever fails, the
# trainer-to-inference contract is broken regardless of unit tests.
#
# Usage:  scripts/train_and_chat.sh [OUT_DIR]
# Env:    INFER_TEST_MODEL_PATH (default: models/Qwen3-0.6B)
#         INFER_TEST_SFT_DATA   (default: models/tiny_sft.jsonl)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-/tmp/train_and_chat_$$}"
MODEL_PATH="${INFER_TEST_MODEL_PATH:-models/Qwen3-0.6B}"
SFT_DATA="${INFER_TEST_SFT_DATA:-models/tiny_sft.jsonl}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "missing model dir: $MODEL_PATH (set INFER_TEST_MODEL_PATH)" >&2
  exit 2
fi
if [[ ! -f "$SFT_DATA" ]]; then
  echo "missing sft data: $SFT_DATA (set INFER_TEST_SFT_DATA)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "[train_and_chat] out=$OUT_DIR"
echo "[train_and_chat] model=$MODEL_PATH  data=$SFT_DATA"

echo "[train_and_chat] === SFT (2 steps, metal backend) ==="
cargo run --release --no-default-features --features metal \
  -p train --bin train_sft -- \
  --model "$MODEL_PATH" \
  --data  "$SFT_DATA" \
  --out   "$OUT_DIR" \
  --steps 2 --batch 1 --seq-len 64 --lr 1e-6 \
  --save-every 2 --log-every 1 --backend metal

CHECKPOINT="$OUT_DIR/step_2"
if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "checkpoint missing: $CHECKPOINT/model.safetensors" >&2
  exit 3
fi

echo "[train_and_chat] === generate 8 tokens from checkpoint ==="
echo "hi" | cargo run --release --no-default-features --features metal,no-cuda \
  -p agent-infer -- \
  --model-path "$CHECKPOINT" \
  --max-tokens 8 --non-interactive

echo "[train_and_chat] OK — train→save→load→generate round-trip green"
