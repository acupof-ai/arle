#!/usr/bin/env bash
# scripts/train_and_chat.sh — 2-step SFT smoke test that exercises the full
# train → save → load → generate loop on Metal. If this ever fails, the
# trainer-to-ARLE runtime contract is broken regardless of unit tests.
#
# DX-1 (docs/plans/train-eval-infer-dx-v1.md) note: we address the checkpoint
# via `$OUT_DIR/latest` — a symlink the trainer refreshes on every save.
# The caller never has to know the step number; `latest` always resolves to
# the newest `step_NNNNNN` dir.
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
# --model       : base weights to fine-tune on top of
# --data        : JSONL of {prompt, response} supervised examples
# --out         : checkpoint root — writes step_NNNNNN/ + `latest` symlink
# --steps       : total optimizer steps (tiny; this is a smoke test)
# --batch       : micro-batch (1 for deterministic per-example loss)
# --seq-len     : max tokens per example (pad/truncate)
# --save-every  : trigger a checkpoint every N steps (here: the only one)
cargo run --release --no-default-features --features metal,no-cuda,cli \
  -p agent-infer --bin arle -- \
  train sft \
  --model "$MODEL_PATH" \
  --data  "$SFT_DATA" \
  --out   "$OUT_DIR" \
  --steps 2 --batch 1 --seq-len 64 --lr 1e-6 \
  --save-every 2 --log-every 1 --backend metal

# DX-1: $OUT_DIR/latest → step_000002. Symlink is refreshed on every save,
# so downstream tooling never hardcodes step numbers. Any non-unix target
# falls back to reading the lex-max `step_*` directory instead.
CHECKPOINT="$OUT_DIR/latest"
if [[ ! -e "$CHECKPOINT" ]]; then
  echo "latest symlink missing at $CHECKPOINT (DX-1 wiring broken?)" >&2
  exit 3
fi
if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "checkpoint weights missing: $CHECKPOINT/model.safetensors" >&2
  exit 3
fi

echo "[train_and_chat] === generate 8 tokens from $(readlink "$CHECKPOINT") ==="
echo "hi" | cargo run --release --no-default-features --features metal,no-cuda,cli \
  -p agent-infer --bin arle -- \
  --model-path "$CHECKPOINT" \
  --max-tokens 8 --non-interactive

echo "[train_and_chat] OK — train→save→load→generate round-trip green"
