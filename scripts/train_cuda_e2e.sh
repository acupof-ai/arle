#!/usr/bin/env bash
# CUDA SFT smoke test for the train -> eval -> infer -> resume contract.
#
# Usage:
#   scripts/train_cuda_e2e.sh [OUT_DIR]
#
# Env:
#   CUDA_HOME            default: /usr/local/cuda
#   CARGO_TARGET_DIR     default: /tmp/agent-infer-target-cuda
#   INFER_TEST_MODEL_ID  default: Qwen/Qwen3-0.6B
#   INFER_TEST_MODEL_PATH default: /tmp/agent-infer-models/Qwen3-0.6B
#   CUDA_MIN_FREE_MIB    default: 4096

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-/tmp/train_cuda_e2e_$$}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/agent-infer-target-cuda}"
MODEL_ID="${INFER_TEST_MODEL_ID:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${INFER_TEST_MODEL_PATH:-/tmp/agent-infer-models/Qwen3-0.6B}"
CUDA_MIN_FREE_MIB="${CUDA_MIN_FREE_MIB:-4096}"
TRAIN_OUT="${OUT_DIR}/train"
SFT_DATA="${OUT_DIR}/tiny_sft.jsonl"
EVAL_DATA="${OUT_DIR}/tiny_eval.jsonl"
EVAL_OUT="${OUT_DIR}/eval.json"
AGENT_OUT="${OUT_DIR}/agent.out"
BIN_DIR="${CARGO_TARGET_DIR}/release"
TRAIN_BIN="${BIN_DIR}/train_sft"
EVAL_BIN="${BIN_DIR}/eval_lm"
AGENT_BIN="${BIN_DIR}/agent-infer"

info() { echo "[train_cuda_e2e] $*"; }
die() { echo "[train_cuda_e2e] error: $*" >&2; exit 1; }

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing dir: $path"
}

latest_step_name() {
  local latest="$1"
  readlink "$latest" | xargs basename
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1
}

gpu_process_table() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}

mkdir -p "$OUT_DIR" "$TRAIN_OUT"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  die "hf / huggingface-cli not found"
fi
[[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"

info "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
info "artifacts: $OUT_DIR"
info "model dir: $MODEL_PATH"

FREE_MIB="$(gpu_free_mib)"
if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB < CUDA_MIN_FREE_MIB )); then
  info "compute apps:"
  gpu_process_table || true
  die "only ${FREE_MIB} MiB free on the GPU; need at least ${CUDA_MIN_FREE_MIB} MiB before running CUDA smoke"
fi

if [[ ! -f "${MODEL_PATH}/model.safetensors" ]]; then
  info "downloading ${MODEL_ID} into ${MODEL_PATH}"
  mkdir -p "$MODEL_PATH"
  "${HF_DOWNLOAD[@]}" "$MODEL_ID" \
    --local-dir "$MODEL_PATH" \
    config.json \
    generation_config.json \
    model.safetensors \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json
fi

require_dir "$MODEL_PATH"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/model.safetensors"
require_file "${MODEL_PATH}/tokenizer.json"

cat > "$SFT_DATA" <<'EOF'
{"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Say hello in one word."},{"role":"assistant","content":"Hello."}]}
{"messages":[{"role":"user","content":"Complete: 2 plus 2 equals"},{"role":"assistant","content":"4."}]}
{"messages":[{"role":"user","content":"Reply with one fruit."},{"role":"assistant","content":"Apple."}]}
EOF

cat > "$EVAL_DATA" <<'EOF'
{"messages":[{"role":"user","content":"Say hello in one word."},{"role":"assistant","content":"Hello."}]}
{"messages":[{"role":"user","content":"Reply with one fruit."},{"role":"assistant","content":"Apple."}]}
EOF

info "building train/eval/infer release binaries"
CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  cargo build --release --features cuda -p train --bin train_sft --bin eval_lm
CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  cargo build --release -p agent-infer --features cli

require_file "$TRAIN_BIN"
require_file "$EVAL_BIN"
require_file "$AGENT_BIN"

info "running CUDA train_sft smoke"
CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  "$TRAIN_BIN" \
  --model "$MODEL_PATH" \
  --data "$SFT_DATA" \
  --out "$TRAIN_OUT" \
  --steps 2 \
  --batch 1 \
  --seq-len 64 \
  --lr 1e-6 \
  --save-every 2 \
  --log-every 1 \
  --backend cuda

LATEST="${TRAIN_OUT}/latest"
[[ -L "$LATEST" || -d "$LATEST" ]] || die "missing latest checkpoint at $LATEST"
require_file "${LATEST}/model.safetensors"
require_file "${LATEST}/adapter_model.safetensors"
require_file "${LATEST}/optimizer.safetensors"
info "latest -> $(latest_step_name "$LATEST")"

info "running CUDA eval_lm"
CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  "$EVAL_BIN" \
  --model-path "$LATEST" \
  --data "$EVAL_DATA" \
  --seq-len 64 \
  --backend cuda | tee "$EVAL_OUT"
grep -q '"loss"' "$EVAL_OUT" || die "eval_lm output missing loss field"

info "running infer-side load/generate smoke"
printf 'hi\n/quit\n' | \
  CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  "$AGENT_BIN" \
  --model-path "$LATEST" \
  --max-tokens 8 \
  --non-interactive > "$AGENT_OUT" 2>&1
grep -q "=== agent-infer REPL ===" "$AGENT_OUT" || die "agent-infer banner missing"
grep -q "Model:" "$AGENT_OUT" || die "agent-infer model line missing"
if grep -q "panicked at" "$AGENT_OUT"; then
  die "agent-infer panicked while loading checkpoint"
fi

info "running CUDA resume smoke"
CUDA_HOME="$CUDA_HOME" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  "$TRAIN_BIN" \
  --model "$MODEL_PATH" \
  --data "$SFT_DATA" \
  --out "$TRAIN_OUT" \
  --steps 4 \
  --batch 1 \
  --seq-len 64 \
  --lr 1e-6 \
  --save-every 2 \
  --log-every 1 \
  --backend cuda \
  --resume-from "$LATEST"

[[ "$(latest_step_name "$LATEST")" == "step_000004" ]] || \
  die "resume did not advance latest to step_000004"
require_file "${LATEST}/trainer_state.json"
require_file "${LATEST}/optimizer.safetensors"

info "OK train_sft -> eval_lm -> agent-infer -> resume"
