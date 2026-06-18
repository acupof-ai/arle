#!/usr/bin/env bash
# ARLE OPD R1 launcher: Qwen3.6-35B-A3B teacher -> Qwen3.5-4B student
# on MATH question-only prompts.
#
# Purpose:
#   Preserve the exact bring-up configuration for the first non-saturated
#   reasoning verdict lane. GSM8K and MMLU were saturated/no-headroom in the
#   4B-vs-35B checks; MATH-500 showed headroom and uses this train prompt
#   corpus shape.
#
# Default paths match the .62 pod layout used during bring-up. Override any
# variable from the environment instead of editing this file.
#
# Example:
#   CUDA_VISIBLE_DEVICES=2 \
#   RUN_ROOT=/data01/arle-opd-runs/math-r1-$(date +%Y%m%d-%H%M%S) \
#   ARLE_BIN=/data01/arle-build/target/release/arle \
#   ./examples/opd/run-math-r1-35b-to-4b.sh
#
# Fit note:
#   rollout_len=1536 is the R1 long-reasoning target. If current autograd
#   backward cannot fit it, lower ROLLOUT_LEN only as an explicit fit probe and
#   record the OOM tensor in the run log. 4096+ belongs to the chunked-forward
#   lane, not this launcher.

set -euo pipefail

ROOT="${ARLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/bench-output/opd-math-r1-35b-to-4b-$(date +%Y%m%d-%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
CONFIG_JSON="${CONFIG_JSON:-$RUN_ROOT/run_config.json}"

ARLE_BIN="${ARLE_BIN:-$ROOT/target/release/arle}"
BUILD_ARLE="${BUILD_ARLE:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TEACHER_MODEL="${TEACHER_MODEL:-/data01/models/Qwen3.6-35B-A3B-FP8}"
STUDENT_MODEL="${STUDENT_MODEL:-/data01/modelscope-cache/Qwen/Qwen3___5-4B}"
TEACHER_RUNTIME="${TEACHER_RUNTIME:-infer}"
PROMPTS_FILE="${PROMPTS_FILE:-/data01/arle-opd-runs/math-train-question-only.jsonl}"

STEPS="${STEPS:-250}"
SAVE_EVERY="${SAVE_EVERY:-50}"
ROLLOUT_LEN="${ROLLOUT_LEN:-1536}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-0}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-1024}"
PROMPT_SEED="${PROMPT_SEED:-0}"

KL_DIRECTION="${KL_DIRECTION:-forward}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
KL_MASK="${KL_MASK:-completion}"
LOGITS_WINDOW_SIZE="${LOGITS_WINDOW_SIZE:-32}"
GKD_LAMBDA="${GKD_LAMBDA:-0.0}"

LR="${LR:-2e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-8}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

LORA_TARGET_SET="${LORA_TARGET_SET:-all-linear}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

TRAIN_BACKEND="${TRAIN_BACKEND:-cuda}"
ENGINE_OFFLOAD="${ENGINE_OFFLOAD:-student}"
STEP_PROFILE="${STEP_PROFILE:-1}"
STEP_TRACE="${STEP_TRACE:-0}"
JSON_OUTPUT="${JSON_OUTPUT:-1}"

die() {
  echo "[opd-math-r1] error: $*" >&2
  exit 1
}

log() {
  echo "[opd-math-r1] $*"
}

[[ -d "$TEACHER_MODEL" ]] || die "TEACHER_MODEL is not a directory: $TEACHER_MODEL"
[[ -d "$STUDENT_MODEL" ]] || die "STUDENT_MODEL is not a directory: $STUDENT_MODEL"
[[ -f "$PROMPTS_FILE" ]] || die "PROMPTS_FILE is not a file: $PROMPTS_FILE"
[[ "$TEACHER_MODEL" != "$STUDENT_MODEL" ]] || die "teacher==student is confounded"
[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 0 ]] || die "STEPS must be a positive integer"
[[ "$SAVE_EVERY" =~ ^[0-9]+$ ]] || die "SAVE_EVERY must be an integer"
[[ "$ROLLOUT_LEN" =~ ^[0-9]+$ && "$ROLLOUT_LEN" -gt 0 ]] || die "ROLLOUT_LEN must be a positive integer"
[[ "$PROMPT_MAX_TOKENS" =~ ^[0-9]+$ && "$PROMPT_MAX_TOKENS" -gt 0 ]] || die "PROMPT_MAX_TOKENS must be a positive integer"
[[ "$ENGINE_OFFLOAD" =~ ^(off|0|false|student|teacher|all|1|true)$ ]] || die "bad ENGINE_OFFLOAD=$ENGINE_OFFLOAD"

mkdir -p "$RUN_ROOT" "$CHECKPOINT_DIR" "$LOG_DIR"

if [[ "$BUILD_ARLE" == "1" ]]; then
  log "building arle release CUDA binary"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12090}" \
  cargo build --release --features cuda --bin arle
fi
[[ -x "$ARLE_BIN" ]] || die "ARLE_BIN is not executable: $ARLE_BIN"

python3 - <<PY
import json
from pathlib import Path

cfg = {
    "teacher_model": "$TEACHER_MODEL",
    "student_model": "$STUDENT_MODEL",
    "teacher_runtime": "$TEACHER_RUNTIME",
    "prompts_file": "$PROMPTS_FILE",
    "steps": int("$STEPS"),
    "save_every": int("$SAVE_EVERY"),
    "rollout_len": int("$ROLLOUT_LEN"),
    "rollout_temperature": float("$ROLLOUT_TEMPERATURE"),
    "rollout_top_p": float("$ROLLOUT_TOP_P"),
    "rollout_top_k": int("$ROLLOUT_TOP_K"),
    "rollout_seed": int("$ROLLOUT_SEED"),
    "prompt_max_tokens": int("$PROMPT_MAX_TOKENS"),
    "prompt_seed": int("$PROMPT_SEED"),
    "kl_direction": "$KL_DIRECTION",
    "kl_temperature": float("$KL_TEMPERATURE"),
    "kl_mask": "$KL_MASK",
    "logits_window_size": int("$LOGITS_WINDOW_SIZE"),
    "gkd_lambda": float("$GKD_LAMBDA"),
    "lr": float("$LR"),
    "lr_schedule": "$LR_SCHEDULE",
    "lr_warmup_steps": int("$LR_WARMUP_STEPS"),
    "grad_clip": float("$GRAD_CLIP"),
    "lora_target_set": "$LORA_TARGET_SET",
    "lora_rank": int("$LORA_RANK"),
    "lora_alpha": int("$LORA_ALPHA"),
    "train_backend": "$TRAIN_BACKEND",
    "engine_offload": "$ENGINE_OFFLOAD",
    "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
    "checkpoint_dir": "$CHECKPOINT_DIR",
}
path = Path("$CONFIG_JSON")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(path)
PY

train_args=(
  train opd
  --backend "$TRAIN_BACKEND"
  --teacher-model "$TEACHER_MODEL"
  --teacher-runtime "$TEACHER_RUNTIME"
  --student-model "$STUDENT_MODEL"
  --steps "$STEPS"
  --rollout-len "$ROLLOUT_LEN"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --rollout-top-p "$ROLLOUT_TOP_P"
  --rollout-top-k "$ROLLOUT_TOP_K"
  --rollout-seed "$ROLLOUT_SEED"
  --kl-direction "$KL_DIRECTION"
  --kl-temperature "$KL_TEMPERATURE"
  --kl-mask "$KL_MASK"
  --logits-window-size "$LOGITS_WINDOW_SIZE"
  --gkd-lambda "$GKD_LAMBDA"
  --lr "$LR"
  --lr-schedule "$LR_SCHEDULE"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --grad-clip "$GRAD_CLIP"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-target-set "$LORA_TARGET_SET"
  --prompts-file "$PROMPTS_FILE"
  --prompt-max-tokens "$PROMPT_MAX_TOKENS"
  --prompt-seed "$PROMPT_SEED"
  --gate-every-n 0
  --save-checkpoint "$CHECKPOINT_DIR"
  --save-every "$SAVE_EVERY"
)
if [[ "$JSON_OUTPUT" == "1" ]]; then
  train_args+=(--json)
fi

log "run_root=$RUN_ROOT"
log "config=$CONFIG_JSON"
log "train log=$LOG_DIR/train.log"
log "teacher=$TEACHER_MODEL"
log "student=$STUDENT_MODEL"
log "prompts=$PROMPTS_FILE"
log "rollout_len=$ROLLOUT_LEN lora=$LORA_TARGET_SET/r$LORA_RANK/a$LORA_ALPHA"
log "cuda_visible_devices=$CUDA_VISIBLE_DEVICES engine_offload=$ENGINE_OFFLOAD"
printf '[opd-math-r1] command: CUDA_VISIBLE_DEVICES=%q ARLE_OPD_ENGINE_OFFLOAD=%q %q' \
  "$CUDA_VISIBLE_DEVICES" "$ENGINE_OFFLOAD" "$ARLE_BIN"
printf ' %q' "${train_args[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
ARLE_OPD_ENGINE_OFFLOAD="$ENGINE_OFFLOAD" \
ARLE_OPD_STEP_PROFILE="$STEP_PROFILE" \
ARLE_OPD_STEP_TRACE="$STEP_TRACE" \
"$ARLE_BIN" "${train_args[@]}" 2>&1 | tee "$LOG_DIR/train.log"

log "done"
