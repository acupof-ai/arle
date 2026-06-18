#!/usr/bin/env bash
# Retained OPD launch recipe for the .62 CUDA pod:
# Qwen3.6-35B-A3B-FP8 teacher -> Qwen3.5-4B student on MATH question-only
# prompts with long on-policy rollout. This wrapper is intentionally concrete;
# it exists so the exact run parameters live in examples/opd instead of in
# transient shell history.

set -euo pipefail

ARLE_ROOT="${ARLE_ROOT:-/data01/arle-opd-runs/agent-infer-73763ee8-longreasoning}"
ARLE_BIN="${ARLE_BIN:-/data01/arle-build/target/release/arle}"
RUN_ROOT="${RUN_ROOT:-/data01/arle-opd-runs/opd-math-r1-35b-to-4b-r2048-$(date +%Y%m%d-%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TMUX_SESSION="${TMUX_SESSION:-opd_math_r1_35b_to_4b_$(date +%H%M%S)}"
DETACH_TMUX="${DETACH_TMUX:-1}"

TEACHER_MODEL="${TEACHER_MODEL:-/data01/models/Qwen3.6-35B-A3B-FP8}"
STUDENT_MODEL="${STUDENT_MODEL:-/data01/modelscope-cache/Qwen/Qwen3___5-4B}"
PROMPTS_FILE="${PROMPTS_FILE:-/data01/arle-opd-runs/math-train-question-only.jsonl}"

STEPS="${STEPS:-250}"
SAVE_EVERY="${SAVE_EVERY:-50}"
ROLLOUT_LEN="${ROLLOUT_LEN:-2048}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-2048}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-0}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
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
TEACHER_RUNTIME="${TEACHER_RUNTIME:-infer}"
ENGINE_OFFLOAD="${ENGINE_OFFLOAD:-teacher}"
STEP_PROFILE="${STEP_PROFILE:-1}"
STEP_TRACE="${STEP_TRACE:-0}"
JSON_OUTPUT="${JSON_OUTPUT:-1}"

env_args=(
  "ARLE_ROOT=$ARLE_ROOT"
  "ARLE_BIN=$ARLE_BIN"
  "RUN_ROOT=$RUN_ROOT"
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "TEACHER_MODEL=$TEACHER_MODEL"
  "STUDENT_MODEL=$STUDENT_MODEL"
  "PROMPTS_FILE=$PROMPTS_FILE"
  "STEPS=$STEPS"
  "SAVE_EVERY=$SAVE_EVERY"
  "ROLLOUT_LEN=$ROLLOUT_LEN"
  "PROMPT_MAX_TOKENS=$PROMPT_MAX_TOKENS"
  "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE"
  "ROLLOUT_TOP_P=$ROLLOUT_TOP_P"
  "ROLLOUT_TOP_K=$ROLLOUT_TOP_K"
  "ROLLOUT_SEED=$ROLLOUT_SEED"
  "PROMPT_SEED=$PROMPT_SEED"
  "KL_DIRECTION=$KL_DIRECTION"
  "KL_TEMPERATURE=$KL_TEMPERATURE"
  "KL_MASK=$KL_MASK"
  "LOGITS_WINDOW_SIZE=$LOGITS_WINDOW_SIZE"
  "GKD_LAMBDA=$GKD_LAMBDA"
  "LR=$LR"
  "LR_SCHEDULE=$LR_SCHEDULE"
  "LR_WARMUP_STEPS=$LR_WARMUP_STEPS"
  "GRAD_CLIP=$GRAD_CLIP"
  "LORA_TARGET_SET=$LORA_TARGET_SET"
  "LORA_RANK=$LORA_RANK"
  "LORA_ALPHA=$LORA_ALPHA"
  "TRAIN_BACKEND=$TRAIN_BACKEND"
  "TEACHER_RUNTIME=$TEACHER_RUNTIME"
  "ENGINE_OFFLOAD=$ENGINE_OFFLOAD"
  "STEP_PROFILE=$STEP_PROFILE"
  "STEP_TRACE=$STEP_TRACE"
  "JSON_OUTPUT=$JSON_OUTPUT"
)

if [[ "$DETACH_TMUX" == "1" ]]; then
  mkdir -p "$RUN_ROOT"
  printf -v command 'cd %q && env' "$ARLE_ROOT"
  for item in "${env_args[@]}"; do
    printf -v command '%s %q' "$command" "$item"
  done
  printf -v command '%s %q' "$command" "./examples/opd/run-math-r1-35b-to-4b.sh"

  tmux new-session -d -s "$TMUX_SESSION" "$command"
  echo "[opd-math-r1-launch] tmux_session=$TMUX_SESSION"
  echo "[opd-math-r1-launch] run_root=$RUN_ROOT"
  echo "[opd-math-r1-launch] log=$RUN_ROOT/logs/train.log"
  exit 0
fi

cd "$ARLE_ROOT"
env "${env_args[@]}" ./examples/opd/run-math-r1-35b-to-4b.sh
