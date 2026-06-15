#!/usr/bin/env bash
# H20-gated OPD capability curve run:
# Qwen3.5-4B teacher -> Qwen3.5-0.8B LoRA student, full-materialized
# step_NNNNNN checkpoints, then scripts/opd_capability_curve.py over the
# base student + each checkpoint (+ optional teacher upper-bound point).
#
# This script is intentionally orchestration-only. It assumes the H20/QAT
# environment has already produced normal HF-layout Qwen3.5 model dirs.

set -euo pipefail

ROOT="${ARLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/bench-output/h20-opd-teacher-student-$(date +%Y%m%d-%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
CURVE_DIR="${CURVE_DIR:-$RUN_ROOT/curve}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
MANIFEST="${MANIFEST:-$RUN_ROOT/curve_manifest.json}"

STUDENT_MODEL="${STUDENT_MODEL:-/data01/models/Qwen3.5-0.8B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-/data01/models/Qwen3.5-4B}"

STEPS="${STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
ROLLOUT_LEN="${ROLLOUT_LEN:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-0}"
ROLLOUT_SEED="${ROLLOUT_SEED:-0}"
KL_DIRECTION="${KL_DIRECTION:-forward}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
LR="${LR:-2e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_SET="${LORA_TARGET_SET:-attention-qv}"
PROMPT_IDS="${PROMPT_IDS:-1,3,8}"
EVAL_IDS="${EVAL_IDS:-}"
GATE_EVERY_N="${GATE_EVERY_N:-0}"

CURVE_TASKS="${CURVE_TASKS:-mmlu,gsm8k}"
N_SAMPLES="${N_SAMPLES:-500}"
SEEDS="${SEEDS:-0,1,2,3,4}"
GSM8K_SHOTS="${GSM8K_SHOTS:-}"
SWE_LIMIT="${SWE_LIMIT:-3}"
INCLUDE_TEACHER="${INCLUDE_TEACHER:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CURVE="${RUN_CURVE:-1}"

ARLE_BIN="${ARLE_BIN:-$ROOT/target/release/arle}"
BUILD_ARLE="${BUILD_ARLE:-1}"
TRAIN_BACKEND="${TRAIN_BACKEND:-auto}"
SERVE_BACKEND="${SERVE_BACKEND:-cuda}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8123}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-900}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$RUN_ROOT" "$CHECKPOINT_DIR" "$CURVE_DIR" "$LOG_DIR"

die() {
  echo "[h20-opd-curve] error: $*" >&2
  exit 1
}

log() {
  echo "[h20-opd-curve] $*"
}

if [[ "${SKIP_H20_CHECK:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; set SKIP_H20_CHECK=1 only for non-H20 dry orchestration"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  [[ "$gpu_name" == *H20* ]] || die "expected an H20 GPU, got '$gpu_name'; set SKIP_H20_CHECK=1 to override"
  log "gpu=$gpu_name"
fi

[[ -d "$STUDENT_MODEL" ]] || die "STUDENT_MODEL is not a directory: $STUDENT_MODEL"
[[ -d "$TEACHER_MODEL" ]] || die "TEACHER_MODEL is not a directory: $TEACHER_MODEL"
[[ "$STUDENT_MODEL" != "$TEACHER_MODEL" ]] || die "teacher==student is confounded; pass a larger TEACHER_MODEL"
[[ "$STEPS" =~ ^[0-9]+$ ]] || die "STEPS must be an integer"
[[ "$SAVE_EVERY" =~ ^[0-9]+$ ]] || die "SAVE_EVERY must be an integer"
[[ "$STEPS" -gt 0 ]] || die "STEPS must be > 0 for the capability run"

if [[ "$BUILD_ARLE" == "1" && "$ARLE_BIN" == "$ROOT/target/release/arle" ]]; then
  log "building arle release CUDA binary"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" cargo build --release --features cuda --bin arle
fi
[[ -x "$ARLE_BIN" ]] || die "ARLE_BIN is not executable: $ARLE_BIN"

log "run_root=$RUN_ROOT"
log "student=$STUDENT_MODEL"
log "teacher=$TEACHER_MODEL"
log "checkpoints=$CHECKPOINT_DIR"

write_manifest() {
  STUDENT_MODEL="$STUDENT_MODEL" \
  TEACHER_MODEL="$TEACHER_MODEL" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  MANIFEST="$MANIFEST" \
  STEPS="$STEPS" \
  SAVE_EVERY="$SAVE_EVERY" \
  INCLUDE_TEACHER="$INCLUDE_TEACHER" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

student = Path(os.environ["STUDENT_MODEL"])
teacher = Path(os.environ["TEACHER_MODEL"])
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
manifest = Path(os.environ["MANIFEST"])
steps = int(os.environ["STEPS"])
save_every = int(os.environ["SAVE_EVERY"])
include_teacher = os.environ["INCLUDE_TEACHER"] == "1"

save_steps = []
if save_every > 0:
    save_steps.extend(range(save_every, steps + 1, save_every))
if steps not in save_steps:
    save_steps.append(steps)
save_steps = sorted(set(save_steps))

points = [
    {
        "label": "base-0p8b",
        "model_path": str(student),
        "model_id": student.name,
    }
]
for step in save_steps:
    path = checkpoint_dir / f"step_{step:06d}"
    points.append(
        {
            "label": f"opd-step-{step:06d}",
            "model_path": str(path),
            "model_id": path.name,
        }
    )
if include_teacher:
    points.append(
        {
            "label": "teacher-4b-upper",
            "model_path": str(teacher),
            "model_id": teacher.name,
        }
    )

manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(points, indent=2) + "\n")
print(manifest)
PY
}

check_checkpoint_dirs() {
  MANIFEST="$MANIFEST" "$PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

missing = []
for point in json.loads(Path(os.environ["MANIFEST"]).read_text()):
    label = point["label"]
    path = point.get("model_path")
    if not path or not label.startswith("opd-step-"):
        continue
    p = Path(path)
    if not p.is_dir():
        missing.append(str(p))
    elif not (p / "config.json").is_file() or not (p / "model.safetensors").is_file():
        missing.append(f"{p} (not a servable HF checkpoint)")
if missing:
    print("missing checkpoint dirs:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    sys.exit(2)
PY
}

write_manifest >/dev/null
log "curve_manifest=$MANIFEST"

if [[ "$RUN_TRAIN" == "1" ]]; then
  train_args=(
    train opd
    --backend "$TRAIN_BACKEND"
    --teacher-model "$TEACHER_MODEL"
    --student-model "$STUDENT_MODEL"
    --steps "$STEPS"
    --rollout-len "$ROLLOUT_LEN"
    --rollout-temperature "$ROLLOUT_TEMPERATURE"
    --rollout-top-p "$ROLLOUT_TOP_P"
    --rollout-top-k "$ROLLOUT_TOP_K"
    --rollout-seed "$ROLLOUT_SEED"
    --kl-direction "$KL_DIRECTION"
    --kl-temperature "$KL_TEMPERATURE"
    --gkd-lambda 0.0
    --lr "$LR"
    --grad-clip "$GRAD_CLIP"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-target-set "$LORA_TARGET_SET"
    --prompt-ids "$PROMPT_IDS"
    --gate-every-n "$GATE_EVERY_N"
    --save-checkpoint "$CHECKPOINT_DIR"
    --save-every "$SAVE_EVERY"
    --json
  )
  if [[ -n "$EVAL_IDS" ]]; then
    train_args+=(--eval-ids "$EVAL_IDS")
  fi

  log "train command:"
  printf '  %q' "$ARLE_BIN" "${train_args[@]}"
  printf '\n'
  "$ARLE_BIN" "${train_args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
  check_checkpoint_dirs
else
  log "RUN_TRAIN=0; manifest was written but checkpoints were not produced"
fi

if [[ "$RUN_CURVE" == "1" ]]; then
  curve_args=(
    scripts/opd_capability_curve.py
    --manifest "$MANIFEST"
    --tasks "$CURVE_TASKS"
    --n-samples "$N_SAMPLES"
    --seeds "$SEEDS"
    --serve-backend "$SERVE_BACKEND"
    --serve-host "$SERVE_HOST"
    --serve-port "$SERVE_PORT"
    --serve-ready-timeout "$SERVE_READY_TIMEOUT"
    --arle-bin "$ARLE_BIN"
    --output "$CURVE_DIR"
  )
  if [[ -n "$GSM8K_SHOTS" ]]; then
    curve_args+=(--gsm8k-shots "$GSM8K_SHOTS")
  fi
  if [[ "$CURVE_TASKS" == *swe_pro* ]]; then
    curve_args+=(--swe-limit "$SWE_LIMIT")
  fi

  log "curve command:"
  printf '  %q' "$PYTHON" "${curve_args[@]}"
  printf '\n'
  "$PYTHON" "${curve_args[@]}" 2>&1 | tee "$LOG_DIR/curve.log"
  [[ -f "$CURVE_DIR/curve.json" ]] || die "curve.json was not written at $CURVE_DIR/curve.json"
else
  log "RUN_CURVE=0; skipping capability curve eval"
fi

log "done"
log "run_root=$RUN_ROOT"
log "manifest=$MANIFEST"
log "curve_json=$CURVE_DIR/curve.json"
