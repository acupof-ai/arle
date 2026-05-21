#!/usr/bin/env bash
# ARLE OPD — one-click distillation example.
#
# Default (no env vars): runs `arle train opd --smoke` end-to-end on the
# embedded tiny Qwen3.5 config. No downloads, no GPU strictly required (CPU
# backend works for the smoke path). Takes < 30 s on a recent laptop —
# verifies your build is sane and you can see an OPD loss curve come out.
#
# Real-models mode: set ARLE_TEACHER + ARLE_STUDENT to HuggingFace or
# ModelScope IDs. The script will auto-download them on first run, then run
# real OPD distillation through `arle train opd --teacher-model
# <teacher-dir> --student-model <student-dir>`. Downloads resume on retry.
#
# Examples:
#
#   # Smoke (default, < 30 s, no internet, no GPU strictly required)
#   ./examples/opd/run-distillation.sh
#
#   # Real distillation Qwen3.5-4B → Qwen3.5-0.8B-Base from ModelScope
#   ARLE_TEACHER=Qwen/Qwen3.5-4B \
#   ARLE_STUDENT=Qwen/Qwen3.5-0.8B-Base \
#   ARLE_STEPS=500 \
#   ./examples/opd/run-distillation.sh
#
#   # Same but HuggingFace
#   ARLE_SOURCE=huggingface \
#   ARLE_TEACHER=Qwen/Qwen3.5-4B \
#   ARLE_STUDENT=Qwen/Qwen3.5-0.8B-Base \
#   ./examples/opd/run-distillation.sh

set -euo pipefail

cd "$(dirname "$0")/../.."   # project root, regardless of where invoked from

ARLE_TEACHER="${ARLE_TEACHER:-}"
ARLE_STUDENT="${ARLE_STUDENT:-}"
ARLE_SOURCE="${ARLE_SOURCE:-modelscope}"     # modelscope | huggingface
ARLE_STEPS="${ARLE_STEPS:-5}"
ARLE_ROLLOUT_LEN="${ARLE_ROLLOUT_LEN:-8}"
ARLE_LR="${ARLE_LR:-1e-4}"
ARLE_GRAD_CLIP="${ARLE_GRAD_CLIP:-1.0}"
ARLE_BACKEND="${ARLE_BACKEND:-auto}"
ARLE_VENV="${ARLE_VENV:-$PWD/.venv}"
ARLE_OUTPUT_DIR="${ARLE_OUTPUT_DIR:-$PWD/opd-output/$(date +%Y%m%d-%H%M%S)}"

mode="smoke"
if [[ -n "$ARLE_TEACHER" && -n "$ARLE_STUDENT" ]]; then
  mode="real"
elif [[ -n "$ARLE_TEACHER" || -n "$ARLE_STUDENT" ]]; then
  echo "[run-distillation] set BOTH ARLE_TEACHER and ARLE_STUDENT for real-models mode, or neither for smoke" >&2
  exit 1
fi

# ─── Build arle if missing ─────────────────────────────────────────────────────
ARLE_BIN="$PWD/target/release/arle"
if [[ ! -x "$ARLE_BIN" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    echo "[run-distillation] building arle (release + cuda)…"
    NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-14}" \
    INFER_TILELANG_PYTHON="${INFER_TILELANG_PYTHON:-$ARLE_VENV/bin/python}" \
    CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13010}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
    cargo build --release --features cuda --bin arle
  else
    echo "[run-distillation] building arle (release, no-cuda)…"
    cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
  fi
fi

mkdir -p "$ARLE_OUTPUT_DIR"

# ─── Smoke mode (default) ─────────────────────────────────────────────────────
if [[ "$mode" == "smoke" ]]; then
  echo "[run-distillation] mode: SMOKE (embedded tiny Qwen3.5 config, no download)"
  echo "[run-distillation] output: $ARLE_OUTPUT_DIR/run.txt"
  "$ARLE_BIN" train opd \
    --smoke \
    --backend "$ARLE_BACKEND" \
    --lr "$ARLE_LR" \
    --steps "$ARLE_STEPS" \
    --rollout-len "$ARLE_ROLLOUT_LEN" \
    --grad-clip "$ARLE_GRAD_CLIP" \
    --json \
    2>&1 | tee "$ARLE_OUTPUT_DIR/run.txt"
  echo "[run-distillation] smoke OK. Loss column should decrease across steps."
  echo "[run-distillation] for real distillation, set ARLE_TEACHER + ARLE_STUDENT (see header comment)."
  exit 0
fi

# ─── Real-models mode ─────────────────────────────────────────────────────────
echo "[run-distillation] mode: REAL  teacher=$ARLE_TEACHER  student=$ARLE_STUDENT  source=$ARLE_SOURCE"

if [[ ! -d "$ARLE_VENV" ]]; then
  echo "[run-distillation] no Python venv at $ARLE_VENV — create one with:" >&2
  echo "    python3 -m venv .venv && .venv/bin/pip install modelscope" >&2
  echo "or override ARLE_VENV=/path/to/venv." >&2
  exit 1
fi

PY="$ARLE_VENV/bin/python"
PIP="$ARLE_VENV/bin/pip"

if [[ "$ARLE_SOURCE" == "modelscope" ]]; then
  $PY -c "import modelscope" 2>/dev/null || {
    echo "[run-distillation] installing modelscope in $ARLE_VENV"
    $PIP install -q modelscope
  }
elif [[ "$ARLE_SOURCE" == "huggingface" ]]; then
  $PY -c "import huggingface_hub" 2>/dev/null || {
    echo "[run-distillation] installing huggingface_hub in $ARLE_VENV"
    $PIP install -q huggingface_hub
  }
else
  echo "[run-distillation] unknown ARLE_SOURCE=$ARLE_SOURCE (expected: modelscope | huggingface)" >&2
  exit 1
fi

resolve_model() {
  local model_id="$1"
  ARLE_RESOLVE_ID="$model_id" ARLE_RESOLVE_SOURCE="$ARLE_SOURCE" $PY - <<'PY'
import os, sys
model_id = os.environ["ARLE_RESOLVE_ID"]
source = os.environ["ARLE_RESOLVE_SOURCE"]
try:
    if source == "modelscope":
        from modelscope import snapshot_download
        path = snapshot_download(
            model_id,
            cache_dir=os.path.expanduser("~/.cache/modelscope/hub"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        )
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        )
    print(path)
except Exception as e:
    print(f"resolve_model failed for {model_id}: {e}", file=sys.stderr)
    sys.exit(2)
PY
}

echo "[run-distillation] resolving teacher: $ARLE_TEACHER"
TEACHER_DIR="$(resolve_model "$ARLE_TEACHER")"
echo "[run-distillation]   → $TEACHER_DIR"

echo "[run-distillation] resolving student: $ARLE_STUDENT"
STUDENT_DIR="$(resolve_model "$ARLE_STUDENT")"
echo "[run-distillation]   → $STUDENT_DIR"

echo "[run-distillation] launching arle train opd…"
echo "[run-distillation] output: $ARLE_OUTPUT_DIR/run.txt"
"$ARLE_BIN" train opd \
  --backend "$ARLE_BACKEND" \
  --teacher-model "$TEACHER_DIR" \
  --student-model "$STUDENT_DIR" \
  --lr "$ARLE_LR" \
  --steps "$ARLE_STEPS" \
  --rollout-len "$ARLE_ROLLOUT_LEN" \
  --grad-clip "$ARLE_GRAD_CLIP" \
  --json \
  2>&1 | tee "$ARLE_OUTPUT_DIR/run.txt"

echo "[run-distillation] done. Output: $ARLE_OUTPUT_DIR/run.txt"
