#!/usr/bin/env bash
# Usage: run_la_ab.sh <label> <gpu> <binary_path> <checkpoint_fn> <serve_port>
set -euo pipefail
LABEL="${1:?label}"; GPU="${2:?gpu}"; BIN="${3:?binary}"; FN="${4:?checkpoint_fn}"; PORT="${5:?serve_port}"
OUT="/host/arle-runs/mlpchunk-ab/$LABEL"
mkdir -p "$OUT"
DS=/host/opd-corpora/staged-sweetspot3/train.jsonl
STAGED=/host/opd-corpora/staged-sweetspot3/staged
LOG="$OUT/run.log"
{
  echo "=== $LABEL gpu=$GPU fn=$FN binary_sha=$(sha256sum "$BIN" | cut -d' ' -f1) start=$(date -u +%FT%TZ) ==="
  rc=0
  CUDA_VISIBLE_DEVICES="$GPU" INFER_CUDA_DEVICE=0 \
  ARLE_OPD_VRAM_TRACE=1 ARLE_AOPD_PROFILE=1 ARLE_OPD_OP_MEM_CHECKPOINT_FN="$FN" RUST_LOG=info \
    "$BIN" train agent-opd \
      --student-model /host/ThinkingCap-Qwen3.6-27B-FP8 \
      --dataset "$DS" --staged-root "$STAGED" \
      --work-root "$OUT/work" \
      --synthetic-writeback-seq 40960 \
      --serve-port "$PORT" \
      --lora-rank 16 --lora-alpha 32 --lora-target-set attention-qv || rc=$?
  echo "RUN_EXIT=$rc"
  exit "$rc"
} >"$LOG" 2>&1
