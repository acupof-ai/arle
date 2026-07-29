#!/usr/bin/env bash
# Usage: run_mlp_ladder.sh <label> <gpu> <seq> <chunk> <serve_port>
set -euo pipefail
LABEL="${1:?label}"; GPU="${2:?gpu}"; SEQ="${3:?seq}"; CHUNK="${4:?chunk}"; PORT="${5:?serve_port}"
OUT="/host/arle-runs/mlp-ladder/$LABEL"
mkdir -p "$OUT"
BIN=/host/arle-build/target/release/arle
DS=/host/opd-corpora/staged-sweetspot3/train.jsonl
STAGED=/host/opd-corpora/staged-sweetspot3/staged
LOG="$OUT/run.log"
{
  echo "=== $LABEL gpu=$GPU seq=$SEQ chunk=$CHUNK binary_sha=$(sha256sum "$BIN" | cut -d' ' -f1) start=$(date -u +%FT%TZ) ==="
  rc=0
  CUDA_VISIBLE_DEVICES="$GPU" INFER_CUDA_DEVICE=0 \
  ARLE_OPD_VRAM_TRACE=1 ARLE_AOPD_PROFILE=1 ARLE_OPD_MLP_SEQ_CHUNK="$CHUNK" RUST_LOG=info \
    "$BIN" train agent-opd \
      --student-model /host/ThinkingCap-Qwen3.6-27B-FP8 \
      --dataset "$DS" --staged-root "$STAGED" \
      --work-root "$OUT/work" \
      --synthetic-writeback-seq "$SEQ" \
      --serve-port "$PORT" \
      --lora-rank 16 --lora-alpha 32 --lora-target-set attention-qv || rc=$?
  echo "RUN_EXIT=$rc"
  exit "$rc"
} >"$LOG" 2>&1
