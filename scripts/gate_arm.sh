#!/usr/bin/env bash
# One needle-gate arm for the B2 CP-decode verification.
# Usage: gate_arm.sh <label> <cp_size> <lengths_csv> <port> <max_total_tokens> <baseline_log_or_empty>
set -uo pipefail
LABEL="$1"; CP="$2"; LENGTHS="$3"; PORT="$4"; MAXTOK="$5"; BASELINE="${6:-}"
cd /host/arle-gates
export BIN=/root/arle-ops/builds/b2decode/arle
export MODEL=/data00/ThinkingCap-Qwen3.6-27B-FP8
export GATE_PROFILE=generic
export INFER_TP_SIZE=2
export INFER_ATTN_CP_SIZE="$CP"
export INFER_CUDA_DEVICES=1,3
export LENGTHS="$LENGTHS"
export RUNS=3
export PORT="$PORT"
export SERVE_FLAGS="--max-total-tokens $MAXTOK"
export TEMPLATE=qwen3_nonthink
export RAW=1
export NEEDLE_MAX_TOKENS=32
export RUST_LOG=info
export OUT="needle_gate_${LABEL}.log"
[ -z "$BASELINE" ] || export BASELINE_LOG="$BASELINE"
echo "=== arm $LABEL cp=$CP lengths=$LENGTHS port=$PORT start $(date -u +%FT%TZ) ==="
bash /host/arle-build/scripts/lever_gate.sh "$LABEL"
rc=$?
echo "=== arm $LABEL GATE_EXIT=$rc end $(date -u +%FT%TZ) ==="
exit "$rc"
