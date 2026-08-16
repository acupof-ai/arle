#!/usr/bin/env bash
# Persistent serve for the B2 decode-perf gates.
# Usage: serve_arm.sh <tag> <cp_size> <max_total_tokens> <port> [extra serve flags...]
# Writes /host/arle-gates/serve_<tag>.{log,pid}; stop with: kill -- -$(cat pidfile)
set -uo pipefail
TAG="$1"; CP="$2"; MAXTOK="$3"; PORT="$4"; shift 4
LOG=/host/arle-gates/serve_${TAG}.log
PIDFILE=/host/arle-gates/serve_${TAG}.pid
cd /host/arle-gates
setsid env INFER_TP_SIZE=2 INFER_ATTN_CP_SIZE="$CP" INFER_CUDA_DEVICES=1,3 RUST_LOG=info \
  /root/arle-ops/builds/b2decode/arle serve --backend cuda \
  --model-path /data00/ThinkingCap-Qwen3.6-27B-FP8 --port "$PORT" \
  --max-total-tokens "$MAXTOK" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "serve $TAG cp=$CP maxtok=$MAXTOK port=$PORT pgid=$(cat $PIDFILE) launched"
