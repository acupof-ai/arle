#!/usr/bin/env bash
# Matched-A/B serve launcher for the W8A16 marlin-vs-scalar decode bench.
# Identical lifecycle both arms; only ARLE_W8A16_DISABLE_MARLIN differs.
# Usage: marlin_serve.sh <gpu> <port> <disable_marlin:0|1> <label>
set -u
GPU="$1"; PORT="$2"; DISABLE="$3"; LABEL="$4"; MODEL_ARG="${5:-}"
TREE=/host/arle-build
MODEL="${MODEL_ARG:-/host/nvme0/models/iso-tc-huihui-w8a16-e739}"
BIN="$TREE/target/release/arle"
LOG="/host/marlin-bench/$LABEL.serve.log"
mkdir -p /host/marlin-bench
# Serve doesn't need tilelang, but source the canonical env for PATH/proxy parity.
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="$GPU"
export INFER_CUDA_DEVICE=0
if [ "$DISABLE" = "1" ]; then
  export ARLE_W8A16_DISABLE_MARLIN=1
else
  unset ARLE_W8A16_DISABLE_MARLIN
fi
echo "=== SERVE $LABEL gpu=$GPU port=$PORT disable_marlin=$DISABLE $(date -u +%FT%TZ) ===" > "$LOG"
echo "=== bin=$BIN model=$MODEL ARLE_W8A16_DISABLE_MARLIN=${ARLE_W8A16_DISABLE_MARLIN:-<unset>} ===" >> "$LOG"
setsid nohup "$BIN" serve --model-path "$MODEL" --port "$PORT" --bind 127.0.0.1 \
  --mem-fraction-static 0.85 --qwen35-decode-graph true \
  --max-running-requests 8 --max-prompt-tokens 40000 --max-total-tokens 40000 \
  >> "$LOG" 2>&1 < /dev/null &
echo "SERVE_PID=$!"
