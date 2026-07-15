#!/usr/bin/env bash
# DSv4 FP32 compressor A/B benchmark.
# Runs guidellm at rate 1,4,8,16 with 60s duration per rate.
# Usage: run_dsv4_bench.sh <label>
set -uo pipefail

LABEL="${1:?usage: run_dsv4_bench.sh <label>}"
MODEL="/host/DeepSeek-V4-Flash-FP8"
TARGET="http://localhost:8000"
DATA="/host/arle-build/bench-prompts.jsonl"
DATE="$(date +%Y-%m-%d)"
OUTPUT_BASE="/host/arle-build/bench-output/${DATE}-${LABEL}"

for RATE in 1 4 8 16; do
  OUTDIR="${OUTPUT_BASE}-rate${RATE}"
  mkdir -p "$OUTDIR"
  echo "=== rate ${RATE} -> ${OUTDIR} ==="
  GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  guidellm benchmark run \
    --target "$TARGET" \
    --model "$MODEL" \
    --processor "$MODEL" \
    --profile concurrent \
    --data "$DATA" \
    --max-seconds 60 \
    --random-seed 20260416 \
    --output-dir "$OUTDIR" \
    --backend openai_http \
    --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
    --disable-console-interactive \
    --outputs 'result.json' --outputs 'result.csv' \
    --rate "$RATE"
  echo "=== rate ${RATE} done ==="
done

echo "=== ALL DONE. Results in ${OUTPUT_BASE}-rate* ==="
