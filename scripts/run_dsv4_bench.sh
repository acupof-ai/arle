#!/usr/bin/env bash
# DSv4 A/B benchmark: guidellm fixed-concurrency sweep against a running server.
# Usage: run_dsv4_bench.sh <label> [data.jsonl]   (env: RATES="1 4 8 16", SECONDS_PER_RATE=60)
# Summary/Δ tables are Python's job (bench_compare.py) — keep this a plain loop.
set -uo pipefail

LABEL="${1:?usage: run_dsv4_bench.sh <label> [data.jsonl]}"
MODEL="/host/DeepSeek-V4-Flash-FP8"
TARGET="http://localhost:8000"
DATA="${2:-/host/arle-build/bench-prompts.jsonl}"
RATES="${RATES:-1 4 8 16}"
SECONDS_PER_RATE="${SECONDS_PER_RATE:-60}"
DATE="$(date +%Y-%m-%d)"
OUTPUT_BASE="/host/arle-build/bench-output/${DATE}-${LABEL}"

for RATE in $RATES; do
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
    --data-args '{"output_tokens_count_column":"output_tokens"}' \
    --max-seconds "$SECONDS_PER_RATE" \
    --random-seed 20260416 \
    --output-dir "$OUTDIR" \
    --backend openai_http \
    --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
    --disable-console-interactive \
    --outputs 'result.json' --outputs 'result.csv' \
    --rate "$RATE"
  # Server-side counters (num_slots, cache hits, preemptions) alongside the
  # client-side result — the bench spec requires reporting capacity, and the
  # slot-clamp regression was only caught from a stale log.
  curl -s "$TARGET/v1/stats" -o "$OUTDIR/server-stats.json" 2>/dev/null || true
  echo "=== rate ${RATE} done ==="
done

echo "=== ALL DONE. Results in ${OUTPUT_BASE}-rate* ==="
