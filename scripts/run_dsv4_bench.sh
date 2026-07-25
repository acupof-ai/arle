#!/usr/bin/env bash
# DSv4 A/B benchmark: native fixed-concurrency sweep against a running server.
# Usage: run_dsv4_bench.sh <label> [data.jsonl]
#   env: RATES="1,4,8,16", SECONDS_PER_RATE=60, MAX_TOKENS=256, PORT=8000
# Runner: scripts/bench_throughput.py (canonical; guidellm removed 2026-07-16 —
# it silently dropped max_tokens, defaulting outputs to 16 tokens).
# Summary/Δ tables are Python's job (bench_compare.py) — keep this a plain loop.
set -uo pipefail

LABEL="${1:?usage: run_dsv4_bench.sh <label> [data.jsonl]}"
MODEL="/host/DeepSeek-V4-Flash-FP8"
PORT="${PORT:-8000}"
TARGET="http://localhost:${PORT}"
DATA="${2:-/host/bench-prompts-20.jsonl}"
# #180: the old default pointed at a dataset that no longer exists anywhere and
# had no generator — a silent substitution makes every Delta% uninterpretable.
if [[ ! -f "$DATA" ]]; then
  echo "dataset $DATA missing. Regenerate the anchored pair:" >&2
  echo "  python3 scripts/gen_bench_prompts.py bench-prompts-64.jsonl 64 13400 256" >&2
  echo "  head -20 bench-prompts-64.jsonl > bench-prompts-20.jsonl" >&2
  exit 1
fi
RATES="${RATES:-1,4,8,16}"
SECONDS_PER_RATE="${SECONDS_PER_RATE:-60}"
MAX_TOKENS="${MAX_TOKENS:-256}"
DATE="$(date +%Y-%m-%d)"
OUTDIR="/host/arle-build/bench-output/${DATE}-${LABEL}"
mkdir -p "$OUTDIR"

sha256sum "$DATA" > "$OUTDIR/dataset.sha256"
echo "=== ${LABEL}: grid ${RATES}, ${SECONDS_PER_RATE}s/point, max_tokens ${MAX_TOKENS} -> ${OUTDIR} ==="
python3 /host/arle-build/scripts/bench_throughput.py \
  --url "$TARGET" \
  --model "$MODEL" \
  --prompts-jsonl "$DATA" \
  --concurrency-grid "$RATES" \
  --seconds-per-concurrency "$SECONDS_PER_RATE" \
  --max-tokens "$MAX_TOKENS" \
  --seed 20260416 \
  --output "$OUTDIR/result.json"

# Server-side counters (num_slots, cache hits, preemptions) alongside the
# client-side result — the bench spec requires reporting capacity, and the
# slot-clamp regression was only caught from a stale log.
curl -s "$TARGET/v1/stats" -o "$OUTDIR/server-stats.json" 2>/dev/null || true
echo "=== ${LABEL} done. Results in ${OUTDIR} ==="
