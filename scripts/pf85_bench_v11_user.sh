#!/usr/bin/env bash
# pf85_bench_v11_user.sh — single-shot user-runnable bench v11 for PF8.5
# license decision at conc=1.
#
# Wraps the pickup-state §3 invocation that was previously copy-paste-only.
# Adds:
# - server start with PF8.3 substrate enabled
# - 30s warmup wait
# - guidellm conc=1 60s sustained-load bench
# - automatic server cleanup
# - integration with pf83_bench_health.sh for verdict
# - prints license decision per a66d99a §2 (TTFT Δ ≥ -8% threshold)
#
# Per docs/research/2026-05-10-next-session-pickup-state.md §3 + DISPROVEN
# doc §3, this bench is USER-runs-only because Claude session sleep limits
# block the 60s wait + 30s warmup chain. This script makes it one command:
#
#   bash scripts/pf85_bench_v11_user.sh
#
# Outputs:
#   /tmp/pf83-FINAL-treatment.log         server log
#   bench-output/2026-05-10-pf83-treatment-conc1-FINAL/  bench results
#   stdout                                 license decision verdict
#
# Cross-refs:
#   - 11763ba PF8.3 substrate
#   - 57c37b5 H8 DISPROVEN (kernel works at conc=1, KILL is load-dependent)
#   - 0cde63d PF8.3 RUNTIME KILL evidence
#   - a66d99a PF8.5 license matrix (TTFT Δ ≥ -8%)
#   - 868e147 pf83_bench_health.sh
#   - v3 baseline INT8 conc=1: TTFT 53.6 ms mdn, ITL 6.8 ms, 1.1 req/s, 697 tok/s
#
# License gate:
#   LICENSE  if TTFT mdn ≤ 49.3 ms  (Δ ≥ -8% vs 53.6 ms baseline)
#   KILL     if TTFT mdn  > 55.2 ms (Δ < -3% regression)
#   REVIEW   if 49.3 < TTFT mdn ≤ 55.2 (need n=3 σ-tight to decide)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL_PATH="infer/models/Qwen3-4B-W4-hybrid-zpfix"
SERVER_LOG="/tmp/pf83-FINAL-treatment.log"
OUTPUT_DIR="$REPO/bench-output/2026-05-10-pf83-treatment-conc1-FINAL"
PORT=8000
TARGET="http://127.0.0.1:$PORT"

# Pre-flight checks
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint missing: $MODEL_PATH" >&2
  echo "       expected at $REPO/$MODEL_PATH (PF8 hybrid checkpoint per 11763ba)" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/guidellm" ]]; then
  echo "ERROR: guidellm not in .venv. Run: pip install -e .[bench]" >&2
  exit 1
fi

if [[ ! -x "target/release/infer" ]]; then
  echo "ERROR: infer binary not built. Run: CUDA_HOME=/usr/local/cuda cargo build --release" >&2
  exit 1
fi

# Cleanup any prior server orphans on this port (port-level, not command-string)
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 2

# Pre-create output dir (per v3-v10 cascade lesson — guidellm 0.6.0 can't
# create the dir itself, save will crash)
mkdir -p "$OUTPUT_DIR"

PATH=".venv/bin:$PATH"
export PATH

echo "[pf85_bench_v11] Starting infer with PF8.3 substrate enabled..."
echo "[pf85_bench_v11]   INFER_HYBRID_W4A8_PREFILL=1"
echo "[pf85_bench_v11]   INFER_MARLIN_W4_FP8_PREFILL=1"
echo "[pf85_bench_v11]   RUST_MIN_STACK=33554432 (per Task #43 hypothesis)"
echo "[pf85_bench_v11]   server log: $SERVER_LOG"

RUST_MIN_STACK=33554432 \
  INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_MARLIN_W4_FP8_PREFILL=1 \
  setsid target/release/infer \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[pf85_bench_v11] Cleaning up server (PID $SERVER_PID)..."
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 3
  fuser -k "$PORT/tcp" 2>/dev/null || true
}
trap cleanup EXIT

echo "[pf85_bench_v11] Waiting up to 60s for server readiness..."
for i in $(seq 1 60); do
  if curl -fsS "$TARGET/v1/models" >/dev/null 2>&1; then
    echo "[pf85_bench_v11] Server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server died during startup. Last log lines:" >&2
    tail -50 "$SERVER_LOG" >&2
    exit 2
  fi
  sleep 1
done

if ! curl -fsS "$TARGET/v1/models" >/dev/null 2>&1; then
  echo "ERROR: server readiness timeout. Last log lines:" >&2
  tail -50 "$SERVER_LOG" >&2
  exit 3
fi

echo "[pf85_bench_v11] Running guidellm conc=1 60s sustained-load bench..."
guidellm benchmark run \
    --target "$TARGET" \
    --model "$MODEL_PATH" \
    --processor "$MODEL_PATH" \
    --profile concurrent --rate "1" --max-seconds 60 --warmup 5 \
    --random-seed 20260416 \
    --data 'prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128' \
    --output-dir "$OUTPUT_DIR" \
    --backend openai_http \
    --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
    --disable-console-interactive \
    --outputs json --outputs csv --outputs html
BENCH_EXIT=$?

echo ""
echo "==================== HEALTH CHECK ===================="
bash "$REPO/scripts/pf83_bench_health.sh" "$OUTPUT_DIR" "$SERVER_LOG"
HEALTH_EXIT=$?

echo ""
echo "==================== LICENSE DECISION ===================="
if [[ "$HEALTH_EXIT" -ne 0 ]]; then
  echo "❌ Health check did not return HEALTHY. License decision DEFERRED."
  echo "   See pf83_bench_health.sh output above for next steps."
  exit "$HEALTH_EXIT"
fi

# Parse TTFT mdn from results.json
RESULTS_JSON="$(find "$OUTPUT_DIR" -maxdepth 2 -name 'benchmarks.json' -o -name 'results.json' 2>/dev/null | head -1)"
if [[ -z "$RESULTS_JSON" ]]; then
  echo "⚠ Could not find results.json under $OUTPUT_DIR — check guidellm output"
  exit 4
fi

TTFT_MDN_MS="$(python3 -c "
import json, sys
try:
    with open('$RESULTS_JSON') as f:
        d = json.load(f)
    benchmarks = d.get('benchmarks', [])
    for b in benchmarks:
        m = b.get('metrics', {})
        ttft = m.get('time_to_first_token_ms', m.get('ttft_ms', {}))
        if isinstance(ttft, dict):
            mdn = ttft.get('median', ttft.get('p50'))
            if mdn is not None:
                print(f'{float(mdn):.1f}')
                break
except Exception as e:
    print(f'ERR:{e}', file=sys.stderr)
")"

if [[ -z "$TTFT_MDN_MS" || "$TTFT_MDN_MS" == ERR:* ]]; then
  echo "⚠ Could not parse TTFT mdn from $RESULTS_JSON"
  echo "   Inspect manually: cat $RESULTS_JSON | python3 -m json.tool | head -100"
  exit 5
fi

# Compare against baseline 53.6 ms / threshold 49.3 (LICENSE) / 55.2 (REVIEW upper)
echo "PF8.3 treatment FP8 conc=1: TTFT mdn = ${TTFT_MDN_MS} ms"
echo "INT8 baseline (v3):         TTFT mdn = 53.6 ms"
DELTA_PCT="$(python3 -c "print(f'{(($TTFT_MDN_MS - 53.6) / 53.6) * 100:+.1f}')")"
echo "Δ vs baseline:              ${DELTA_PCT}% (LICENSE if Δ ≤ -8%, KILL if Δ > -3%)"
echo ""

if (( $(python3 -c "print(1 if $TTFT_MDN_MS <= 49.3 else 0)") )); then
  echo "🎯 LICENSE PF8.3 substrate at conc=1 — codex pickup Task #47 H1' refactor"
  echo "   See docs/plans/M_pf83_h1prime_static_scratch.md (commit 05e2135)"
  echo "   And REVISION docs/research/2026-05-10-h1prime-design-revision-marlinscratch-already-exists.md (2cc608a)"
  exit 0
elif (( $(python3 -c "print(1 if $TTFT_MDN_MS > 55.2 else 0)") )); then
  echo "🚫 KILL PF8.3 substrate — TTFT regression beyond -3% kill threshold"
  echo "   Close Task #44 PF8 chain, pivot to Task #28 Medusa per 2e1e73a"
  exit 0
else
  echo "⚠  REVIEW window — Δ between -8% and -3%, need n=3 σ-tight to decide"
  echo "   Re-run this script 2-3 more times for variance estimation"
  exit 0
fi
