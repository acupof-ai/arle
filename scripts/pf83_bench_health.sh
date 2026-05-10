#!/usr/bin/env bash
# pf83_bench_health.sh — operationalizes SKILL kernel-optimization v1.12.0 #34b:
# when bench reports 0 successful → CHECK SERVER LOG FIRST before debugging tool.
#
# Single-shot diagnostic for any guidellm bench-output dir + paired server log.
# Outputs 3-line verdict so you know whether to debug bench-tool quirks vs
# debug kernel failure vs proceed to license decision.
#
# Usage:
#   scripts/pf83_bench_health.sh <bench-output-dir> [<server-log-path>]
#
# Examples:
#   scripts/pf83_bench_health.sh bench-output/2026-05-10-pf83-treatment-direct-v8/
#   scripts/pf83_bench_health.sh bench-output/2026-05-10-pf83-treatment-conc1-FINAL /tmp/pf83-FINAL-treatment.log
#
# Exit codes:
#   0 = healthy bench (succeeded, ready for license decision)
#   1 = bench produced 0 successful requests AND server log shows kernel failure (substrate KILL signal)
#   2 = bench produced 0 successful requests but server log clean (likely tool quirk — check guidellm CLI)
#   3 = bench produced no output at all (didn't run / save crash)
#   4 = usage error

set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <bench-output-dir> [<server-log-path>]" >&2
  exit 4
fi

BENCH_DIR="$1"
SERVER_LOG="${2:-}"

if [[ ! -d "$BENCH_DIR" ]]; then
  echo "ERROR: bench-output dir does not exist: $BENCH_DIR" >&2
  exit 4
fi

# 1. Did guidellm produce a results file?
RESULTS_JSON="$(find "$BENCH_DIR" -maxdepth 2 -name 'results.json' -o -name 'benchmarks.json' 2>/dev/null | head -1)"
RESULTS_HTML="$(find "$BENCH_DIR" -maxdepth 2 -name '*.html' 2>/dev/null | head -1)"

if [[ -z "$RESULTS_JSON" && -z "$RESULTS_HTML" ]]; then
  echo "VERDICT: BENCH-NO-OUTPUT"
  echo "DETAIL:  no results.json/benchmarks.json/*.html in $BENCH_DIR — guidellm crashed before save (per 7f7a58e v7-v9 cascade pattern)"
  echo "NEXT:    re-run bench with --outputs html + absolute --output-dir + pre-mkdir (lessons from v3-v10)"
  exit 3
fi

# 2. Count successful vs failed requests from results.json (preferred) or html (fallback)
SUCCESS_COUNT=0
FAIL_COUNT=0

if [[ -n "$RESULTS_JSON" ]]; then
  # guidellm 0.6.0 results.json structure: benchmarks[].metrics.requests_successful_total / requests_errored_total
  SUCCESS_COUNT="$(python3 -c "
import json, sys
try:
    with open('$RESULTS_JSON') as f:
        data = json.load(f)
    total_succ = 0
    total_fail = 0
    benchmarks = data.get('benchmarks', [])
    for b in benchmarks:
        m = b.get('metrics', {}) or b.get('run_stats', {}) or {}
        request_totals = m.get('request_totals', {}) or {}
        total_succ += int(
            request_totals.get(
                'successful',
                m.get('requests_successful_total', m.get('successful_requests', 0)),
            )
            or 0
        )
        total_fail += int(
            request_totals.get(
                'errored',
                m.get('requests_errored_total', m.get('errored_requests', 0)),
            )
            or 0
        )
    print(f'{total_succ}|{total_fail}')
except Exception as e:
    print('ERR|' + str(e))
" 2>/dev/null)"
  if [[ "$SUCCESS_COUNT" == ERR* ]]; then
    SUCCESS_COUNT=0
    FAIL_COUNT=0
  else
    FAIL_COUNT="${SUCCESS_COUNT##*|}"
    SUCCESS_COUNT="${SUCCESS_COUNT%|*}"
  fi
fi

# 3. Server log pattern check (only if log path provided)
KERNEL_FAIL_COUNT=0
LIVE_KERNEL_FAIL_COUNT=0
KERNEL_FAIL_PATTERN=""
LIVE_KERNEL_FAIL_PATTERN=""
if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
  # Match kernel failure patterns (PF8.3 + W4A8 + W4A16 marlin)
  KERNEL_FAIL_COUNT="$(grep -cE 'failed with code|gemm.*failed|prefill batch failed|cudaError' "$SERVER_LOG" 2>/dev/null || true)"
  KERNEL_FAIL_COUNT="${KERNEL_FAIL_COUNT:-0}"
  KERNEL_FAIL_PATTERN="$(grep -mE 1 'failed with code|gemm.*failed|prefill batch failed' "$SERVER_LOG" 2>/dev/null | head -1 | cut -c1-120)"
  # Warmup may intentionally probe a shape and back off on OOM. Treat only
  # live-request failures as substrate health failures.
  LIVE_KERNEL_FAIL_COUNT="$(grep -E 'failed with code|gemm.*failed|prefill batch failed|cudaError' "$SERVER_LOG" 2>/dev/null | grep -vc 'Pass 3 prefill warmup' || true)"
  LIVE_KERNEL_FAIL_COUNT="${LIVE_KERNEL_FAIL_COUNT:-0}"
  LIVE_KERNEL_FAIL_PATTERN="$(grep -E 'failed with code|gemm.*failed|prefill batch failed|cudaError' "$SERVER_LOG" 2>/dev/null | grep -v 'Pass 3 prefill warmup' | head -1 | cut -c1-120)"
fi

# 4. Verdict
echo "BENCH:   success=$SUCCESS_COUNT  fail=$FAIL_COUNT  output=$BENCH_DIR"
if [[ -n "$SERVER_LOG" ]]; then
  echo "SERVER:  log=$SERVER_LOG  kernel_failures=$KERNEL_FAIL_COUNT  live_kernel_failures=$LIVE_KERNEL_FAIL_COUNT"
fi

if [[ "$LIVE_KERNEL_FAIL_COUNT" -gt 0 ]]; then
  echo "VERDICT: SUBSTRATE-KILL ($SUCCESS_COUNT successful, server log shows $LIVE_KERNEL_FAIL_COUNT live kernel failures)"
  echo "DETAIL:  $LIVE_KERNEL_FAIL_PATTERN"
  echo "NEXT:    debug kernel/runtime memory path (NOT bench tool) — see SKILL v1.12.0 #34b"
  exit 1
fi

if [[ "$SUCCESS_COUNT" -gt 0 ]]; then
  echo "VERDICT: HEALTHY ($SUCCESS_COUNT successful requests)"
  echo "NEXT:    proceed to license-or-kill decision per a66d99a §2 (TTFT Δ ≥ -8% threshold)"
  exit 0
fi

# 0 successful requests
if [[ "$KERNEL_FAIL_COUNT" -gt 0 ]]; then
  echo "VERDICT: SUBSTRATE-KILL ($SUCCESS_COUNT successful, server log shows $KERNEL_FAIL_COUNT kernel failures)"
  echo "DETAIL:  $KERNEL_FAIL_PATTERN"
  echo "NEXT:    debug kernel (NOT bench tool) — see SKILL v1.12.0 #34b. PF8.3 example: H1' static-scratch refactor per docs/plans/M_pf83_h1prime_static_scratch.md"
  exit 1
fi

# 0 successful + no kernel failures (or no server log provided)
if [[ -z "$SERVER_LOG" ]]; then
  echo "VERDICT: AMBIGUOUS (0 successful, no server log to cross-check)"
  echo "NEXT:    re-run with server log path: $0 $BENCH_DIR /tmp/<server>.log"
  exit 2
fi

echo "VERDICT: TOOL-QUIRK ($SUCCESS_COUNT successful but server log clean — bench-tool issue not substrate)"
echo "NEXT:    debug guidellm CLI (--backend-kwargs validate_backend, --outputs html, absolute --output-dir, pre-mkdir per v3-v10 cascade)"
exit 2
