#!/usr/bin/env bash
# PF8.3 full license sequence — runs greedy + PPL + bench A/B in order, exits early on KILL.
#
# Sequence per docs/research/2026-05-10-pf83-ppl-gate-methodology.md (aebd4a5) §4:
#   1. greedy_consistency PASS with INFER_MARLIN_W4_FP8_PREFILL=1
#   2. PPL gate via eval_ppl_pf83.py    (Δ% ≤ +1.0% wikitext = LICENSE, > +5% = KILL)
#   3. e2e bench A/B via bench_pf83_ab.sh (TTFT Δ% ≥ -8% σ < 5% n=3 = LICENSE)
#   4. License decision left to human/codex review of bench output table
#
# Usage:
#   scripts/pf83_license_sequence.sh           # full sequence (greedy + PPL + bench-full)
#   scripts/pf83_license_sequence.sh --quick   # ~2-min triage bench
#   scripts/pf83_license_sequence.sh --skip-greedy  # if greedy already verified this build
#   scripts/pf83_license_sequence.sh --skip-ppl     # if PPL already gated separately
#
# Exit codes:
#   0 — full sequence completed (manual review of bench output for final decision)
#   1 — KILL (greedy FAIL OR PPL Δ% > +5%)
#   2 — pre-flight error (missing model, unbuilt binary, etc)

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_GREEDY=0
SKIP_PPL=0
BENCH_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-greedy) SKIP_GREEDY=1 ;;
        --skip-ppl)    SKIP_PPL=1 ;;
        *)             BENCH_FLAGS+=("$arg") ;;
    esac
done

if [[ $SKIP_GREEDY -eq 0 ]]; then
    echo "=== Step 1/3: greedy_consistency with INFER_MARLIN_W4_FP8_PREFILL=1 ==="
    if ! INFER_MARLIN_W4_FP8_PREFILL=1 cargo test --release --test greedy_consistency w4a8 -- --nocapture; then
        echo "FAIL: greedy_consistency — KILL PF8.3" >&2
        exit 1
    fi
    echo "PASS: greedy_consistency"
fi

if [[ $SKIP_PPL -eq 0 ]]; then
    echo ""
    echo "=== Step 2/3: PPL gate via eval_ppl_pf83.py ==="
    if ! python3 scripts/eval_ppl_pf83.py; then
        echo "FAIL: PPL gate Δ% > +5% — KILL PF8.3" >&2
        exit 1
    fi
    echo "PASS: PPL gate"
fi

echo ""
echo "=== Step 3/3: e2e bench A/B via bench_pf83_ab.sh ==="
exec scripts/bench_pf83_ab.sh "${BENCH_FLAGS[@]}"
