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
#   scripts/pf83_license_sequence.sh --dry-run      # print invocation plan, do not run anything
#
# Exit codes:
#   0 — full sequence completed (manual review of bench output for final decision)
#   1 — KILL (greedy FAIL OR PPL Δ% > +5%)
#   2 — pre-flight error (missing model, unbuilt binary, etc)

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Per bench_guidellm.sh:53-56 docstring (M_e gauntlet 2026-05-07): prepend
# .venv/bin to PATH so guidellm + datasets + httpx are resolved consistently.
# Without this, bench wrappers fail with "guidellm not on PATH" even though
# pip install -e .[bench] put it in .venv/bin.
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

SKIP_GREEDY=0
SKIP_PPL=0
DRY_RUN=0
BENCH_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-greedy) SKIP_GREEDY=1 ;;
        --skip-ppl)    SKIP_PPL=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        *)             BENCH_FLAGS+=("$arg") ;;
    esac
done

# Hybrid checkpoint default per 473081d + e99e5a5 — required so greedy_consistency
# actually exercises the PF8 path (anti-pattern #29: W4A8-only checkpoint silently
# keeps the new branch INACTIVE per linear.rs:86 hybrid_w4_fp8_aligned guard).
HYBRID_MODEL="${INFER_TEST_W4A8_MODEL_PATH:-/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix}"

# Stale-binary check: warn if target/release/infer is older than PF8.3 source
# (prevents running PF8.5 against a binary that doesn't have the new dispatch).
PF83_KERNEL=crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu
if [[ -f $PF83_KERNEL && -x target/release/infer ]]; then
    if [[ $PF83_KERNEL -nt target/release/infer ]]; then
        echo "WARN: target/release/infer is OLDER than $PF83_KERNEL" >&2
        echo "  → PF8.3 dispatch may not be present in this binary" >&2
        echo "  → rebuild: CUDA_HOME=/opt/cuda cargo build --release -p infer --features cuda" >&2
        echo "  (continuing in 5s; Ctrl-C to abort)" >&2
        [[ $DRY_RUN -eq 1 ]] || sleep 5
    fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN — would execute (in order):"
    echo "  HYBRID_MODEL=$HYBRID_MODEL"
    echo "  Step 1 [skip=$SKIP_GREEDY]: INFER_MARLIN_W4_FP8_PREFILL=1 INFER_TEST_W4A8_MODEL_PATH=\$HYBRID_MODEL cargo test --release --test greedy_consistency w4a8 -- --nocapture"
    echo "  Step 2 [skip=$SKIP_PPL]: python3 scripts/eval_ppl_pf83.py"
    echo "  Step 3: scripts/bench_pf83_ab.sh ${BENCH_FLAGS[*]:-}"
    echo ""
    echo "Pre-flight checks:"
    [[ -d "$HYBRID_MODEL" ]] && echo "  OK: HYBRID_MODEL dir present" || echo "  FAIL: HYBRID_MODEL dir missing at $HYBRID_MODEL"
    [[ -x target/release/infer ]] && echo "  OK: target/release/infer present" || echo "  FAIL: target/release/infer missing or not executable"
    command -v python3 >/dev/null 2>&1 && echo "  OK: python3 in PATH" || echo "  FAIL: python3 not in PATH"
    [[ -f scripts/eval_ppl_pf83.py ]] && echo "  OK: scripts/eval_ppl_pf83.py present" || echo "  FAIL: scripts/eval_ppl_pf83.py missing"
    [[ -x scripts/bench_pf83_ab.sh ]] && echo "  OK: scripts/bench_pf83_ab.sh present" || echo "  FAIL: scripts/bench_pf83_ab.sh missing or not executable"
    exit 0
fi

if [[ $SKIP_GREEDY -eq 0 ]]; then
    echo "=== Step 1/3: greedy_consistency with INFER_MARLIN_W4_FP8_PREFILL=1 + hybrid checkpoint ==="
    echo "  INFER_TEST_W4A8_MODEL_PATH=$HYBRID_MODEL"
    if ! INFER_MARLIN_W4_FP8_PREFILL=1 \
         INFER_HYBRID_W4A8_PREFILL=1 \
         INFER_TEST_W4A8_MODEL_PATH="$HYBRID_MODEL" \
         cargo test --release --test greedy_consistency w4a8 -- --nocapture; then
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
