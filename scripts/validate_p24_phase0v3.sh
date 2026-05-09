#!/usr/bin/env bash
# Phase 0v3 5-gate validation runner for #24 W4A8 prefill graph capture hoist.
# Per docs/plans/2026-05-09-prefill-graph-phase0v3-validation-protocol.md.
#
# Usage:  ./scripts/validate_p24_phase0v3.sh [--skip-bench]
# Output: stdout = colored gate results; exit 0 on full pass, 1 on any fail.
#
# Gates:
#   1. cargo check --release -p infer --features cuda
#   2. cargo clippy --release -p infer --features cuda -- -D warnings
#   3. cargo test --test e2e + --test greedy_consistency (with INFER_PREFILL_GRAPH=1)
#   4. functional smoke: server boot + 200 OK + capture key log line
#   5. matched-control bench (skipable via --skip-bench, ~3min)
#
# Pass-all → brief #37 throughput bench (next phase).
# Fail-any → KILL with errors entry per failure mode.

set -euo pipefail

SKIP_BENCH=${1:-}
LOG_DIR="/tmp/p24-validate-$(date +%s)"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
FAILURES=()

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { PASS=$((PASS + 1)); echo -e "${GREEN}✓${NC} $*"; }
fail() { FAIL=$((FAIL + 1)); FAILURES+=("$*"); echo -e "${RED}✗${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }

CARGO_ENV='env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9'

# ──────────────────────── Gate 1 — cargo check ────────────────────────
info "Gate 1: cargo check --release -p infer --features cuda"
if eval $CARGO_ENV cargo check --release -p infer --features cuda > "$LOG_DIR/gate1.log" 2>&1; then
    pass "cargo check (CUDA release)"
else
    fail "cargo check failed; see $LOG_DIR/gate1.log"
fi

# ──────────────────────── Gate 2 — cargo clippy ───────────────────────
info "Gate 2: cargo clippy --release -p infer --features cuda -- -D warnings"
if eval $CARGO_ENV cargo clippy --release -p infer --features cuda -- -D warnings > "$LOG_DIR/gate2.log" 2>&1; then
    pass "cargo clippy -D warnings clean"
else
    fail "cargo clippy failed; see $LOG_DIR/gate2.log"
fi

# ──────────────────── Gate 3 — correctness tests ──────────────────────
info "Gate 3a: cargo test --test e2e (INFER_PREFILL_GRAPH=1)"
if INFER_PREFILL_GRAPH=1 eval $CARGO_ENV cargo test --release -p infer --features cuda --test e2e -- --test-threads=1 > "$LOG_DIR/gate3a.log" 2>&1; then
    pass "e2e tests pass with INFER_PREFILL_GRAPH=1"
else
    fail "e2e tests failed; see $LOG_DIR/gate3a.log"
fi

info "Gate 3b: cargo test --test greedy_consistency test_greedy_solo_vs_concurrent"
if INFER_PREFILL_GRAPH=1 eval $CARGO_ENV cargo test --release -p infer --features cuda --test greedy_consistency test_greedy_solo_vs_concurrent -- --test-threads=1 --nocapture > "$LOG_DIR/gate3b.log" 2>&1; then
    pass "greedy_solo_vs_concurrent pass"
else
    fail "greedy_solo_vs_concurrent failed; see $LOG_DIR/gate3b.log"
fi

# ──────────────────── Gate 4 — server smoke ────────────────────────────
info "Gate 4: server boot + 200 OK + capture key in log"
SERVER_LOG="$LOG_DIR/server-smoke.log"
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
  CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  RUST_LOG=info \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8765 --num-slots 4 --max-seq-len 5120 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

sleep 35  # model load + tilelang AOT compile

if curl -fsS --max-time 30 -X POST http://127.0.0.1:8765/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"Tell me a short story about a painter","max_tokens":8,"temperature":0,"stream":false}' \
    > "$LOG_DIR/smoke.response" 2>&1; then
    pass "server 200 OK + completion received"
else
    fail "server smoke failed; see $LOG_DIR/smoke.response"
fi

# Verify capture key log line (codex draft format: "prefill graph capture key: tokens=N ... marlin_scratch=true")
if grep -qE 'prefill graph capture key.*marlin_scratch=true' "$SERVER_LOG"; then
    pass "Qwen3 prefill graph capture key with marlin_scratch=true logged"
else
    fail "no 'marlin_scratch=true' capture key in server log; see $SERVER_LOG"
fi

# Anti-pattern check #6 (skill v1.7.0): capture exists != capture reused
CAPTURE_COUNT=$(grep -c 'prefill graph capture key' "$SERVER_LOG" || echo 0)
LAUNCH_COUNT=$(grep -c 'prefill graph replay' "$SERVER_LOG" || echo 0)
info "capture count=$CAPTURE_COUNT, replay count=$LAUNCH_COUNT (informational)"

kill -TERM $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

# ──────────────────── Gate 5 — bench (optional) ────────────────────────
if [[ "$SKIP_BENCH" == "--skip-bench" ]]; then
    info "Gate 5: bench SKIPPED (--skip-bench)"
else
    info "Gate 5: matched-control bench c=4 4k/256 (~3min, server boot needed)"
    info "  Skip with: $0 --skip-bench"
    if PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
       INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
       scripts/bench_guidellm.sh p24-validate-graph-on \
       --concurrencies 4 --max-seconds 60 --warmup 10 \
       --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256' \
       > "$LOG_DIR/gate5-bench.log" 2>&1; then
        TTFT=$(grep -E 'TTFT.*p50' "$LOG_DIR/gate5-bench.log" | head -1 || echo 'TBD')
        pass "bench completed (informational, throughput license is #37): $TTFT"
    else
        fail "bench failed; see $LOG_DIR/gate5-bench.log"
    fi
fi

# ──────────────────────── Summary ─────────────────────────────────────
echo
echo "============================================="
echo "Phase 0v3 Validation Summary: $PASS pass / $FAIL fail"
echo "Logs: $LOG_DIR"
if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}✓ ALL GATES PASS${NC} — brief #37 throughput bench next"
    exit 0
else
    echo -e "${RED}✗ FAILURES:${NC}"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    echo "→ KILL #24 with errors entry per failure mode (see anti-pattern check matrix)"
    exit 1
fi
