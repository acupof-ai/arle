#!/usr/bin/env bash
# PF8.5 e2e bench — A/B INFER_MARLIN_W4_FP8_PREFILL=0 (baseline INT8) vs =1 (treatment FP8).
#
# Thin wrapper around bench_ab.sh — pre-fills PF8.3 invocation.
# Per a66d99a §2 license matrix + aebd4a5 §3 PPL gate:
#   LICENSE: TTFT p50 Δ ≥ -8%  σ < 5%  n=3
#   KILL:    TTFT p50 Δ < -3%  OR any ITL/decode regression
#
# Usage:
#   scripts/bench_pf83_ab.sh           # full preset (4k prompt, c=4, 120s)
#   scripts/bench_pf83_ab.sh --quick   # ~2-min preset for triage
#
# Env:
#   MODEL              path to W4A8-marlin checkpoint (default models/Qwen3-4B-W4A8-marlin)
#   PORT               server port (default 8000)
#   BIN                infer binary (default target/release/infer)

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# NOTE: default = HYBRID checkpoint (quant_type=marlin_w4_hybrid). PF8 dispatch
# only activates on hybrid weights per linear.rs:86 hybrid_w4_fp8_aligned() guard.
# Using a W4A8-only checkpoint silently keeps the new PF8 branch INACTIVE
# (anti-pattern #29 per b551bea + 473081d).
MODEL="${MODEL:-infer/models/Qwen3-4B-W4-hybrid-zpfix}"
BIN="${BIN:-target/release/infer}"
PORT="${PORT:-8000}"

if [[ ! -d "$MODEL" ]]; then
    echo "error: model dir not found at $MODEL" >&2
    echo "  set MODEL=<path> to a hybrid W4 marlin checkpoint" >&2
    echo "  (config.json must have \"quant_type\": \"marlin_w4_hybrid\")" >&2
    exit 2
fi
if [[ ! -x "$BIN" ]]; then
    echo "error: infer binary not found/executable at $BIN" >&2
    echo "  build with: CUDA_HOME=/opt/cuda cargo build --release" >&2
    exit 2
fi

# RUST_MIN_STACK=8MB: defensive against Task #43 stack overflow under
# sustained W4A16 4k-token bench load (PID 1816462 crash at ~7min uptime
# preceded by prefix cache pressure fallback + 331ms cleanup). Same value
# applied to both A/B servers so it's invariant across the comparison
# (doesn't bias TTFT/ITL measurements). Per task #43 documented mitigations.
PORT="$PORT" exec scripts/bench_ab.sh \
    pf83-baseline-int8 \
    pf83-treatment-fp8 \
    --model "$MODEL" \
    --processor "$MODEL" \
    --concurrencies 4 \
    --max-seconds 120 \
    --warmup 10 \
    --cmd-a "RUST_MIN_STACK=8388608 INFER_MARLIN_W4_FP8_PREFILL=0 $BIN --model-path $MODEL --port $PORT \
             > /tmp/pf83-baseline-int8.log 2>&1 &" \
    --cmd-b "RUST_MIN_STACK=8388608 INFER_MARLIN_W4_FP8_PREFILL=1 $BIN --model-path $MODEL --port $PORT \
             > /tmp/pf83-treatment-fp8.log 2>&1 &" \
    "$@"
