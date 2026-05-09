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

MODEL="${MODEL:-models/Qwen3-4B-W4A8-marlin}"
BIN="${BIN:-target/release/infer}"
PORT="${PORT:-8000}"

if [[ ! -d "$MODEL" ]]; then
    echo "error: model dir not found at $MODEL" >&2
    echo "  set MODEL=<path> or run: arle model download <hf-id-of-w4a8-marlin>" >&2
    exit 2
fi
if [[ ! -x "$BIN" ]]; then
    echo "error: infer binary not found/executable at $BIN" >&2
    echo "  build with: CUDA_HOME=/opt/cuda cargo build --release" >&2
    exit 2
fi

PORT="$PORT" exec scripts/bench_ab.sh \
    pf83-baseline-int8 \
    pf83-treatment-fp8 \
    --model "$MODEL" \
    --processor "$MODEL" \
    --concurrencies 4 \
    --max-seconds 120 \
    --warmup 10 \
    --cmd-a "INFER_MARLIN_W4_FP8_PREFILL=0 $BIN --model-path $MODEL --port $PORT \
             > /tmp/pf83-baseline-int8.log 2>&1 &" \
    --cmd-b "INFER_MARLIN_W4_FP8_PREFILL=1 $BIN --model-path $MODEL --port $PORT \
             > /tmp/pf83-treatment-fp8.log 2>&1 &" \
    "$@"
