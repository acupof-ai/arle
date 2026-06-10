#!/usr/bin/env bash
# #58 lever gate — license-or-kill correctness gate for DSv4 kernel-path levers.
#
# Gate semantics (strategy v2 §2.6): correct inference, NOT byte-identity:
#   1. needle ladder ×3 same-config repeats (scripts/dsv4_needle_gate.py)
#   2. baseline same-config envelope = the non-determinism floor
#   3. self-consistency: the lever path's own greedy output is the reference
#
# Usage (on the 8xH20 pod, serve script mirrors needle_serve.sh):
#   scripts/dsv4_lever_gate.sh baseline
#   scripts/dsv4_lever_gate.sh flashmla  ARLE_DSV4_FLASHMLA_DECODE=1
#   scripts/dsv4_lever_gate.sh wqkv      ARLE_DSV4_FUSED_WQKV_DECODE=1
#   scripts/dsv4_lever_gate.sh contigmoe ARLE_DSV4_GPU_ROUTER=1
#
# Each invocation: boots a serve with the lever env, runs the gate matrix,
# writes needle_gate_<label>.log, tears the serve down. Compare lever logs
# against the baseline log's SUMMARY distribution: PASS = exact/partial
# counts within the baseline envelope (±1 per length), zero garbage-class
# outputs. Any new miss class or looping output = KILL.
set -uo pipefail

LABEL="${1:?usage: dsv4_lever_gate.sh <label> [ENV=V ...]}"
shift || true
BIN="${BIN:-/data01/build/arle/target/release/arle}"
MODEL="${MODEL:-/data01/models/DeepSeek-V4-Flash}"
PORT="${PORT:-18189}"
LENGTHS="${LENGTHS:-115,300,446,2000,8000}"
RUNS="${RUNS:-3}"
OUT="${OUT:-needle_gate_${LABEL}.log}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export INFER_TP_SIZE="${INFER_TP_SIZE:-8}"
export INFER_DSV4_MAX_SEQ_LEN="${INFER_DSV4_MAX_SEQ_LEN:-16384}"
export RUST_LOG="${RUST_LOG:-info}"
export ARLE_DSV4_MOE_BACKEND="${ARLE_DSV4_MOE_BACKEND:-allreduce}"
export ARLE_DSV4_INCREMENTAL_KV=1
export ARLE_DSV4_EXPERT_BACKEND="${ARLE_DSV4_EXPERT_BACKEND:-deepgemm}"
# Lever env flips ride the CLI: dsv4_lever_gate.sh <label> KEY=VAL ...
for kv in "$@"; do export "${kv?}"; done

"$BIN" serve --backend cuda --model-path "$MODEL" --port "$PORT" \
    > "serve_${LABEL}.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVE_PID 2>/dev/null || { echo "[gate] serve died; see serve_${LABEL}.log"; exit 3; }
    sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[gate] serve never ready"; exit 3; }

PORT="$PORT" python3 "$ROOT/scripts/dsv4_needle_gate.py" "$LENGTHS" "$RUNS" 0.0 2>&1 | tee "$OUT"
echo "[gate] $LABEL done -> $OUT"
