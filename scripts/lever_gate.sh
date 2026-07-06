#!/usr/bin/env bash
# lever gate — model/backend-neutral license-or-kill correctness gate
# (generalized from dsv4_lever_gate.sh per #68; original #58).
#
# Gate semantics (strategy v2 §2.6): correct inference, NOT byte-identity:
#   1. needle ladder ×3 same-config repeats (scripts/needle_gate.py)
#   2. baseline same-config envelope = the non-determinism floor
#   3. self-consistency: the lever path's own greedy output is the reference
#
# Usage:
#   # DSv4 (8xH20, TP=8) — the dsv4 profile carries the multi-GPU + kernel env:
#   GATE_PROFILE=dsv4 scripts/lever_gate.sh baseline
#   GATE_PROFILE=dsv4 scripts/lever_gate.sh flashmla ARLE_DSV4_FLASHMLA_DECODE=1
#   # Qwen (single GPU) KV-dtype matrix — pass serve flags, generic profile.
#   # MODEL is any on-pod single-GPU Qwen: Qwen3-0.6B (dense, cleanest gate) or
#   # Qwen3.6-35B-A3B (qwen3_5_moe family). No Qwen3.5-4B checkpoint on the pod.
#   GATE_PROFILE=generic MODEL=/data01/models/Qwen3-0.6B \
#     SERVE_FLAGS="--kv-cache-dtype fp8" scripts/lever_gate.sh qwen_fp8
#
# Each invocation: boots a serve with the lever env, runs the gate matrix,
# writes needle_gate_<label>.log, tears the serve down. Compare lever logs
# against the baseline log's SUMMARY distribution: PASS = exact/partial
# counts within the baseline envelope (±1 per length), zero garbage-class
# outputs. Any new miss class or looping output = KILL.
set -uo pipefail

LABEL="${1:?usage: lever_gate.sh <label> [ENV=V ...]}"
shift || true
# Default to THIS tree's release binary; the old /data01 absolute default
# broke on every other box layout (silent exit 3 at serve boot).
BIN="${BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/target/release/arle}"
MODEL="${MODEL:-/data01/models/DeepSeek-V4-Flash}"
PORT="${PORT:-18189}"
LENGTHS="${LENGTHS:-115,300,446,2000,8000}"
RUNS="${RUNS:-3}"
SERVE_FLAGS="${SERVE_FLAGS:-}"
OUT="${OUT:-needle_gate_${LABEL}.log}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RUST_LOG="${RUST_LOG:-info}"

# DSv4 multi-GPU profile: the TP=8 + DSv4 kernel env. Defaults to dsv4 so the
# existing DSv4 gate is byte-identical; any other GATE_PROFILE (generic, qwen)
# skips it for a single-GPU model that brings its own config via SERVE_FLAGS.
DSV4_FLAGS=()
if [ "${GATE_PROFILE:-dsv4}" = "dsv4" ]; then
    export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    export INFER_TP_SIZE="${INFER_TP_SIZE:-8}"
    export ARLE_DSV4_MOE_BACKEND="${ARLE_DSV4_MOE_BACKEND:-allreduce}"
    export ARLE_DSV4_INCREMENTAL_KV=1
    export ARLE_DSV4_EXPERT_BACKEND="${ARLE_DSV4_EXPERT_BACKEND:-deepgemm}"
    DSV4_FLAGS=(--max-total-tokens "${MAX_TOTAL_TOKENS:-16384}")
fi
# Lever env flips ride the CLI: lever_gate.sh <label> KEY=VAL ...
for kv in "$@"; do export "${kv?}"; done

# shellcheck disable=SC2086  # SERVE_FLAGS is an intentional word-split passthrough
"$BIN" serve --backend cuda --model-path "$MODEL" --port "$PORT" "${DSV4_FLAGS[@]}" $SERVE_FLAGS \
    > "serve_${LABEL}.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVE_PID 2>/dev/null || { echo "[gate] serve died; see serve_${LABEL}.log"; exit 3; }
    sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[gate] serve never ready"; exit 3; }

PORT="$PORT" python3 "$ROOT/scripts/needle_gate.py" "$LENGTHS" "$RUNS" 0.0 2>&1 | tee "$OUT"
echo "[gate] $LABEL done -> $OUT"
