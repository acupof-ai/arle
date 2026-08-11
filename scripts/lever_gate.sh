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
#   GATE_PROFILE=dsv4 SERVE_FLAGS="--dsv4-flashmla-decode false" \
#     scripts/lever_gate.sh scalar
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
STATS_OUT="${STATS_OUT:-}"
EXPECTED_PRODUCT_SHA256="${EXPECTED_PRODUCT_SHA256:-}"
EXPECTED_KERNEL_BUNDLE_ID="${EXPECTED_KERNEL_BUNDLE_ID:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RUST_LOG="${RUST_LOG:-info}"

validate_summary() {
    local log=$1
    local baseline=${2:-}
    python3 - "$log" "$RUNS" "$LENGTHS" "$baseline" "${LEVER_GATE_REQUIRE_EXACT:-0}" <<'PY'
import re, sys

path, runs, lengths, baseline, require_exact = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5] == "1"
expected_lengths = [int(x) for x in lengths.split(",") if x]
summary = re.compile(
    r"^SUMMARY len=(\d+) .* exact=(\d+) partial=(\d+) miss=(\d+) (?:DET|NONDET)(?: kv=.*)?$"
)

def load(path):
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if " ERROR " in line:
                raise SystemExit(f"[gate] request error: {line.strip()}")
            if not line.startswith("SUMMARY "):
                continue
            match = summary.match(line.strip())
            if match is None:
                raise SystemExit(f"[gate] malformed summary: {line.strip()}")
            length, exact, partial, miss = map(int, match.groups())
            if exact + partial + miss != runs:
                raise SystemExit(
                    f"[gate] incomplete summary: {exact}+{partial}+{miss} != runs={runs}"
                )
            if length in counts:
                raise SystemExit(f"[gate] duplicate summary length {length}")
            counts[length] = (exact, partial, miss)
    if sorted(counts) != sorted(expected_lengths):
        raise SystemExit(
            f"[gate] summary lengths {sorted(counts)} != expected {sorted(expected_lengths)}"
        )
    return counts

counts = load(path)
if require_exact:
    for length, (exact, partial, miss) in counts.items():
        if exact != runs or partial or miss:
            raise SystemExit(
                f"[gate] len={length} requires exact={runs}, got exact={exact} partial={partial} miss={miss}"
            )
if baseline:
    baseline_counts = load(baseline)
    for length in expected_lengths:
        exact, partial, miss = counts[length]
        base_exact, base_partial, base_miss = baseline_counts[length]
        if miss > base_miss or abs(exact - base_exact) > 1 or abs(partial - base_partial) > 1:
            raise SystemExit(
                f"[gate] len={length} outside baseline envelope: "
                f"exact={exact} baseline={base_exact}, partial={partial} baseline={base_partial}, "
                f"miss={miss} baseline={base_miss}"
            )
print(f"[gate] correctness PASS: summaries={len(counts)}")
PY
}

if [ -n "${LEVER_GATE_VALIDATE_LOG:-}" ]; then
    validate_summary "$LEVER_GATE_VALIDATE_LOG" "${BASELINE_LOG:-}"
    exit
fi
if [ -n "$EXPECTED_PRODUCT_SHA256" ] || [ -n "$EXPECTED_KERNEL_BUNDLE_ID" ]; then
    [ -n "$EXPECTED_PRODUCT_SHA256" ] && [ -n "$EXPECTED_KERNEL_BUNDLE_ID" ] || {
        echo "[gate] expected product SHA and kernel bundle ID must be provided together" >&2
        exit 2
    }
    [ -n "$STATS_OUT" ] || { echo "[gate] STATS_OUT is required with expected identities" >&2; exit 2; }
fi

# DSv4 multi-GPU profile: the TP=8 + DSv4 kernel env. Defaults to dsv4 so the
# existing DSv4 gate is byte-identical; any other GATE_PROFILE (generic, qwen)
# skips it for a single-GPU model that brings its own config via SERVE_FLAGS.
DSV4_FLAGS=()
DSV4_CLAIMS=""
if [ "${GATE_PROFILE:-dsv4}" = "dsv4" ]; then
    export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    [ "${INFER_TP_SIZE:-8}" = 8 ] || { echo "[gate] DSv4 requires INFER_TP_SIZE=8" >&2; exit 2; }
    export INFER_TP_SIZE=8
    gate_op="lever-$LABEL-$$-$RANDOM"
    ARLE_OP_ID="$gate_op" ARLE_OWNER="$(id -u):$(id -un)" ARLE_CLAIM_PID="$$" \
        bash "$ROOT/scripts/pick-gpu.sh" reserve-set "$INFER_CUDA_DEVICES" >/dev/null || {
        echo "[gate] DSv4 requires eight free unique SM90 GPUs" >&2
        exit 2
    }
    DSV4_CLAIMS="$INFER_CUDA_DEVICES"
    export ARLE_DSV4_MOE_BACKEND="${ARLE_DSV4_MOE_BACKEND:-allreduce}"
    export ARLE_DSV4_INCREMENTAL_KV=1
    export ARLE_DSV4_EXPERT_BACKEND="${ARLE_DSV4_EXPERT_BACKEND:-deepgemm}"
    DSV4_FLAGS=(--max-total-tokens "${MAX_TOTAL_TOKENS:-16384}")
fi
# Lever env flips ride the CLI: lever_gate.sh <label> KEY=VAL ...
for kv in "$@"; do export "${kv?}"; done

if [ "${GATE_PROFILE:-dsv4}" = "dsv4" ]; then
    # shellcheck disable=SC2086  # SERVE_FLAGS is an intentional word-split passthrough
    "$BIN" serve --backend cuda --model-path "$MODEL" --port "$PORT" "${DSV4_FLAGS[@]}" $SERVE_FLAGS \
        > "serve_${LABEL}.log" 2>&1 &
else
    # shellcheck disable=SC2086  # SERVE_FLAGS is an intentional word-split passthrough
    "$BIN" serve --backend cuda --model-path "$MODEL" --port "$PORT" $SERVE_FLAGS \
        > "serve_${LABEL}.log" 2>&1 &
fi
SERVE_PID=$!
cleanup() {
    kill "$SERVE_PID" 2>/dev/null
    wait "$SERVE_PID" 2>/dev/null
    old_ifs="$IFS"; IFS=','
    for gpu in $DSV4_CLAIMS; do
        claim="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$gpu"
        [ "$(awk -F= '$1=="op" {print $2}' "$claim" 2>/dev/null)" = "${gate_op:-}" ] && rm -f "$claim"
    done
    IFS="$old_ifs"
}
trap cleanup EXIT

for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVE_PID 2>/dev/null || { echo "[gate] serve died; see serve_${LABEL}.log"; exit 3; }
    sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[gate] serve never ready"; exit 3; }

if [ -n "$EXPECTED_PRODUCT_SHA256" ]; then
    stats_tmp="$(mktemp "${STATS_OUT}.tmp.XXXXXX")" || exit 2
    curl -sf "http://127.0.0.1:${PORT}/v1/stats" -o "$stats_tmp" || {
        rm -f "$stats_tmp"
        echo "[gate] failed to capture /v1/stats" >&2
        exit 3
    }
    expected_sha="${EXPECTED_PRODUCT_SHA256#sha256:}"
    jq -e --arg sha "$expected_sha" --arg kernel "$EXPECTED_KERNEL_BUNDLE_ID" '
        ($sha | test("^[0-9a-f]{64}$")) and
        (.build_identity.product_binary_sha256 | type == "string") and
        ((.build_identity.product_binary_sha256 | sub("^sha256:"; "")) == $sha) and
        .build_identity.kernel_bundle_id == $kernel
    ' "$stats_tmp" >/dev/null || {
        rm -f "$stats_tmp"
        echo "[gate] runtime build identity mismatch" >&2
        exit 3
    }
    mv "$stats_tmp" "$STATS_OUT"
fi

PORT="$PORT" python3 "$ROOT/scripts/needle_gate.py" "$LENGTHS" "$RUNS" 0.0 2>&1 | tee "$OUT"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "[gate] needle gate failed with status $status"
    exit "$status"
fi
validate_summary "$OUT" "${BASELINE_LOG:-}"
echo "[gate] $LABEL done -> $OUT"
