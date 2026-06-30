#!/usr/bin/env bash
# DSv4-Flash resident scalar-vs-FlashMLA A/B launcher (8xH20, TP=8/EP=8).
#
# Build first:
#   cargo build --release -p infer-cuda --features cuda,nccl,deepep \
#       --example dsv4_resident_ab
#
# Run:
#   INFER_DSV4_MODEL_PATH=/data01/models/DeepSeek-V4-Flash \
#     scripts/dsv4_resident_ab.sh
#
# Key knobs:
#   DSV4_AB_BIN                    path to dsv4_resident_ab binary
#   WORLD_SIZE                     default 8
#   INFER_DSV4_AB_VARIANTS         default scalar,flashmla
#   INFER_DSV4_AB_MAX_NEW          default 128
#   INFER_DSV4_AB_WARMUP_NEW       default 16
#   INFER_DSV4_AB_REPEAT           repeat variant list after one load, default 1
#   INFER_DSV4_AB_PROFILE_VARIANT  optional scalar|flashmla profiler window
#   INFER_DSV4_DUMP_TOPK_POSITIONS optional comma-separated sample positions
#                                  for rank-0 logits top-k diagnostics
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${DSV4_AB_BIN:-$ROOT/target/release/examples/dsv4_resident_ab}"
WORLD_SIZE="${WORLD_SIZE:-8}"

[[ -n "${INFER_DSV4_MODEL_PATH:-}" ]] || {
    echo "ERROR: set INFER_DSV4_MODEL_PATH to the DSv4 FP8 safetensors dir" >&2
    exit 1
}
[[ -x "$BIN" ]] || {
    echo "ERROR: $BIN not found/executable; build it first" >&2
    exit 1
}

DEVICES="$(seq -s, 0 $((WORLD_SIZE - 1)))"
WORK="$(mktemp -d -t dsv4-ab.XXXXXX)"
ID_FILE="$WORK/nccl_id.hex"
trap 'rm -rf "$WORK"' EXIT

echo "[launcher] resident A/B binary: $BIN" >&2
echo "[launcher] NCCL file-rendezvous at $ID_FILE" >&2

declare -a PIDS
declare -a LOGS
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    LOG="$WORK/rank_$r.log"
    LOGS[r]="$LOG"
    CUDA_VISIBLE_DEVICES="$r" \
    INFER_CUDA_DEVICE=0 \
    INFER_CUDA_DEVICES="$DEVICES" \
    INFER_TP_SIZE="$WORLD_SIZE" \
    INFER_TP_RANK="$r" \
    INFER_NCCL_ID_FILE="$ID_FILE" \
    ARLE_DSV4_FLASHMLA_DECODE_ALLOC=1 \
        "$BIN" >"$LOG" 2>&1 &
    PIDS[r]=$!
    echo "[launcher] spawned rank $r (pid ${PIDS[r]}, gpu $r)" >&2
done

FAIL=0
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    if ! wait "${PIDS[r]}"; then
        echo "[launcher] rank $r FAILED - log:" >&2
        cat "${LOGS[r]}" >&2
        FAIL=1
    fi
done
[[ $FAIL -eq 0 ]] || {
    echo "[launcher] one or more ranks failed" >&2
    exit 1
}

echo "===== rank 0 log =====" >&2
cat "${LOGS[0]}" >&2
echo "======================" >&2

grep -E '^ab_variant=' "${LOGS[0]}" || {
    echo "[launcher] no ab_variant= lines found in rank 0 log" >&2
    exit 1
}
