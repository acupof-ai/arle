#!/usr/bin/env bash
# Long-context OPD writeback across a context-parallel group.
#
# Every knob below is either measured on H20 (see the table) or a default this
# script deliberately does NOT override. The three that actually decide whether
# a run fits and finishes are CP_SIZE, CP_DEVICES, and MAX_UPDATE_SEQ.
#
#   SEQ=262144 CP_SIZE=4 CP_DEVICES=0,1,2,3 scripts/train_longctx_cp.sh
#
# Measured ladder (ThinkingCap-Qwen3.6-27B, synthetic writeback, H20 96 GB):
#
#   seq      cp  GPUs  loss      forward  backward  peak
#   4096      2    2   9.857565        -     9.0 s      -
#   16384     2    2  11.229959        -    35.0 s  44.5 GB
#   131072    2    2   3.034898    125 s   391.0 s  64.6 GB
#   262144    2    2   1.561557    280 s  1083.0 s  85.7 GB
#   262144    4    4   1.560897    154 s   539.0 s  65.1 GB
set -euo pipefail

ROOT="${ARLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

# Global sequence length of the synthetic writeback trajectory. The sequence is
# zigzag-sharded across the CP group: rank r owns global chunks r and 2N-1-r.
SEQ="${SEQ:-262144}"

# Context-parallel group size, one process per rank. Single-card peaks past one
# 96 GB card at ~seq 49152, so anything long needs cp>1. 262144 fits on 2.
CP_SIZE="${CP_SIZE:-4}"
CP_DEVICES="${CP_DEVICES:-0,1,2,3}"

# Records longer than this are SKIPPED. The 23000-token default silently drops
# every long sample, so it has to clear SEQ or the run trains on nothing.
MAX_UPDATE_SEQ="${MAX_UPDATE_SEQ:-$SEQ}"

STUDENT_MODEL="${STUDENT_MODEL:-/data00/ThinkingCap-Qwen3.6-27B-FP8}"
REPLAY_RECORDS="${REPLAY_RECORDS:-}"
LORA_RANK="${LORA_RANK:-8}"
LOG="${LOG:-/tmp/arle-longctx-cp${CP_SIZE}-${SEQ}.log}"

# Left at their shipped defaults on purpose:
#   --opd-seq-chunk 0        derive the recompute chunk from this rank's own
#                            sequence length. Measured: cp=4 692.9 -> 618.4 s,
#                            cp=2 1362.8 -> 1163.4 s, peak VRAM unchanged. A
#                            fixed value is strictly slower.
#   --gradient-checkpointing true    off = OOM.
#   --cuda-mempool-retain true       keeps the async pool across syncs.
#   --checkpoint-reload-device true  recompute forward takes the device path.
#   --gdr-chunkwise-prefill true     FlashQLA chunkwise GDN prefill.
#
# Deliberately NOT enabled:
#   --fp8-native-gemm    -5.5% step time, but loss moves 0.23% (6-10x the
#                        matched-A/B envelope) and its own forward is 2.7%
#                        slower; cause unknown. Opt-in only.
#   --tape-precision bf16  inert under gradient checkpointing (activations are
#                        not resident), and VRAM is no longer the constraint.
args=(
  train agent-opd
  --student-model "$STUDENT_MODEL"
  --synthetic-writeback-seq "$SEQ"
  --cp-size "$CP_SIZE"
  --cp-devices "$CP_DEVICES"
  --max-update-seq "$MAX_UPDATE_SEQ"
  --lora-rank "$LORA_RANK"
)
[ -n "$REPLAY_RECORDS" ] && args+=(--replay-records "$REPLAY_RECORDS")

echo "seq=$SEQ cp=$CP_SIZE devices=$CP_DEVICES max_update_seq=$MAX_UPDATE_SEQ -> $LOG"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  ./target/release/arle "${args[@]}" "$@" 2>&1 | tee "$LOG"
