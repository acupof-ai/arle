#!/bin/bash
# Online iterative cc-as-harness OPD. Per round: collect cc-harness rollouts with
# the CURRENT adapter (cc_run.sh: serve-restart -> cc -> cc-convert -> records),
# then train ONE round on those records starting from the current adapter and save
# the next. Per-round serve RESTART keeps the RadixCache fresh across the adapter
# epoch (sidesteps #92 stale-prefix; the token-keyed cache has no version).
# Plan: docs/plans/2026-07-13-cc-as-harness-online-opd.md
set -euo pipefail

# ===== CONFIG (override via env) =====
ROUNDS=${ROUNDS:-8}
GPU=${GPU:-0}
PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-/host/Qwen3.6-27B-FP8}
DATASET=${DATASET:-/host/opd-sweetspot3/split/tasks_train.jsonl}
STAGED=${STAGED:-/host/opd-sweetspot3/staged-sweetspot3/staged}
EVAL_DATASET=${EVAL_DATASET:-/host/opd-sweetspot3/split/tasks_eval.jsonl}
EVAL_N=${EVAL_N:-33}
EVAL_EVERY=${EVAL_EVERY:-2}
PYTHONPATH_TASK=${PYTHONPATH_TASK:-lib:test}
UPDATE_STRATEGY=${UPDATE_STRATEGY:-rejection-ce}   # rejection-ce | sao-dis | sao-value
# rejection-CE keeps the 1-attempt default (byte-identical); SAO needs >=2 for within-task reward variance.
SAMPLES=${SAMPLES:-$([ "$UPDATE_STRATEGY" = rejection-ce ] && echo 1 || echo 2)}
SAO_GAMMA=${SAO_GAMMA:-1.0}; SAO_LAMBDA=${SAO_LAMBDA:-0.95}; VALUE_LR=${VALUE_LR:-0.0003}
CC_TIMEOUT=${CC_TIMEOUT:-600}; DECODE_GRAPH=${DECODE_GRAPH:-1}   # rollout speed: per-task cap + decode graph
FROZEN_PROMPT_KV=${FROZEN_PROMPT_KV:-1}   # SAO writeback skips the shared 17K-prompt forward (needs the port)
LR=${LR:-0.00001}
LORA_RANK=${LORA_RANK:-16}; LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET_SET=${LORA_TARGET_SET:-attention-qv}
ARLE=${ARLE:-/host/arle-build/target/release/arle}
RUN=${RUN:-/host/cc_opd/$(date -u +%Y%m%d_%H%M%S)}
INIT_ADAPTER=${INIT_ADAPTER:-}          # empty = start from base
# ===== end CONFIG =====

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$RUN"
adapter="$INIT_ADAPTER"
echo "[cc-opd] run=$RUN rounds=$ROUNDS gpu=$GPU model=$MODEL_PATH init_adapter=${adapter:-<base>}"

for round in $(seq 0 $((ROUNDS - 1))); do
  rdir="$RUN/round-$round"; mkdir -p "$rdir"
  echo "[cc-opd] === round $round === adapter=${adapter:-<base>}"

  # 1. Collect cc-harness rollouts with the current adapter (fresh serve each round).
  GPU="$GPU" PORT="$PORT" MODEL_PATH="$MODEL_PATH" DATASET="$DATASET" STAGED="$STAGED" \
    PYTHONPATH_TASK="$PYTHONPATH_TASK" SAMPLES="$SAMPLES" LORA="$adapter" \
    CC_TIMEOUT="$CC_TIMEOUT" DECODE_GRAPH="$DECODE_GRAPH" \
    OUT_DIR="$rdir/collect" ARLE="$ARLE" \
    bash "$ROOT/scripts/cc_run.sh"

  records="$rdir/collect/records.jsonl"
  if [ ! -s "$records" ]; then
    echo "[cc-opd] round $round: no records (no cc rollouts) — stopping."
    break
  fi

  # 2. Train ONE round on the cc records, chaining the adapter forward. sao-value
  # also passes the critic GAE/lr knobs; rejection-ce/sao-dis ignore them.
  sao_args=()
  if [ "$UPDATE_STRATEGY" = "sao-value" ]; then
    sao_args=(--sao-gamma "$SAO_GAMMA" --sao-lambda "$SAO_LAMBDA" --value-lr "$VALUE_LR")
  fi
  next_adapter="$rdir/adapter"
  CUDA_VISIBLE_DEVICES="$GPU" "$ARLE" train agent-opd \
    --student-model "$MODEL_PATH" --replay-records "$records" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-target-set "$LORA_TARGET_SET" \
    --lr "$LR" --update-strategy "$UPDATE_STRATEGY" ${sao_args[@]+"${sao_args[@]}"} \
    ${FROZEN_PROMPT_KV:+--writeback-frozen-prompt-kv} \
    ${adapter:+--lora-adapters "$adapter"} \
    --save-lora-adapters "$next_adapter" 2>&1 | tee "$rdir/train.log"
  # --save-lora-adapters writes <dir>/adapters_replay/{adapter_model.safetensors,..}
  # + a <dir>/latest symlink; the serve + resume loaders read the adapter dir
  # directly (no symlink-follow / descend), so chain the real subdir.
  adapter="$next_adapter/adapters_replay"

  # 3. Periodic held-out eval — ALSO cc-harness (train==eval==serve under cc).
  # NOT `agent-opd --rounds 0`: that eval uses the inference engine, which loads
  # the base from student_dir un-synced, so it would score the BASE, not the
  # resumed adapter. cc_run.sh over the eval set serves+scores the adapter directly.
  if [ $(((round + 1) % EVAL_EVERY)) -eq 0 ]; then
    GPU="$GPU" PORT="$PORT" MODEL_PATH="$MODEL_PATH" DATASET="$EVAL_DATASET" STAGED="$STAGED" \
      PYTHONPATH_TASK="$PYTHONPATH_TASK" LORA="$adapter" OUT_DIR="$rdir/eval" ARLE="$ARLE" \
      CC_TIMEOUT="$CC_TIMEOUT" DECODE_GRAPH="$DECODE_GRAPH" SAMPLES=1 \
      bash "$ROOT/scripts/cc_run.sh" || true
    echo "[cc-opd] round $round held-out eval -> $rdir/eval/results.jsonl"
  fi
done
echo "[cc-opd] done. final adapter=$adapter run=$RUN"
