#!/usr/bin/env bash
# Wait for the v8 train (rollout=64 × 200 steps) to finish, then run
# multi-seed paired eval against the existing base 5-seed at
# runs/2026-05-28-base-multiseed-eval/capability_seeds.
#
# Designed to run as a long-lived background companion to the v8 train:
# poll for the TRAIN_EXIT_STATUS marker, then auto-eval step_100 and
# step_200 at matched seeds {0..4}, then paired-vs-base for both.
#
# Usage:
#   setsid bash scripts/eval_v8_after_train.sh > runs/2026-05-28-rollout64-200steps-v8/auto_eval.log 2>&1 < /dev/null &
#   disown
#
# Exits non-zero if the train exited non-zero (no eval done) or if a
# ckpt is missing (eval skipped). Otherwise exits 0 once both evals +
# both paired analyses have printed.
set -uo pipefail

RUN_DIR="${RUN_DIR:-runs/2026-05-28-rollout64-200steps-v8}"
BASE_OUT="${BASE_OUT:-runs/2026-05-28-base-multiseed-eval/capability_seeds}"
# SEEDS via env: SEEDS="0 1 2 3 4" bash scripts/eval_v8_after_train.sh
# Default matches the v4 + base 5-seed evals (matched seeds → paired analysis).
read -ra SEEDS <<< "${SEEDS:-0 1 2 3 4}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "error: $RUN_DIR not found" >&2
    exit 2
fi
if [[ ! -d "$BASE_OUT" ]]; then
    echo "error: base 5-seed not found at $BASE_OUT" >&2
    exit 2
fi

printf '[auto-eval] waiting for v8 train completion (TRAIN_EXIT_STATUS marker in %s/run.txt)\n' "$RUN_DIR"
while ! grep -q '^TRAIN_EXIT_STATUS=' "$RUN_DIR/run.txt" 2>/dev/null; do
    sleep 60
done
STATUS=$(grep '^TRAIN_EXIT_STATUS=' "$RUN_DIR/run.txt" | tail -1 | cut -d= -f2)
printf '[auto-eval] train exited with status=%s\n' "$STATUS"
if [[ "$STATUS" != "0" ]]; then
    echo "[auto-eval] train failed; skipping eval"
    exit 1
fi

eval_ckpt() {
    local ckpt_label="$1"  # e.g. step_000100
    local out_label="$2"   # e.g. capability_seeds_step100
    local ckpt="$RUN_DIR/student/$ckpt_label"
    local out="$RUN_DIR/$out_label"
    if [[ ! -d "$ckpt" ]]; then
        echo "[auto-eval] ckpt $ckpt missing, skipping"
        return 1
    fi
    printf '\n[auto-eval] ============ %s ============\n' "$ckpt_label"
    bash scripts/eval_opd_ckpt_seeds.sh "$ckpt" "$out" "${SEEDS[@]}"
    printf '\n[auto-eval] PAIRED %s vs base\n' "$ckpt_label"
    .venv/bin/python scripts/analyze_multi_seed.py "$out" --paired-vs "$BASE_OUT" || true
}

eval_ckpt "step_000100" "capability_seeds_step100"
eval_ckpt "step_000200" "capability_seeds_step200"

printf '\n[auto-eval] done\n'
