#!/usr/bin/env bash
# End-to-end agentic-OPD capability curve on the 27B student.
#
#   scripts/agent_opd_curve.sh <label>
#
# gen corpus (+self-check) -> baseline-envelope repeats -> train+held-out-eval
# -> curve PNG/JSON. One H20 GPU. Plan:
# docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md
#
# Models auto-download from HF if not local (set HF_ENDPOINT=https://hf-mirror.com
# for the mirror).
#
# Default = ALL FEATURES ON (canonical config): grpo on-policy at
# --rollout-temperature 1.0 (#167 fixed 2026-07-21 — the final-norm w-1 corruption
# that scrambled temp>0 is gone), --samples-per-prompt 4 (grpo needs G>=4 for group
# variance), --sync every-group (faithful pi_behavior), spec-decode via TC's own
# MTP head (--mtp-draft-tokens, aligned to the student — see SPEC below), eval
# --eval-concurrency 8. Fall back to the cheap SFT-on-wins baseline with
# UPDATE_STRATEGY=rejection-ce ROLLOUT_TEMPERATURE=0.0 SAMPLES=2.
# Optional env (full-run defaults; SMOKE=1 = 2-round sizing run):
#   ARLE_BIN=target/release/arle  OUT_ROOT=runs  GPU=0  MODEL_CACHE=models
#   STUDENT_MODEL=<dir>           override student (else fetch STUDENT_MODEL_HF_ID)
#   STUDENT_MODEL_HF_ID=bottlecapai/ThinkingCap-Qwen3.6-27B-FP8
#   UPDATE_STRATEGY=grpo          {grpo,rejection-ce,dapo,dr-grpo,gspo,cispo,...}
#   ROLLOUT_TEMPERATURE=1.0       grpo needs >0; rejection-ce uses 0.0 (greedy)
#   SPEC=mtp                      spec-decode via TC's built-in MTP head (default,
#                                 aligned, no download). dflash=external draft
#                                 (needs C4 retrain vs TC). off=disable.
#   MTP_DRAFT_TOKENS=3            MTP draft depth (SPEC=mtp)
#   DSPARK_DRAFT_HF_ID=z-lab/Qwen3.6-27B-DFlash   DSPARK_CONF_THRESHOLD=0.0  (SPEC=dflash)
#   ROUNDS=16 SAMPLES=4 EVAL_EVERY=2 EVAL_N=24 EVAL_CONCURRENCY=8
#   TASK_LIMIT=12 WRITEBACK_CAP=8 BASE_REPEATS=2 DIFFICULTY=easy SEED=0
set -euo pipefail

LABEL=${1:?usage: agent_opd_curve.sh <label>}
ARLE_BIN=${ARLE_BIN:-target/release/arle}
OUT=${OUT_ROOT:-runs}/agent-opd-"$LABEL"
GPU=${GPU:-0}

# Models: use a local dir if given, else auto-fetch from HF (honors HF_ENDPOINT
# mirror, e.g. https://hf-mirror.com). Only the student auto-downloads by default
# (spec-decode uses the student's own MTP head); SPEC=dflash also fetches the draft.
MODEL_CACHE=${MODEL_CACHE:-models}
STUDENT_MODEL_HF_ID=${STUDENT_MODEL_HF_ID:-bottlecapai/ThinkingCap-Qwen3.6-27B-FP8}
DSPARK_DRAFT_HF_ID=${DSPARK_DRAFT_HF_ID:-z-lab/Qwen3.6-27B-DFlash}

ensure_hf_model() {  # <hf_id> -> echoes local dir, downloads if absent
    local hf_id=$1 dst="$MODEL_CACHE/${1##*/}"
    if [[ ! -f $dst/config.json ]]; then
        echo "[curve] fetching $hf_id -> $dst (HF_ENDPOINT=${HF_ENDPOINT:-huggingface.co})" >&2
        huggingface-cli download "$hf_id" --local-dir "$dst" >&2 \
            || { echo "download failed: $hf_id" >&2; return 1; }
    fi
    echo "$dst"
}

command -v huggingface-cli >/dev/null || { echo "huggingface-cli missing (pip install huggingface_hub)" >&2; exit 1; }
STUDENT_MODEL=${STUDENT_MODEL:-$(ensure_hf_model "$STUDENT_MODEL_HF_ID")}

if [[ ${SMOKE:-0} == 1 ]]; then
    ROUNDS=${ROUNDS:-2} SAMPLES=${SAMPLES:-4} EVAL_EVERY=${EVAL_EVERY:-1}
    EVAL_N=${EVAL_N:-8} TASK_LIMIT=${TASK_LIMIT:-4} BASE_REPEATS=${BASE_REPEATS:-0}
else
    ROUNDS=${ROUNDS:-16} SAMPLES=${SAMPLES:-4} EVAL_EVERY=${EVAL_EVERY:-2}
    EVAL_N=${EVAL_N:-24} TASK_LIMIT=${TASK_LIMIT:-12} BASE_REPEATS=${BASE_REPEATS:-2}
fi
# All-features-on canonical default: grpo on-policy at temp=1.0 (#167 fixed).
UPDATE_STRATEGY=${UPDATE_STRATEGY:-grpo}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-8}
WRITEBACK_CAP=${WRITEBACK_CAP:-8} DIFFICULTY=${DIFFICULTY:-easy} SEED=${SEED:-0}
# Spec-decode default = TC's OWN MTP head (--mtp-draft-tokens): the mtp.* tensors
# ship inside ThinkingCap (mtp_num_hidden_layers=1), aligned to the student by
# construction -> no external draft, no download, NO weight adjustment. verify is
# distribution-preserving so results are unchanged either way; this is purely the
# aligned-acceptance choice.
#   SPEC=dflash  external z-lab/Qwen3.6-27B-DFlash draft. It is trained for BASE
#                Qwen3.6-27B; on a TC student its acceptance drops (shifted
#                distribution) and it needs retraining against TC (DSpark C4
#                draft-head harness) before it beats the MTP head. Opt in only
#                after that retrain.
#   SPEC=off     no spec-decode.
SPEC=${SPEC:-mtp}
MTP_DRAFT_TOKENS=${MTP_DRAFT_TOKENS:-3}
DSPARK_CONF_THRESHOLD=${DSPARK_CONF_THRESHOLD:-0.0}
DSPARK_DRAFT_MODEL=""
if [[ $SPEC == dflash ]]; then
    DSPARK_DRAFT_MODEL=${DSPARK_DRAFT_MODEL:-$(ensure_hf_model "$DSPARK_DRAFT_HF_ID")}
    echo "[curve] WARNING: DFlash draft is trained for BASE Qwen3.6-27B; on a TC" \
         "student it needs C4 retraining or acceptance will be low." >&2
fi

command -v python3 >/dev/null || { echo "python3 missing" >&2; exit 1; }
python3 -m pytest --version >/dev/null 2>&1 || { echo "pytest missing (scoring needs it)" >&2; exit 1; }
[[ -x $ARLE_BIN ]] || { echo "arle binary missing at $ARLE_BIN" >&2; exit 1; }

mkdir -p "$OUT"
echo "[curve] out=$OUT strategy=$UPDATE_STRATEGY temp=$ROLLOUT_TEMPERATURE spec=$SPEC rounds=$ROUNDS samples=$SAMPLES tasks=$TASK_LIMIT eval_n=$EVAL_N eval_conc=$EVAL_CONCURRENCY gpu=$GPU"

# 1. Corpus (deterministic; self-check = base-FAILS / gold-PASSES gate).
python3 scripts/gen_agent_opd_tasks.py \
    --out "$OUT/corpus" --seed "$SEED" --difficulty "$DIFFICULTY" --self-check

train_args=(
    train agent-opd
    --student-model "$STUDENT_MODEL"
    --dataset "$OUT/corpus/tasks_train.jsonl"
    --staged-root "$OUT/corpus/staged"
    --work-root "$OUT/work"
    --task-limit "$TASK_LIMIT"
    --eval-dataset "$OUT/corpus/tasks_eval.jsonl"
    --eval-n "$EVAL_N"
    --eval-concurrency "$EVAL_CONCURRENCY"
    --update-strategy "$UPDATE_STRATEGY"
    --rollout-temperature "$ROLLOUT_TEMPERATURE"
    --samples-per-prompt "$SAMPLES"
    --sync every-group
    --test-timeout-secs 60
    --writeback-cap "$WRITEBACK_CAP"
    --lora-rank 16
    --lora-alpha 32
    --lora-target-set attention-qv
    --save-lora-adapters "$OUT/adapters"
    --save-every 0
)
case $SPEC in
    mtp)
        train_args+=(--mtp-draft-tokens "$MTP_DRAFT_TOKENS")
        echo "[curve] spec-decode: MTP head (TC built-in, aligned) draft_tokens=$MTP_DRAFT_TOKENS" ;;
    dflash)
        train_args+=(--dspark-draft-model "$DSPARK_DRAFT_MODEL"
                     --dspark-conf-threshold "$DSPARK_CONF_THRESHOLD")
        echo "[curve] spec-decode: DFlash draft=$DSPARK_DRAFT_MODEL conf=$DSPARK_CONF_THRESHOLD" ;;
    off)
        echo "[curve] spec-decode: OFF" ;;
    *)
        echo "unknown SPEC=$SPEC (want mtp|dflash|off)" >&2; exit 1 ;;
esac

# 2. Baseline non-determinism envelope: same-config eval-only repeats
#    (--rounds 0 runs just the round-0 baseline eval, trains nothing).
for i in $(seq 1 "$BASE_REPEATS"); do
    echo "[curve] baseline envelope repeat $i/$BASE_REPEATS"
    CUDA_VISIBLE_DEVICES=$GPU "$ARLE_BIN" "${train_args[@]}" \
        --rounds 0 --eval-every 0 --eval-out-dir "$OUT/base_rep$i" \
        2>&1 | tee "$OUT/base_rep$i.log"
done

# 3. The training run (baseline eval + per-round held-out evals inside).
CUDA_VISIBLE_DEVICES=$GPU "$ARLE_BIN" "${train_args[@]}" \
    --rounds "$ROUNDS" --eval-every "$EVAL_EVERY" --eval-out-dir "$OUT/eval" \
    2>&1 | tee "$OUT/train.log"

# 4. Curve.
base_extra=()
for d in "$OUT"/base_rep*/; do [[ -d $d ]] && base_extra+=("$d"); done
python3 scripts/plot_agent_opd_curve.py \
    --eval-dir "$OUT/eval" --train-log "$OUT/train.log" \
    ${base_extra:+--baseline-extra "${base_extra[@]}"} \
    --out "$OUT/curve.png"

echo "[curve] done: $OUT/curve.png $OUT/curve.json"
