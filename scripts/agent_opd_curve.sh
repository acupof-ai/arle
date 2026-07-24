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
# Defaults come from the 2026-07-21 SOTA review
# (docs/research/2026-07-21-rl-algo-infra-deepresearch.md). Each knob is explained
# where it is set below; override any via env. SMOKE=1 = quick 2-round sizing run.
# Optional env (full-run defaults):
#   ARLE_BIN=target/release/arle  OUT_ROOT=runs  GPU=0  MODEL_CACHE=models
#   WORK_ROOT=/tmp/agent-opd     sandbox root — MUST be outside any repo tree
#                                 (an ancestor CLAUDE.md becomes ~31K/request CC preamble)
#   SERVE_PORT=8000               in-process serve port (bump when another lane holds 8000)
#   STUDENT_MODEL=<dir>           override student (else fetch STUDENT_MODEL_HF_ID)
#   STUDENT_MODEL_HF_ID=bottlecapai/ThinkingCap-Qwen3.6-27B-FP8
#   UPDATE_STRATEGY=dapo          policy-update rule {dapo,gspo,dr-grpo,grpo,rejection-ce,...}
#   STALENESS=1                   1 = train on a batch while the next generates
#                                 (dapo/dr-grpo/gspo only); 0 = wait for each batch
#   TASK_SELECTION=true           skip prompts that always pass or always fail
#   COMFORT_BAND=1                profile 1 round, then train only the "band": drop
#                                 always-pass/always-fail AND >CB_MAX_SEQ tasks the
#                                 writeback skips (0=off; auto-off under SMOKE)
#   CB_MAX_SEQ=22000 CB_PASS_LO=0.2 CB_PASS_HI=0.8 CB_MIN_TESTS=2   comfort-band thresholds
#   REWARD_SHAPE=dense            dense|anchored|binary (dense/anchored = more reward-bearing groups)
#   REPLAY_REUSE=0                reuse N past variance-bearing groups per fresh one (opt-in, needs dapo)
#   ROLLOUT_TEMPERATURE=1.0       >0 samples (dapo/grpo); 0.0 is greedy (rejection-ce)
#   SPEC=mtp                      spec-decode via TC's built-in MTP head (default,
#                                 aligned, no download). dflash=external draft
#                                 (needs C4 retrain vs TC). off=disable.
#   MTP_DRAFT_TOKENS=3            MTP draft depth (SPEC=mtp)
#   DSPARK_DRAFT_HF_ID=z-lab/Qwen3.6-27B-DFlash   DSPARK_CONF_THRESHOLD=0.0  (SPEC=dflash)
#   ROUNDS=16 SAMPLES=8 EVAL_EVERY=2 EVAL_N=24 EVAL_CONCURRENCY=8
#   TASK_LIMIT=12 WRITEBACK_CAP=8 BASE_REPEATS=2 DIFFICULTY=easy SEED=0
set -euo pipefail

LABEL=${1:?usage: agent_opd_curve.sh <label>}
ARLE_BIN=${ARLE_BIN:-target/release/arle}
OUT=${OUT_ROOT:-runs}/agent-opd-"$LABEL"
WORK=${WORK_ROOT:-/tmp/agent-opd}/"$LABEL"
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
        hf download "$hf_id" --local-dir "$dst" >&2 \
            || { echo "download failed: $hf_id" >&2; return 1; }
    fi
    echo "$dst"
}

command -v hf >/dev/null || { echo "hf CLI missing (pip install -U huggingface_hub; huggingface-cli download is removed)" >&2; exit 1; }
STUDENT_MODEL=${STUDENT_MODEL:-$(ensure_hf_model "$STUDENT_MODEL_HF_ID")}

if [[ ${SMOKE:-0} == 1 ]]; then
    ROUNDS=${ROUNDS:-2} SAMPLES=${SAMPLES:-8} EVAL_EVERY=${EVAL_EVERY:-1}
    EVAL_N=${EVAL_N:-8} TASK_LIMIT=${TASK_LIMIT:-4} BASE_REPEATS=${BASE_REPEATS:-0}
else
    ROUNDS=${ROUNDS:-16} SAMPLES=${SAMPLES:-8} EVAL_EVERY=${EVAL_EVERY:-2}
    EVAL_N=${EVAL_N:-24} TASK_LIMIT=${TASK_LIMIT:-12} BASE_REPEATS=${BASE_REPEATS:-2}
fi
# dapo: the update rule most recent coding-RL work uses; it also lets STALENESS=1
# overlap generation with training (rejection-ce cannot). G=8, not 4: with a
# pass/fail reward, 4 samples per prompt too often come out all-same → no signal.
# For a quick supervised-style baseline instead, set UPDATE_STRATEGY=rejection-ce
# ROLLOUT_TEMPERATURE=0.0 STALENESS=0 SAMPLES=2.
UPDATE_STRATEGY=${UPDATE_STRATEGY:-dapo}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
STALENESS=${STALENESS:-1}
TASK_SELECTION=${TASK_SELECTION:-true}
# Reward shape: dense (default, partial pass-fraction → more reward-bearing groups) |
# anchored (keeps sub-test variance in all-failing groups) | binary (collapses to 0/1, lowers ratio).
REWARD_SHAPE=${REWARD_SHAPE:-dense}
# Experience replay: reuse N past variance-bearing groups per fresh group (opt-in; needs a
# ratio-weighted strategy — dapo default satisfies it). 0 = off, amortizes spent rollout.
REPLAY_REUSE=${REPLAY_REUSE:-0}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-8}
WRITEBACK_CAP=${WRITEBACK_CAP:-8} DIFFICULTY=${DIFFICULTY:-easy} SEED=${SEED:-0}
# Faster generation via speculative decoding — it never changes the output, only
# speed. Default uses the student's own MTP head, which ships inside ThinkingCap,
# so it matches the model and needs no extra download or setup.
#   SPEC=dflash  a separate draft model. The public one is trained for the BASE
#                model, so it guesses worse for our fine-tuned student and helps
#                little until retrained. Prefer mtp.
#   SPEC=off     plain decoding.
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

# Pre-staged corpus: CORPUS_ROOT points at a dir with train/eval jsonl + staged/
# already built (data/opd-corpora/staged-sweetspot3), skipping synthetic generation.
# Such roots name their splits train.jsonl/eval.jsonl; synthetic uses tasks_*.jsonl.
CORPUS_ROOT=${CORPUS_ROOT:-}
if [[ -n $CORPUS_ROOT ]]; then
    TRAIN_JSONL=${TRAIN_JSONL:-train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-eval.jsonl}
else
    TRAIN_JSONL=${TRAIN_JSONL:-tasks_train.jsonl}
    EVAL_JSONL=${EVAL_JSONL:-tasks_eval.jsonl}
fi

mkdir -p "$OUT"
echo "[curve] out=$OUT strategy=$UPDATE_STRATEGY staleness=$STALENESS temp=$ROLLOUT_TEMPERATURE spec=$SPEC rounds=$ROUNDS samples=$SAMPLES tasks=$TASK_LIMIT eval_n=$EVAL_N eval_conc=$EVAL_CONCURRENCY gpu=$GPU"

# 1. Corpus (deterministic; self-check = base-FAILS / gold-PASSES gate).
if [[ -n $CORPUS_ROOT ]]; then
    echo "[curve] using pre-staged corpus $CORPUS_ROOT (splits $TRAIN_JSONL/$EVAL_JSONL), skipping generation"
else
    python3 scripts/gen_agent_opd_tasks.py \
        --out "$OUT/corpus" --seed "$SEED" --difficulty "$DIFFICULTY" --self-check
fi

# Spec-decode args, shared by the comfort-band profile round and the training run.
spec_args=()
case $SPEC in
    mtp)    spec_args+=(--mtp-draft-tokens "$MTP_DRAFT_TOKENS")
            echo "[curve] spec-decode: MTP head (TC built-in, aligned) draft_tokens=$MTP_DRAFT_TOKENS" ;;
    dflash) spec_args+=(--dspark-draft-model "$DSPARK_DRAFT_MODEL" --dspark-conf-threshold "$DSPARK_CONF_THRESHOLD")
            echo "[curve] spec-decode: DFlash draft=$DSPARK_DRAFT_MODEL conf=$DSPARK_CONF_THRESHOLD" ;;
    off)    echo "[curve] spec-decode: OFF" ;;
    *)      echo "unknown SPEC=$SPEC (want mtp|dflash|off)" >&2; exit 1 ;;
esac

# Args shared by the comfort-band profile round and the training run. Per-invocation
# flags (dataset/work-root/staleness/task-selection/eval/rounds) are appended below.
common_args=(
    train agent-opd
    --student-model "$STUDENT_MODEL"
    --task-limit "$TASK_LIMIT"
    --update-strategy "$UPDATE_STRATEGY"
    --reward-shape "$REWARD_SHAPE"
    --rollout-temperature "$ROLLOUT_TEMPERATURE"
    --samples-per-prompt "$SAMPLES"
    --serve-port "${SERVE_PORT:-8000}"
    --sync every-group
    --test-timeout-secs 60
    --writeback-cap "$WRITEBACK_CAP"
    --lora-rank 16
    --lora-alpha 32
    --lora-target-set attention-qv
    --save-every 0
    "${spec_args[@]}"
)

# 1b. Comfort-band corpus filter (default on; COMFORT_BAND=0 or SMOKE=1 to skip).
# One unbiased profile round (task-selection off) → drop always-pass / always-fail
# tasks AND >CB_MAX_SEQ trajectories the writeback would skip → train only the
# trainable band. Closes the length-waste gap the runtime P5 selector cannot: a
# variance-bearing 30K task passes P5 but its 40%-of-round rollout is thrown away
# every round (errors/2026-07-22). See docs/research/2026-07-23-online-rl-acceleration.md.
CORPUS="${CORPUS_ROOT:-$OUT/corpus}"
if [[ ${COMFORT_BAND:-1} == 1 && ${SMOKE:-0} != 1 ]]; then
    echo "[curve] comfort-band: profiling $CORPUS (1 round, task-selection off) → filter"
    CUDA_VISIBLE_DEVICES=$GPU "$ARLE_BIN" "${common_args[@]}" \
        --dataset "$CORPUS/$TRAIN_JSONL" --staged-root "$CORPUS/staged" \
        --work-root "$WORK/cb_work" --staleness 0 --task-selection false \
        --rounds 1 --eval-every 0 --eval-out-dir "$OUT/cb_profile" \
        2>&1 | tee "$OUT/cb_profile.log"
    if python3 scripts/comfort_band.py \
            --metrics "$OUT/cb_profile/metrics.jsonl" --corpus "$CORPUS" --out "$OUT/corpus-band" \
            --train-name "$TRAIN_JSONL" --eval-name "$EVAL_JSONL" \
            --max-seq "${CB_MAX_SEQ:-22000}" --pass-lo "${CB_PASS_LO:-0.2}" --pass-hi "${CB_PASS_HI:-0.8}" \
            --min-tests "${CB_MIN_TESTS:-2}"; then
        CORPUS="$OUT/corpus-band"
        echo "[curve] comfort-band: training on filtered corpus $CORPUS"
    else
        echo "[curve] comfort-band: empty/failed band (rc=$?) — falling back to full corpus $CORPUS" >&2
    fi
fi

train_args=(
    "${common_args[@]}"
    --dataset "$CORPUS/$TRAIN_JSONL"
    --staged-root "$CORPUS/staged"
    --work-root "$WORK/work"
    --eval-dataset "$CORPUS/$EVAL_JSONL"
    --eval-n "$EVAL_N"
    --eval-concurrency "$EVAL_CONCURRENCY"
    --staleness "$STALENESS"
    --task-selection "$TASK_SELECTION"
    --replay-reuse "$REPLAY_REUSE"
    --save-lora-adapters "$OUT/adapters"
)

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

# 4. Curve (plot script optional — training results stand without it).
if [[ -f scripts/plot_agent_opd_curve.py ]]; then
    base_extra=()
    for d in "$OUT"/base_rep*/; do [[ -d $d ]] && base_extra+=("$d"); done
    python3 scripts/plot_agent_opd_curve.py \
        --eval-dir "$OUT/eval" --train-log "$OUT/train.log" \
        ${base_extra:+--baseline-extra "${base_extra[@]}"} \
        --out "$OUT/curve.png"
    echo "[curve] done: $OUT/curve.png $OUT/curve.json"
else
    echo "[curve] done (plot skipped: scripts/plot_agent_opd_curve.py missing): $OUT/eval"
fi
