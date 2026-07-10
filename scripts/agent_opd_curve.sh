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
# for the mirror). DSpark spec decode is ON by default (draft auto-fetched).
# Optional env (full-run defaults; SMOKE=1 = 2-round sizing run):
#   ARLE_BIN=target/release/arle  OUT_ROOT=runs  GPU=0  MODEL_CACHE=models
#   STUDENT_MODEL=<dir>           override student (else fetch STUDENT_MODEL_HF_ID)
#   STUDENT_MODEL_HF_ID=Qwen/Qwen3.6-27B-FP8
#   DSPARK=                       auto: on iff SAMPLES==1 (serial); at SAMPLES>1
#                                 plain-batched wins ~2.4x. 1/0 to force.
#   DSPARK_DRAFT_HF_ID=z-lab/Qwen3.6-27B-DFlash   DSPARK_CONF_THRESHOLD=0.0
#   ROUNDS=16 SAMPLES=2 MAX_TURNS=8 MAX_TOKENS=768 EVAL_EVERY=2 EVAL_N=24
#   TASK_LIMIT=12 WRITEBACK_CAP=8 BASE_REPEATS=2 DIFFICULTY=easy SEED=0
set -euo pipefail

LABEL=${1:?usage: agent_opd_curve.sh <label>}
ARLE_BIN=${ARLE_BIN:-target/release/arle}
OUT=${OUT_ROOT:-runs}/agent-opd-"$LABEL"
GPU=${GPU:-0}

# Models: use a local dir if given, else auto-fetch from HF (honors HF_ENDPOINT
# mirror, e.g. https://hf-mirror.com). STUDENT + DSpark draft both auto-download.
MODEL_CACHE=${MODEL_CACHE:-models}
STUDENT_MODEL_HF_ID=${STUDENT_MODEL_HF_ID:-Qwen/Qwen3.6-27B-FP8}
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
    ROUNDS=${ROUNDS:-2} SAMPLES=${SAMPLES:-2} EVAL_EVERY=${EVAL_EVERY:-1}
    EVAL_N=${EVAL_N:-8} TASK_LIMIT=${TASK_LIMIT:-4} BASE_REPEATS=${BASE_REPEATS:-0}
else
    ROUNDS=${ROUNDS:-16} SAMPLES=${SAMPLES:-2} EVAL_EVERY=${EVAL_EVERY:-2}
    EVAL_N=${EVAL_N:-24} TASK_LIMIT=${TASK_LIMIT:-12} BASE_REPEATS=${BASE_REPEATS:-2}
fi
MAX_TURNS=${MAX_TURNS:-8} MAX_TOKENS=${MAX_TOKENS:-768}
WRITEBACK_CAP=${WRITEBACK_CAP:-8} DIFFICULTY=${DIFFICULTY:-easy} SEED=${SEED:-0}
# DSpark spec decode wins ONLY on serial c=1 decode (licensed 2026-07-10:
# ~1.9x single-stream). At SAMPLES>1 the sample group decodes CONCURRENTLY and
# plain-batched beats dspark 2.37x (spike 2026-07-10: per-row spec throws away
# the batch dim, aggregate flat ~40 vs plain 94 tok/s @ C=8). So default DSpark
# ON only when SAMPLES==1; DSPARK=1 forces it, DSPARK=0 disables.
DSPARK_CONF_THRESHOLD=${DSPARK_CONF_THRESHOLD:-0.0}
DSPARK_DEFAULT=$([[ $SAMPLES == 1 ]] && echo 1 || echo 0)
if [[ ${DSPARK:-$DSPARK_DEFAULT} == 1 ]]; then
    [[ $SAMPLES != 1 ]] && echo "[curve] WARN: DSpark forced ON with SAMPLES=$SAMPLES — plain-batched is ~2.4x faster at concurrency (see 2026-07-10-dspark-concurrency-derisk-kill)"
    DSPARK_DRAFT_MODEL=${DSPARK_DRAFT_MODEL:-$(ensure_hf_model "$DSPARK_DRAFT_HF_ID")}
else
    DSPARK_DRAFT_MODEL=""
fi

command -v python3 >/dev/null || { echo "python3 missing" >&2; exit 1; }
python3 -m pytest --version >/dev/null 2>&1 || { echo "pytest missing (scoring needs it)" >&2; exit 1; }
[[ -x $ARLE_BIN ]] || { echo "arle binary missing at $ARLE_BIN" >&2; exit 1; }

mkdir -p "$OUT"
echo "[curve] out=$OUT rounds=$ROUNDS samples=$SAMPLES tasks=$TASK_LIMIT eval_n=$EVAL_N gpu=$GPU"

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
    --samples-per-prompt "$SAMPLES"
    --max-turns "$MAX_TURNS"
    --max-tokens "$MAX_TOKENS"
    --bash-timeout-secs 30
    --test-timeout-secs 60
    --writeback-cap "$WRITEBACK_CAP"
    --rollout-temperature 1.0
    --rollout-seed "$SEED"
    --lora-rank 16
    --lora-alpha 32
    --lora-target-set attention-qv
    --save-lora-adapters "$OUT/adapters"
    --save-every 0
)
if [[ -n $DSPARK_DRAFT_MODEL ]]; then
    train_args+=(--dspark-draft-model "$DSPARK_DRAFT_MODEL"
                 --dspark-conf-threshold "$DSPARK_CONF_THRESHOLD")
    echo "[curve] DSpark spec decode ON: draft=$DSPARK_DRAFT_MODEL conf=$DSPARK_CONF_THRESHOLD"
fi

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
