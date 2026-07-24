#!/usr/bin/env bash
# Data-parallel comfort-band profile across N GPUs (#173 phase 1).
#
#   GPUS="0 2 3 4 5 6 7" CORPUS_ROOT=/host/opd-corpora/staged-sweetspot3 \
#   STUDENT_MODEL=/host/ThinkingCap-Qwen3.6-27B-FP8 OUT_ROOT=/host/runs \
#   scripts/agent_opd_profile_sharded.sh sweetspot3-profile
#
# The 32 profile tasks are independent, so we shard train.jsonl by stride across
# the free GPUs (stride keeps the repo-interleaved corpus balanced per shard),
# run one PROFILE_METRICS_ONLY leg per GPU concurrently, then merge every shard's
# metrics.jsonl and cut the band ONCE. arle is single-card single-process; this is
# process-level data parallelism — no Rust change, no cross-process data movement
# (each worker loads the static FP8 student from disk and serves on its own port).
set -euo pipefail

LABEL=${1:?usage: agent_opd_profile_sharded.sh <label>}
GPUS=${GPUS:?set GPUS="0 2 3 ..." (space-separated free GPU indices)}
CORPUS_ROOT=${CORPUS_ROOT:?set CORPUS_ROOT=<pre-staged corpus dir>}
STUDENT_MODEL=${STUDENT_MODEL:?set STUDENT_MODEL=<local 27B dir> (workers must not race HF)}
OUT_ROOT=${OUT_ROOT:-runs}
WORK_ROOT=${WORK_ROOT:-/tmp/agent-opd}
ARLE_BIN=${ARLE_BIN:-target/release/arle}
TASK_LIMIT=${TASK_LIMIT:-32}          # total tasks to profile across all shards
SAMPLES=${SAMPLES:-8}
SPEC=${SPEC:-off}
BASE_PORT=${BASE_PORT:-8000}          # shard i serves on BASE_PORT+i
TRAIN_JSONL=${TRAIN_JSONL:-train.jsonl}
EVAL_JSONL=${EVAL_JSONL:-eval.jsonl}

read -ra GPU_ARR <<<"$GPUS"
N=${#GPU_ARR[@]}
OUT=$OUT_ROOT/agent-opd-"$LABEL"
SHARDS=$OUT/shards
mkdir -p "$SHARDS"
[[ -x $ARLE_BIN ]] || { echo "arle binary missing at $ARLE_BIN" >&2; exit 1; }

echo "[sharded] $N GPUs ($GPUS), $TASK_LIMIT tasks, corpus=$CORPUS_ROOT out=$OUT"

# 1. Shard the head TASK_LIMIT rows of train.jsonl by stride into N corpus roots,
#    each a lightweight tree: its own train.jsonl + eval.jsonl + a staged/ of
#    symlinks back to the real trees (no 313 MB copy per shard).
python3 - "$CORPUS_ROOT" "$SHARDS" "$N" "$TASK_LIMIT" "$TRAIN_JSONL" "$EVAL_JSONL" <<'PY'
import json, os, sys
corpus, shards_dir, n, limit, train_name, eval_name = sys.argv[1:7]
n, limit = int(n), int(limit)
rows = [l for l in open(os.path.join(corpus, train_name)) if l.strip()][:limit]
staged_src = os.path.join(corpus, "staged")
eval_src = os.path.join(corpus, eval_name)
for i in range(n):
    d = os.path.join(shards_dir, f"shard{i}")
    st = os.path.join(d, "staged")
    os.makedirs(st, exist_ok=True)
    shard_rows = rows[i::n]  # stride keeps the repo-interleaved order balanced
    with open(os.path.join(d, train_name), "w") as f:
        f.writelines(shard_rows)
    for r in shard_rows:
        iid = json.loads(r)["instance_id"]
        link = os.path.join(st, iid)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(staged_src, iid)), link)
    # eval split is copied whole into every shard (unused in profile, but keeps
    # the corpus schema valid for the curve script's train_args resolution).
    if os.path.exists(eval_src) and not os.path.lexists(os.path.join(d, eval_name)):
        os.symlink(os.path.abspath(eval_src), os.path.join(d, eval_name))
    print(f"  shard{i}: {len(shard_rows)} tasks")
PY

# 2. One profile leg per GPU, concurrent. PROFILE_METRICS_ONLY stops each worker
#    right after it writes its shard metrics — no per-shard band cut.
pids=()
for i in "${!GPU_ARR[@]}"; do
    gpu=${GPU_ARR[$i]}
    port=$((BASE_PORT + i))
    log=$OUT/shard$i.log
    echo "[sharded] launch shard$i on GPU $gpu port $port -> $log"
    CORPUS_ROOT="$SHARDS/shard$i" STUDENT_MODEL="$STUDENT_MODEL" \
        ARLE_BIN="$ARLE_BIN" OUT_ROOT="$OUT/w$i" WORK_ROOT="$WORK_ROOT/$LABEL/w$i" \
        GPU="$gpu" SERVE_PORT="$port" PROFILE_METRICS_ONLY=1 \
        TASK_LIMIT=1000 SAMPLES="$SAMPLES" SPEC="$SPEC" ROUNDS=1 \
        TRAIN_JSONL="$TRAIN_JSONL" EVAL_JSONL="$EVAL_JSONL" \
        bash scripts/agent_opd_curve.sh "$LABEL-w$i" >"$log" 2>&1 &
    pids+=($!)
done

# 3. Wait for all shards; report per-shard exit so a crashed GPU isn't silent.
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[sharded] shard$i OK"; else echo "[sharded] shard$i FAILED (see $OUT/shard$i.log)" >&2; fail=1; fi
done

# 4. Merge every shard's metrics.jsonl, then cut the band once over the full corpus.
MERGED=$OUT/cb_profile
mkdir -p "$MERGED"
: >"$MERGED/metrics.jsonl"
for i in "${!GPU_ARR[@]}"; do
    m="$OUT/w$i/agent-opd-$LABEL-w$i/cb_profile/metrics.jsonl"
    [[ -f $m ]] && cat "$m" >>"$MERGED/metrics.jsonl" || echo "[sharded] WARN: shard$i metrics missing ($m)" >&2
done
groups=$(grep -c '"kind":"group"\|"kind": "group"' "$MERGED/metrics.jsonl" || true)
echo "[sharded] merged $groups group rows -> $MERGED/metrics.jsonl"

python3 scripts/comfort_band.py \
    --metrics "$MERGED/metrics.jsonl" --corpus "$CORPUS_ROOT" --out "$OUT/corpus-band" \
    --train-name "$TRAIN_JSONL" --eval-name "$EVAL_JSONL" \
    --max-seq "${CB_MAX_SEQ:-22000}" --pass-lo "${CB_PASS_LO:-0.2}" --pass-hi "${CB_PASS_HI:-0.8}" \
    --min-tests "${CB_MIN_TESTS:-2}"
echo "[sharded] band: $OUT/corpus-band (train.jsonl below) — run opd_security_filter --scan before phase 3"
exit $fail
