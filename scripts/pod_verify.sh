#!/usr/bin/env bash
# DSv4-Flash pod verify PIPELINE — re-run after any code change.
#   scripts/pod_verify.sh [label] [--nsys]
# Steps: sync working tree (incl. uncommitted) -> build (BUILD_EXIT gate) ->
#        serve TP4 GPUs 0-3 -> needle x3 + tok/s [-> nsys kernel breakdown] -> kill.
# Kill-group safe (only this run's serve, by unique port; never pkill). Reads a
# local summary at /tmp/pipe_<label>.txt. Idempotent; the proven recipe, scripted.
set -uo pipefail
POD="$HOME/bin/pod"
TREE=/data01/arle-gpu-verify-087df440
MODEL=/data01/models/DeepSeek-V4-Flash
PORT=18293
LABEL="${1:-verify}"
WANT_NSYS=0; [ "${2:-}" = "--nsys" ] && WANT_NSYS=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUM=/tmp/pipe_${LABEL}.txt
: > "$SUM"
log(){ echo "[pipe $LABEL] $*" | tee -a "$SUM"; }

# ---------- 1. SYNC working tree (uncommitted included) ----------
cd "$REPO"
STASH=$(git stash create 2>/dev/null || true); REF=${PIPE_REF:-${STASH:-HEAD}}
git branch -f __pipe "$REF" >/dev/null 2>&1
POD_HEAD=$($POD "cd $TREE && git rev-parse HEAD" 2>/dev/null | tr -d '[:space:]')
log "sync: local $(git rev-parse --short "$REF")  ->  pod ${POD_HEAD:0:8}"
git bundle create /tmp/pipe.bundle __pipe --not "$POD_HEAD" >/dev/null 2>&1 \
  || git bundle create /tmp/pipe.bundle __pipe >/dev/null 2>&1
git branch -D __pipe >/dev/null 2>&1
tn push /tmp/pipe.bundle /data01/pipe.bundle >/dev/null 2>&1 || { log "tn push FAILED"; exit 2; }
NEW=$($POD "cd $TREE && git fetch /data01/pipe.bundle __pipe >/dev/null 2>&1; git reset --hard FETCH_HEAD >/dev/null 2>&1; git rev-parse --short HEAD" 2>&1 | tail -1)
log "sync: pod now at $NEW"

# ---------- 2. BUILD ----------
log "build: dsv4_fast_build (release-fast) ..."
$POD "cd $TREE && rm -f /data01/pipe_build.log && setsid bash -c 'CUDA_HOME=/usr/local/cuda-12.9 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 PROFILE=release-fast FEATURES=cuda,nccl bash scripts/dsv4_fast_build.sh > /data01/pipe_build.log 2>&1; echo BUILD_EXIT=\$? >> /data01/pipe_build.log' </dev/null >/dev/null 2>&1 & echo launched" >/dev/null
BE=""
for _ in $(seq 1 90); do
  sleep 8
  BE=$($POD "grep -o 'BUILD_EXIT=[0-9]*' /data01/pipe_build.log 2>/dev/null | tail -1" 2>/dev/null | tr -d '[:space:]')
  [ -n "$BE" ] && break
done
log "build: ${BE:-TIMEOUT}"
if [ "$BE" != "BUILD_EXIT=0" ]; then
  log "BUILD FAILED — tail:"; $POD "tail -25 /data01/pipe_build.log" | tee -a "$SUM"
  exit 1
fi

# ---------- 3. SERVE (TP4 GPUs 0-3, kill-group by PORT) ----------
log "serve: TP4 on GPUs 0-3, port $PORT ..."
$POD "cd $TREE && rm -f /data01/pipe_serve.log && setsid bash -lc '
  export CUDA_VISIBLE_DEVICES=0,1,2,3 INFER_CUDA_DEVICES=0,1,2,3 INFER_TP_SIZE=4 INFER_DSV4_MAX_SEQ_LEN=8192 \
    ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_EXPERT_BACKEND=deepgemm ARLE_DSV4_LOCAL_GROUPED_EXPERTS=0 \
    ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_DEEPGEMM_ROOT=\$PWD/crates/cuda-kernels/vendor/deepgemm \
    ARLE_DEEPGEMM_LIBRARY_ROOT=\$PWD/crates/cuda-kernels/vendor/deepgemm/deep_gemm DG_JIT_CACHE_DIR=/data01/deepgemm-warm \
    CUDA_HOME=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu NCCL_DEBUG=WARN RUST_LOG=warn;
  exec target/release-fast/arle serve --backend cuda --model-path $MODEL --port $PORT --num-slots 4 --kv-cache-dtype auto
' </dev/null >/data01/pipe_serve.log 2>&1 & echo launched" >/dev/null
killserve(){ $POD "pg=\$(ps -eo pid,pgid,args | grep '[a]rle serve' | grep '$PORT' | awk '{print \$2}' | sort -u | head -1); [ -n \"\$pg\" ] && kill -9 -\$pg 2>/dev/null; echo killed \$pg" ; sleep 10; }
log "serve: launched (probe retries ready internally) — running verify ..."

# ---------- 4. VERIFY: needle x3 + tok/s ----------
tn push "$REPO/scripts/pod_verify_probe.py" /data01/pod_verify_probe.py >/dev/null 2>&1
log "verify: needle + tok/s ..."
$POD "python3 /data01/pod_verify_probe.py $PORT 2>&1" | tee -a "$SUM"

# ---------- 5. (optional) nsys kernel breakdown ----------
if [ "$WANT_NSYS" = 1 ]; then
  log "nsys: capturing decode kernel breakdown ..."
  $POD "nsys profile --duration=10 --inherit=true --trace=cuda --sample=none --force-overwrite=true -o /data01/pipe_nsys \
        curl -s -m 30 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' -d '{\"model\":\"x\",\"prompt\":\"Count:\",\"max_tokens\":400,\"temperature\":0}' >/dev/null 2>&1; \
        nsys stats --report cuda_gpu_kern_sum --format table /data01/pipe_nsys.nsys-rep 2>/dev/null | head -25" | tee -a "$SUM"
fi

# ---------- 6. KILL (group, mine only) ----------
killserve | tee -a "$SUM"
log "DONE. summary: $SUM"
