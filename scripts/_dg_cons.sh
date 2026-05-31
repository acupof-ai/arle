#!/usr/bin/env bash
# native-deepep coherence validation after Finding-1 fix (local_expert_start=0).
# Serves native-deepep TP=8 multiproc, probes factual + arithmetic prompts with
# the CORRECT model name, expects coherent output (was ': is?' garbage before).
set -u
ROOT=/data01/build/arle
BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash
PORT=18200
LOG=/tmp/dgc_serve.log
RESP=/tmp/dgc_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
sleep 3

cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_MULTIPROC_SERVE=1 \
ARLE_DSV4_MOE_BACKEND=native-deepep \
ARLE_DSV4_EXPERT_BACKEND=deepgemm ARLE_DEEPGEMM_CONSERVATIVE_LAYOUT=1 \
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1 \
ARLE_DEEPEP_DIR=/data01/build/DeepEP \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 \
ARLE_DSV4_GPU_FULL_LAYERS=43 \
ARLE_DSV4_INCREMENTAL_KV=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 \
  --max-seq-len 4096 --mem-fraction-static 0.10 \
  --kv-cache-dtype bf16 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
echo "serve pid=$SVPID" | tee -a "$RESP"

ready=0
for i in $(seq 1 100); do
  if curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SVPID" 2>/dev/null; then echo "SERVER_EXITED_EARLY" | tee -a "$RESP"; break; fi
  sleep 3
done
echo "ready=$ready" | tee -a "$RESP"

if [ "$ready" = "1" ]; then
  for q in "The capital of France is" "Compute 137 + 269. Answer with only the number." "List the first five prime numbers."; do
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'DeepSeek-V4-Flash','messages':[{'role':'user','content':sys.argv[1]}],'temperature':0,'max_tokens':24,'stream':False}))" "$q")
    out=$(curl -sS -m 90 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" 2>&1)
    ans=$(echo "$out" | python3 -c "import json,sys
try:
  d=json.load(sys.stdin);print(repr(d['choices'][0]['message']['content']))
except Exception as e:
  print('ERR '+str(e))" 2>/dev/null)
    echo "Q: $q" | tee -a "$RESP"
    echo "A: $ans" | tee -a "$RESP"
  done
fi

echo "=== boot ranks ===" | tee -a "$RESP"
grep -c "booted (device_id" "$LOG" | tee -a "$RESP"
echo "=== IMA grep ===" | tee -a "$RESP"
grep -icE "illegal memory|combine failed|unspecified launch" "$LOG" | tee -a "$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
echo "DGC_DONE" | tee -a "$RESP"
