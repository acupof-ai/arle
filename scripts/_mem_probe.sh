#!/usr/bin/env bash
# Memory probe: baseline after boot + single 16K-token prompt (single chunk, V2.4 scenario).
set -u
ROOT=/data01/build/arle; BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash; PORT=18202
LOG=/tmp/mp_serve.log; RESP=/tmp/mp_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true; sleep 3
cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 --max-seq-len 17408 \
  --mem-fraction-static 0.10 --kv-cache-dtype fp8 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
ready=0
for i in $(seq 1 100); do
  curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SVPID" 2>/dev/null || { echo "EXITED_EARLY" | tee -a "$RESP"; break; }
  sleep 3
done
echo "ready=$ready" | tee -a "$RESP"
echo "=== nvidia-smi BASELINE (after boot, idle) ===" | tee -a "$RESP"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -8 | tee -a "$RESP"
if [ "$ready" = "1" ]; then
  body=$(python3 -c "import json;c='word '*15500;print(json.dumps({'model':'DeepSeek-V4-Flash','messages':[{'role':'user','content':c+' What is 2+2? Answer with only the number.'}],'temperature':0,'max_tokens':8,'stream':False}))")
  out=$(curl -sS -m 285 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" 2>&1)
  ans=$(echo "$out" | python3 -c "import json,sys
try:
  d=json.load(sys.stdin);print('OK prompt_tok='+str(d['usage']['prompt_tokens'])+' text='+repr(d['choices'][0]['message']['content']))
except Exception as e: print('ERR '+str(e))" 2>/dev/null)
  echo "single-16K result: $ans" | tee -a "$RESP"
fi
echo "=== nvidia-smi PEAK (after request) ===" | tee -a "$RESP"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -8 | tee -a "$RESP"
echo "=== OOM grep ===" | tee -a "$RESP"
grep -icE "out of memory|Alloc failed" "$LOG" | tee -a "$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
echo "MP_DONE" | tee -a "$RESP"
