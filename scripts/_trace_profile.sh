#!/usr/bin/env bash
# Per-phase prefill profile: ARLE_DSV4_TRACE_LAYER=1, 4K prompt, sum per-phase ms.
set -u
ROOT=/data01/build/arle; BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash; PORT=18203
LOG=/tmp/tp_serve.log; RESP=/tmp/tp_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f release/infer 2>/dev/null || true; sleep 4
cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 ARLE_DSV4_TRACE_LAYER=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 --max-seq-len 8192 \
  --mem-fraction-static 0.10 --kv-cache-dtype fp8 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
ready=0
for i in $(seq 1 100); do
  curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SVPID" 2>/dev/null || break
  sleep 3
done
echo "ready=$ready" | tee -a "$RESP"
if [ "$ready" = "1" ]; then
  body=$(python3 -c "import json;c='word '*4000;print(json.dumps({'model':'DeepSeek-V4-Flash','messages':[{'role':'user','content':c+' 2+2?'}],'temperature':0,'max_tokens':4,'stream':False}))")
  t0=$(date +%s)
  curl -sS -m 200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >/tmp/tp_out.txt 2>&1
  t1=$(date +%s)
  echo "request_wall_s=$((t1-t0))" | tee -a "$RESP"
fi
echo "=== per-phase totals (sum elapsed_ms across all layers, prefill only) ===" | tee -a "$RESP"
grep "dsv4_trace" "$LOG" | python3 -c "
import sys,re,collections
agg=collections.defaultdict(lambda:[0.0,0])
for ln in sys.stdin:
    m=re.search(r'phase=(\S+) tokens=(\d+) elapsed_ms=([\d.]+)',ln)
    if not m: continue
    ph,tok,ms=m.group(1),int(m.group(2)),float(m.group(3))
    if tok<=1: continue   # prefill only (skip decode tokens=1)
    agg[ph][0]+=ms; agg[ph][1]+=1
for ph,(ms,n) in sorted(agg.items(),key=lambda x:-x[1][0]):
    print(f'{ph:20s} total={ms:9.1f}ms  calls={n}  avg={ms/max(n,1):7.2f}ms')
" | tee -a "$RESP"
pkill -9 -f release/infer 2>/dev/null || true
echo "TP_DONE" | tee -a "$RESP"
