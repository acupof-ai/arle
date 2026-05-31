#!/usr/bin/env bash
# native-deepep per-phase prefill profile (4K, ARLE_DSV4_TRACE_LAYER=1).
set -u
ROOT=/data01/build/arle; BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash; PORT=18205
LOG=/tmp/ndt_serve.log; RESP=/tmp/ndt_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f release/infer 2>/dev/null || true; sleep 4
cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 ARLE_MULTIPROC_SERVE=1 \
ARLE_DSV4_MOE_BACKEND=native-deepep ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1 ARLE_DEEPEP_DIR=/data01/build/DeepEP \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 ARLE_DSV4_TRACE_LAYER=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 --max-seq-len 8192 \
  --mem-fraction-static 0.10 --kv-cache-dtype bf16 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
ready=0
for i in $(seq 1 120); do
  curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SVPID" 2>/dev/null || break; sleep 3
done
echo "ready=$ready ranks=$(grep -c 'booted (device_id' "$LOG")" | tee -a "$RESP"
if [ "$ready" = "1" ]; then
  body=$(python3 -c "import json;c='word '*4000;print(json.dumps({'model':'DeepSeek-V4-Flash','messages':[{'role':'user','content':c+' 2+2?'}],'temperature':0,'max_tokens':4,'stream':False}))")
  t0=$(date +%s); curl -sS -m 250 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >/tmp/ndt_out.txt 2>&1; t1=$(date +%s)
  echo "request_wall_s=$((t1-t0))" | tee -a "$RESP"
fi
echo "=== per-phase (prefill, tokens>1) ===" | tee -a "$RESP"
grep "dsv4_trace" "$LOG" | python3 -c "
import sys,re,collections
agg=collections.defaultdict(lambda:[0.0,0])
for ln in sys.stdin:
 m=re.search(r'phase=(\S+) tokens=(\d+) elapsed_ms=([\d.]+)',ln)
 if not m or int(m.group(2))<=1: continue
 agg[m.group(1)][0]+=float(m.group(3)); agg[m.group(1)][1]+=1
for ph,(ms,n) in sorted(agg.items(),key=lambda x:-x[1][0])[:14]:
 print(f'{ph:34s} total={ms:9.1f}ms calls={n} avg={ms/max(n,1):7.2f}')
" | tee -a "$RESP"
pkill -9 -f release/infer 2>/dev/null || true
echo "NDT_DONE" | tee -a "$RESP"
