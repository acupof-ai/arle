#!/usr/bin/env bash
# native-deepep + FlashMLA + OOM-fix >24K prefill test (the real fast path).
set -u
ROOT=/data01/build/arle; BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash; PORT=18204
LOG=/tmp/nd24_serve.log; RESP=/tmp/nd24_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f release/infer 2>/dev/null || true; sleep 4
cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 ARLE_MULTIPROC_SERVE=1 \
ARLE_DSV4_MOE_BACKEND=native-deepep ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1 ARLE_DEEPEP_DIR=/data01/build/DeepEP \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 --max-seq-len 32768 \
  --mem-fraction-static 0.10 --kv-cache-dtype bf16 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
ready=0
for i in $(seq 1 120); do
  curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SVPID" 2>/dev/null || { echo "EXITED_EARLY" | tee -a "$RESP"; break; }
  sleep 3
done
echo "ready=$ready boot_ranks=$(grep -c 'booted (device_id' "$LOG")" | tee -a "$RESP"
if [ "$ready" = "1" ]; then
  python3 - "$PORT" "$RESP" <<'PY'
import json,sys,time,urllib.request
port,resp=sys.argv[1],sys.argv[2]
chunk="data record entry log value metric sample point reading observation note "
prompt=chunk*1550+" The secret access code for the vault is 74925. "+chunk*620+" Question: What is the secret access code for the vault? Answer with only the number."
payload={"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":16,"stream":False}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
t0=time.perf_counter()
try:
  with urllib.request.urlopen(req,timeout=290) as r: b=json.loads(r.read())
  dt=time.perf_counter()-t0
  txt=b["choices"][0]["message"]["content"]; u=b.get("usage",{})
  print(f"RESULT elapsed={dt:.1f}s prompt_tok={u.get('prompt_tokens')} needle_hit={'74925' in txt} text={txt!r}")
except Exception as e:
  print(f"RESULT ERROR {e}")
PY
fi >> "$RESP" 2>&1
cat "$RESP" | grep RESULT
echo "=== IMA/OOM/timeout grep ===" | tee -a "$RESP"
grep -icE "illegal memory|out of memory|Alloc failed|timed out|combine failed" "$LOG" | tee -a "$RESP"
tail -3 "$LOG" | tee -a "$RESP"
pkill -9 -f release/infer 2>/dev/null || true
echo "ND24_DONE" | tee -a "$RESP"
