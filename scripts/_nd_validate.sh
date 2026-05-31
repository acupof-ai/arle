#!/usr/bin/env bash
# native-deepep cross-stream race fix validation (one-shot, pod-side).
# Launches TP=8 multiproc native-deepep serve, smoke + forward-window needle,
# reports combine-IMA presence. Run INSIDE a pod tmux session.
set -u
ROOT=/data01/build/arle
BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash
PORT=18200
LOG=/tmp/nd_serve.log
RESP=/tmp/nd_resp.txt
: >"$LOG"; : >"$RESP"

# Clean any stale server bound to our binary (never pkill by script name).
pkill -f "target-pod/release/infer" 2>/dev/null || true
sleep 3

cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_MULTIPROC_SERVE=1 \
ARLE_DSV4_MOE_BACKEND=native-deepep \
ARLE_DSV4_EXPERT_BACKEND=native \
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

# Wait up to 240s for HTTP readiness.
ready=0
for i in $(seq 1 80); do
  if curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SVPID" 2>/dev/null; then echo "SERVER_EXITED_EARLY" | tee -a "$RESP"; break; fi
  sleep 3
done
echo "ready=$ready boot_ranks=$(grep -c 'booted (peer_handles' "$LOG" 2>/dev/null)" | tee -a "$RESP"

if [ "$ready" = "1" ]; then
  # Smoke + 3 forward-window probes (16, 400, 1000 tokens of filler).
  for fw in 16 400 1000; do
    filler=$(python3 -c "print('word '*$fw)")
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'dsv4','messages':[{'role':'user','content':'$filler Compute 137 + 269. Answer with only the number.'}],'temperature':0,'max_tokens':16,'stream':False}))")
    out=$(curl -sS -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$body" 2>&1)
    ans=$(echo "$out" | python3 -c "import json,sys;
try:
  d=json.load(sys.stdin);print('OK:'+d['choices'][0]['message']['content'].strip()[:40])
except Exception as e:
  print('ERR:'+sys.stdin.read()[:120] if False else 'ERR:'+str(e))" 2>/dev/null)
    echo "fw=$fw -> $ans" | tee -a "$RESP"
  done
fi

echo "=== combine IMA grep ===" | tee -a "$RESP"
grep -iE "illegal memory access|combine failed|unspecified launch failure|sync after combine" "$LOG" | tail -5 | tee -a "$RESP"
echo "=== server tail ===" | tee -a "$RESP"
tail -8 "$LOG" | tee -a "$RESP"
pkill -f "target-pod/release/infer" 2>/dev/null || true
echo "VALIDATE_DONE" | tee -a "$RESP"
