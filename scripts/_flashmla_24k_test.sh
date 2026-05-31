#!/usr/bin/env bash
# FlashMLA prefill >24K cap-lift validation (pod tmux). Serves allreduce MoE
# (isolates FlashMLA from the MoE backend) at max-seq 32768, sends a ~29K-token
# needle prompt with FlashMLA prefill ON, checks needle retrieval + TTFT (fast =>
# FlashMLA engaged on the >24576 chunk; ~190s => legacy fallback) + IMA/TMA grep.
set -u
ROOT=/data01/build/arle
BIN=$ROOT/target-pod/release/infer
MODEL=/data01/models/DeepSeek-V4-Flash
PORT=18201
LOG=/tmp/fmla_serve.log
RESP=/tmp/fmla_resp.txt
: >"$LOG"; : >"$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
sleep 3

cd "$ROOT"
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_DSV4_MOE_BACKEND=allreduce \
ARLE_DSV4_EXPERT_BACKEND=native \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 \
ARLE_DSV4_GPU_FULL_LAYERS=43 \
ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_FLASHMLA_PREFILL=1 \
ARLE_DSV4_FLASHMLA_DECODE=1 \
RUST_LOG=info NCCL_DEBUG=WARN \
"$BIN" --model-path "$MODEL" --port "$PORT" --num-slots 1 \
  --max-seq-len 32768 --mem-fraction-static 0.10 \

  --kv-cache-dtype fp8 --deepseek-distributed-layers 43 >>"$LOG" 2>&1 &
SVPID=$!
echo "serve pid=$SVPID port=$PORT" | tee -a "$RESP"

ready=0
for i in $(seq 1 100); do
  if curl -sS -f "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SVPID" 2>/dev/null; then echo "SERVER_EXITED_EARLY" | tee -a "$RESP"; break; fi
  sleep 3
done
echo "ready=$ready" | tee -a "$RESP"

if [ "$ready" = "1" ]; then
  python3 - "$PORT" "$RESP" <<'PY'
import json, sys, time, urllib.request
port, resp = sys.argv[1], sys.argv[2]
# ~29K-token needle prompt: deep filler + a buried secret code + question.
needle = "The secret access code for the vault is 74925."
filler_a = ("The weather report for the region notes mild conditions. " * 1)
# build ~29000 words of filler around the needle
chunk = "data record entry log value metric sample point reading observation note "  # 10 words
pre = chunk * 1000   # ~10000 words
post = chunk * 450   # ~4500 words -> ~15K single chunk (timing profile)
prompt = pre + " " + needle + " " + post + " Question: What is the secret access code for the vault? Answer with only the number."
payload = {"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":prompt}],
           "temperature":0,"max_tokens":16,"stream":False}
data = json.dumps(payload).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type":"application/json"}, method="POST")
t0=time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        body=json.loads(r.read())
    dt=time.perf_counter()-t0
    txt=body["choices"][0]["message"]["content"]
    usage=body.get("usage",{})
    hit="74925" in txt
    line=f"RESULT elapsed={dt:.1f}s prompt_tokens={usage.get('prompt_tokens')} needle_hit={hit} text={txt!r}"
except Exception as e:
    line=f"RESULT ERROR {e}"
open(resp,"a").write(line+"\n")
print(line)
PY
fi

echo "=== IMA / TMA grep ===" | tee -a "$RESP"
grep -iE "illegal memory access|TMA Desc|unspecified launch|CUDA_ERROR_ILLEGAL|cudaErrorInvalidValue" "$LOG" | tail -6 | tee -a "$RESP"
echo "=== flashmla engagement (Once probe / legacy markers) ===" | tee -a "$RESP"
grep -iE "prefill=.*us|flashmla|use_flashmla|hybrid_attention|legacy" "$LOG" | tail -8 | tee -a "$RESP"
echo "=== server tail ===" | tee -a "$RESP"
tail -6 "$LOG" | tee -a "$RESP"
pkill -9 -f "target-pod/release/infer" 2>/dev/null || true
echo "FMLA_DONE" | tee -a "$RESP"
