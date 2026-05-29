#!/usr/bin/env bash
# dsv4_beat_sglang_bench.sh — apples-to-apples DSv4 throughput A/B:
# ARLE (target-pod/release/infer) vs SGLang (editable /sgl-workspace/sglang),
# same 8×H20 TP=8, same model, same ISL/OSL, same concurrency sweep.
#
# Goal: ARLE decode throughput > SGLang × 1.30 (campaign target).
#
# Runs INSIDE the pod (invoke via `~/bin/pod 'bash /data01/build/arle/scripts/dsv4_beat_sglang_bench.sh <engine> <phase>'`).
#   engine: arle | sglang
#   phase : serve | bench | both   (default both)
#
# Standard SLO shape (decode-throughput-dominant): ISL=1024 OSL=512,
# concurrency sweep {1,8,32}. Writes JSON results to
# /data01/build/arle/docs/trace-artifacts/beat-sglang/<engine>-<ts>.json
set -uo pipefail

ENGINE="${1:-arle}"
PHASE="${2:-both}"
MODEL="/data01/models/DeepSeek-V4-Flash"
OUTDIR="/data01/build/arle/docs/trace-artifacts/beat-sglang"
mkdir -p "$OUTDIR"

ARLE_PORT=18300
SGL_PORT=30000
ISL=1024; OSL=512
CONCURRENCY="1 8 32"

serve_arle() {
  cd /data01/build/arle
  pkill -9 -f target-pod/release/infer 2>/dev/null; sleep 2
  INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
  ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 \
  ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 \
  ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native \
  ./target-pod/release/infer --model-path "$MODEL" --port $ARLE_PORT \
    --num-slots 4 --max-seq-len 4096 --mem-fraction-static 0.80 \
    --kv-cache-dtype fp8 --deepseek-distributed-layers 43
}

serve_sglang() {
  pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2
  cd /sgl-workspace/sglang
  python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 \
    --trust-remote-code --port $SGL_PORT --mem-fraction-static 0.80 \
    --kv-cache-dtype fp8e4m3 2>&1
}

# Minimal OpenAI-compat throughput bench (no external deps): fire N concurrent
# completion requests with fixed ISL/OSL, measure aggregate output tok/s.
bench() {
  local port="$1" tag="$2"
  local out="$OUTDIR/${tag}.json"
  python3 - "$port" "$ISL" "$OSL" "$out" "$CONCURRENCY" <<'PY'
import json, sys, time, urllib.request, threading
port, isl, osl, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
conc_list = [int(x) for x in sys.argv[5].split()]
prompt = "The history of computing is " + " ".join(["word%d" % (i % 97) for i in range(isl)])
def one(results, idx, port):
    body = json.dumps({"model":"DeepSeek-V4-Flash","prompt":prompt,
        "max_tokens":osl,"temperature":0,"ignore_eos":True,"stream":False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=body, headers={"Content-Type":"application/json"}, method="POST")
    t0=time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d=json.loads(r.read()); ct=d.get("usage",{}).get("completion_tokens",0)
        results[idx]=(time.perf_counter()-t0, ct)
    except Exception as e:
        results[idx]=(time.perf_counter()-t0, 0, str(e)[:80])
summary={}
for c in conc_list:
    res=[None]*c; ths=[threading.Thread(target=one,args=(res,i,port)) for i in range(c)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.perf_counter()-t0
    toks=sum(r[1] for r in res); errs=[r[2] for r in res if len(r)>2]
    summary[f"c{c}"]={"wall_s":round(wall,3),"out_tokens":toks,
        "out_tok_per_s":round(toks/wall,2) if wall>0 else 0,
        "per_req_tok_per_s":round(toks/wall/c,2) if wall>0 else 0,"errors":errs[:3]}
    print(f"c={c}: {summary[f'c{c}']['out_tok_per_s']} tok/s agg, {summary[f'c{c}']['per_req_tok_per_s']}/req, errs={len(errs)}")
json.dump(summary, open(out,"w"), indent=2)
print("wrote", out)
PY
}

case "$ENGINE" in
  arle)   [[ "$PHASE" =~ (serve|both) ]] && serve_arle ;;
  sglang) [[ "$PHASE" =~ (serve|both) ]] && serve_sglang ;;
esac
case "$ENGINE-$PHASE" in
  arle-bench)   bench $ARLE_PORT "arle" ;;
  sglang-bench) bench $SGL_PORT "sglang" ;;
esac
