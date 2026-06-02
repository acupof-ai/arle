#!/usr/bin/env bash
# dsv4_beat_sglang_bench.sh — apples-to-apples DSv4 throughput A/B:
# ARLE vs SGLang, same 8×H20 TP=8, same model, same ISL/OSL, same
# concurrency sweep. This helper is contract-aware: ARLE is always launched
# through the DSv4 SGLang best-practice profile, so an unsupported binary fails
# at startup instead of serving on the replicated-token debug lane.
#
# Goal: ARLE decode throughput > SGLang × 1.30 (campaign target).
#
# Runs INSIDE the pod (invoke via `~/bin/pod 'bash /data01/build/arle/scripts/dsv4_beat_sglang_bench.sh <engine> <phase>'`).
#   engine: arle | sglang
#   phase : serve | bench | both   (default both; both starts a server,
#           waits for readiness, runs the bench, then stops the server)
#
# Standard SLO shape (decode-throughput-dominant): ISL=1024 OSL=512,
# concurrency sweep {1,8,32}. Writes JSON results to
# /data01/build/arle/docs/trace-artifacts/beat-sglang/<engine>-<ts>.json
set -euo pipefail

ENGINE="${1:-arle}"
PHASE="${2:-both}"
MODEL="/data01/models/DeepSeek-V4-Flash"
OUTDIR="/data01/build/arle/docs/trace-artifacts/beat-sglang"
mkdir -p "$OUTDIR"

ARLE_BIN="${ARLE_BIN:-/data01/build/arle/target/release-fast/infer}"
SGLANG_DIR="${SGLANG_DIR:-/workspace/sglang@0d51db3}"
ARLE_PORT=18300
SGL_PORT=30000
ISL=1024; OSL=512
CONCURRENCY="1 8 32"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
SERVER_PID=""
SERVER_LOG=""

usage() {
  cat >&2 <<'EOF'
usage: dsv4_beat_sglang_bench.sh <arle|sglang> <serve|bench|both>

  serve: start the selected server in the foreground
  bench: run the bench against an already-running selected server
  both : start the selected server in the background, wait /v1/models,
         run the bench, then stop the server
EOF
}

case "$ENGINE" in
  arle|sglang) ;;
  *) usage; exit 2 ;;
esac
case "$PHASE" in
  serve|bench|both) ;;
  *) usage; exit 2 ;;
esac

run_arle_server() {
  [[ -x "$ARLE_BIN" ]] || { echo "ARLE_BIN is not executable: $ARLE_BIN" >&2; exit 2; }
  INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
  ARLE_MULTIPROC_SERVE=1 ARLE_DSV4_PERFORMANCE_PROFILE=sglang \
  ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 \
  ARLE_DSV4_SHARED_KV_POOL=1 \
  ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 \
  ARLE_DSV4_MOE_BACKEND=native-deepep ARLE_DSV4_EXPERT_BACKEND=deepgemm \
  "$ARLE_BIN" --model-path "$MODEL" --port $ARLE_PORT \
    --num-slots 128 --max-seq-len 4096 --mem-fraction-static 0.80 \
    --kv-cache-dtype fp8 --cuda-graph-max-bs 16 \
    --deepseek-distributed-layers 43
}

run_sglang_server() {
  [[ -d "$SGLANG_DIR" ]] || { echo "SGLANG_DIR does not exist: $SGLANG_DIR" >&2; exit 2; }
  python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 \
    --trust-remote-code --port $SGL_PORT --mem-fraction-static 0.80 \
    --kv-cache-dtype fp8_e4m3 2>&1
}

serve_arle() {
  cd /data01/build/arle
  run_arle_server
}

serve_sglang() {
  cd "$SGLANG_DIR"
  run_sglang_server
}

start_arle() {
  SERVER_LOG="$OUTDIR/arle-serve-$(date +%Y%m%d-%H%M%S).log"
  (cd /data01/build/arle && run_arle_server) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "started ARLE pid=$SERVER_PID log=$SERVER_LOG"
}

start_sglang() {
  SERVER_LOG="$OUTDIR/sglang-serve-$(date +%Y%m%d-%H%M%S).log"
  (cd "$SGLANG_DIR" && run_sglang_server) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "started SGLang pid=$SERVER_PID log=$SERVER_LOG"
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_ready() {
  local port="$1" tag="$2"
  local waited=0
  while (( waited < READY_TIMEOUT_SEC )); do
    if python3 - "$port" <<'PY'
import sys, urllib.request
port = sys.argv[1]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "$tag ready on port $port after ${waited}s"
      return 0
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$tag server exited before readiness; see $SERVER_LOG" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "$tag server did not become ready within ${READY_TIMEOUT_SEC}s; see $SERVER_LOG" >&2
  return 1
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

bench_tag="${ENGINE}-$(date +%Y%m%d-%H%M%S)"
case "$ENGINE-$PHASE" in
  arle-serve) serve_arle ;;
  sglang-serve) serve_sglang ;;
  arle-bench) bench $ARLE_PORT "$bench_tag" ;;
  sglang-bench) bench $SGL_PORT "$bench_tag" ;;
  arle-both)
    trap cleanup_server EXIT
    start_arle
    wait_ready $ARLE_PORT "ARLE"
    bench $ARLE_PORT "$bench_tag"
    cleanup_server
    trap - EXIT
    ;;
  sglang-both)
    trap cleanup_server EXIT
    start_sglang
    wait_ready $SGL_PORT "SGLang"
    bench $SGL_PORT "$bench_tag"
    cleanup_server
    trap - EXIT
    ;;
esac
