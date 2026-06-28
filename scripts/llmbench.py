#!/usr/bin/env python3
"""Closed-loop LLM serving micro-bench — stdlib only.

arle serve has streaming deferred (stream=true -> 400), so we use NON-STREAMING
requests + the two-point method to decompose prefill vs decode:
  - run with --max-tokens 1   -> latency == TTFT (prefill + 1 token)
  - run with --max-tokens N    -> latency == prefill + N-token decode
    => ITL (ms/tok) = (lat_N - lat_1) / (N - 1)   [computed in the report]
Per concurrency level it reports latency p50/p99, mean completion tokens, the
aggregate output tok/s (sum completion_tokens / wall), and req/s.

Usage: llmbench.py --model NAME [--port 8000] [--prompt-tokens 512]
                   [--max-tokens 128] [--duration 20] [--concurrencies 1,2,4,8,16]
                   [--json OUT.json]
"""
import argparse, json, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--duration", type=float, default=20.0)
ap.add_argument("--concurrencies", default="1,2,4,8,16")
ap.add_argument("--json", default="")
A = ap.parse_args()
URL = f"http://127.0.0.1:{A.port}/v1/chat/completions"
CONC = [int(x) for x in A.concurrencies.split(",")]
PROMPT = "Numbers: " + " ".join(f"w{i}" for i in range(A.prompt_tokens)) + \
         "\nContinue the sequence and describe the pattern in detail."

def one_request():
    body = json.dumps({
        "model": A.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": A.max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read().decode("utf-8", "ignore"))
    lat = time.perf_counter() - t0
    u = obj.get("usage") or {}
    return {"lat": lat, "out": u.get("completion_tokens"), "prompt": u.get("prompt_tokens")}

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]

try:
    w = one_request()
    print(f"# warmup ok: lat={w['lat']:.3f}s out={w['out']} prompt={w['prompt']}", flush=True)
except Exception as e:
    print(f"# warmup FAILED: {e}", flush=True); sys.exit(1)

print(f"# model={A.model} prompt~{A.prompt_tokens}tok(actual={w['prompt']}) "
      f"max_tokens={A.max_tokens} dur={A.duration}s", flush=True)
print(f"{'conc':>4} {'reqs':>5} {'lat_p50ms':>10} {'lat_p99ms':>10} {'out_tok':>7} {'out_tok/s':>9} {'req/s':>6}",
      flush=True)
rows = []
for C in CONC:
    results = []; lock = threading.Lock(); stop = time.perf_counter() + A.duration
    def worker():
        while time.perf_counter() < stop:
            try:
                r = one_request()
            except Exception:
                continue
            with lock:
                results.append(r)
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(C)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    rs = [r for r in results if r["out"]]
    if not rs:
        print(f"{C:>4} {0:>5}  (no valid completions)", flush=True); continue
    lats = [r["lat"] for r in rs]
    tot_out = sum(r["out"] for r in rs)
    row = {"conc": C, "reqs": len(rs),
           "lat_p50_ms": round(pct(lats, 0.5) * 1000, 1),
           "lat_p99_ms": round(pct(lats, 0.99) * 1000, 1),
           "out_tok_mean": round(statistics.mean(r["out"] for r in rs), 1),
           "out_tok_s": round(tot_out / wall, 1),
           "req_s": round(len(rs) / wall, 2)}
    rows.append(row)
    print(f"{C:>4} {row['reqs']:>5} {row['lat_p50_ms']:>10} {row['lat_p99_ms']:>10} "
          f"{row['out_tok_mean']:>7} {row['out_tok_s']:>9} {row['req_s']:>6}", flush=True)

if A.json:
    with open(A.json, "w") as f:
        json.dump({"model": A.model, "prompt_tokens_actual": w["prompt"],
                   "max_tokens": A.max_tokens, "duration_s": A.duration, "rows": rows}, f, indent=2)
    print(f"# wrote {A.json}", flush=True)
