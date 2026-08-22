#!/usr/bin/env python3
"""Non-streaming single-user latency/throughput probe for the Metal serve path.

The rewrite Metal backend defers `stream=true` (R5 tranche 2), so the native
streaming TTFT/ITL collection can't run against it. This derives the same
single-stream numbers from two non-streaming latency points:

    lat(max_tokens=1)  ~= TTFT            (prefill of P tokens + 1 decode)
    lat(max_tokens=N)   = TTFT + (N-1)*TPOT

  => TPOT       = (lat_N - lat_1) / (N - 1)
     decode tok/s = 1000 / TPOT          (steady-state, prefill excluded)

All requests are temp=0 + ignore_eos so output length is exactly N.
Reports the median over K repeats (after one warmup).
"""
import json
import statistics
import sys
import time
from urllib import request

TARGET = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen"
PROMPT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
K = int(sys.argv[5]) if len(sys.argv) > 5 else 5

# A prompt ~PROMPT_TOKENS long. " word" is ~1 token under the Qwen BPE, so
# repeating it PROMPT_TOKENS times lands close to the target. Every call gets a
# UNIQUE nonce prefix so the RadixCache never hits — each request does a full
# uncached prefill of ~P tokens. This keeps the prefill cost identical between
# the max_tokens=1 (TTFT) and max_tokens=N (E2E) points so it cancels cleanly in
# TPOT = (lat_N - lat_1)/(N-1); a cache hit on one point but not the other would
# corrupt the subtraction.
_BODY = "word " * PROMPT_TOKENS
_nonce = [0]


def call(max_tokens):
    _nonce[0] += 1
    prompt = f"Story #{_nonce[0]}-{time.time_ns()}. Tell me a long story. " + _BODY
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }).encode()
    req = request.Request(f"{TARGET}/v1/completions", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    dt = (time.perf_counter() - t0) * 1000.0  # ms
    usage = body.get("usage", {})
    return dt, usage.get("completion_tokens"), usage.get("prompt_tokens")


def median_lat(max_tokens, label):
    call(max_tokens)  # warmup (not counted)
    lats, comp, prompt_toks = [], None, None
    for _ in range(K):
        dt, c, p = call(max_tokens)
        lats.append(dt)
        comp, prompt_toks = c, p
    med = statistics.median(lats)
    print(f"  {label}: median {med:.1f} ms  (completion_tokens={comp}, "
          f"prompt_tokens={prompt_toks}, runs={['%.0f'%x for x in lats]})", file=sys.stderr)
    return med, comp, prompt_toks


lat1, _, prompt_toks = median_lat(1, "TTFT (max_tokens=1)")
latN, compN, _ = median_lat(N, f"E2E  (max_tokens={N})")

decode_ms = (latN - lat1) / (N - 1)
tpot = decode_ms
decode_tps = 1000.0 / tpot if tpot > 0 else float("nan")

out = {
    "model": MODEL,
    "prompt_tokens": prompt_toks,
    "output_tokens": N,
    "ttft_ms": round(lat1, 1),
    "tpot_ms": round(tpot, 2),
    "decode_tok_s": round(decode_tps, 1),
}
print(json.dumps(out))
