#!/usr/bin/env python3
"""TTFT probe: wall time of max_tokens=1 completions (prefill + 1 decode + HTTP)
across prompt lengths. Usage: ttft_probe.py <arm-label>"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:18188"
ARM = sys.argv[1] if len(sys.argv) > 1 else "unknown"


def req(prompt, timeout=300):
    body = json.dumps({"model": "DeepSeek-V4-Flash", "prompt": prompt,
                       "max_tokens": 1, "temperature": 0}).encode()
    r = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        p = json.loads(resp.read().decode())
    wall = time.perf_counter() - t0
    return wall, (p.get("usage", {}) or {}).get("prompt_tokens", -1)


print(f"##### TTFT ARM={ARM} #####", flush=True)
# Filler word -> ~1 token each; lengths approximate, report actual usage.
for label, prompt in [
    ("short", "The capital of France is"),
    ("~512", "Paris is a city. " * 128 + "The capital of France is"),
    ("~2k", "Paris is a city. " * 512 + "The capital of France is"),
]:
    walls = []
    ptok = -1
    for i in range(4):
        w, ptok = req(prompt)
        walls.append(w)
    walls_sorted = sorted(walls[1:])  # drop warmup
    p50 = walls_sorted[len(walls_sorted) // 2]
    print(f"  {label}: prompt_tokens={ptok} ttft_p50={p50 * 1000:.0f}ms "
          f"(all={['%.0f' % (w * 1000) for w in walls]}ms, req0=warmup)", flush=True)
print(f"##### TTFT ARM={ARM} DONE #####", flush=True)
