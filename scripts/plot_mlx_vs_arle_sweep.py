#!/usr/bin/env python3
"""Plot the single-machine (c=1) ARLE-vs-mlx-lm sweep — corrected metric.

Three panels:
  * TTFT — time to first token (prefill latency), s, lower better.
  * TPOT — steady-state time per output token (token 2 onward; the token1→2
           prefill-tail interval is excluded), ms/token, lower better.
  * first-interval — the ARLE token1→token2 gap (prefill front-loaded by the
           pipelined scheduler AFTER token 1). This is the real long-context
           "slow to start" cost; mlx-lm has no equivalent (its prefill is all in
           TTFT). Diagnostic panel, ARLE only.

Usage: python3 scripts/plot_mlx_vs_arle_sweep.py [results.json] [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx_arle_sweep.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mlx_arle_sweep.png"

d = json.load(open(RESULTS))
lens = [int(x) for x in d["lens"]]
arle = d.get("arle", {})
mlx = d.get("mlx", {})            # engine (in-process)
mlx_http = d.get("mlx_http", {})  # HTTP (transport-matched), optional


def g(src, n, k):
    return (src.get(str(n)) or {}).get(k)


def ser(src, key):
    xs, ys = [], []
    for n in lens:
        v = g(src, n, key)
        if v is not None:
            xs.append(n); ys.append(v)
    return xs, ys


def tpot(src, key="decode_tps"):
    xs, ys = [], []
    for n in lens:
        v = g(src, n, key)
        if v:
            xs.append(n); ys.append(1000.0 / v)
    return xs, ys


def xt(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens)
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in lens])
    ax.set_xlabel("prompt length (tokens)")
    ax.grid(True, which="both", alpha=0.3)


ARLE_C, HTTP_C, ENG_C = "#d6336c", "#1c7ed6", "#868e96"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
fig.suptitle(
    "ARLE Metal vs mlx-lm — Qwen3.6-35B-A3B-4bit, single machine c=1 (48 GB)",
    fontsize=13, fontweight="bold")

# Panel 1: TTFT
ax = ax1
ax.plot(*ser(arle, "ttft_s"), "o-", color=ARLE_C, lw=2, ms=6, label="ARLE")
if mlx_http:
    ax.plot(*ser(mlx_http, "ttft_s"), "s-", color=HTTP_C, lw=2, ms=6, label="mlx-lm (HTTP)")
ax.plot(*ser(mlx, "prefill_s"), "^--", color=ENG_C, lw=1.5, ms=5, label="mlx-lm (engine)")
xt(ax); ax.set_ylabel("TTFT (s) — lower better"); ax.set_title("TTFT — time to first token"); ax.legend()

# Panel 2: TPOT (steady-state)
ax = ax2
ax.plot(*tpot(arle), "o-", color=ARLE_C, lw=2, ms=6, label="ARLE")
if mlx_http:
    ax.plot(*tpot(mlx_http), "s-", color=HTTP_C, lw=2, ms=6, label="mlx-lm (HTTP)")
ax.plot(*tpot(mlx), "^--", color=ENG_C, lw=1.5, ms=5, label="mlx-lm (engine)")
xt(ax); ax.set_ylabel("TPOT (ms/token) — lower better")
ax.set_title("TPOT — steady-state decode (token 2+)")
ax.set_ylim(0, max(20, max(tpot(arle)[1] + tpot(mlx)[1]) * 1.2)); ax.legend()

# Panel 3: ARLE first-interval (prefill-tail front-load)
ax = ax3
x, y = ser(arle, "first_interval_ms")
ax.plot(x, [v / 1000.0 for v in y], "o-", color=ARLE_C, lw=2, ms=6, label="ARLE token1→2 gap")
xt(ax); ax.set_ylabel("first inter-token gap (s)")
ax.set_title("ARLE prefill-tail (front-loaded after token 1)")
ax.legend()

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")

# text table
def f(x, w=9, p=3):
    return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else " " * (w - 1) + "-"


print("\n          TTFT(s)              TPOT(ms/tok)          ARLE")
print("len    ARLE   mlxH   mlxE    ARLE   mlxH   mlxE    first_iv(s)")
for n in lens:
    at, ht, et = g(arle, n, "ttft_s"), g(mlx_http, n, "ttft_s"), g(mlx, n, "prefill_s")
    ad, hd, ed = g(arle, n, "decode_tps"), g(mlx_http, n, "decode_tps"), g(mlx, n, "decode_tps")
    fi = g(arle, n, "first_interval_ms")
    tp = lambda v: (1000.0 / v) if v else None
    print(f"{n:6d} {f(at,6,2)} {f(ht,6,2)} {f(et,6,2)}  "
          f"{f(tp(ad),6,1)} {f(tp(hd),6,1)} {f(tp(ed),6,1)}   {f((fi/1000.0) if fi else None,6,1)}")
