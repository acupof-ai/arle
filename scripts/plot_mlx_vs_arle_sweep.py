#!/usr/bin/env python3
"""Plot the single-machine (c=1) ARLE-vs-mlx-lm sweep — TTFT and TPOT.

Two panels, the two canonical serving-latency metrics:
  * TTFT  — time to first token (prefill latency), seconds, lower is better.
  * TPOT  — time per output token (1000 / decode_tps), ms/token, lower is better.

Series, in fairness order:
  * ARLE          — metal_serve /v1/completions, client SSE timing.
  * mlx-lm (HTTP) — mlx_lm.server /v1/completions, SAME client (transport-matched;
                    the apples-to-apples comparison). Present iff bench_mlx_http_decode ran.
  * mlx-lm (engine) — in-process stream_generate (no HTTP); shown as a dashed
                    reference for the raw-engine ceiling.

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
mlx = d.get("mlx", {})          # engine (in-process)
mlx_http = d.get("mlx_http", {})  # HTTP (transport-matched)


def g(src, n, k):
    return (src.get(str(n)) or {}).get(k)


def ttft_series(src, key="ttft_s"):
    xs, ys = [], []
    for n in lens:
        v = g(src, n, key)
        if v is not None:
            xs.append(n); ys.append(v)
    return xs, ys


def tpot_series(src, key="decode_tps"):
    xs, ys = [], []
    for n in lens:
        v = g(src, n, key)
        if v:  # tok/s > 0
            xs.append(n); ys.append(1000.0 / v)
    return xs, ys


def xticks(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens)
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in lens])
    ax.set_xlabel("prompt length (tokens)")
    ax.grid(True, which="both", alpha=0.3)


ARLE_C, HTTP_C, ENG_C = "#d6336c", "#1c7ed6", "#868e96"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.suptitle(
    "ARLE Metal vs mlx-lm — Qwen3.6-35B-A3B-4bit, single machine c=1 (48 GB)",
    fontsize=13, fontweight="bold")

# ---- Panel 1: TTFT ----
ax = ax1
x, y = ttft_series(arle)
ax.plot(x, y, "o-", color=ARLE_C, lw=2, ms=6, label="ARLE (metal_serve)")
if mlx_http:
    x, y = ttft_series(mlx_http)
    ax.plot(x, y, "s-", color=HTTP_C, lw=2, ms=6, label="mlx-lm (HTTP)")
x, y = ttft_series(mlx, "prefill_s")
ax.plot(x, y, "^--", color=ENG_C, lw=1.5, ms=5, label="mlx-lm (engine ref)")
xticks(ax)
ax.set_ylabel("TTFT (s) — lower is better")
ax.set_title("TTFT — time to first token (prefill)")
ax.legend()
# annotate ARLE-vs-(HTTP or engine) ratio
ref = mlx_http if mlx_http else None
am = dict(zip(*ttft_series(arle)))
rm = dict(zip(*(ttft_series(ref) if ref else ttft_series(mlx, "prefill_s"))))
for n in lens:
    if n in am and n in rm and rm[n]:
        ax.annotate(f"{am[n]/rm[n]:.1f}×", (n, am[n]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=ARLE_C)

# ---- Panel 2: TPOT ----
ax = ax2
x, y = tpot_series(arle)
ax.plot(x, y, "o-", color=ARLE_C, lw=2, ms=6, label="ARLE (metal_serve)")
if mlx_http:
    x, y = tpot_series(mlx_http)
    ax.plot(x, y, "s-", color=HTTP_C, lw=2, ms=6, label="mlx-lm (HTTP)")
x, y = tpot_series(mlx)
ax.plot(x, y, "^--", color=ENG_C, lw=1.5, ms=5, label="mlx-lm (engine ref)")
xticks(ax)
ax.set_yscale("log")
ax.set_ylabel("TPOT (ms/token) — lower is better")
ax.set_title("TPOT — time per output token (decode)")
ax.legend()

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")

# ---- compact text table ----
def f(x, w=9, p=3):
    return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else " " * (w - 1) + "-"


print("\n            TTFT (s)                      TPOT (ms/token)")
print("len      ARLE   mlxHTTP  mlxEng      ARLE   mlxHTTP  mlxEng")
for n in lens:
    at = g(arle, n, "ttft_s")
    ht = g(mlx_http, n, "ttft_s")
    et = g(mlx, n, "prefill_s")
    ad = g(arle, n, "decode_tps")
    hd = g(mlx_http, n, "decode_tps")
    ed = g(mlx, n, "decode_tps")
    tp = lambda v: (1000.0 / v) if v else None
    print(f"{n:6d} {f(at,8)} {f(ht,8)} {f(et,7)}   "
          f"{f(tp(ad),8,1)} {f(tp(hd),8,1)} {f(tp(ed),7,1)}")
