#!/usr/bin/env python3
"""Plot the single-machine (c=1) ARLE-vs-mlx-lm sweep — three-phase latency view.

Long-context c=1 on ARLE has THREE distinct phases (server request_trace +
client per-token timing, both agree):
  A. TTFT (token 1)          — prefill. ARLE ≈ mlx-lm (at parity).
  B. token1 → token2 gap     — a stall that scales with context (~3× a full
                               prefill). NOT decode. This is the entire ARLE
                               long-context disadvantage. mlx-lm has no analogue.
  C. token 2 onward (TPOT)   — steady decode. ARLE ≈ mlx-lm (at parity).

Panels:
  1. TTFT vs length (A) — ARLE vs mlx-lm.
  2. TPOT vs length (C) — ARLE vs mlx-lm (steady state, token 2+).
  3. token1→token2 gap (B) — ARLE only; the real long-context cost.
  4. cumulative time-to-40-tokens (A+B+C) — the end-to-end the user feels.

Usage: python3 scripts/plot_mlx_vs_arle_sweep.py [results.json] [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx_arle_sweep.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mlx_arle_sweep.png"
NTOK = 40  # cumulative horizon for panel 4

d = json.load(open(RESULTS))
lens = [int(x) for x in d["lens"]]
arle = d.get("arle", {})
mlx = d.get("mlx", {})


def g(src, n, k):
    return (src.get(str(n)) or {}).get(k)


def xt(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens)
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in lens])
    ax.set_xlabel("prompt length (tokens)")
    ax.grid(True, which="both", alpha=0.3)


ARLE_C, MLX_C = "#d6336c", "#1c7ed6"
fig, axs = plt.subplots(1, 4, figsize=(22, 5.2))
fig.suptitle(
    "ARLE Metal vs mlx-lm — Qwen3.6-35B-A3B-4bit, single machine c=1 (48 GB)",
    fontsize=14, fontweight="bold")

# Panel 1: TTFT (phase A)
ax = axs[0]
ax.plot(lens, [g(arle, n, "ttft_s") for n in lens], "o-", color=ARLE_C, lw=2, ms=6, label="ARLE")
ax.plot(lens, [g(mlx, n, "prefill_s") for n in lens], "s--", color=MLX_C, lw=2, ms=6, label="mlx-lm")
xt(ax); ax.set_ylabel("TTFT (s)"); ax.set_title("A. TTFT — time to token 1 (prefill)"); ax.legend()

# Panel 2: TPOT (phase C)
ax = axs[1]
ax.plot(lens, [1000.0 / g(arle, n, "decode_tps") if g(arle, n, "decode_tps") else None for n in lens],
        "o-", color=ARLE_C, lw=2, ms=6, label="ARLE")
ax.plot(lens, [1000.0 / g(mlx, n, "decode_tps") if g(mlx, n, "decode_tps") else None for n in lens],
        "s--", color=MLX_C, lw=2, ms=6, label="mlx-lm")
xt(ax); ax.set_ylabel("TPOT (ms/token)"); ax.set_ylim(0, 20)
ax.set_title("C. TPOT — steady decode (token 2+)"); ax.legend()

# Panel 3: token1->token2 gap (phase B) — ARLE only
ax = axs[2]
fi = [(g(arle, n, "first_interval_ms") or 0) / 1000.0 for n in lens]
ax.plot(lens, fi, "o-", color=ARLE_C, lw=2, ms=6, label="ARLE token1→token2")
ax.plot(lens, [0.012] * len(lens), "s--", color=MLX_C, lw=1.5, ms=5, label="mlx-lm (~12 ms)")
xt(ax); ax.set_ylabel("token1→token2 gap (s)")
ax.set_title("B. THE GAP — long-context stall (not decode)"); ax.legend()

# Panel 4: cumulative time-to-NTOK (A+B+C)
ax = axs[3]
def arle_e2e(n):
    t1 = g(arle, n, "ttft_s"); gap = (g(arle, n, "first_interval_ms") or 0) / 1000.0
    tp = g(arle, n, "decode_tps")
    return t1 + gap + (NTOK - 2) / tp if (t1 and tp) else None
def mlx_e2e(n):
    p = g(mlx, n, "prefill_s"); tp = g(mlx, n, "decode_tps")
    return p + (NTOK - 1) / tp if (p and tp) else None
ax.plot(lens, [arle_e2e(n) for n in lens], "o-", color=ARLE_C, lw=2, ms=6, label="ARLE")
ax.plot(lens, [mlx_e2e(n) for n in lens], "s--", color=MLX_C, lw=2, ms=6, label="mlx-lm")
xt(ax); ax.set_ylabel(f"time to {NTOK} tokens (s)")
ax.set_title(f"A+B+C — end-to-end ({NTOK} tokens)"); ax.legend()
for n in lens:
    a, m = arle_e2e(n), mlx_e2e(n)
    if a and m:
        ax.annotate(f"{a/m:.1f}×", (n, a), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=ARLE_C)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")

# table
print("\nlen    TTFT_A(s)        gap_B(s)   TPOT_C(ms)      e2e40(s)        e2e_ratio")
print("       ARLE   mlx       ARLE       ARLE   mlx       ARLE   mlx")
for n in lens:
    at, mp = g(arle, n, "ttft_s"), g(mlx, n, "prefill_s")
    gap = (g(arle, n, "first_interval_ms") or 0) / 1000.0
    ad, md = g(arle, n, "decode_tps"), g(mlx, n, "decode_tps")
    a4, m4 = arle_e2e(n), mlx_e2e(n)
    def f(x, w=6, p=2):
        return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else " " * (w - 1) + "-"
    r = f"{a4/m4:.2f}x" if (a4 and m4) else "-"
    print(f"{n:5d} {f(at)} {f(mp)}   {f(gap,7,2)}   "
          f"{f(1000/ad if ad else None)} {f(1000/md if md else None)}   "
          f"{f(a4)} {f(m4)}   {r}")
