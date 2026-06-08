#!/usr/bin/env python3
"""DeepSeek-V4-Flash B=1 latency optimization journey on 8xH20 (TP=8/EP=8, FP8 MoE).

A single three-panel figure that tells the whole 2026-06-06 -> 2026-06-08 arc,
every number traced to a committed wins/ or errors/ entry:

  A. Decode stops scaling with context  (official DSA indexer)
       legacy csa_select single-SM -> 124 ms @4096; official multi-SM flat ~26 ms.
       Source: wins/2026-06-06-dsv4-official-dsa-decode-flat-26ms-variable-shape.md
  B. Prefill projections: scalar GEMV -> tensor-core DeepGEMM
       per-stage M=1024 prefill kernel time, wq_b/wo_a/wo_b -94..95%.
       Source: wins/2026-06-08-dsv4-prefill-proj-deepgemm-clean.md
  C. The B=1 decode wall: amortization moved it, per-kernel tuning washed
       27 -> 15 ms via MTP depth-1 batched verify (+71%); 8 levers wall-neutral.
       Source: wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md,
               wins/2026-06-08-dsv4-mtp-batched-verify-lands.md

Usage: python3 scripts/plot_dsv4_perf_journey.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/dsv4-perf-journey.png"

# House palette (matches scripts/plot_mlx_vs_arle_sweep.py)
BEFORE = "#868e96"   # legacy / scalar  (muted grey)
AFTER = "#d6336c"    # optimized        (ARLE magenta)
WIN = "#2f9e44"      # target / win     (green)
INK = "#212529"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#495057",
    "axes.linewidth": 0.9,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
fig.suptitle(
    "DeepSeek-V4-Flash  -  B=1 latency optimization journey  (8xH20, TP=8 / EP=8, FP8 MoE,  2026-06-06 -> 06-08)",
    fontsize=15.5, fontweight="bold", y=0.99)

# ----------------------------------------------------------------------------
# Panel A: decode stops scaling with context  (official DSA indexer)
# ----------------------------------------------------------------------------
ax = axs[0]
ctx = [256, 512, 1024, 2048, 4096]
legacy = [28.16, 37.11, 25.77, 48.03, 124.37]   # csa_select, single CTA / 1-SM
dsa = [26.02, 26.06, 26.05, 26.11, 26.15]        # official fp8_paged_mqa_logits, multi-SM

ax.plot(ctx, legacy, "o-", color=BEFORE, lw=2.4, ms=8,
        label="legacy csa_select (1-SM, scales)")
ax.plot(ctx, dsa, "s-", color=AFTER, lw=2.4, ms=8,
        label="official DSA indexer (multi-SM, flat)")
ax.set_xscale("log", base=2)
ax.set_xticks(ctx)
ax.set_xticklabels(["256", "512", "1k", "2k", "4k"])
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("decode latency (ms / token)")
ax.set_title("A.  Decode stops scaling with context\n(official DSA indexer, default-on)")
ax.set_ylim(0, 135)
ax.grid(True, which="both", axis="y", alpha=0.25)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
# 4.8x callout at 4096
ax.annotate("124 ms", (4096, 124.37), textcoords="offset points", xytext=(-6, 4),
            ha="right", fontsize=9.5, color=BEFORE, fontweight="bold")
ax.annotate("~26 ms", (4096, 26.15), textcoords="offset points", xytext=(-4, -16),
            ha="right", fontsize=9.5, color=AFTER, fontweight="bold")
ax.annotate("4.8x faster\n@ 4k ctx", (4096, 75), ha="center", fontsize=11,
            color=WIN, fontweight="bold")
ax.add_patch(FancyArrowPatch((4096, 119), (4096, 31), arrowstyle="-|>",
             mutation_scale=16, lw=1.6, color=WIN, shrinkA=0, shrinkB=0))

# ----------------------------------------------------------------------------
# Panel B: prefill projections  scalar GEMV -> DeepGEMM  (per-stage, M=1024)
# ----------------------------------------------------------------------------
ax = axs[1]
stages = ["wq_b", "wo_a", "wo_b"]
scalar = [138.3, 132.0, 137.7]     # dsv4_fp8_gemv_batch (M=1 GEMV run at M=1024)
deepgemm = [8.4, 8.6, 6.3]         # tensor-core DeepGEMM
x = range(len(stages))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], scalar, w, color=BEFORE,
            label="scalar fp8_gemv_batch")
b2 = ax.bar([i + w / 2 for i in x], deepgemm, w, color=AFTER,
            label="tensor-core DeepGEMM")
ax.set_xticks(list(x))
ax.set_xticklabels(stages)
ax.set_xlabel("MLA / output projection (M=1024 prefill)")
ax.set_ylabel("per-stage kernel time (ms)")
ax.set_title("B.  Prefill projections -> DeepGEMM\nrank-0 LINEAR_PROFILE, load/JIT excluded")
ax.set_ylim(0, 178)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
for i, (s, d) in enumerate(zip(scalar, deepgemm)):
    ax.annotate(f"{s:.0f}", (i - w / 2, s), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9, color=BEFORE)
    ax.annotate(f"{d:.1f}", (i + w / 2, d), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9, color=AFTER, fontweight="bold")
    pct = 100 * (s - d) / s
    ax.annotate(f"-{pct:.0f}%", (i, max(s, 60) * 0.55), ha="center",
                fontsize=10.5, color=WIN, fontweight="bold")
ax.text(0.30, 0.985, "TTFT outcome:  prefill ~23 ms\n(projections were 62%\nof prefill GPU)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.0, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f3", ec=AFTER, lw=1.0))

# ----------------------------------------------------------------------------
# Panel C: the B=1 decode wall  -  one lever moved it, eight washed
# ----------------------------------------------------------------------------
ax = axs[2]
labels = ["baseline\n(DSA + DeepGEMM\nprojections)", "MTP depth-1\nbatched verify"]
vals = [27.0, 15.0]
bars = ax.bar([0, 1], vals, 0.5, color=[BEFORE, AFTER], zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("B=1 decode latency (ms / token)")
ax.set_title("C.  The B=1 decode wall\namortization moved it, per-kernel tuning washed")
ax.set_ylim(0, 39)
ax.grid(True, axis="y", alpha=0.25)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f} ms", (i, v), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=11, fontweight="bold",
                color=BEFORE if i == 0 else AFTER, zorder=4)
# +71% arrow
ax.add_patch(FancyArrowPatch((0.0, 27), (1.0, 15.6), arrowstyle="-|>",
             mutation_scale=18, lw=2.0, color=WIN, shrinkA=4, shrinkB=4, zorder=5))
ax.annotate("+71% tok/s\n(39.9 -> 64.2,\nbyte-identical)", (0.42, 21.0), ha="center",
            fontsize=9.5, color=WIN, fontweight="bold", zorder=6)
# reference lines
ax.axhline(15, color=AFTER, ls=":", lw=1.3, zorder=1)
ax.annotate("sound B=1 ceiling", (1.46, 15), va="center", ha="right",
            fontsize=8.5, color=AFTER)
ax.axhline(6, color=WIN, ls="--", lw=1.5, zorder=1)
ax.annotate("6 ms target  -  tree-EAGLE + mega-kernel,\nor batching (M=N)", (1.46, 6.6),
            va="bottom", ha="right", fontsize=8.5, color=WIN)
ax.annotate("HBM active-weight floor ~0.3 ms (87x below the wall)", (-0.45, 1.1),
            va="bottom", ha="left", fontsize=8, color="#868e96", style="italic")
# washed levers callout
washed = ("8 levers WASHED (wall-neutral):\n"
          "whole-step CUDA graph . mhc_params uint4 (-25% iso)\n"
          "M=1 GEMV (1.8x iso) . comm-overlap . mempool\n"
          "launch removal . per-layer graph . DSA-skip\n"
          "=> GPU-bound critical path; gpukernsum != crit path")
ax.text(0.5, 0.985, washed, transform=ax.transAxes, ha="center", va="top",
        fontsize=7.6, color=INK,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f1f3f5", ec="#adb5bd", lw=1.0))

fig.tight_layout(rect=(0, 0.0, 1, 0.955))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
