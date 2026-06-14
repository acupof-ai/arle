#!/usr/bin/env python3
"""DeepSeek-V4-Flash B=1 decode optimization journey on 8xH20 (TP=8/EP=8, FP8 MoE).

ONE compact figure (small-display-friendly). A milestone waterfall / step-line of
B=1 decode tok/s across the campaign, each step annotated with the lever name +
the sourced number + a tiny date. EVERY number is quoted verbatim from a committed
docs/experience/wins/ entry (no fabrication, no interpolation):

  33.5  regressed base                wins/2026-06-13-dsv4-decode-band-64align-regression-fix.md
                                       (also -fp8-decode-moe-lane-44.md "pre-campaign 33.5")
  36.9  64-align packing fix          wins/2026-06-13-dsv4-decode-band-64align-regression-fix.md
                                       (a1e15307; "36.90 / 35.95")
  44.0  FP8 decode-band MoE lane      wins/2026-06-13-dsv4-fp8-decode-moe-lane-44.md
                                       ("44.11 / 43.76 / 44.18", 3 boots)
  44.5  + automatic NUMA pin          wins/2026-06-13-dsv4-numa-pin-all-threads-ab.md
                                       ("44.54 / 44.52 / 44.52", mean 44.48)
  53.3  d2 chain-fold MTP (default)   wins/2026-06-13-dsv4-mtp-d2-chain-fold-53.md
                                       ("53.37 / 53.30 / 53.32")
  49..55.6 confirmed 06-14 baseline   wins/2026-06-14-dsv4-config-mechanism-classification.md
                                       (forward floor 42.0 ms/step; tok/s 43.2->55.6 by
                                        MTP acceptance; code-prompt hits 55.6)

CROSS-CAMPAIGN CAVEAT (encoded as a separate side note, NOT spliced into the line
above): the 2026-06-08 decode-6ms campaign reported 27 -> 15 ms/token / "39.9 ->
64.2 tok/s" via MTP depth-1 on a DIFFERENT base + spec config.
  Source: wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md (27 -> 15 ms),
          wins/2026-06-08-dsv4-mtp-batched-verify-lands.md ("39.9 | 64.2 | +61%").

Usage: python3 scripts/plot_dsv4_perf_journey.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/dsv4-perf-journey.png"

# House palette (matches scripts/plot_mlx_vs_arle_sweep.py)
BEFORE = "#868e96"   # legacy / regressed  (muted grey)
AFTER = "#d6336c"    # optimized           (ARLE magenta)
WIN = "#2f9e44"      # win / confirmed     (green)
INK = "#212529"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#495057",
    "axes.linewidth": 0.9,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
})

# Campaign milestones (chronological). Each (x_index, tok/s, lever, date, source-tag).
# tok/s values quoted verbatim from the cited wins/ entry (representative number).
steps = [
    (0, 33.5, "regressed\nbase", "06-13", BEFORE),
    (1, 36.9, "64-align\npacking fix", "06-13", AFTER),
    (2, 44.0, "FP8 decode-band\nMoE lane", "06-13", AFTER),
    (3, 44.5, "+ NUMA pin\n(all-threads)", "06-13", AFTER),
    (4, 53.3, "d2 chain-fold\nMTP (default-on)", "06-13", WIN),
]
xs = [s[0] for s in steps]
ys = [s[1] for s in steps]

fig, ax = plt.subplots(figsize=(10.0, 5.0))
fig.suptitle(
    "DeepSeek-V4-Flash — B=1 decode optimization journey  (8×H20, TP=8 / EP=8, FP8 MoE)",
    fontsize=13.0, fontweight="bold", y=0.985)

# --- step-line connecting the campaign milestones --------------------------------
ax.plot(xs, ys, "-", color="#ced4da", lw=2.0, zorder=1)
for x, y, lever, date, col in steps:
    ax.plot(x, y, "o", color=col, ms=11, zorder=3,
            markeredgecolor="white", markeredgewidth=1.3)
    # tok/s value above the marker
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=11, fontweight="bold", color=col, zorder=4)
    # lever name + tiny date: below for high markers, above the value for the two
    # lowest (no room below near the axis floor)
    if y < 40:
        ax.annotate(f"{lever} {date}", (x, y), textcoords="offset points",
                    xytext=(0, 26), ha="center", va="bottom", fontsize=8.0,
                    color=INK, zorder=4)
    else:
        ax.annotate(f"{lever}\n{date}", (x, y), textcoords="offset points",
                    xytext=(0, -32), ha="center", va="top", fontsize=8.0,
                    color=INK, zorder=4)

# --- delta callouts between consecutive steps ------------------------------------
deltas = [
    (0, 1, "+10%"),   # 33.5 -> 36.9
    (1, 2, "+19%"),   # 36.9 -> 44.0 (entry: "+19.5% over 64-align")
    (3, 4, "+18-20%"),  # 44.5 -> 53.3 (entry: "+18-20% over no-spec base")
]
for i, j, txt in deltas:
    xm = (steps[i][0] + steps[j][0]) / 2
    ym = (steps[i][1] + steps[j][1]) / 2
    ax.annotate(txt, (xm, ym), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.6, color=WIN, fontweight="bold",
                style="italic", zorder=4)

# --- 2026-06-14 confirmed-baseline acceptance band (49 .. 55.6) ------------------
# Source: wins/2026-06-14-dsv4-config-mechanism-classification.md
band_x = 5.05
ax.add_patch(plt.Rectangle((band_x - 0.12, 49.0), 0.24, 55.6 - 49.0,
             facecolor=WIN, alpha=0.16, edgecolor=WIN, lw=1.1, zorder=2))
ax.plot([band_x], [53.3], marker="_", ms=18, color=WIN, mew=2.0, zorder=3)
ax.annotate("06-14 confirmed\nacceptance band\n49 → 55.6 tok/s",
            (band_x, 55.6), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8.0, color=WIN, fontweight="bold", zorder=4)
ax.annotate("forward floor 42.0 ms/step;\ntok/s swings by MTP acceptance",
            (band_x, 49.0), textcoords="offset points", xytext=(0, -8),
            ha="center", va="top", fontsize=7.0, color="#495057",
            style="italic", zorder=4)

# --- faint reference: the "old 32.52 base" (not a campaign A/B point) -------------
ax.axhline(32.52, color="#ced4da", ls=":", lw=1.0, zorder=0)
ax.annotate("old 32.52 base (reference only — prior to this campaign)",
            (5.62, 32.52), textcoords="offset points", xytext=(0, 2),
            ha="right", va="bottom", fontsize=6.8, color="#adb5bd", style="italic")

ax.set_xlim(-0.55, 5.65)
ax.set_ylim(30.5, 63)
ax.set_xticks([])
ax.set_ylabel("B=1 decode throughput (tok/s)")
ax.grid(True, axis="y", alpha=0.22)

# --- cross-campaign caveat box (separate arc, NOT spliced into the line) ----------
caveat = ("Separate arc — 2026-06-08 decode-6ms campaign (different base + spec config):\n"
          "decode 27 → 15 ms/token via MTP depth-1 batched verify; \"39.9 → 64.2 tok/s\".\n"
          "Not continuous with the 33.5→53.3 line above — distinct harness/base/prompt.")
ax.text(0.015, 0.975, caveat, transform=ax.transAxes, ha="left", va="top",
        fontsize=6.9, color=INK,
        bbox=dict(boxstyle="round,pad=0.42", fc="#f1f3f5", ec="#adb5bd", lw=0.9))

fig.tight_layout(rect=(0, 0.0, 1, 0.95))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
