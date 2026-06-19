#!/usr/bin/env python3
"""OPD multi-seed lock — Qwen3.5-4B student ← Qwen3.6-35B-A3B teacher, MATH-500.
3 recipe arms × 5 seeds × step50, each n=500 greedy @4096 tokens, 0 request_error.
Regenerate: python3 docs/assets/plot_opd_multiseed_curve.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE, TEACHER = 0.518, 0.82
# per-seed step50 accuracies (n=500 each)
arms = [
    ("reverse-KL\n(greedy)", [0.786, 0.790, 0.792, 0.794, 0.798], "#d62728"),
    ("forward-KL\n(greedy)", [0.772, 0.778, 0.780, 0.786, 0.788], "#1f77b4"),
    ("stochastic\n(temp 0.9)", [0.758, 0.770, 0.786, 0.788], "#7f7f7f"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=140)
ax.axhline(TEACHER, ls="--", lw=1.4, color="#2ca02c", alpha=0.9)
ax.text(2.62, TEACHER + 0.004, "teacher 35B-A3B = 0.82", color="#2ca02c", fontsize=9, ha="right", va="bottom")
ax.axhline(BASE, ls=":", lw=1.3, color="#888888", alpha=0.9)
ax.text(2.62, BASE - 0.012, "base 4B = 0.518 (n=500)", color="#666666", fontsize=9, ha="right", va="top")

for i, (name, accs, color) in enumerate(arms):
    accs = np.array(accs)
    m, sd = accs.mean(), accs.std(ddof=1)
    ax.scatter(np.full_like(accs, i) + np.linspace(-0.06, 0.06, len(accs)), accs,
               s=26, color=color, alpha=0.55, zorder=3)
    ax.errorbar(i, m, yerr=sd, marker="D", ms=10, color=color, capsize=5, lw=2, elinewidth=1.6, zorder=4)
    ax.text(i, m + sd + 0.006, f"{m:.3f}±{sd:.3f}", color=color, fontsize=9, ha="center", va="bottom", fontweight="bold")

ax.annotate("", xy=(0, BASE + 0.01), xytext=(0, 0.788),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.3, alpha=0.7))
ax.text(0.10, (BASE + 0.792) / 2, "+27.4 pp\nCI-separated", color="#d62728", fontsize=9.5, va="center", fontweight="bold")

ax.set_xticks(range(len(arms)))
ax.set_xticklabels([a[0] for a in arms], fontsize=10)
ax.set_ylabel("MATH-500 accuracy (greedy, n=500/seed)")
ax.set_title("On-Policy Distillation lock: Qwen3.5-4B ← Qwen3.6-35B-A3B teacher\n"
             "5 seeds/arm, step50 — base 0.518 → reverse-KL 0.792 (error bars = ±1σ across seeds)", fontsize=11)
ax.set_xlim(-0.5, 2.7)
ax.set_ylim(0.46, 0.86)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
out = "docs/assets/opd-multiseed-curve.png"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
