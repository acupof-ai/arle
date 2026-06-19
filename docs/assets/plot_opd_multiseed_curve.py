#!/usr/bin/env python3
"""OPD multi-seed capability curve — Qwen3.5-4B student ← Qwen3.6-35B-A3B teacher, MATH-500.
3 recipe arms × 5 seeds × {step25, step50}, each n=500 greedy @4096 tokens, 0 request_error.
Trajectory base(step0) → step25 → step50, error bars = ±1σ across seeds.
Regenerate: python3 docs/assets/plot_opd_multiseed_curve.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE, TEACHER = 0.518, 0.82
# per-seed accuracies: arm -> {step: [5 seeds]}
data = {
    "reverse-KL (greedy)":  ({25: [0.782, 0.796, 0.778, 0.790, 0.768], 50: [0.786, 0.790, 0.792, 0.794, 0.798]}, "#d62728", "s"),
    "forward-KL (greedy)":  ({25: [0.752, 0.786, 0.774, 0.678, 0.750], 50: [0.772, 0.778, 0.780, 0.786, 0.788]}, "#1f77b4", "o"),
    "stochastic (temp 0.9)":({25: [0.762, 0.770, 0.750, 0.710, 0.752], 50: [0.770, 0.788, 0.758, 0.786, 0.782]}, "#7f7f7f", "^"),
}

fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=140)
ax.axhline(TEACHER, ls="--", lw=1.4, color="#2ca02c", alpha=0.9)
ax.text(50.5, TEACHER + 0.004, "teacher 35B-A3B = 0.82", color="#2ca02c", fontsize=9, ha="right", va="bottom")
ax.axhline(BASE, ls=":", lw=1.3, color="#888888", alpha=0.9)
ax.text(50.5, BASE - 0.012, "base 4B = 0.518 (n=500)", color="#666666", fontsize=9, ha="right", va="top")

for name, (steps, color, mk) in data.items():
    xs = [0, 25, 50]
    ys = [BASE] + [np.mean(steps[s]) for s in (25, 50)]
    err = [0] + [np.std(steps[s], ddof=1) for s in (25, 50)]
    ax.errorbar(xs, ys, yerr=err, marker=mk, color=color, capsize=4, lw=2, ms=7,
                label=f"{name} → {ys[-1]:.3f}±{err[-1]:.3f}", elinewidth=1.4, alpha=0.95)

ax.annotate("+27.4 pp\n(reverse-KL)", xy=(50, 0.792), xytext=(38, 0.66), fontsize=9.5,
            color="#d62728", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

ax.set_xlabel("OPD training step")
ax.set_ylabel("MATH-500 accuracy (greedy, n=500/seed)")
ax.set_title("On-Policy Distillation: Qwen3.5-4B ← Qwen3.6-35B-A3B teacher\n"
             "3 recipe arms × 5 seeds — base 0.518 → reverse-KL 0.792 (error bars = ±1σ across seeds)", fontsize=11)
ax.set_xticks([0, 25, 50])
ax.set_xlim(-3, 53)
ax.set_ylim(0.46, 0.86)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
fig.tight_layout()
out = "docs/assets/opd-multiseed-curve.png"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
