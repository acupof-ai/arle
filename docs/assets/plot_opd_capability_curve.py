#!/usr/bin/env python3
"""OPD capability curve — Qwen3.5-4B student distilled from Qwen3.6-35B-A3B teacher.
MATH-500 greedy exact-match @4096 tokens, 0 request_error (retry-clean eval);
student ckpts n=100/point, anchors base n=40 / teacher n=50.
Single-seed bring-up run (2026-06-19). Regenerate: python3 docs/assets/plot_opd_capability_curve.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (step, acc, ci_lo, ci_hi)
baseline = [(0, 0.60, 0.45, 0.74), (25, 0.78, 0.689, 0.850), (50, 0.80, 0.711, 0.867), (75, 0.77, 0.678, 0.842)]
reverse  = [(0, 0.60, 0.45, 0.74), (25, 0.78, 0.689, 0.850), (50, 0.75, 0.657, 0.825)]
BASE_ACC, TEACHER_ACC = 0.60, 0.82

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)

# ceiling + floor reference lines
ax.axhline(TEACHER_ACC, ls="--", lw=1.4, color="#2ca02c", alpha=0.9)
ax.text(76, TEACHER_ACC + 0.006, "teacher 35B-A3B = 0.82", color="#2ca02c", fontsize=9, ha="right", va="bottom")
ax.axhline(BASE_ACC, ls=":", lw=1.2, color="#888888", alpha=0.9)
ax.text(76, BASE_ACC - 0.018, "base 4B = 0.60", color="#666666", fontsize=9, ha="right", va="top")

for name, data, color, mk in [("forward-KL (greedy)", baseline, "#1f77b4", "o"),
                              ("reverse-KL (greedy)", reverse, "#d62728", "s")]:
    xs = [d[0] for d in data]; ys = [d[1] for d in data]
    lo = [d[1] - d[2] for d in data]; hi = [d[3] - d[1] for d in data]
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=color, capsize=3, lw=2,
                ms=7, label=name, elinewidth=1, alpha=0.95)

# annotate the headline lift
ax.annotate("+18pp in 25 steps\n(~82% of base→teacher gap)",
            xy=(25, 0.78), xytext=(34, 0.66), fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))

ax.set_xlabel("OPD training step")
ax.set_ylabel("MATH-500 accuracy (greedy, n=100)")
ax.set_title("On-Policy Distillation: Qwen3.5-4B student ← Qwen3.6-35B-A3B teacher\n"
             "(single-seed bring-up, 2026-06-19; error bars = Wilson 95% CI)", fontsize=11)
ax.set_xlim(-3, 80); ax.set_ylim(0.40, 0.92)
ax.set_xticks([0, 25, 50, 75])
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
out = "docs/assets/opd-capability-curve.png"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
