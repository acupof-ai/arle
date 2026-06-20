#!/usr/bin/env python3
"""Agentic OPD (think-on) capability curve — Qwen3.5-4B ← Qwen3.6-35B-A3B teacher, BFCL live.
Clean think-on eval (single-thread, 600s, 0 errors), 206 run-ids. base→step25→step50.
Headline: abstention (live_irrelevance) 0.60→1.00. step25 = sweet spot (+3.9pp agg);
step50 over-trains (over-thinking truncates tool-calls). Regenerate: python3 docs/assets/plot_agentic_thinkon_curve.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = [0, 25, 50]
# category: (accs over [base,25,50], n, color, lw)
cats = {
    "live_irrelevance (abstain)": ([0.60, 1.00, 1.00], 50, "#2ca02c", 3.0),
    "live_multiple":              ([0.86, 0.82, 0.80], 50, "#1f77b4", 1.3),
    "live_simple":                ([0.86, 0.74, 0.76], 50, "#17becf", 1.3),
    "live_parallel_multiple":     ([0.792,0.833,0.625],24, "#9467bd", 1.3),
    "live_relevance":             ([0.875,0.75, 0.625],16, "#ff7f0e", 1.3),
    "live_parallel":              ([0.812,0.625,0.50], 16, "#8c564b", 1.3),
}
# weighted aggregate over 206 run-ids
N = sum(n for _,n,_,_ in cats.values())
agg = [sum(a[i]*n for a,n,_,_ in cats.values())/N for i in range(3)]

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=140)
for name,(accs,n,color,lw) in cats.items():
    ax.plot(steps, accs, marker="o", color=color, lw=lw, ms=6 if lw>2 else 4,
            label=f"{name}", alpha=0.95, zorder=3 if lw>2 else 2)
ax.plot(steps, agg, marker="D", color="#111111", lw=2.4, ms=8, ls="--",
        label=f"AGGREGATE (n=206)", zorder=4)
for i,v in enumerate(agg):
    ax.text(steps[i], v-0.03, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

ax.axhline(0.93, ls=":", lw=1.1, color="#2ca02c", alpha=0.6)
ax.text(50.4, 0.93, "teacher think-on\nabstain 0.93", color="#2ca02c", fontsize=8, ha="right", va="center")
ax.annotate("abstention 0.60→1.00\n(was 0.00 in no-think arm)", xy=(25,1.0), xytext=(28,0.55),
            fontsize=9, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
ax.annotate("step25 sweet spot (+3.9pp);\nstep50 over-trains\n(over-thinking truncates tool-calls)", xy=(50,0.781),
            xytext=(6,0.45), fontsize=8.5, color="#444",
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))

ax.set_xlabel("OPD training step (think-on)")
ax.set_ylabel("BFCL live accuracy (clean think-on eval, n per category)")
ax.set_title("Agentic OPD (think-on): Qwen3.5-4B ← Qwen3.6-35B-A3B teacher\n"
             "abstention solved (0.60→1.00); aggregate base 0.786 → step25 0.825", fontsize=11)
ax.set_xticks(steps); ax.set_xlim(-3, 56); ax.set_ylim(0.35, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, framealpha=0.95)
fig.tight_layout()
out = "docs/assets/opd-agentic-thinkon-curve.png"
fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")
