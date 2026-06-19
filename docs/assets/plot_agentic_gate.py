#!/usr/bin/env python3
"""Agentic gate figure — BFCL single-turn (no-think), base Qwen3.5-4B landscape +
the Qwen3.6-35B-A3B teacher's headroom on realistic live queries.
The 4B saturates synthetic AST (no OPD room) but is weak on realistic live tool-use,
where the teacher has +42pp — the agentic OPD target. (OPD student lift: in flight.)
Regenerate: python3 docs/assets/plot_agentic_gate.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, base acc, group) — base single-turn no-think, n per category
cats = [
    ("Non-Live Simple (n=400)",   0.930, "ast"),
    ("Non-Live Multiple (n=200)", 0.925, "ast"),
    ("Non-Live Parallel (n=200)", 0.870, "ast"),
    ("Live Relevance (n=16)",     0.750, "live"),
    ("Live Parallel (n=16)",      0.750, "live"),
    ("Live Simple (n=50)",        0.700, "live"),
    ("Live Multiple (n=50)",      0.700, "live"),
    ("Live Irrelevance (n=50)",   0.460, "room"),
    ("Non-Live Irrelevance (n=240)", 0.279, "room"),
]
TEACHER_LIVE_IRREL = 0.882  # 15/17

color = {"ast": "#9aa7b4", "live": "#1f77b4", "room": "#d62728"}
fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=140)
ys = list(range(len(cats)))[::-1]
for y, (lbl, acc, g) in zip(ys, cats):
    ax.barh(y, acc, color=color[g], alpha=0.9, height=0.62, zorder=3)
    ax.text(acc + 0.008, y, f"{acc:.2f}", va="center", fontsize=8.5, color="#333")

# teacher headroom marker on Live Irrelevance
li_y = ys[7]
ax.barh(li_y, TEACHER_LIVE_IRREL, color="none", edgecolor="#2ca02c", lw=1.8, height=0.62, zorder=4, linestyle="--")
ax.plot(TEACHER_LIVE_IRREL, li_y, marker="D", ms=9, color="#2ca02c", zorder=5)
ax.annotate("teacher 35B = 0.88\n+42pp → OPD target",
            xy=(TEACHER_LIVE_IRREL, li_y), xytext=(0.66, li_y - 1.45), fontsize=9, color="#2ca02c",
            fontweight="bold", ha="center", arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3))

ax.set_yticks(ys)
ax.set_yticklabels([c[0] for c in cats], fontsize=9)
ax.set_xlabel("BFCL accuracy (base Qwen3.5-4B, single-turn, no-think)")
ax.set_xlim(0, 1.0)
ax.axvline(0.85, ls=":", color="#888", lw=1)
ax.text(0.855, ys[0] + 0.2, "saturated →\nno OPD room", fontsize=8, color="#666", va="bottom")
ax.set_title("Agentic gate: where can OPD lift the 4B?  (BFCL single-turn)\n"
             "synthetic AST is saturated; realistic Live queries are the headroom (teacher +42pp)", fontsize=11)
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#9aa7b4", label="synthetic AST — saturated"),
                   Patch(color="#1f77b4", label="realistic Live — partial room"),
                   Patch(color="#d62728", label="irrelevance — most room"),
                   Patch(facecolor="none", edgecolor="#2ca02c", label="teacher 35B (measured)")],
          loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, framealpha=0.95)
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout()
out = "docs/assets/opd-agentic-gate.png"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
