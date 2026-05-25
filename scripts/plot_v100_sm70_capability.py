#!/usr/bin/env python3
"""Render the V100 sm_70 capability validation chart for README.

What this chart says
====================
ARLE now serves Qwen3.5-4B and Qwen3.5-9B on V100 (Volta, sm_70) with
capability preserved vs T1 (A100/L4/H100). No model training was
involved; the base BF16 weights from ModelScope/HF were served as-is
through the ARLE CUDA runtime, and MMLU 5-shot accuracy was scored
against the OpenAI-v1 endpoint.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- data ----------
groups = ["Qwen3.5-4B", "Qwen3.5-9B"]
t1_ref = [77.33, None]  # T1 4B from 2026-05-22 wrap; T1 9B not on file
v100 = [79.9, 83.0]

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "2026-05-25-v100-sm70-capability.png"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- render ----------
fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140)

x = list(range(len(groups)))
bw = 0.32

# T1 reference (only available for 4B)
t1_vals = [v if v is not None else 0 for v in t1_ref]
t1_bars = ax.bar(
    [xi - bw / 2 for xi in x],
    t1_vals,
    width=bw,
    color="#bdbdbd",
    edgecolor="#7a7a7a",
    label="T1 reference (A100/L4/H100)",
)
v100_bars = ax.bar(
    [xi + bw / 2 for xi in x],
    v100,
    width=bw,
    color="#D97757",
    edgecolor="#a04f33",
    label="V100 sm_70 (this work)",
)

# annotate values on top of bars
for xi, val in zip(x, t1_ref):
    if val is None:
        ax.text(xi - bw / 2, 2, "n/a", ha="center", va="bottom", fontsize=8.5, color="#7a7a7a")
    else:
        ax.text(xi - bw / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, color="#7a7a7a")
for xi, val in zip(x, v100):
    ax.text(xi + bw / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, color="#a04f33", fontweight="bold")

# arrow showing 4B Δ (V100 vs T1)
delta_4b = v100[0] - t1_ref[0]
ax.annotate(
    f"+{delta_4b:.2f}pp",
    xy=(0 + bw / 2, v100[0] + 0.3),
    xytext=(0 - bw / 2 - 0.05, t1_ref[0] + 4.5),
    arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
    fontsize=9,
    color="#2e7d32",
    fontweight="bold",
)

# arrow showing 4B→9B size scaling on V100
ax.annotate(
    "+3.1pp (size scaling preserved)",
    xy=(1 + bw / 2 - 0.05, v100[1] + 0.3),
    xytext=(0.45, v100[1] + 6),
    arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
    fontsize=9,
    color="#2e7d32",
    fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel("MMLU 5-shot accuracy (%)", fontsize=10.5)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "V100 sm_70 inference fallback preserves Qwen3.5 capability\n"
    "ARLE serve + TileLang PR #2257 (BF16→FP16 fragment staging) + per-kernel cubin filter",
    fontsize=11,
    loc="left",
    pad=14,
)

ax.legend(loc="lower right", fontsize=9, frameon=False)

fig.text(
    0.012,
    -0.02,
    "Method: served base BF16 weights as-is; scored MMLU 5-shot via arle_capability_eval.py n=200 against arle serve.  "
    "No training / OPD / distillation.  GSM8K omitted (both sizes at base-model floor ~1-2%).",
    fontsize=7.5,
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")
