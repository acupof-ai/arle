#!/usr/bin/env python3
"""Render the Route B vs full-logit V100 GPU memory chart for README.

Data source: bench-output/2026-05-26-opd-chunked-kl-route-b-* on V100
(see docs/experience/wins/2026-05-26-opd-chunked-kl-route-b-bench.md).
"""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "img"
    / "2026-05-26-opd-chunked-kl-route-b-v100-memory.png"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

modes = ["fullogit\n(T5b shape)", "windowed\n(Route B)"]
peak_mib = [31506, 20800]
colors = ["#c62828", "#2e7d32"]
verdicts = [
    "VRAM OOM:\ncuda alloc_zeros\nfailed (slice)",
    "fits, ~11 GB\nheadroom",
]

V100_CAP_MIB = 32 * 1024  # 32 GiB

fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)

x = list(range(len(modes)))
bw = 0.55
bars = ax.bar(x, peak_mib, width=bw, color=colors, edgecolor="black", linewidth=0.4)

ax.axhline(V100_CAP_MIB, color="#7a7a7a", linestyle="--", linewidth=1.0)
ax.text(
    1.45,
    V100_CAP_MIB + 250,
    "V100 32 GB capacity",
    fontsize=9,
    color="#555555",
    ha="right",
)

for xi, val, txt, c in zip(x, peak_mib, verdicts, colors):
    gib = val / 1024
    ax.text(xi, val + 600, f"{val:,} MiB\n({gib:.1f} GiB)", ha="center", va="bottom", fontsize=10, color=c, fontweight="bold")
    ax.text(xi, val / 2, txt, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

# annotate savings
ax.annotate(
    "−34 % peak GPU\n(−10 706 MiB)",
    xy=(1, peak_mib[1] + 200),
    xytext=(0.5, 28000),
    arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5),
    fontsize=10.5,
    color="#2e7d32",
    fontweight="bold",
    ha="center",
)

ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.set_ylabel("Peak GPU memory (MiB)", fontsize=10.5)
ax.set_ylim(0, V100_CAP_MIB * 1.07)
ax.set_yticks([0, 8192, 16384, 24576, 32768])
ax.set_yticklabels(["0", "8 GiB", "16 GiB", "24 GiB", "32 GiB"])
ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "OPD GKD chunked-KL Route B fits on V100; full-logit OOMs even at 32 GB\n"
    "Qwen3.5-4B teacher → 0.8B-Base student, 512-token corpus, rollout 8, gkd-lambda 0.3, sft-anchor corpus-truth",
    fontsize=10.5,
    loc="left",
    pad=14,
)

fig.text(
    0.012,
    -0.02,
    "Source: bench-output/2026-05-26-opd-chunked-kl-route-b-{wA-windowed-noeval, wB-fullogit-noeval} on V100 (32 GB SXM2, CUDA 12.4).  "
    "Same shape across rows; only --logits-window-size varied.  Train-step KL parity + wall-clock pending separate host-RAM fix.",
    fontsize=7.5,
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")
