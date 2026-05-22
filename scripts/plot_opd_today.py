#!/usr/bin/env python3
"""Generate the 2026-05-22 OPD cycle headline chart for README.

Two panels:
  Left  — OPD distill trajectory on 4B → 0.8B (U-curve valley + recovery)
          + base / teacher floor / ceiling, + held-out KL on second axis.
  Right — Cross-engine validation: ARLE serve vs HF transformers on the
          same Qwen3.5-4B checkpoint (statistical equivalence).

Output: docs/projects/img/2026-05-22-arle-opd-distill-trajectory.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs/projects/img/2026-05-22-arle-opd-distill-trajectory.png"


def panel_trajectory(ax):
    steps = np.array([0, 500, 1000, 2000])
    # MMLU accuracy (%): base=51.4, step1000=47.9, step2000=50.0 measured;
    # step 500 not eval'd → interpolate as the segment between 51.4 → 47.9.
    mmlu_distill = np.array([51.4, np.nan, 47.9, 50.0])
    base_floor = 51.4
    teacher_ceiling = 77.33

    # KL trajectory (×1e-5): heldout
    kl_heldout = np.array([1.739, 1.606, 1.598, 1.599])

    # MMLU axis
    ax.set_xlim(-50, 2150)
    ax.set_ylim(45, 80)
    ax.set_xlabel("training step")
    ax.set_ylabel("MMLU 5-shot accuracy (%)", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")

    # Teacher ceiling line
    ax.axhline(teacher_ceiling, color="#3a7d3a", linestyle=":", linewidth=2, label=f"Teacher 4B ceiling = {teacher_ceiling:.1f}%")
    # Base floor line
    ax.axhline(base_floor, color="#999999", linestyle="--", linewidth=2, label=f"Base 0.8B floor = {base_floor:.1f}%")

    # MMLU distill points (only at measured steps)
    mask = ~np.isnan(mmlu_distill)
    ax.plot(steps[mask], mmlu_distill[mask], "o-", color="#1f4e79", linewidth=2.5, markersize=10,
            label="Distilled 0.8B (lr=2e-5)")

    # Annotations
    ax.annotate("base 51.4%\n(starting point)", xy=(0, 51.4), xytext=(150, 53.5),
                fontsize=9, ha="left", color="#1f4e79",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8))
    ax.annotate("valley\n-3.5pp", xy=(1000, 47.9), xytext=(700, 46),
                fontsize=9, ha="center", color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax.annotate("recovering\n+2.1pp", xy=(2000, 50.0), xytext=(1650, 53),
                fontsize=9, ha="center", color="#2980b9", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2980b9", lw=0.8))
    # Gap annotation
    ax.annotate("",  xy=(2100, 77.33), xytext=(2100, 50.0),
                arrowprops=dict(arrowstyle="<->", color="#3a7d3a", lw=1.5))
    ax.text(2120, 63.6, "+27.3pp\ngap to\nclose", fontsize=9, color="#3a7d3a", va="center")

    # KL secondary axis
    ax2 = ax.twinx()
    ax2.set_ylim(1.55, 1.78)
    ax2.set_ylabel("held-out KL ×1e-5", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.plot(steps, kl_heldout, "s--", color="#8e44ad", linewidth=1.5, markersize=7, alpha=0.7,
             label="KL held-out")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8.5, framealpha=0.9)

    ax.set_title("OPD distill trajectory: U-curve valley → recovery\n4B teacher → 0.8B LoRA student, 2 000 steps, lr=2e-5",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def panel_cross_validation(ax):
    labels = ["ARLE\nserve", "HF\ntransformers"]
    accuracies = [77.33, 78.18]
    invalid_pct = [12.3, 3.6]
    colors = ["#1f4e79", "#e67e22"]

    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", fontsize=11, fontweight="bold")
    for i, (acc, inv) in enumerate(zip(accuracies, invalid_pct)):
        ax.text(i, acc / 2, f"invalid\n{inv:.1f}%", ha="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("MMLU 5-shot accuracy (%)")
    ax.set_ylim(0, 95)
    ax.axhline(0, color="black", linewidth=0.5)
    # Add a delta annotation
    ax.annotate("", xy=(1, 78.18), xytext=(0, 77.33),
                arrowprops=dict(arrowstyle="<->", color="#666666", lw=1))
    ax.text(0.5, 82, "Δ = +0.85 pp\n(within ±5 pp 95 % CI)",
            ha="center", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f4e6", edgecolor="#cccccc"))
    ax.set_title("Cross-engine validation: ARLE serve ≈ HF transformers\nsame Qwen3.5-4B checkpoint, MMLU 5-shot, n=171",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    panel_trajectory(axes[0])
    panel_cross_validation(axes[1])

    fig.suptitle("ARLE OPD cycle 2026-05-22 — train→save→load→eval pipeline closed, GDR prefill bug fixed",
                 fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
