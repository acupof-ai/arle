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
    steps = np.array([0, 1000, 2000])
    mmlu_lr2e5 = np.array([51.4, 47.9, 50.0])
    mmlu_lr1e5 = np.array([51.4, 50.6, 48.5])
    base_floor = 51.4
    teacher_ceiling = 77.33

    ax.set_xlim(-50, 2200)
    ax.set_ylim(45, 80)
    ax.set_xlabel("training step")
    ax.set_ylabel("MMLU 5-shot accuracy (%)")

    # Teacher ceiling + base floor
    ax.axhline(teacher_ceiling, color="#3a7d3a", linestyle=":", linewidth=2,
               label=f"Teacher 4B ceiling = {teacher_ceiling:.1f}%")
    ax.axhline(base_floor, color="#999999", linestyle="--", linewidth=2,
               label=f"Base 0.8B floor = {base_floor:.1f}%")

    # lr=2e-5 (deeper valley + recovery)
    ax.plot(steps, mmlu_lr2e5, "o-", color="#1f4e79", linewidth=2.5, markersize=10,
            label="lr=2e-5 (deep valley → recovery)")
    # lr=1e-5 (shallow valley + regression)
    ax.plot(steps, mmlu_lr1e5, "s-", color="#c0392b", linewidth=2.5, markersize=10,
            label="lr=1e-5 (shallow valley → REGRESSION)")

    # Annotations on key points
    ax.annotate("lr=2e-5 valley\n47.9%", xy=(1000, 47.9), xytext=(700, 46),
                fontsize=9, ha="center", color="#1f4e79",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8))
    ax.annotate("lr=2e-5 recovery\n50.0%", xy=(2000, 50.0), xytext=(2000, 53),
                fontsize=9, ha="center", color="#1f4e79", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8))
    ax.annotate("lr=1e-5 shallow\nvalley 50.6%", xy=(1000, 50.6), xytext=(550, 56),
                fontsize=9, ha="center", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax.annotate("lr=1e-5 REGRESSED\n48.5%", xy=(2000, 48.5), xytext=(1500, 45.5),
                fontsize=9, ha="center", color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    # Capability gap to close
    ax.annotate("", xy=(2150, 77.33), xytext=(2150, 50.0),
                arrowprops=dict(arrowstyle="<->", color="#3a7d3a", lw=1.5))
    ax.text(2170, 63.6, "+27.3pp\ngap\nto close", fontsize=9, color="#3a7d3a", va="center")

    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.set_title("OPD distill trajectory: lr sweep shows valley is NOT just lr-driven\n4B teacher → 0.8B LoRA student, 2 000 steps",
                 fontsize=10.5, fontweight="bold")
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
