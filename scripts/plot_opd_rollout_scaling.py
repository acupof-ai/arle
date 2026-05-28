#!/usr/bin/env python3
"""Plot the OPD student_rollout O(n²) scaling measurement.

Data from `runs/2026-05-28-rollout-scale-bench/run.log` + the v4
production point (`runs/2026-05-26-rollout128-v4-diverse1k-train-60/run.txt`,
warm step-2 average). Fit derived in
`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md`.

Output: docs/figures/2026-05-28-opd-rollout-scaling.png

Usage: python scripts/plot_opd_rollout_scaling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Measured (rollout_len, n_gen_warm, student_rollout_warm) from this session.
# n_gen = teacher_seq_len - prompt_len for step 2 (warm) of each sweep.
POINTS = [
    # (rollout_len, n_gen, student_rollout_seconds, label_for_chart)
    (8,   6.5,   2.45,  "rollout=8"),
    (16,  14.5,  6.30,  "rollout=16"),
    (32,  30.5,  19.26, "rollout=32"),
    (64,  64.0,  60.73, "rollout=64 (held-out)"),
    (128, 130.0, 208.0, "rollout=128 (v4 production)"),
]

# Fit derived from {8, 16, 32, 128} — rollout=64 was held-out validation.
FIT_A = 0.31     # linear coefficient (s/token)
FIT_B = 0.0099   # quadratic coefficient (s/token²)


def fit(n: float) -> float:
    return FIT_A * n + FIT_B * n * n


def main() -> int:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2026-05-28-opd-rollout-scaling.png"

    n_arr = [p[1] for p in POINTS]
    y_arr = [p[2] for p in POINTS]

    n_smooth = np.linspace(0, 150, 300)
    y_smooth = FIT_A * n_smooth + FIT_B * n_smooth ** 2
    y_linear_only = FIT_A * n_smooth
    y_quadratic_only = FIT_B * n_smooth ** 2

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8.5, 7.0), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True
    )

    # ── Top: measured points + fit + decomposition ─────────────────
    ax_top.plot(
        n_smooth, y_smooth, color="#D97757", linewidth=2.0,
        label=f"fit: {FIT_A:.2f}·n + {FIT_B:.4f}·n²",
    )
    ax_top.plot(
        n_smooth, y_linear_only, color="#888", linestyle="--", linewidth=1.0,
        label=f"linear term: {FIT_A:.2f}·n",
    )
    ax_top.plot(
        n_smooth, y_quadratic_only, color="#888", linestyle=":", linewidth=1.0,
        label=f"quadratic term: {FIT_B:.4f}·n²",
    )
    ax_top.scatter(
        n_arr, y_arr, color="#1F4D7A", s=80, zorder=5, label="measured (step-2 warm)",
    )
    # Per-point label placement avoids the bottom-left overlap (rollout=8/16
    # land within a few px of each other on the y axis).
    label_offsets = {
        "rollout=8":  (12, -6),
        "rollout=16": (12, 12),
        "rollout=32": (12, -2),
        "rollout=64 (held-out)": (12, -2),
        "rollout=128 (v4 production)": (-150, -16),
    }
    for n, y, label in [(p[1], p[2], p[3]) for p in POINTS]:
        dx, dy = label_offsets.get(label, (8, 4))
        ax_top.annotate(
            label, xy=(n, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color="#1F4D7A",
        )
    ax_top.set_ylabel("student_rollout (seconds per step)")
    ax_top.set_title(
        "OPD student rollout is O(n²) in rollout length\n"
        "Qwen3.5-0.8B-Base (LoRA) student, Qwen3.5-4B teacher, RTX 4070 Ti SUPER",
        fontsize=11,
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.set_xlim(0, 150)
    ax_top.set_ylim(0, 240)

    # Crossover annotation: where quadratic == linear
    n_cross = FIT_A / FIT_B  # ≈ 31.3
    ax_top.axvline(n_cross, color="#888", alpha=0.4, linewidth=0.8)
    ax_top.annotate(
        f"quadratic > linear\nabove n≈{n_cross:.0f}",
        xy=(n_cross, 25),
        xytext=(n_cross + 4, 30),
        fontsize=8.5,
        color="#555",
    )

    # ── Bottom: residuals ─────────────────────────────────────────
    residuals_pct = [100 * (y - fit(n)) / fit(n) for n, _, y, _ in [(p[1], None, p[2], p[3]) for p in POINTS]]
    ax_bottom.bar(
        n_arr, residuals_pct, width=4, color=["#1F4D7A" if abs(r) < 3 else "#D97757" for r in residuals_pct],
    )
    ax_bottom.axhline(0, color="black", linewidth=0.5)
    ax_bottom.set_xlabel("n_gen (tokens generated during rollout)")
    ax_bottom.set_ylabel("residual\n(measured − fit) / fit %")
    ax_bottom.set_ylim(-6, 6)
    ax_bottom.grid(True, alpha=0.3)
    for n, r in zip(n_arr, residuals_pct):
        ax_bottom.annotate(
            f"{r:+.1f}%", xy=(n, r), xytext=(0, 4 if r >= 0 else -12),
            textcoords="offset points", ha="center", fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
