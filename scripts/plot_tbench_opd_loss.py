#!/usr/bin/env python3
"""TB-OPD distill loss curve: the +5.1pp Terminal-Bench run (41 records × 3 epochs).
Per-step masked-CE loss (light) + EMA trend (bold) + per-epoch means. Real data
from pod `/host/tb_distill.log`. -> docs/assets/tbench-opd-loss-curve.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-step masked-CE loss, 41 records/epoch × 3 epochs (pod tb_distill.log)
STEPS = [
    0.109966,0.393166,0.098294,0.309965,0.151355,0.214967,0.068518,0.208573,0.177254,0.307647,
    0.383780,0.298524,0.354334,0.066735,0.383964,0.037783,0.055860,0.147447,0.170871,0.115744,
    0.351565,0.386141,0.388251,0.368012,0.050061,0.155664,0.211472,0.141866,0.218666,0.340489,
    0.144823,0.204356,0.322525,0.076233,0.096270,0.169610,0.183097,0.196764,0.370549,0.167333,0.278694,
    0.098890,0.337900,0.087984,0.261815,0.127258,0.197637,0.062152,0.191571,0.149294,0.267084,
    0.328802,0.250116,0.296294,0.064251,0.318709,0.033693,0.047788,0.123221,0.144638,0.102356,
    0.294697,0.323951,0.306742,0.277459,0.048032,0.127323,0.169029,0.127654,0.187041,0.285593,
    0.116069,0.145909,0.244969,0.072370,0.086696,0.111945,0.158488,0.157922,0.290304,0.115824,0.225324,
    0.084250,0.276568,0.072546,0.200440,0.104216,0.158484,0.051021,0.161306,0.114486,0.205894,
    0.262541,0.180848,0.224819,0.061072,0.202470,0.028812,0.037745,0.101106,0.107089,0.091167,
    0.240990,0.283175,0.231864,0.217663,0.046674,0.114170,0.154359,0.118456,0.140657,0.241177,
    0.078740,0.106377,0.188776,0.067932,0.081819,0.092513,0.127007,0.132253,0.263337,0.097954,0.204244,
]
EPOCH_MEANS = [0.2165, 0.1796, 0.1453]
EP = len(STEPS) // 3  # 41

ema, a, e = [], 0.15, STEPS[0]
for v in STEPS:
    e = a * v + (1 - a) * e
    ema.append(e)

fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=140)
x = range(1, len(STEPS) + 1)
ax.plot(x, STEPS, color="#c7d2fe", lw=1.0, label="per-step masked-CE")
ax.plot(x, ema, color="#4f46e5", lw=2.2, label="EMA trend (α=0.15)")
for i, m in enumerate(EPOCH_MEANS):
    xc = i * EP + EP / 2
    ax.plot(xc, m, "o", color="#dc2626", ms=7, zorder=5)
    ax.annotate(f"epoch {i}\nmean {m:.4f}", (xc, m), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8, color="#dc2626")
    if i:
        ax.axvline(i * EP + 0.5, color="#e5e7eb", lw=1, ls="--", zorder=0)

ax.set_title("Terminal-Bench OPD distill loss — 27B student, +5.1pp run",
             fontsize=11, weight="bold")
ax.set_xlabel("training step (41 records × 3 epochs)", fontsize=9)
ax.set_ylabel("masked-CE loss", fontsize=9)
ax.set_ylim(0, 0.45)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("docs/assets/tbench-opd-loss-curve.png", bbox_inches="tight")
print("wrote docs/assets/tbench-opd-loss-curve.png  epoch means:", EPOCH_MEANS)
