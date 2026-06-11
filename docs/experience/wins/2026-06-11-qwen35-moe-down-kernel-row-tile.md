# Qwen3.5/3.6 MoE down kernel 4-row warp tile — +6.6% c=1, +11% c≥2

**Date:** 2026-06-11. **Commit:** `30611ad4`. **Backend:** CUDA H20,
Qwen3.6-35B-A3B. **Status: LICENSED** (c≥2 clears the +10% bar; c=1 +6.6% >
2σ; no axis regressed; no tradeoff — same launch count, same numerics
contract, +30 LOC tile loop).

## Context

Post-license re-profile
(`docs/reviews/2026-06-11-qwen35-post-license-reprofile-rerank.md`) split the
licensed MoE decode pair: swiglu healthy (32.9 µs/layer, 25% of HBM peak),
down broken (69.3 µs/layer for half the bytes, 6% of peak, σ 0.6%). Binding
constraint: K=512 → 2 v-loop iterations/lane → one-row-per-warp keeps ~2
weight loads in flight. Fix: each warp owns a 4-row weight tile
(`MOE_DECODE_ROW_TILE=4`), activation vector loaded once per v-iteration and
reused across the tile. Per-element reduction order unchanged (same lane→k
mapping, same butterfly).

## Results (pod A/B, same harness, sequential same-session, n=3 at c=1)

| arm | c=1 tok/s (×3) | c=2 agg | c=4 agg |
|---|---|---:|---:|
| baseline `9e37bc77` | 90.4 / 92.1 / 91.0 (μ 91.2, σ 1%) | 141.1 | 138.1 |
| downtile `30611ad4` | 96.6 / 97.8 / 97.3 (μ 97.2, σ <1%) | 156.8 | 153.8 |
| Δ | **+6.6%** | **+11.1%** | **+11.4%** |

Mechanism (nsys, 64-token decode): down 69.3 → **45.7 µs/layer (−34%)**;
swiglu control byte-unchanged (32.9 µs). Savings 23.6 µs × 40 layers =
0.94 ms/token, consistent with the wall delta. Needle gate ×2 PASS
(QZK-7341 retrieved), same-config-twice byte-identical tails.
Campaign cumulative: 36.0 → **97.2 tok/s single-stream (2.70×)**.

c=2/c=4 aggregates here are NOT comparable to the 142/185 in
`2026-06-11-qwen35-decode-moe-kernel-licensed.md` — different driver (inline
ThreadPool wall-clock vs the sweep harness); within-table Δ is the claim.

## Formula post-mortem (predicted +18–20%, measured +6.6%)

The MLP model assumed in-flight bytes/SM scale ×4 with the row tile. Wrong:
grid.x shrank ×4, dropping resident warps from the 64/SM occupancy cap to
~53/SM — per-SM in-flight grew ~3×, and the kernel only reached 9% of peak
(369 GB/s), not swiglu's 25%. The per-warp MLP model must multiply by
resident warps, and something beyond MLP still binds (L2 latency on the
1-KB-strided 4-row pattern is the open hypothesis). Remaining headroom
~25–40 µs/layer (~1–1.6 ms/token) — parked: prefill levers (attention 42%,
GDR 28%) dominate the next marginal hour.

## Rule

- An MLP formula that changes the grid must account for resident-warp count,
  not just per-warp loads in flight: total in-flight = warps/SM × per-warp,
  and tiling trades one for the other.
- Keep the sibling kernel as the in-run control: swiglu byte-identical at
  32.9 µs across both arms validates the harness before crediting the delta.
