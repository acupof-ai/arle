---
name: Metal Qwen3.5 concurrency sweep — c=3/c=5 valleys
description: HTTP throughput sweep c=1..8 in one thermal-matched session; c=3 and c=5 are regression points, c=2/4/6 are peaks. GDR kernel has ~2-wide batch granularity.
type: project
---

# Metal Qwen3.5-4B-MLX-4bit concurrency sweep — c=3 / c=5 are valleys

**Date**: 2026-04-19
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA)
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` (24 GDR + 8 full-attn)
**Binary**: `metal_serve` on HEAD `6b4f305`
**Bench**: `scripts/bench_throughput.py`, synthetic, N=24, max_tokens=128, temp=0
**Session**: single server, no restarts — thermal-matched

## Context

`2026-04-19-metal-qwen35-final-state.md` only measured c∈{1,2,4,8}. User
asked to fill in c=3. Bench revealed a non-linearity, so the full odd+even
sweep c=1..8 was run in the same session for apples-to-apples comparison.

## Results (thermal-matched, same server)

| c | tok/s | ITL p50 (ms) | per-req tok/s | ΔTP vs c-1 |
|---|-------|--------------|---------------|------------|
| 1 | 70.9  | 13.8         | 70.9          | —          |
| 2 | 130.2 | 14.3         | 65.1          | **+83.6%** |
| **3** | **140.4** | **20.0**     | 46.8          | **+7.8%** ← valley |
| 4 | 160.3 | 23.5         | 40.1          | +14.2%     |
| **5** | **149.4** | **25.6**     | 29.9          | **−6.8%** ← regression |
| 6 | 160.1 | 23.6         | 26.7          | +7.2%      |
| 7 | 159.2 | 23.7         | 22.7          | −0.6%      |
| 8 | 153.5 | 24.3         | 19.2          | −3.6%      |

## What Worked (findings)

1. **c=3 pays c=4 per-step cost for c=2 throughput.** Only +7.8% over c=2,
   while c=2→c=1 gives +84%. Per-request ITL jumps 14.3 → 20.0 ms (+40%).
2. **c=5 *regresses* below c=4.** 149.4 < 160.3. Odd counts between 4 and 6
   cost throughput.
3. **c=2/4/6 are the sweet spots.** Post-c=4, throughput saturates in a
   155 ± 5 tok/s band — the GDR-kernel compute ceiling reported in the
   final-state doc.
4. **Even/odd pattern suggests ~2-wide batch granularity** in the packed
   decode kernel. c=3 ≈ "c=4 with 1 idle lane"; c=5 ≈ "c=6 with 1 idle +
   setup-cost penalty."

## Implication for users

Published concurrency guidance should be **c ∈ {1, 2, 4, 6}** for this
model, not "any c up to 8." c=3, 5, 7 all land on plateaus or valleys.

c=1 stays the correct choice for strict latency (ITL 13.8 ms vs 20+ ms
at c≥3). c=4 is the throughput sweet spot (160 tok/s, ITL 23.5 ms).

## Rule

When publishing concurrent-decode throughput numbers, **sweep the full
range including odd c-values** — kernel batch-tile granularity creates
valleys at odd counts that power-of-2 benches miss entirely. This is
especially important for Qwen3.5 hybrid (GDR + full-attn) where the
GDR kernel dominates and has its own tile shape.

## Cross-references

- `2026-04-19-metal-qwen35-final-state.md` — c=1/2/4/8 baselines (matches this session within 2%)
- `2026-04-18-metal-qwen35-concurrent-decode-ceiling.md` — "GDR kernel 6.1 ms/row" ceiling analysis
