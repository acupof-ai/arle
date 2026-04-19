---
name: Qwen3.5 tg_y=2 c=3/c=5 wins failed to reproduce
description: Two 2026-04-19 tg_y-class wins (c=3 +10%, c=5 +5.8%) from single-session sweeps both collapsed to 0% under matched same-binary env A/B in a second session. GDR decode default tg_y=4 stays. Originating wins docs deleted; this errors doc is the record.
type: project
---

# GDR tg_y=2 wins (c=3, then c=5) failed to reproduce

**Date**: 2026-04-19
**Hot-path file**: `crates/mlx-sys/src/mlx_qwen35_model.cpp` — no change landed.

## Context

A post-archival /loop iter flagged c=3 scaling at only +7.8% over c=2 and
c=5 regressing below c=4 in a single consecutive c=1..8 HTTP sweep. A
three-way tg_y=2/4/8 bench then seemed to show tg_y=2 as a +10% win at
c=3 and +5% at c=5 over default tg_y=4.

## What happened

### Session 1 — c=3 claim died

After landing a parity-adaptive tg_y change and rebuilding, c=3 throughput
came in *below* the prior tg_y=4 baseline (136.7 vs 140.4). I forced
env=4 vs env=2 on the same binary, back-to-back, matched N=24/max=128:

| c | env=4 | env=2 | Δ        |
|---|-------|-------|----------|
| 3 | 152.9 | 152.6 | ~noise   |
| 4 | 154.4 | 154.3 | ~noise   |
| 5 | 150.8 | 159.6 | **+5.8%** |

c=3 at env=4 jumped from 140 → 153 — the prior "140" was a thermal low
point, not a peak. c=4's 160 peak also dropped to 154. The code change
was reverted.

### Session 2 — c=5 claim also died

Fresh server restart, matched env=4 vs env=2 at c=5, same N/max:

| session | env=4 | env=2 | Δ at c=5 |
|---------|-------|-------|----------|
| 1       | 150.8 | 159.6 | +5.8%    |
| 2       | 145.9 | 145.9 | **0.0%** |

Zero delta in session 2. No tg_y signal survives at any c.

## Root Cause

The single-pass c=1..8 sweep was consecutive bench invocations (~27 s each,
~4 minutes total) on one server. GPU thermal state drifts monotonically
across the sweep. The apparent "valleys" at c=3/c=5 vs "peaks" at c=4/c=6
lined up with where the GPU happened to cool between runs or heat during
them — an artifact of **sweep ordering × thermals**, not a GDR-kernel
wave-packing structure.

## Fix

1. Reverted the parity-adaptive code change in
   `crates/mlx-sys/src/mlx_qwen35_model.cpp` — never landed.
2. Deleted the two originating wins docs
   (`2026-04-19-metal-qwen35-concurrency-sweep-c3-c5-valleys.md`,
   `2026-04-19-metal-qwen35-gdr-tgy-sweep.md`). This errors doc is the
   canonical record of the investigation.
3. GDR decode default `tg_y=4` stays. `AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y`
   env override remains for future exploration.

## Rule

**For c-sweep Metal benches, a single consecutive pass is thermally biased.**
Effects ≤10% need reproduction by matched same-binary env A/B in ≥2
independent sessions before landing a code change. Specifically:

- Don't trust a per-c "valley" or tg_y-class win from one sweep.
- Use N ≥ 24 prompts and at least 2 independent server sessions.
- Always A/B on the *same binary* (env-override path), not across builds —
  build-to-build throughput variance exceeds the effects being chased.
- Prior-session numbers are NOT a valid baseline for current claims —
  thermal state decays across hours.

Captured as `feedback_matched_ab_for_small_bench_effects.md`.
