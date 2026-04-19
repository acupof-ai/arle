---
name: tg_y=2 c=3 win failed to reproduce
description: The c=3 valley in the concurrency-sweep doc was thermal/noise; matched N=24 A/B on the same binary shows env=4 and env=2 land within 0.2% at c=3. Only c=5 still benefits modestly from tg_y=2. Parity-adaptive change was reverted before shipping.
type: project
---

# GDR tg_y=2 "c=3 win" failed to reproduce under a matched A/B

**Date**: 2026-04-19
**Relates to**:
- `docs/experience/wins/2026-04-19-metal-qwen35-gdr-tgy-sweep.md` (findings now in question)
- `docs/experience/wins/2026-04-19-metal-qwen35-concurrency-sweep-c3-c5-valleys.md` (source of the c=3 "140.4" baseline)

## Context

After the first /loop iteration flagged c=3 scaling at only +7.8% over c=2
and c=5 regressing below c=4, a three-way sweep at tg_y=2/4/8 seemed to
show tg_y=2 as a ~+10% win at c=3 and ~+5% at c=5. The next /loop iter
attempted to land a parity-adaptive change in
`crates/mlx-sys/src/mlx_qwen35_model.cpp:79` — but it FAILED to reproduce.

## What happened

After shipping the code change and rebuilding, the new binary's c=3
throughput came in *below* the prior tg_y=4 baseline (136.7 vs 140.4).
To isolate thermal drift from the code change, I forced env=4 and env=2
on the same binary, back-to-back, matched N=24/max_tokens=128:

| c | env=4 this session | env=2 this session | Δ        | env=4 prior | env=2 prior |
|---|--------------------|--------------------|----------|-------------|-------------|
| 3 | 152.9              | 152.6              | ~noise   | 140.4       | 154.4       |
| 4 | 154.4              | 154.3              | ~noise   | 160.3       | 157.1       |
| 5 | 150.8              | 159.6              | **+5.8%** | 149.4       | 156.4       |

Three things stand out:

1. **c=3 at env=4 jumped from 140 → 153.** The prior "140" was a low point,
   not a peak. Under thermal stability the two tg_y values converge at c=3.
2. **c=4 peak at env=4 dropped from 160 → 154.** The prior 160 was itself
   a thermal-peak reading, not the steady-state ceiling.
3. **c=5 tg_y=2 win persists** (+5.8%) across both sessions — this one signal
   is probably real, but too small to justify a one-off code path on its own.

### Update — session 2 (same day) kills c=5 too

Re-ran matched env=4 vs env=2 at c=5 in a fresh server-restart session
(N=24, max=128, same binary):

| session | env=4 | env=2 | Δ at c=5 |
|---------|-------|-------|----------|
| 1       | 150.8 | 159.6 | **+5.8%** |
| 2       | 145.9 | 145.9 | **0.0%**  |

Session 2 shows **zero** difference — the session-1 tg_y=2 advantage at c=5
was also thermal noise. No signal remains across any c. The entire
tg_y-class investigation closes: **GDR decode default tg_y=4 is fine**.
No code change. This confirms the feedback-memory rule —
effects ≤10% in a single matched A/B need ≥2 independent sessions, and
session 2 here did the disproving work.

## Root Cause

The single-session c=1..8 sweep in `concurrency-sweep-c3-c5-valleys.md`
was run as **consecutive bench invocations** (~27 s each) on one server.
GPU thermal state drifts monotonically across a ~4-minute sweep. The
"valleys" at c=3/c=5 vs "peaks" at c=4/c=6 line up with where the GPU
happened to cool between runs (after longer c≥5 benches) or heat during
runs. The apparent pattern is an artifact of **sweep ordering × thermals**,
not a GDR-kernel wave-packing structure.

## Fix

1. **Reverted the parity-adaptive code change** in
   `crates/mlx-sys/src/mlx_qwen35_model.cpp` — never landed.
2. Leave the two wins docs in place as a record of the investigation, but
   this errors doc is the canonical interpretation; anyone reading those
   wins docs should follow the cross-reference here.

## Rule

**For c-sweep benches, a single consecutive pass is thermally biased.**
Before declaring a per-c "valley" or a tg_y-class win, re-run the A/B
back-to-back on the SAME server with the variable you're isolating, at
matched N and max_tokens. If the effect is ≤10% and doesn't survive a
matched A/B, it was noise.

Specifically:
- Don't trust a c=3 "valley" from a single sweep. Valleys need ≥2 matched
  A/Bs in separate sessions.
- For effects ≤10% on Metal throughput, use N ≥ 24 prompts AND at least
  2 independent server sessions before committing a code change.
- Always run the A/B on the *same binary* (env-override path), not on
  different builds. Build-to-build throughput variance is wider than
  the effects you're chasing.

## Cross-references

- `2026-04-19-metal-qwen35-concurrency-sweep-c3-c5-valleys.md` — original
  single-sweep finding that this doc retracts
- `2026-04-19-metal-qwen35-gdr-tgy-sweep.md` — three-way tg_y bench; the
  +2.1% overall-mean claim survives in weaker form (~+1% at c=5 only) but
  is not worth a code change
