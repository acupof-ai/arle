# Device-native cat: matched A/B proves a strict win (faster AND less host RAM)

**Date:** 2026-08-02 · **Commit:** 7276fa081 · **Pod:** 8×H20, real 27B cp=2

## Context

7276fa081 made `ops::cat` device-lazy (D2D concat, no host round-trip) for the
CP linear-attn reorder. The first real-27B cp=2 seq=131072 run after it was
memcg-killed at 343 GB host anon-RSS, while the old host-cat path had once
completed the same invocation. Suspicion: the removed host round-trip doubled as
a memory-release valve. No host-RSS baseline existed, so per
[[feedback_optimization_that_regresses_working_axis]] the claim needed a matched
A/B, run at seq=32768 (fits host RAM).

## What Worked

Matched A/B, identical invocation, real 27B cp=2 seq=32768:

| arm | peak host RSS | fwd | bwd | exit | losses |
|-----|--------------|-----|-----|------|--------|
| device-cat (HEAD 7276fa081) | 46.2 GB | 102.6 s | 386.5 s | 0 | 4.805667 / 6.064441 |
| host-cat (pre-7276fa081) | 56.8 GB | 670.0 s | 2054.1 s | done, marker unwritten | identical |

Device-cat is **−10.6 GB host RSS and ~5.6× faster** — a strict win on both
axes. The 343 GB OOM at seq=131072 is inherent to sequence-length scaling
(superlinear: 4× seq from 32768 would extrapolate ~185 GB, actual crossed 343 GB
at 42% of forward), not introduced by the cat path. Fix direction is memcg
headroom or checkpoint-offload footprint, a separate decision.

## Rule

When a perf change is accused of regressing a formerly-working axis, the verdict
comes from a matched A/B at a size both arms survive — the accusation can be
exactly backwards (here the "regression" arm used *less* of the contested
resource).
