# Group-stagger admission KILLED — the prefill waste it fixed never existed

## Context

Cross-sample LCP measurement showed a CC group's 8 samples share 18,176 of
~21.1K first-turn prompt tokens (86%). Since the radix cache publishes only at
request finish, the inference was: 8 concurrent cold starts each pay the full
21K prefill — ~127K wasted tokens/group. Built `--group-stagger` (hold K−1
samples until sample 0's first request publishes), landed as 1566bb175 +
93bb17726 + 294cfd90b, ran the pod A/B (H20, ThinkingCap-27B-FP8, SMOKE
SAMPLES=8, 4 groups, same GPU/port/binary, sequential arms).

## Root Cause

**The premise was never measured on the baseline.** The A/B counters show OFF
and ON with identical cache behavior — hits 1/7/14 vs 1/7/13 at t+2/4/5 min
from serve start; cumulative `hit_tokens` 4.243M vs 4.208M (−0.8%). Baseline
sample starts are already serialized ~20–30 s apart by `claude` CLI boot, and
sample 0's turn 1 finishes and publishes at ~90 s — before most other samples'
first requests are processed. The gate re-implemented an already-occurring
stagger. Rollout wall −9% was box drift: the eval segment, where the gate is
inert (k=1), moved −14% in the same direction.

## Fix

Reverted all three commits (2ab7883f1). The LCP measurement itself stands
(86% shared preamble) — only the "waste exists" inference was wrong.

## Rule

Before building a cache/prefill lever, curl the existing observable on the
**baseline** first — `arle_prefix_cache_hits_total` on `/metrics` would have
falsified the premise in 60 s, before 200 lines and a 2-run pod A/B. A
publish-timing code read (publish-on-finish) proves what *can't* be shared at
t=0, not that requests actually arrive at t=0.
