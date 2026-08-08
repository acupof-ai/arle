# Checkpoint offload pool retained one host buffer per size class ever seen

**Date:** 2026-08-08 · **Pod:** 8×H20 GPUs 4–7, ThinkingCap-Qwen3.6-27B-FP8, cp=4

## Context

The 449-task agent-OPD production run (`fulltrain11`) reported host RSS
plateauing at ~208 GiB over its first 30 groups, then resuming a climb:
233 → 264 → 286 → 306 GiB by group 58, ≈0.86 GiB per group per rank. Linear
extrapolation crossed the box's memory before round 0 would finish. Every prior
run of this lane was 16–24 groups and ended inside the apparent plateau, so the
growth had never been visible.

## Root Cause

`CheckpointOffloadPool` (`crates/autograd/src/tensor.rs`, added by the L3 commit
`ac348032c`) held idle host buffers for reuse with **no bound at all**:
`recycle` pushed every non-empty buffer onto a free list that nothing ever
drained, and `take` was **first-fit** over that list.

Both halves matter. OPD trajectories run ~5K–30K tokens, so a parked activation
is 100–600 MB and each new, longer trajectory asks for a capacity no pooled
buffer has — a fresh allocation, with the smaller ones retained forever. First
fit compounds it: a short trajectory can claim the largest pooled buffer, so the
next long one allocates again. The pool therefore converges on one live buffer
per size class ever observed, which on a 449-task corpus is unbounded in
practice.

The same defect class was fixed during Lever 2 in `PinnedCheckpointPool`
(64 MiB slot granularity + best-fit + a byte budget) because exact-length reuse
exhausted the pinned budget. The pageable pool had the identical varying-length
problem and no budget, so instead of failing it grew.

## Fix

Best-fit `take`, plus a byte budget (4 GiB/rank) with smallest-first eviction on
`recycle` — a large buffer serves any smaller take, so capacity coverage is what
the cap should buy. The cap is a field so the unit test can drive eviction
without allocating 4 GiB; `pool_is_bounded_and_best_fit` asserts the bound holds
across 64 growing size classes and that a 512-element take does not strand a
4096-element buffer.

Not restarted on the running job: the pool holds *idle* buffers and
`max_update_seq` caps the largest class, so the curve saturates once every size
class has been seen. The run continues under a system-memory guard instead
(abort on available memory, not on the RSS sum, which counts four ranks).

## Rule

A reuse pool over variable-size buffers needs both halves: best fit so a small
request cannot strand a large buffer, and a byte budget so retention cannot
outlive the size classes. Growth of this shape is invisible in short runs — it
saturates exactly when the workload stops producing new size classes, which is
what makes a 16-task plateau look like a bound.
