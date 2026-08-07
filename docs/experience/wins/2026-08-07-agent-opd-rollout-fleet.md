# Agent-OPD cp rollout fleet — every rank serves, rank 0 keeps the harness

**Date:** 2026-08-07 · **Pod:** 8×H20 GPUs 4+5, ThinkingCap-Qwen3.6-27B-FP8, cp=2

> Status: accepted (correct + per-sample faster), but the round wall barely
> moved — the bottleneck is rollout concurrency, not per-sample throughput.
> See Results.

## Context

cp=2 is the preferred operating point for the cc-rollout lane (update wall 7.1×
vs single-GPU, [errors entry](../errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock.md)),
but after the deadlock fix the follower idled through the rollout phase — 75%
of the round wall (round-0 `cc_rollout` 4576 s on subset16).

## Change

`7aef20557` + simplify pass: every cp rank loads the rollout engine + cc serve
(`load_agent_opd_serve_student`, follower port = base + rank) and rank 0's
harness spreads each group's samples round-robin across the fleet endpoints
(`CcHarness.base_urls`). Scheduling, filtering, GRESO, replay, and saves stay
on rank 0; followers mirror engine lifecycle via update-stream flags
(`MeshMsg::Update.release_engines`, `MeshMsg::GroupEnd.synced`), with the
shared sequences in `quiesce_and_release_engines` / `sync_and_restore_engines`
so both sides stay aligned by construction. Dump filenames are pid-tagged
(`coordinator.rs`) so the fleet shares one dump dir; sidecar attribution is
per-sample model tag + time window, endpoint-agnostic. Consumed-file GC is
anchored to Update collectives only (a `GroupEnd` proves nothing about other
ranks' progress).

## A/B contract

Same binary, same subset16 manifest, cp=2 GPUs 4/5, one config change per arm:
- Baseline: fulltrain5 (follower idle), round-0 `cc_rollout` 4576 s, round
  walls 6131/1892/2262 s.
- Fleet arm: expect round-0 `cc_rollout` ≈ 50–60% of baseline (2 sessions per
  endpoint instead of 4); correctness license = the fulltrain5 signature
  (paired writeback losses identical across ranks, follower end-of-stream,
  `RUN_EXIT=0`, loss values in family with fulltrain4/5 round trajectories).

## Results — measured 2026-08-07/08, pod GPUs 4+5, tree `af9e48246`, `fulltrain6`

Correctness: 3 rounds, `RUN_EXIT=0`, both serve pids present in the shared dump
dir, rank-0 and follower writeback losses identical to all printed decimals
(e.g. 0.013347/0.013347), 24 mirrored updates, follower `end of stream`, mesh
dir removed. Peak VRAM symmetric 78.4 / 78.5 GB (both ranks now hold
engine + student).

Round 0 (same 16 tasks as the `fulltrain5` baseline), matched per task:

| Scope | baseline | fleet | Δ |
|---|---:|---:|---:|
| round-0 `cc_rollout` wall | 4576 s | 4348 s | −5% |
| 11 matched groups, cumulative | 3283 s | 3451 s | +5% |
| the 7 of those under the cap in both arms | 1367 s | 1051 s | **−23%** |
| aggregate throughput on those 7 | 47.2 tok/s | 67.6 tok/s | **+43%** |

The per-sample mechanism works; the round wall does not follow it because the
wall is straggler-bound. A group's wall is `max` over its samples, and 9 of the
run's 96 samples sat at the 600 s `--cc-timeout` cap (1 of them still passed —
`rejection-ce` keeps a truncated pass, `SampleFilter::PassOnly` does not filter
on `truncated`). In round 0 six capped samples set 4 of 11 group walls = 70% of
the wall. The two tasks that flipped from finished to capped had already run
489–571 s in prior runs — inside their own spread, not a fleet regression.

**Rollout capacity utilization ≈ 29%**: run-wide sample wall 17078 s against a
7289 s `cc_rollout` wall = 2.34 samples in flight on average, against 8 fleet
slots (4 per engine × 2 ranks). Two independent causes, both structural:
- only one group (4 samples) is ever in flight, so half the fleet's slots are
  never used;
- the group barrier idles the fast samples' slots until the straggler ends.

## Follow-up

The next lever is rollout concurrency, not per-sample throughput: batch G
prompts per update (the verl shape — roll the whole batch at one policy
version, then a single update), which fills the slots and amortizes the
straggler tail while staying strictly on-policy. `ScoredTrajectory.group_id`
already carries the per-prompt key the group baselines need. Filed as the
production blocker for the 449-task run: at this config round 0 alone
extrapolates to ~36 h, most of it idle slots.
