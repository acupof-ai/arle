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

## Follow-up — `--prompts-per-update` (`f996e6826`)

The lever is rollout concurrency, not per-sample throughput. The round loop now
windows the verl way: G groups roll concurrently under one policy version, then
a single update trains their merged batch. Staleness stays 0 (the window shares
its launch version), and `rejection-ce` steps per trajectory inside `update_ce`,
so the merged batch does not raise writeback VRAM — only the engine's extra
slots do.

Measured 2026-08-08, `fulltrain7` at `5b1cd473d`, G=4 vs `fulltrain6` G=1
(same subset16, cp=2, GPUs 4+5):

| round 0 | G=1 | G=4 | Δ |
|---|---:|---:|---:|
| `cc_rollout` wall | 4348.5 s | **2101.5 s** | **−51.7%** |
| realized concurrency | 2.44× | **9.05×** | 3.7× |
| aggregate throughput | 38.4 tok/s | **72.4 tok/s** | +88% |
| samples at the 600 s cap | 6 / 64 | 9 / 64 | +50% |
| passing samples | 9 | 5 | **−44%** |

Correctness clean: 28 writeback DONE lines in 14 identical rank pairs, follower
losses matching rank 0 to six decimals, `end of stream`, `RUN_EXIT=0`, the
windowing invariant silent, every group carrying exactly 4 records, and 16
distinct `#g{nonce}s{sample}` tags inside one window (the P1 fix engaged).
VRAM peak 79.0 / 78.8 GB — only +0.2 GB over G=1, and host RSS *fell* to
153 GiB from 270 GiB.

**The wall win is real but the signal cost is the headline.** Per-sample walls
rose 10621 → 19014 s (each session ~79% slower under 8-way contention), and
because `--cc-timeout` is a wall-clock cap, contention pushes borderline
sessions past it: `9b5ribm7`'s samples ran 162–223 s at G=1 and all four pinned
at 600 s at G=4, taking that task from 3 passes to 1. Measured on the metric
that matters for RFT — passing samples per wall-hour — G=4 is 8.6/h against
G=1's 7.5/h, **+15%, not the +88% throughput suggests**.

Conclusion: a wall-clock rollout cap is the wrong control under variable
contention — it makes the training signal a function of scheduling load. Next
arm (`fulltrain8`): G=4 with `--cc-timeout 1200`, to test whether the pass loss
is entirely cap-induced. Also unresolved: aggregate throughput scaled +88% for
4× the concurrency, so 8 sessions/engine is past the knee; G=2 is the missing
third point.

## Production config for the 449-task run

Two numbers decide it, both from this run:
- **Windowing is the wall lever.** At G=1 round 0 extrapolates to ~36 h for 449
  groups. At G=4 the window wall is bounded below by `max(sum of the window's
  sample walls / 8 slots, longest sample)`, which on this corpus is the cap:
  ≈ 600 s × 113 windows ≈ 19 h.
- **After windowing the `--cc-timeout` cap binds.** With 16 samples per window
  and ~9% of samples capped, almost every window contains a capped sample, so
  the window wall sits at the cap. Evidence for lowering it: of the run's 8
  passing samples, 7 finished within 386 s and only 1 passed at the 600 s cap;
  of the 9 capped samples, 8 failed. A 400 s cap keeps 7 of 8 passes and takes
  the window wall to ≈ 400 s (≈ 12.5 h for round 0). The cost is coverage of
  genuinely long trajectories, which are also the hard tasks that carry the
  most signal — so this is a config call, not a default flip.
