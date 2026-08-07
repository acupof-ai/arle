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

## Concurrency sweep — round 0, one rep per arm

| | G=1/600 | G=4/600 | G=4/1200 | G=2/600 |
|---|---:|---:|---:|---:|
| `cc_rollout` wall | 4348.5 s | **2101.5 s** | 3502.4 s | 3538.2 s |
| update wall | 894.4 s | 452.5 s | 526.0 s | 1421.3 s |
| **total round wall** | 5242 s | **2554 s** | 4028 s | 4959 s |
| sum(sample walls) | 10621 s | 19014 s | 21410 s | 14103 s |
| realized concurrency | 2.44× | **9.05×** | 6.11× | 3.99× |
| aggregate throughput | 38.4 tok/s | **72.4 tok/s** | 59.9 tok/s | 49.7 tok/s |
| samples at cap | 6/64 | 9/64 | **3/64** | 7/64 |
| passing samples | 9 | 5 | 5 | 10 |
| passes / total-wall hour | 6.18 | 7.05 | 4.47 | 7.26 |

VRAM peak sat at 78.6–79.1 GB in every arm (13 GB under the abort line) and is
effectively invariant to G, so slot state is not the binding cost at these
widths. Correctness signature clean in all arms.

**What is established, and what is not.** The wall, concurrency and throughput
figures are mechanistic and large. The pass counts are not resolved at one rep:
64 samples at a ~11% pass rate carry a binomial SD of ≈2.5, so 9 vs 5 is 1.6 SD
and 10 vs 9 is noise. Read the passes-per-hour column as indistinguishable
between G=2 and G=4, and both marginally above G=1.

**The 1200 s cap is strictly worse** — the one clean refutation in the sweep.
It did what it was supposed to mechanically (capped samples 9 → 3, and two
sessions at 604 s and 623 s converted to passes the 600 s cap would have cut)
but total passes stayed at 5 and realized concurrency *fell* 9.05× → 6.11×:
long stragglers hold slots while the rest of the window drains. So the G=4 pass
count is not primarily cap-induced. A longer leash buys wandering (one sample
burned 82 turns), not solutions.

**Throughput and pass yield knee at different points.** Aggregate throughput
keeps climbing to 8 sessions/engine (+88%), while per-sample walls degrade
(each session ~79% slower at G=4 than G=1). Optimizing tok/s past ~4 sessions
per engine buys wall at the expense of the thing being trained on.

## Accepted config — cp=4 × G=2 (`fulltrain10`)

8 concurrent sessions spread as **2 per engine over 4 engines**. The sweep above
only ever raised per-engine pressure; adding engines instead buys concurrency at
the pressure that keeps sessions healthy. No code change — the fleet and mesh are
rank-agnostic and per-rank sizing is `G × ceil(K/cp)` (`slots=2` confirmed on all
four ranks).

| round 0 | best of the cp=2 sweep | cp=4 × G=2 |
|---|---:|---:|
| sum(sample walls) | 10621 s (G=1, uncontended) | **9350 s** |
| samples at the 600 s cap | 3/64 (G=4/1200) | **1/64** |
| aggregate throughput | 72.4 tok/s (G=4) | **88.7 tok/s** |
| total round wall | 2659 s (G=4) | 3468 s |
| VRAM peak per card | 78.6–79.1 GB | **76.2–76.9 GB** |
| passing samples | 10 (G=2) | 10 |

The decisive figure is sum-of-sample-walls: 9350 s is lower than the
*uncontended* G=1 baseline, i.e. sessions are healthier than in any cp=2 arm
while running 8 at once. Per-sample walls collapse (tenacity 45–58 s against
G=4's 157–175 s) and straggler pressure nearly vanishes. Correctness at 4 ranks:
212 writeback DONE lines in 53 four-way identical groups, all three followers at
`end of stream`, `staleness: 0`, every group carrying exactly 4 records, VRAM
2 GB *lower* per card than cp=2 (smaller shard).

Cost to watch at scale: host RSS peaked 230.6 GiB (4 ranks × engine+student),
second-highest of the campaign; VRAM is not the binding resource.

**New bottleneck, measured:** realized concurrency is only 3.63× of 8 slots — the
rollout phase is no longer engine-bound. The residual wall is host-side (sandbox
boot, tool exec, pytest), so further GPU concurrency has little left to buy. The
untested wall lever is cp=4 × G=4 (16 sessions, still 4 per engine); the host
floor may absorb it.

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
