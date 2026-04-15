# 2026-04-16 · Tiered KV Tier A/B/C/D rollup

## Context

The 2026-04-16 tiered-KV follow-on work split the post-M3c local runtime
promotion tranche into small, reviewable pieces instead of one oversized
batch. The per-tier notes in tree are:
[`2026-04-16-tiered-kv-tier-a-coordinator-local.md`](2026-04-16-tiered-kv-tier-a-coordinator-local.md),
[`2026-04-16-tiered-kv-tier-b-fingerprint-local.md`](2026-04-16-tiered-kv-tier-b-fingerprint-local.md),
and
[`2026-04-16-tiered-kv-tier-c-hygiene-local.md`](2026-04-16-tiered-kv-tier-c-hygiene-local.md).
Tier D remains the next M4-facing handoff rather than a shipped local note in
this batch: publish-time fingerprints now exist, but cross-restart
reconciliation and coordinator-driven disk staging are still deferred.

## What Worked

Tier A landed the scheduler-visible coordinator loop without pretending the
real transport path exists yet: staged hits park on tickets, the coordinator
thread replays `StagingCompleted`, and the scheduler re-admits cleanly. Tier B
landed identity scaffolding where it matters now: fingerprints are computed at
publish time, threaded through radix insert, and preserved by the local
DiskStore round trip. Tier C cleaned up the hot path around that new metadata:
`find_block_node_mut` is O(1) via `block_index`, and the old hardcoded
watermark / keepalive constants became `SchedulerConfig` fields with
validation. The result is a cleaner local M3 runtime-promotion stack on
`main`, while the remaining unsolved work stays explicit: real async
completion, coordinator-driven disk stage wiring, and the final M4/Tier D
fingerprint story.

## Rule

When a tiered-cache milestone spans control flow, identity, and hot-path
hygiene, ship them as narrow vertical slices: first the ticketed control path,
then the stable metadata plumbing, then the constant-time/runtime-config
cleanup. Keep the remote CUDA acceptance gate separate from the local landing
so the docs can say exactly which behavior is real today and which behavior is
still a stub.
