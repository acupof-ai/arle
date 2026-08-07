# Agent-OPD cp rollout fleet — every rank serves, rank 0 keeps the harness

**Date:** 2026-08-07 · **Pod:** 8×H20 GPUs 4+5, ThinkingCap-Qwen3.6-27B-FP8, cp=2

> Status: pending-remote. Pod A/B in flight (subset16 × 4 samples × 3 rounds,
> fleet binary vs the fulltrain5 baseline).

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

## Results

Pending-remote: pod run dispatched 2026-08-07 (fulltrain6 series); numbers land
here when the report returns.
