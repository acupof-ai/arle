# agent-opd cp>1 deadlocks at the first passing-group writeback

**Date:** 2026-08-07 · **Pod:** 8×H20 GPUs 4+5, ThinkingCap-Qwen3.6-27B-FP8, cp=2

## Context

First multi-update run of the cc-rollout agent-OPD lane under cp=2 (both the
449-task run and the 16-task subset). Rollouts complete; the first group with
passed trajectories (tenacity, passed=3/4) enters masked-CE writeback
(`seq_len=21536, total_targets=80`) and hangs 47+ min with zero log advance.
`masked-writeback DONE` count over both runs: 0. dmesg clean — no ELKEID kill,
no OOM.

## Root Cause

Under cp=2 the lane forks two full agent-opd instances, each running its own
serve, its own cc rollouts, and its own sample filter. Rollouts are stochastic
(temp 0.3, cc tool loop), so the two ranks accept different trajectory sets
with different lengths — confirmed by paired `rounds.jsonl` lines for the same
`task_id` with different `completion_tokens`, and a single-rank
`behavior_version` advancing alone. The cp collective inside the writeback then
has divergent participants/shapes: cp1 spins in the collective (1,222 threads,
GPU 100%), cp0 sleeps on a child pipe — one rank is inside a collective the
other never joins.

The `--synthetic-writeback-seq` path dodges this by construction (identical
fixed trajectory on both ranks), and the 12-round loop-stability win was
single-card — the cp>1 path of this lane had never been exercised.

## Fix

Not yet landed. Two viable shapes: rank-0-only rollout+filter with a
deterministic trajectory broadcast before the cp group joins, or per-rank
rollout with a rank-0 arbitration barrier. Validation meanwhile: run the lane
single-GPU — real trajectories cap at `max_update_seq 23000`, which fits one
card.

Evidence: `/host/lever2-out/fulltrain3-wedged-round0.log`,
`fulltrain3-rounds.jsonl`, `fulltrain3-metrics.csv` (and `fulltrain2.log`, the
449-task run, same wedge).

## Rule

A cp>1 lane whose per-rank inputs come from a stochastic source must
synchronize the accepted set before any collective; a synthetic single-step
probe cannot stand in for the multi-rank path because it is deterministic by
construction.
