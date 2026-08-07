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

Rank-0-only rollout with a file-based update stream (`MeshUpdateChannel`,
train_cli.rs): cp rank 0 keeps the whole lane (serve, rollouts, filtering,
saves) and publishes every update's batch under the coordinator-minted
`ARLE_TRAIN_MESH_DIR` (write-then-rename); follower ranks skip the rollout
engine entirely, load only the autograd student + optimizer, and mirror the
update calls until an end marker — the cp collectives inside the writeback see
identical call sequences by construction. The coordinator no longer tears the
group down on a clean follower exit (only rank 0's exit, or any nonzero exit,
ends the run). dp>1 in this lane now fails fast with an explicit error.
Pod cp=2 validation (2026-08-07, subset16 × 4 samples × 3 rounds, GPUs 4+5,
tree `9da8ff777`): **confirmed.** The former wedge point — first passing-group
writeback — completed in ~30 s; 46 writebacks, 22 mirrored updates, rank-0 and
follower losses identical to all printed decimals (e.g. 0.085379/0.085379,
0.118810/0.118810); follower reached `end of stream`, run exited 0, mesh dir
consumed and removed. VRAM leader 78.4 GB peak vs follower 43.0 GB.

Side effect: cp=2 is the better operating point for this lane, not only a
parity mode — round-0 update wall 1119 s vs 7892 s single-GPU (7.1×; backward
28–30 s vs 213–252 s at the same ~21 K seq). The ~10.5 K/rank shard stays out
of the checkpoint-offload regime; host RSS peak 206.9 vs 270.5 GB.

Untested surfaces (recorded, not blocking): follower `--lora-adapters` resume;
ValueGae critic under cp (zero-init is deterministic in theory; the shipped
lane is rejection-ce).

Evidence: `/host/lever2-out/fulltrain3-wedged-round0.log`,
`fulltrain3-rounds.jsonl`, `fulltrain3-metrics.csv` (and `fulltrain2.log`, the
449-task run, same wedge).

## Rule

A cp>1 lane whose per-rank inputs come from a stochastic source must
synchronize the accepted set before any collective; a synthetic single-step
probe cannot stand in for the multi-rank path because it is deterministic by
construction.
