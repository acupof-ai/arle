# agent-opd dapo null gradient — long agentic trajectories exceed the writeback VRAM cap

> Status: Root-caused + code fix landed (3e2388f77); H20 validation pending.

## Context

sm_120 G4 (RTX PRO 6000, 96 GB) agent-RL validation: `arle train agent-opd`,
`ThinkingCap-Qwen3.6-27B-FP8` student, `--update-strategy dapo`
`--rollout-temperature 1.0 --samples-per-prompt 8`, hard-difficulty synthetic
bug-fix corpus. Goal: observe a real non-zero dapo PG gradient from within-group
reward variance.

The reward variance was real and confirmed (baseline held-out 66–83%; decoded
training group `billing-roundhours`: 6×reward=1.0 + 2×reward=0.5). Yet:

```
round 0: tasks=2 rollouts=16 passed=14 tasks_passed=2 zero_variance_groups=1 mean_loss=0.0000
[agent-opd] SKIP trajectory pre-capture: seq 30923 > max_update_seq 23000 (VRAM wall)   ← ×many
```

## Root Cause

**Not a dapo defect. The variance-bearing trajectories were skipped for length.**
hard difficulty stages 18 gold-scenery distractor modules; the agent reads them
across 4–5 turns, so a single agentic rollout transcript reaches ~30K tokens.
The masked-writeback backward has a VRAM guard `max_update_seq` (default 23 000;
`update_strategy.rs:650` — the 27B LoRA backward OOMs at seq≈30K even with
offload+trims on 96 GB). Every >23K trajectory is skipped → the `billing`
variance group never reaches the backward → `mean_loss=0.0000`.

Decoded, not inferred: the `round 0` summary + per-trajectory `SKIP … seq 30923 >
max_update_seq 23000` lines are the ground truth. `zero_variance_groups=1` was the
*other* train task (`paging-count`, 8/8 pass) — a red herring; the real blocker
was length, not missing variance.

Two levers that DON'T apply: `--max-turns` is a top-level arg for the local agent
(the cc-harness rollout is a shelled-out `claude` CLI that owns its own turns);
`--kv-cache-dtype` was ServeArgs-only, not wired to the agent-opd rollout engine.

Confounder: Colab G4 reclaims the GPU at ~1h while actively polled (two sessions
`session_terminated` mid-rollout). A single H20 is also 96 GB — same writeback
wall — so H20 only removes the reclaim, not the VRAM wall.

## Fix

`3e2388f77` — plumb `--kv-cache-dtype` into the agent-opd/rubric-opd rollout
engine (`OpdRuntimeArgs` field → both `EngineLoadConfig` sites; shared
`From<ServeKvCacheDtypeArg>`). Quantized KV (int8/fp8) frees the rollout engine's
KV-pool headroom (~15–20 GB) so the 96 GB writeback fits longer trajectories.
Default Auto = bf16, byte-identical. Run with `--kv-cache-dtype fp8
--max-update-seq 40000` to admit the ~30K trajectories.

## Rule

A dapo `mean_loss=0.0000` with confirmed reward variance is a **length-skip**, not
a dead gradient — grep `SKIP trajectory … > max_update_seq` before concluding the
update is broken. Agentic PG rollouts run 20–30K tokens; the single-GPU writeback
VRAM wall (~23–30K on 96 GB), not the algorithm, is the binding constraint. Free
KV headroom (quantized KV) or shorten trajectories; raising `--max-update-seq`
alone risks the OOM the guard exists to prevent.
