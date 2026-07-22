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

## Fix — attempted, KERNEL-BLOCKED for Qwen35

`3e2388f77` plumbed `--kv-cache-dtype` into the agent-opd/rubric-opd rollout
engine (valid feature; default Auto = bf16 byte-identical) to free KV-pool
headroom via quantized KV. **But it does NOT solve this on the 27B** — decoded
2026-07-22 on Colab G4:

```
ERROR infer_server::execution: Qwen35 full-attn paged: unsupported pool format INT8
[agent-opd] ensure KV pool (round 0) failed: engine thread closed
[ARLE train] error: engine thread closed  → all 16 rollouts reward=0.000, rc=1
```

The Qwen35 full-attn paged attention kernel does not support INT8/FP8 KV pools
(#68's kv-dtype landed for other paths, not this one). Controlled 3-run
comparison, hard corpus, temp=1.0, task-limit 2:

| KV dtype | result |
|----------|--------|
| **bf16 (no flag)** | **14/16 pass, real variance** (billing 6×1.0+2×0.5) — but 30K trajectories SKIPPED (>23K cap) → mean_loss=0.0000 |
| int8 (`--kv-cache-dtype int8`) | engine crash "unsupported pool format INT8" → 0/16, rc=1 |
| fp8 (`--kv-cache-dtype fp8`) | 0/16 (same crash class) |

So quantized KV frees no headroom here — it kills the rollout engine. The
plumbing stays (works for kernels that support quantized pools); it is not the
solution for Qwen35.

## Real path (unresolved — needs a stable box)

1. **bf16 + `--max-update-seq 40000`** — correct generation (14/16 variance), but
   admits 30K to the writeback → likely the OOM the guard exists to prevent
   (`update_strategy.rs:650`). UNTESTED to completion (every Colab G4 session
   died: hard ~1h runtime cap + mid-run VM resets; keepalive at 100% util did NOT
   defeat the cap).
2. **Comfort-band corpus (shorter trajectories)** — the project's planned fix
   (2026-07-03 curve plan, task "Sweet-spot corpus: stage → 27B-profile →
   comfort-band filter"): tasks tuned to intermediate pass rate with FEWER
   distractor modules → trajectories fit under 23K with bf16, no OOM, keeps
   variance. This is the right structural fix.
3. **Writeback sequence-offload** — a code change to raise the 96 GB writeback
   ceiling above 30K (activation offload/checkpointing), decoupling variance
   (needs hard) from trajectory length.

## Rule

A dapo `mean_loss=0.0000` with confirmed reward variance is a **length-skip**, not
a dead gradient — grep `SKIP trajectory … > max_update_seq` first. Agentic PG
rollouts run 20–30K tokens; the single-GPU writeback VRAM wall (~23–30K on 96 GB)
is the binding constraint, and **quantized KV can't buy past it on Qwen35 (paged
attention rejects INT8/FP8 pools)**. The fix is shorter trajectories
(comfort-band corpus), not a KV-dtype flag. Colab G4 is unfit for this loop
(hard ~1h cap + VM resets); use a persistent box (H20).
