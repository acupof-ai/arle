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

## Fix — attempted (`3e2388f77`), then REVERTED (`5240a79c5`, `64013549c`)

Plumbed `--kv-cache-dtype` into the agent-opd/rubric-opd rollout engine to free
KV-pool headroom via quantized KV. **Reverted** — adversarial review (codex) +
empirical runs killed it on three independent counts:

1. **Premise invalid (the killer).** agent-opd ALREADY drops the entire rollout
   KV pool before the co-resident writeback — `InferStudent::release_kv_pool`
   (`crates/train/src/infer_student.rs:96`, called `train_cli.rs:3240,3527`):
   *"the pool is DEAD during the writeback — freeing it (~KV-pool GB) is the
   agent-OPD writeback headroom lever."* The pool is fully released regardless of
   dtype, so quantized KV frees **zero** additional writeback memory. Even with
   working kernels it could never make the 30K writeback fit.
2. **Kernel-blocked on Qwen35.** `Qwen35 full-attn paged: unsupported pool format
   INT8` → engine step fails → thread closes → all 16 rollouts reward=0.000,
   rc=1 (#68's kv-dtype landed for other paths, not this one). Controlled 3-run,
   hard corpus, temp=1.0: **bf16 → 14/16 with real variance** (billing 6×1.0+2×0.5,
   but 30K trajectories SKIPPED >23K cap → mean_loss=0.0000); int8/fp8 → engine
   crash → 0/16.
3. **Silent no-op leak.** The field went on the shared `OpdRuntimeArgs`, so
   `train opd` / `self-opd` accepted `--kv-cache-dtype` but `load_opd_infer_student`
   (`EngineLoadConfig::single_sequence`) never read it → silently used Auto.

The rollout engine's KV **format** is a rollout-time lever, never a writeback
lever — that VRAM is already reclaimed by `release_kv_pool`.

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
is the binding constraint. **Quantized KV cannot buy past it** — not because of a
kernel gap (though Qwen35 has one too), but because the rollout KV pool is already
fully released before the writeback (`release_kv_pool`), so its dtype changes
nothing downstream. Before proposing a memory lever, trace where the peak
actually lives: the writeback peak is activation memory with the KV pool already
gone. The real fix is shorter trajectories (comfort-band corpus) or writeback
activation-offload, not a KV-dtype flag. Colab G4 is unfit for this loop (hard
~1h cap + VM resets); use a persistent box (H20).
