# Batched checkpoint round: LA backward crashes at B=4-long, gate over-fires at B=4-short — ×4 reverted

## Context

`8ba5e45b7` closed both checkpoint gates (capability default on + ×4 estimate)
to fix the B=4 long-completion writeback OOM (97.5 GB, full-tape). Pod round
on snapshot `8ba5e45b7` (GPU 0, sha-verified, zero contention): the OOM is
gone — and both newly-engaged branches are broken.

## Findings (measured)

| Run | RUN_EXIT | loss | Peak | Phase-C |
|---|---|---|---|---|
| long B=4×3153 | 1 — `CUDA_ERROR_ILLEGAL_ADDRESS` (linear_attention dqkv) | — | 41.1 GB (OOM gone) | crash 354 s into mb1 |
| long B=1 control | 0 | 0.1115 | 59.1 GB | ~4 s/row, full-tape |
| short B=4 | 0 | 0.1317 (parity ✓) | 29.5 GB | **1003 s vs ≲460 — 337 s/micro-batch vs <1 s** |
| self-opd smoke (dense default) | 0 | 1.435 ≈ fused 1.440 | 88.2 GB transient (flagged) | 13 s |

- Crash surfaces at the `d_dqkv` dtoh readback (`backend_cuda.rs:4777`) after
  the LA scan-backward kernel — an async fault from the preceding kernel.
  Single-trajectory checkpointing has run for weeks at 22K (B=1); the fault is
  **batch>1-specific** in the LA backward under checkpoint replay.
- Short B=4 gate arithmetic says no-fire (modeled ×4 ≈ 14.4 GB vs ~66–77 GB
  free) yet the slow branch ran — **mechanism unpinned**; the 300× per-mb cost
  smells like the LA host fallback (`ensure_host` + host scan,
  `linear_attention.rs:1240+`), which also implicates ctx population under
  replay (`try_linear_attention_backward_device` bails to host when any saved
  ctx field is None).

## Root Cause

Not fully attributed — two candidate mechanisms, one probe short: ① LA saved
device ctx (preact/qkv_conv/…) not populated on the checkpoint-replay inner
tape → host fallback (explains 300×) whose CUDA scan assist faults at the
12.6K-token batched shape (explains the crash); ② scan-backward kernel batch
indexing at B>1. One log line printing `should_checkpoint` inputs + which LA
backward path ran pins both.

## Fix

Reverted the ×4 multiplier (this commit) — restores yesterday's exact
semantics on every measured shape: short B=4 fast path, long B=4 back to the
documented OOM (use `--writeback-batch 1`, 59 GB, verified). Kept: capability
default-on (no-op vs the deleted per-lane default-true flags), the CPU
checkpoint-parity gate, the flag collapse. Next lever: fix the LA backward
under batched checkpoint (probe → attribute → kernel/ctx fix), then re-tighten
the gate.

## Rule

Before tightening an engage-gate, run the newly-engaged branch at the shapes
the tightening admits — a gate fix that routes into an unverified backward is
a regression with extra steps. 300×-slow + a dtoh label in the crash =
suspect the host fallback and check saved-ctx population first.
