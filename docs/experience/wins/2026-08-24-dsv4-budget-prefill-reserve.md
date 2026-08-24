# DSv4 slot budget: prefill-transient reserve — CUDA, 2026-08-24

> Status: Shipped (`94a15d415`), validated on the new 8xH20 box (`budget-v2`).

## Context

The slot solve handed every budget byte to slot state + FlashMLA pool. On the
old box, FP8 TP=4 `--max-total-tokens 131072` planned 18 slots (18x1049MB in a
19.7GB budget) and c=8 died mid-serve with `CUDA_ERROR_OUT_OF_MEMORY` on every
rank: the first long prefill's chunk transients allocate outside the budget.

## What Worked

`prefill_transient_reserve_bytes()` (budget.rs): one prefill chunk's peak
working set, itemized from the allocation sites (MoE packed/FP8/w13/route
buffers + attention activations at the 4096-token chunk ceiling), subtracted
from `budget_bytes` before the slot solve and logged at boot. Deterministic
from config, so rank-identical with no extra reduce. Computes **1352 MB** for
DSv4-Flash at TP=4.

Validation on the new box (GPUs 1,3,6,7; GPU1 shared with a ~21GB sft job):

- **Fail-closed path (FP8, 287GB ckpt):** the shared rank had 3303MB free
  after weights; budget minus reserve cannot hold one slot's band, so boot
  rejects with "Lower --max-total-tokens or free VRAM". The pre-fix plan
  admitted 1 slot + 718MB pool and would have met the ~1.4GB prefill
  transients at the first long prompt.
- **Positive path (NVFP4, same flags, no --max-running-requests):** reserve
  logged, 256 requested slots clamped to 27 (state-affordable 29), c=8
  16 prompts of 28568 tokens: **16/16 complete, 0 OOM, 0 preempts**.
  Decode 10.9 tok/s per request is contention with the co-located sft job,
  not a perf claim.

**Exact original config on 4 clean GPUs (3,5,6,7, run after the sft job
ended):** FP8 TP=4 `--max-total-tokens 131072`, no `--max-running-requests`.
Pre-fix this planned 18 slots and OOMed on every rank at tick 4432; with the
reserve it plans **17 slots** (state-affordable 18) and c=8 over 16 prompts of
28568 tokens completes **16/16 with 0 OOM and 0 preempts**. One slot of
capacity bought the whole config's stability.

## Rule

A budget planner that admits N units must first reserve the transient working
set of the operation that fills them; otherwise the plan is only valid for an
idle engine. Reserve terms are itemized from allocation sites, never a factor.

## Artifacts

- `/root/arle-ops/runs/budgetval/serve.log` (reject), `budgetval-nvfp4/`
  (bench-c8, serve.log), `budgetval-fp8clean/` (exact-config run), build `budget-v2` (head `0c31d5cde`)
