# Host KV pool single writer (Step 1b) — pending remote validation

Status: **pending-remote** — code landed on `lane/step1b` (PR open); the GPU
correctness gate below has not run. Do not read the correctness claims as
verified.

## Context

Speculative-decode KV accounting had multiple writers: the engine pre-budgets
the full spec chain through its reclaim path (`allocate_for_plan`, #197), and
the qwen35 executor corrected the host pool inline during `submit` — nine
write sites across MTP/DSpark warm-step, accept, and rollback paths, all
reaching the pool through `submit`'s `&mut dyn KvSlotAccounting` parameter.
Step 1b of the 2026-09-09 architecture refactor collapses this to one writer.

## What changed

- `BackendExecutor::submit` drops the `kv` parameter; backends never write the
  host pool.
- The qwen35 executor's nine write sites record the per-slot length they
  actually reached into `StepOutput::kv_actual` (last entry wins per slot
  within a step).
- The engine applies `kv_actual` after poll, before `apply_output`, with the
  same truncate/alloc-delta compare the backend ran inline (`Engine::
  apply_kv_actual`). The pre-budget means applied entries only trim a
  warm/short row's over-budget; growth into an empty pool cannot happen.
- Metal / Vulkan / HIP / diffusion / DSv4 never wrote through this surface;
  their signatures drop the unused parameter.

Host-side checks that ran locally (Mac, no GPU): CPU/Vulkan/HIP/seam/plan
test lanes green; infer-core unit tests cover the shrink/noop/grow compare and
last-entry-wins; CUDA-gated clippy lint (`-D warnings`) green.

## Remote gate (to run on the next GPU window)

1. `scripts/needle_gate.py` ×3 same-config, spec arm vs the baseline envelope.
2. `scripts/spec_parity.py` — token-exact equality between the speculative
   and greedy arms (the gate from the refactor plan §Step 1b).
3. One MTP c-sweep confirming accept counts and decode ms/token are unchanged
   from the 2026-09-10 batched-MTP acceptance entry.

What a regression looks like: host pool length diverges from device truth →
KV corruption, premature free, or a spec chain that reads stale pages. The
needle ladder and parity gate both fail loud in that case.

## Rule

The engine is the sole writer of the host KV pool. A backend that needs the
pool's length corrected reports the target in `StepOutput::kv_actual`; it
never holds a write capability across the seam.
