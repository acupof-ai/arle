# CUDA async prefill overlap rewrite

## Context

- Current `Qwen3-4B` / L4 `c16` baseline is `107.71 tok/s`, still `-22.1%`
  behind `sglang`'s `137.07 tok/s`.
- Recent `c16` trace shows the remaining gap is not decode latency. The loaded
  window stays near the physical active-set ceiling, but `prefill_active` step
  p50 remains around `2.0s`, so long-prompt refill waves are still too
  expensive.
- Our scheduler already overlaps decode launch and readback, but prefill is
  still one synchronous block: launch, sync, completion sampling, then next
  loop turn. That keeps CPU admission and cleanup off the critical path during
  the heaviest batch type.
- Qwen3 paged prefill also still has a stale model-owned `paged_prefill_plan`
  reuse path. That is the wrong owner once the scheduler needs prefill work to
  survive across loop turns.

## Goal

Move prefill onto one canonical async lifecycle that mirrors decode:

1. scheduler plans and launches batched prefill;
2. model-owned prefill context keeps GPU metadata and temporary buffers alive;
3. scheduler returns to the loop and overlaps CPU intake/admission/cleanup;
4. next turn completes prefill, samples first tokens, and transitions requests.

## Scope

This is a delete-style refactor. The end state keeps exactly one prefill flow.

### Track A — model / ops contract

- `infer/src/model.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/src/model/qwen3/prefill.rs`
- `infer/src/model/qwen3/weights.rs`
- `infer/src/ops/attention.rs`

### Track B — scheduler lifecycle

- `infer/src/scheduler/cuda/core.rs`
- `infer/src/scheduler/cuda/execution.rs`
- `infer/src/scheduler/cuda/prefill.rs`
- `infer/src/scheduler/cuda/runtime.rs`

### Track C — validation / records

- `docs/experience/wins/...`
- `docs/plans/2026-04-22-sglang-gap-closure-execution.md`
- `docs/plans/2026-04-22-cuda-end-to-end-trace.md`

## Design

### 1. Canonical prefill contract

- `ModelForward` already exposes `PrefillContext`,
  `create_prefill_context`, `supports_async_prefill_batch`,
  `launch_prefill_batch`, and `complete_prefill_batch`.
- The scheduler owns one `PrefillContext` for the lifetime of the run,
  analogous to `DecodeContext`.
- The model owns any GPU scratch/metadata that must survive until queued
  prefill kernels finish.

### 2. Qwen3 paged prefill ownership

- Remove `Qwen3Model::paged_prefill_plan`.
- Move plan lifetime into `Qwen3PrefillContext`.
- `PagedPrefillForward` keeps only per-forward uploaded indptr / page metadata
  and never borrows the plan.
- Qwen3 paged prefill writes first-token logits directly into
  `states[slot].base.prefill_logits` during launch.
- Completion only synchronizes and releases pending resources; it does not
  recompute logits.

### 3. Scheduler prefill lifecycle

- Add `pending_prefill` state alongside `pending_decode`.
- Add `prefill_ctx: Option<M::PrefillContext>` lazy-initialized against the
  active paged pool.
- `step()` order becomes:
  1. complete pending decode;
  2. complete pending prefill;
  3. wait on emit gates;
  4. plan next step;
  5. launch prefill and/or decode.
- Sync fallback remains for models without async prefill support.

### 4. Completion semantics

- Completion keeps the current canonical scheduler behavior:
  - incomplete chunks only advance `progress`;
  - full prompt completion migrates/snapshots as needed;
  - first-token selection stays batched;
  - finish / emit / move-to-decode semantics do not change.
- No second waiting queue, no second admission planner, no model-specific
  scheduler branches.

## Acceptance

- `cargo build --release -p infer --bin infer` passes.
- Relevant scheduler tests pass.
- `c16` trace shows prefill launch/readback spanning loop turns instead of a
  single synchronous block.
- One new `guidellm` bench entry lands under `docs/experience/wins/` with
  before/after deltas against the latest `c16` baseline.
- Obsolete `paged_prefill_plan` ownership is deleted.

## Bench anchor

- Baseline: `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`
- Latest trace diagnosis: `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-unified-budget-bottleneck.md`
- Prior overlap attempt: `docs/plans/2026-04-22-cuda-prefill-overlap-and-prefix-aware-queue.md`
