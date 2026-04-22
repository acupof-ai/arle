# CUDA prefill overlap path + prefix-aware waiting queue

## Context

- Current `c16` trace on `Qwen3-4B` / L4 still trails `sglang` on throughput:
  `106.81 tok/s` vs `137.07 tok/s`.
- Unified page/token budgeting removed the earlier admission collapse, but the
  trace still spends `16s / 64s` loaded wall time in `prefill_rows>0` and
  shows `prefill_active` step p50 around `2071.4 ms`.
- Recent SGLang code review points to two missing pieces on our side:
  1. cache-aware waiting/admission ordering;
  2. a tighter refill tail, especially around long-prompt prefill completion.

## Goal

Close the remaining high-concurrency control-plane gap without reintroducing
duplicate scheduler paths.

## Scope

Two linked changes, both in CUDA scheduler code:

1. **Prefix-aware waiting queue**
   - Keep the queue incrementally ordered, but add a stable same-priority hint
     so requeued/deferred requests with stronger reusable prefixes are admitted
     earlier.
   - Preserve the current rule that `assign_slots()` does not do a full queue
     re-sort each tick.

2. **Prefill completion batching**
   - Replace the current per-request serial first-token completion tail after a
     batched prefill with a single batched sampling path.
   - Keep one canonical prefill flow: batched prefill forward, batched first
     token selection, then per-request finish/decode transition.

## Planned file ownership

- **Track A — prefix-aware waiting queue**
  - `infer/src/scheduler/types.rs`
  - `infer/src/scheduler/cuda/runtime.rs`

- **Track B — prefill completion batching**
  - `infer/src/scheduler/cuda/prefill.rs`

- **Integration / validation / docs**
  - `docs/experience/wins/...`
  - `docs/plans/2026-04-22-sglang-gap-closure-execution.md`
  - any small glue edits needed after parallel landing

## Design notes

### A. Prefix-aware waiting queue

- Extend `IncomingRequest` with scheduler-owned waiting hints.
- Hints are updated only when the scheduler has already computed a concrete
  admission plan for the request.
- Queue insertion remains incremental:
  - primary key: request priority;
  - secondary key: stronger prefix reuse / GPU-ready reuse first;
  - stable FIFO inside equal keys, except existing requeue bias rules.
- This matches the scheduler invariant that ingress/requeue maintains queue
  shape and avoids per-tick full sorting.

### B. Prefill completion batching

- `step_prefill_batch()` already launches a batched prefill forward.
- The current tail then serially calls `select_token()` per request.
- Replace that tail with a single `select_tokens_batch()` over every request
  whose current chunk completed the full prompt.
- Keep chunk-incomplete rows unchanged: only advance `progress`.
- Keep finish / emit / decode transition semantics identical after the batched
  token vector is produced.

## Acceptance

- Code stays on one canonical scheduler path; no parked alternate queue or
  second admission planner.
- Targeted scheduler tests pass.
- `cargo build --release -p infer --bin infer` passes.
- One new `guidellm` benchmark entry records the net result against the latest
  `c16` unified-budget baseline.

## Bench anchor

- Current diagnosis: `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-unified-budget-bottleneck.md`
- Current baseline: `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`
- This change's local regression check: `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8.md`
