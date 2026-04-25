# `infer::scheduler` — Agent Guide

CUDA multi-request continuous batching + policy/accounting scaffolding that
works with any backend. Load before editing any scheduler internals.

## Refactor posture

- Keep scheduler logic simple and uniform. Prefer deletion-style refactors:
  remove parked or temporary admission paths, collapse duplicate planning
  branches, and keep one canonical request flow instead of parallel queues.

## Module map

| Path | Role |
|------|------|
| `scheduler.rs` | Module root + `pub use` surface. |
| `batch.rs` | **Backend-agnostic** CPU accounting scheduler (`BatchScheduler`) for lifecycle events + dry-run testing. |
| `types.rs` | `IncomingRequest`, `SchedulerHandle`, `SchedulerConfig`, `SchedulerFull`. The config defaults live in `SchedulerConfig::runtime_defaults(num_slots)`. |
| `policy.rs` | `SchedulerSignals`, `AdmissionPolicy`, `ChunkingPolicy`, `DecodeAwareChunking`. `DecodeAwareChunking` is only for the backend-agnostic CPU accounting scheduler in `batch.rs`; the production CUDA runtime uses explicit `SchedulerConfig` token/request budgets. Agent-aware fields (`prefix_hit_tokens`, `session_affinity_slot`, `turn_depth`) are plumbed but only wired under the tiered-KV project (`docs/projects/agent-first-architecture.md::B3`). |
| `metrics.rs` | Scheduler metrics accounting. |
| `cuda/core.rs` | CUDA `Scheduler<M: ModelForward>` struct + construction. Owns slots, paged KV pool, radix prefix cache, `block_owner_slots`. |
| `cuda/prefill.rs` | `step_new` — chunked prefill + prefix-hit paths (exact-full, prompt-prefix-of-cached, partial). |
| `cuda/decode.rs` | Batched decode + retract/requeue under KV pressure. |
| `cuda/request.rs` | Per-request state (`QueuedRequest`, `ActiveRequest`, `Phase`). |
| `cuda/runtime.rs` | Single-writer scheduler thread: intake, prompt-length normalization, priority-ordered admission, spill completions, cleanup. |
| `cuda/execution.rs` | Per-step execution glue: decode launch/readback, prefill budgets, waiting-queue admission. |

## Invariants you will break if you're not careful

1. **The scheduler thread is the only writer** to `states`, `prefix_cache`,
   `block_to_pages`, `block_owner_slots`, `paged_kv_pool`. Taking any of
   these behind an `Arc<Mutex<…>>` is a design change — don't do it without
   reading `docs/projects/tiered-kv-cache.md §5.2`.
2. **`BlockId` = physical pool page index** (`u32`), not a content hash.
   Content hashing uses `crate::types::BlockFingerprint` and only exists at
   persist/migrate boundaries (M4/M5). See `infer/src/kv_tier/AGENTS.md`.
3. **Prefix-cache retention caps** (`SchedulerConfig::runtime_defaults`):
   - `prefix_cache_high_water = 0.75` → cleanup trigger
   - `prefix_cache_low_water = 0.50` → cleanup target
   - `prefix_cache_retain_hard_cap = 0.90` → new prompts no longer publish
     above this, so fresh admissions can't starve on pinned-cold pages.
   These are tuned — change only with a bench snapshot.
4. **`PREFIX_CACHE_BLOCK_SIZE = 16` matches the paged-pool page size.**
   Changing one without the other breaks M2 dual residency.
5. **Do not project `batch.rs` policy defaults onto CUDA runtime behavior.**
   `ChunkingPolicy` / `DecodeAwareChunking` belongs to the backend-agnostic
   CPU accounting scheduler only. The production CUDA runtime does not have a
   "decode active => chunk = 64" rule; `chunked_prefill_size` caps one
   request's prefill chunk, `max_num_batched_tokens` caps the whole step token
   budget, and the planner derives one mutable prefill budget by clamping that
   step budget with `max_prefill_tokens`. `prefill_max_requests` then limits
   how many prefilling requests advance in one planned tick.
6. **Hybrid models (Qwen3.5) cannot truncate recurrent state.** `prefill.rs`
   downgrades partial prefix hits to full MISS when
   `!state.supports_partial_prefix()`. Only full-prefix hits benefit from
   `save_prefix_snapshot` / `restore_prefix_snapshot`.
7. **Decode retract is recompute-mode requeue.** Victim selection now mirrors
   the current sglang-alignment heuristic: retract the least-progressed request
   first, tie-breaking toward longer prompts. If you change it, update
   `docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`.
8. **There are now two prefix reuse modes.** `block_owner_slots` still tracks
   the non-paged same-slot contiguous-state fallback, but paged-prefill models
   may also directly attach radix-backed GPU pages to a fresh slot and rely on
   `paged_kv` tail-page COW before append. Keep those two paths explicit: the
   contiguous fallback is model-compatibility glue, the paged attach path is
   the canonical shared-page flow.
9. **`runtime.rs` ingress owns waiting-queue normalization; `assign_slots()`
   owns admission only.** Tokenization, prompt-length rejection/clamping, and
   cancellation skip happen when requests enter the scheduler so the waiting
   queue always carries normalized prompt tokens. `assign_slots()` then does
   radix classification and slot materialization before `execution.rs::plan_step()`
   decides the current tick's prefill/decode mix. The waiting queue itself now
   stays priority-ordered incrementally on ingress/requeue; `assign_slots()` is
   no longer allowed to re-sort the whole queue every tick. Do not recreate a
   second waiting-queue planner in `execution.rs`.
10. **Eviction never touches pages backing an active slot.** Radix eviction
   only frees pages whose `block_owner_slots` entry is either missing (the
   slot has already been freed) or points at a slot currently in `Idle`
   state. The eviction path confirms this before calling
   `release_pages`. Mid-request eviction would corrupt a running decode
   — if you add a new eviction trigger (e.g. tier-demotion under pool
   pressure), preserve this gate. Verified statically at the
   paged-prefill lifecycle audit (2026-04-18); no property test locks
   it in yet.

## Common mistakes

- Putting model-specific code in `scheduler/cuda/*`. Decode-batch kernel
  invocation lives on `M::DecodeContext` via the `DecodeContextOps` trait —
  add methods there, not `if model_type == …` here.
- Adding a second `HashMap<BlockId, ...>`. There are already two
  (`block_to_pages`, `block_owner_slots`) with distinct roles; the radix
  itself is the third source of truth. A new one usually means you are
  duplicating existing state.
- Calling `SchedulerHandle::submit` from the scheduler thread itself. The
  handle is for *external* submitters (HTTP, CLI). Internal resubmission
  (e.g. preemption recompute) pushes back onto `waiting` directly.

## Tests

- `scheduler/tests.rs` — unit tests for admission + chunking policy.
- `infer/tests/e2e*.rs` — full E2E against JSON baselines; run on GPU hosts.
- `infer/tests/greedy_consistency.rs` — regression gate for scheduler vs
  single-request numerical drift.

## Pointers

- `docs/projects/tiered-kv-cache.md` — project driving scheduler internals right now (also the milestone ledger).
- `docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md` — what changed at M2b.
- `docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md` —
  preemption policy rationale.
