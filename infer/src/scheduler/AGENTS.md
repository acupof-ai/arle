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
   "decode active => chunk = 64" rule; it uses `chunked_prefill_size`,
   `max_prefill_tokens`, `prefill_max_requests`, and `enable_mixed_chunk`.
6. **Hybrid models (Qwen3.5) cannot truncate recurrent state.** `prefill.rs`
   downgrades partial prefix hits to full MISS when
   `!state.supports_partial_prefix()`. Only full-prefix hits benefit from
   `save_prefix_snapshot` / `restore_prefix_snapshot`.
7. **Decode retract is recompute-mode requeue.** Victim selection now mirrors
   the current sglang-alignment heuristic: retract the least-progressed request
   first, tie-breaking toward longer prompts. If you change it, update
   `docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`.
8. **Slot reuse is single-slot-only.** `block_owner_slots` tracks which free
   slot can reuse a radix block's contiguous state. Cross-slot page aliasing
   is intentionally unsupported — M2b closed that door deliberately
   (`docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md`).
9. **`assign_slots()` owns waiting-queue normalization.** Tokenization,
   prompt-length rejection/clamping, cancellation skip, and priority ordering
   all happen before a request becomes an `ActiveRequest`. Do not recreate a
   second waiting-admission path in `execution.rs`.
10. **Eviction never touches pages backing an active slot.** Radix eviction
   only frees pages whose `block_owner_slots` entry is either missing (the
   slot has already been freed) or points at a slot currently in `Idle`
   state. The eviction path confirms this before calling
   `release_pages`. Mid-request eviction would corrupt a running decode
   — if you add a new eviction trigger (e.g. tier-demotion under pool
   pressure), preserve this gate. Verified statically at the
   paged-prefill lifecycle audit (2026-04-18); no property test locks
   it in yet (backlog item §4 in
   `docs/plans/paged-prefill-followups-2026-04-18.md`).

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

- `docs/projects/tiered-kv-cache.md` — project driving scheduler internals right now.
- `docs/plans/tiered-kv-cache-tasks.md` — milestone ledger.
- `docs/experience/wins/2026-04-15-tiered-kv-m2b-local.md` — what changed at M2b.
- `docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md` —
  preemption policy rationale.
