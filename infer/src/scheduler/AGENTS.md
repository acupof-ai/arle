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
| `forward_batch.rs` | **Inert metadata** for future TP/PP execution (`ForwardBatchKind`, `IntermediateTensorMeta`). F0.7 stage-boundary type slot only — no consumer in the CUDA forward path yet. Do not lift it onto the hot path until P0' multi-GPU F2 collectives ship. |
| `cuda/core.rs` + `cuda/core/` | CUDA `Scheduler<M: ModelForward>` struct. Root file holds the type + `pub use`; the `core/` siblings are pure structural splits (`construction.rs` constructors, `emit_worker.rs` completion-delta worker, `helpers.rs` watermark/spill math, `session_slots.rs` sticky-slot tracking, `state_types.rs` `PendingDecode`/dedupe keys, `warmup.rs` CUDA-graph + cublasLt warmup). One owner; do not introduce a second `Scheduler` impl block outside this directory. |
| `cuda/prefill.rs` | `step_new` — chunked prefill + prefix-hit paths (exact-full, prompt-prefix-of-cached, partial). |
| `cuda/decode.rs` | Batched decode + retract/requeue under KV pressure. |
| `cuda/spec_path.rs` | Speculative-decode admission/draft glue: sparse-KV draft views (`SparseDraftView`), external-draft proposal tracking, verifier batch handoff. Phase 2 surface — see Invariant 11. |
| `cuda/budget.rs` | Page-budget accounting helpers (`estimated_request_target`, `clipped_max_new_tokens_estimate`, page-count math). SGLang-style admission charges prefill cost only; decode pages allocate lazily and OOM is caught by `retract_decode_to_fit`. |
| `cuda/policy.rs` | `TieredKvPolicy` — wires `kv_tier::policy::{PrefetchPolicy, WritePolicy}` into the scheduler's prefetch/store gates with soft-saturation thresholds. Owns the scheduler-side wiring only; the policy enums themselves stay in `kv_tier`. |
| `cuda/request.rs` | Per-request state (`QueuedRequest`, `ActiveRequest`, `Phase`). |
| `cuda/runtime.rs` + `cuda/runtime/` | Single-writer scheduler thread. Root file is the `pub use` surface; the `runtime/` siblings split the loop into `scheduler_loop.rs` (`run` driver + slot assignment + cleanup), `admission.rs` (waiting-queue normalization, prefix admission, cold-prefill fallback, staged-prefix promotion), `fetch.rs` (staged-prefix adopt path, coordinator/emit drains, intake normalization), and `helpers.rs` (`FetchWaiter`, `DeferredWaitingRequest`, session-affinity helpers). |
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
11. **Spec-decode admission lives in `cuda/spec_path.rs`, not `decode.rs`.**
   External-draft proposal lifecycle, sparse-KV view collection, and the
   verifier micro-batch hand-off all enter through `SpecPath`. The decode
   loop only invokes verifier launch + bonus-token commit. Phase 2 status:
   plumbing landed but throughput regressed (-62.8 % vs Phase 1 close on
   the first end-to-end bench, see
   `docs/experience/errors/2026-05-01-phase2-real-spec-regression.md`).
   Throughput claims are paused until a packed K+1 verifier or a
   MagicDec-style sparse-KV self-spec path lands; see Common-mistakes
   bullet on `DraftMode::SelfSpec` for the no-op trap.
12. **Tiered-KV prefetch/store goes through `cuda/policy.rs::TieredKvPolicy`**,
   which wires `kv_tier::policy::{PrefetchPolicy, WritePolicy}` into the
   scheduler-side decision points. The scheduler never asks the
   coordinator to move bytes directly — it submits commands via the
   policy gates and the coordinator owns the byte movement (see
   `kv_tier/AGENTS.md` invariant 2).

## Active priorities touching this module

- **P0 — long-context 32k–128k leadership.** Phase 1 SGLang-row closed
  2026-05-01 (W1/c4 mean 1.609× SGLang); Phase 2 spec-decode plumbing
  landed in `cuda/spec_path.rs` but the first end-to-end bench
  regressed. Verifier admission, K+1 batch packing, and sparse-KV
  scheduling work all land here. Plan:
  [`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](../../../docs/plans/2026-05-01-longctx-spec-decode-phase2.md).
- **P0' — multi-GPU single-node F0–F4.** F0–F2 landed; the scheduler
  side currently exposes `forward_batch.rs` as inert TP/PP metadata and
  routes TP rank work through CUDA bootstrap. Production multi-rank
  serving + the TP=2 throughput bench gate on F2 NCCL forward
  collectives wiring through `LayerCommunicator` (see `model/AGENTS.md`).
  Plan: [`docs/plans/2026-04-28-single-node-multi-gpu.md`](../../../docs/plans/2026-04-28-single-node-multi-gpu.md).
- **P2 — tiered KV staged readmission.** `cuda/runtime/fetch.rs` owns the
  staged-prefix promotion adopt path; `cuda/policy.rs::TieredKvPolicy`
  carries the prefetch/write gates. M0–M4 local landed; M5 RDMA design-ready.

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
- **Picking `DraftMode::SelfSpec` without MagicDec-style sparse KV is
  architecturally a no-op.** Plain self-spec runs the *target* model K
  times to draft + 1 time to verify = K+1 forwards of the **same** model
  → no speedup, often a net regression (we observed −8.7 % on Qwen3-4B
  longctx-32k c=4 with K=1 canary at commit `5eddaab8`/`0cc41f6f`).
  Real speedup requires either (a) `DraftMode::External(path)` with a
  genuinely smaller draft (e.g. Qwen3-0.5B drafting for Qwen3-4B target)
  or (b) MagicDec-style sparse-KV self-speculation that makes the draft
  pass cheap. Plain SelfSpec is only useful as a single-token bit-ident
  canary (`global_spec_draft_k == 1`); the path in
  `cuda/decode.rs::step_decode_launch_with_spec_flag` enforces this gate
  for that reason. Do not raise `K > 1` on `SelfSpec` without first
  landing a real draft cheapening mechanism.
- Treating `acceptance_rate = 100 %` as a Phase 2 win. The single-token
  canary above always reports 100 % because every position is verified
  against the target's own argmax. The metric is meaningful only once
  multi-token speculation runs against an *independent* draft source.
  Cite throughput numbers (effective `total_output_tokens / window`)
  versus the Phase 1 baseline when claiming spec-decode value, not the
  bare acceptance gauge.

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
