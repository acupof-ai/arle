# Active KV swap-out — unify prefix-cache demotion with preemption

## Thesis

KV pages are pages. The storage layer (T0 paged pool, T1 host-pinned
pool, T2 disk store, plus COW + refcount + LRU) has no idea whether a
page belongs to a "completed prefix cache entry" or a "currently-active
decode slot whose owner is asleep". Both are immutable once written
(WORM), both can be demoted under memory pressure, both can be
re-hydrated on access.

Today the demote/promote machinery only fires for the prefix-cache
case. Active in-flight requests can't be evicted from T0 — when the
GPU pool is full, admission stops admitting and lets clients pile up
in `Phase::WaitingFetch` *only* when their prefix is staged on T1/T2.
A new request with no prefix overlap waits forever (or rather, until
some active request finishes naturally).

That's the gap. The 2026-04-26 SGLang head-to-head exposed it
directly:

```
peak_active = 5         # 16 client requests, only 5 admitted
peak_waiting = 11       # 11 sit in queue
peak_kv_util = 98 %     # T0 pool saturated
kv_store = sub:0        # T1 NEVER got a demote during the bench
prefix_hit_rate = 0 %   # synthetic prompts → no prefix sharing
```

The 604 MB host pinned pool sat idle while 11 requests waited 16 s for
a turn at first token, all because the storage layer's promote/demote
hooks were never wired to the admission/preemption path.

## What already exists

Inventory of the pieces this work consumes:

- `cuda_kernels::PagedKVPool` — T0 (GPU paged), already supports `cow_*`
  / `alloc_tokens` / `free_slot` / `release_pages` / `retain_pages`.
- `infer::kv_tier::SharedHostPinnedPool` — T1 (host pinned, 604 MB
  default), with refcount + LRU + region allocation already wired.
- `infer::kv_tier::backend::KVBackendStore` /
  `infer::kv_tier::backend::KVBackendFetch` — page-level store/fetch
  primitives, async, with completion tracking. Currently driven only
  by the prefix-cache demotion coordinator.
- `infer::kv_tier::ReadmissionPlan` /
  `infer::kv_tier::ReadmissionSource` — the staged-readmission state
  that drives `Phase::WaitingFetch → Phase::Decoding` transition.
  Source variants: `HostPinned { region }` (T1) and
  `Disk { .. } / Remote { .. }` (T2). All wired already.
- `infer::scheduler::cuda::request::Phase` — the per-request state
  machine. `Phase::WaitingFetch` is the "my staged prefix blocks are
  parked off-T0, waiting to be promoted back" state; today
  `runtime.rs::adopt_promoted_prefix` always lands a fetched plan in
  `Phase::Prefilling` (because the staged plan only covers sealed
  prompt blocks; the decode-side state past the prompt boundary
  isn't preserved). `Phase::Decoding` is "I'm in the running batch".
  Phase 2 needs a NEW promotion arm that lands directly in
  `Phase::Decoding` for the WholeKv plan variant.
- `infer::scheduler::cuda::decode::retract_decode_to_fit` —
  retraction-with-recompute for the case where a decode batch can't
  fit. Triggered DURING decode-step planning, not from admission.
- `PreemptionMode { Recompute, Swap }` enum — defined, plumbed through
  `SchedulerConfig` / `BatchSchedulerConfig` / re-exports / tests.
  The actionable gap is narrower: **the CUDA runtime admission and
  decode paths never branch on the configured mode** — admission is
  hard "fail and queue" regardless of mode, and the decode-time
  `retract_decode_to_fit` is recompute-only with no Swap arm.

## What's missing

Two trigger points and one transition.

### Trigger 1 — Admission preemption

Today (`runtime.rs::admit_one_waiting`):
```text
if pool.free_pages() < required_for_prompt:
    return Err(NoCapacity)   # → request stays in waiting queue
```

Needed:
```text
if pool.free_pages() < required_for_prompt:
    let need = required_for_prompt - pool.free_pages();
    if let Some(victims) = preempt_active_to_release(need, mode):
        for victim in victims:
            preempt_active(victim, mode)   # → Phase::WaitingFetch
        admit()                            # now succeeds
    else:
        return Err(NoCapacity)             # truly nothing to free
```

`preempt_active(slot, mode)`:
- `mode = Recompute`: call the existing
  `decode.rs::requeue_preempted_decode(slot)`. It already does the
  right thing for the mid-decode retract case — frees the slot,
  re-queues the request at the front of the waiting queue with
  cached prompt + sampler state, so the next admission tick picks it
  up and re-prefills from scratch. No `Phase::WaitingFetch` involved.
- `mode = Swap`: emit `KVBackendStore` for slot's owned page blocks → T1
  (or T2 if T1 full), capture decode cursor into the new
  `ReadmissionPlan::WholeKv`, set `req.phase = Phase::WaitingFetch`.
  Page release MUST NOT call `requeue_preempted_decode` — that helper
  finishes the request and re-enqueues it for recompute, which would
  discard the staged plan. Phase 1 needs to factor a smaller
  `release_slot_pages_only(slot)` helper out of the existing
  recompute path; Swap calls that helper, leaves the request
  attached to the staged plan, and lets the promotion handler resume
  Decoding (Phase 2).

### Trigger 2 — Decoding-step swap-in (NEW path required for Phase 2)

The existing `Phase::WaitingFetch → Phase::Decoding` path is **not**
direct — it's `WaitingFetch → fetch staged prefix → Phase::Prefilling
(re-prefill the suffix the staged plan didn't cover) → Decoding`. The
plan validation explicitly requires `matched_len <= prompt_tokens.len()`,
i.e. the staged blocks are *sealed prompt prefix* only. There's no
guarantee the cache holds anything past `prompt_tokens.len()`.

For Swap-mode active preemption the stored KV runs **past** the prompt
boundary into generated-token state. We need a new readmission variant
that:
1. Tags the staged plan as `WholeKv { last_token_pos: usize }` (vs the
   existing `Prefix { matched_len }`).
2. Promotes T1/T2 → T0 the same way (the page-store transport doesn't
   care).
3. Restores the slot's `position` / `generated_tokens` accounting from
   the saved metadata, then jumps **directly** to `Phase::Decoding`
   without going through prefill.

This is the meaningful new code at the readmission side. The
page-level `KVBackendFetch` path is reused as-is; the new bit is the
"land in Decoding, not Prefilling" branch in the readmission state
machine.

`Recompute` mode is simpler: drop pages, re-enter as if the prompt
just arrived (`Phase::Prefilling`). The existing
`requeue_preempted_decode` already does this for the mid-decode
retract case; admission preemption can call the same helper.

### Transition — victim selection

The decode-side `retract_victim_pos` exists, picks lowest-priority by
recency / decoded-tokens score. Admission preemption can reuse the
same scoring but applied to the running batch.

## Implementation plan

### Phase 1 — Recompute admission preemption (smaller blast radius)

1. Lift `retract_victim_pos` from `decode.rs` into a shared module so
   admission can reuse it.
2. In `runtime.rs::admit_one_waiting` (or wherever capacity check
   happens), on capacity miss: call `try_preempt_for_admission(need)`
   that loops `retract_victim_pos` → `requeue_preempted_decode`
   until enough pages free or no more victims.
3. Drive by `SchedulerConfig::preemption_mode`. When
   `PreemptionMode::Recompute` (default), use the existing recompute
   path (drop pages, reset to Phase::Prefilling).

**Bench gate**: c=16 `peak_active` should rise from 5 → 16, TTFT p50
should drop substantially. Recompute means the preempted requests
re-prefill, so total throughput might stay flat or dip slightly; the
win is fair admission and lower tail TTFT.

### Phase 2 — Swap admission preemption

1. Add a `WholeKv` variant to `ReadmissionPlan` (alongside the
   existing prefix-cache plan) carrying the slot's saved cursor
   (`position`, `generated_tokens.len()`, sampler state). The slot
   stays in `Phase::WaitingFetch` while staged, but on promotion
   takes the new "skip prefill, resume decode" branch.
2. `preempt_active_swap(slot)`:
   - Allocate T1 region(s) for the slot's pages via `SharedHostPinnedPool`.
   - Issue `KVBackendStore` for each owned page block.
   - Capture decode cursor into the new `WholeKv` plan.
   - On store completion, set `req.phase = Phase::WaitingFetch` with
     `ReadmissionPlan::WholeKv { ... }` pointing at the just-stored
     T1 region.
   - Free GPU pages via existing `pool.free_slot(slot)`.
3. Promotion-side: extend `runtime.rs` readmission handler to switch
   on plan variant — `Prefix` → existing `Phase::Prefilling` path;
   `WholeKv` → new `Phase::Decoding` path that restores the slot's
   cursor and re-enters the running batch directly.
4. Driven by `SchedulerConfig::preemption_mode = Swap`.

**Bench gate**: vs Phase 1, throughput should improve (no recompute
cost). Tail TTFT might rise slightly because swap-in adds a host→GPU
copy, but that's bounded by HBM-bandwidth × stored-page-count
(~10s-100s of microseconds per slot's-worth of pages).

### Phase 3 — Tier-aware victim selection + T2 spillover

1. When T1 is full, T2 (disk) takes the spillover. Existing
   `KVBackendStore` already supports disk.
2. Victim selection prefers retraction targets whose pages are
   already on T1 (no GPU eviction needed). Avoid double-write.

## Risk gates

- **R1 — Recompute storms**: at low pool capacity, every admission
  could preempt a near-finished request, then re-prefill it,
  thrashing. Mitigate by:
  - Don't preempt requests that have generated > N tokens (sunk cost);
    pick fresher ones.
  - Cap preemption rate per second.
- **R2 — Swap copy bandwidth**: T0↔T1 is ~25 GB/s on host pinned. A
  full slot's KV (16-page slot × 64 KB/page × 36 layers ≈ 36 MB)
  takes ~1.5 ms one direction. The realistic bursty case — swap all
  16 active slots out and back to admit a wave — moves
  16 × 36 MB ≈ 576 MB ≈ **23 ms one way** (~46 ms round trip). For
  steady-state swap churn (a few preemptions per second) the absolute
  bandwidth cost is negligible. Worst case (whole pool ~4 GB) is
  ~160 ms — plan for it with batch copies and async overlap with
  decode.
- **R3 — T1 / T2 admission contention**: same store backend is used
  by prefix-cache demotion. Need fair scheduling between the two
  use cases; right now prefix demotion has priority.

## What gets retired

Once Phase 2 ships, the existing `decode.rs::retract_decode_to_fit`
mid-decode-step retract becomes a fallback for "even after
preemption, we still can't fit". The healthy path is admission
preemption, not decode-time panic.

## Acceptance bench

Recipe: same matched flags as the 2026-04-26 SGLang head-to-head.

| metric | before (`b73c4e97`) | after Phase 1 (target) | after Phase 2 (target) |
|---|---:|---:|---:|
| peak_active @ c=16 | 5 | 16 | 16 |
| peak_waiting @ c=16 | 11 | 0 | 0 |
| TTFT p50 @ c=16 | 16 357 ms | < 8 000 ms | < 6 000 ms |
| out tok/s @ c=16 | 65.92 | > 90 (parity ITL × 16 active) | > 110 |
| infer/sglang ratio @ c=16 | 0.474 | > 0.65 | > 0.79 |

SGLang baseline: 139 tok/s @ c=16.

## Cross-references

- SGLang head-to-head:
  [`wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-vs-sglang-c1-c16.md)
- Workspace over-reservation errors entry:
  [`errors/2026-04-27-prefill-activation-budget-undersize.md`](../experience/errors/2026-04-27-prefill-activation-budget-undersize.md)
- Existing tier coordinator:
  `infer/src/kv_tier/coordinator.rs` (and the
  `coordinator::tests::store_roundtrip_through_disk_store` smoke test
  that already proves the page-level path works end-to-end).
- COW insight memory:
  `memory/project_cow_kv_cache_insight.md`.
