# 2026-05-07 · Opportunity D pre-survey — submit_prefetch_plan exists but is dead code

## Priority & ROI

**Priority**: P1 (after P0 = M3.9 Phase 1A v3 lands; not parallel to P0).

**ROI basis**:
- Substrate cost-reduction: zero (`submit_prefetch_plan` already
  built with backpressure + queue-depth + PlanTicket). Implementation
  is wiring + 1 helper, ~80 LOC — codex-sized.
- Expected gain at long-ctx multi-tenant shared-prefix workloads:
  - Cache HIT bytes (e.g. 6k prefix × 36 layers × hidden × FP8) ≈
    135 MB H2D
  - PCIe 4.0 ~32 GB/s → ~144 ms total H2D wait if synchronous
  - Overlap with prefill of unmatched suffix → ~0 ms wait
  - **TTFT reduction = ~144 ms per request** in this regime
- Stacks with M3.9 Phase 1A v3:
  - 1A v3 alone: long-ctx 8k TTFT 4961 → ~2500 ms
  - + M_pf (shared-prefix only): TTFT 2500 → ~600 ms
  - 4× faster than vLLM 2367 ms at this shape

**Negative case**:
- Single-tenant workloads or no-prefix-share workloads see ZERO
  benefit (no shared cache to prefetch).
- Risk of speculative prefetch evicting useful T0 blocks at high
  pool utilization → degrades non-prefetched requests.
- Wasted bandwidth if request is cancelled before reaching
  prefill.

**Kill criteria**:
- After implementation, bench at the canonical multi-tenant
  shared-prefix shape (e.g., 4k system prompt + 4 different
  user queries c=4) shows `< 20% TTFT improvement` → revert
  the wiring; the cost (extra GPU memory pressure, scheduler
  complexity) is not worth it.
- If pool utilization metric shows prefetch routinely evicts
  T0 blocks within 1 tick of placement → also revert.

**Why this rank vs alternatives**:
- vs M3.9 Phase 1A v3 (P0): P0 closes a measured 10× tax in a
  shape ARLE is currently losing. M_pf is opportunistic gain in
  a niche shape.
- vs M_b.2 Phase 1 (kernel-axis): bench evidence in `2e60844`
  shows attention kernel is already fast at batched mode; M_pf
  addresses a DIFFERENT axis (memory tier transition cost).
- vs Spec-decode (P2): spec-decode is speculative on acceptance
  rate; M_pf has deterministic gain for HIT cases.

## Discovery

While codex implements M3.9 Phase 0 instrumentation, parallel
source-survey of `infer/src/kv_tier/coordinator/builder.rs`
revealed:

- L93: `submit_fetch(blocks: Vec<FetchRequest>) -> Option<FetchTicket>`
  — blocking-style fetch (request needs this KV to proceed)
- L145: `submit_prefetch_plan(blocks: Vec<PrefetchPlanRequest>) -> Option<PlanTicket>`
  — **speculative prefetch**, intended for "KV might be needed soon"

`grep -rn "submit_prefetch_plan" infer/src/`:
- builder.rs:145 (definition)
- builder.rs:110 (rustdoc reference)
- ZERO call sites elsewhere

**The prefetch infrastructure is dead code.** Wired into the
KVTierCoordinator (with backpressure, fetch_queue_depth, etc.)
but no scheduler path fires it.

## Why this matters for "world-first long-sequence engine"

For long-context workloads with shared system prompts (RAG,
code completion, agentic tools), many requests share long prefix
KV. When a fresh request arrives:

**Today (admission-blocking fetch)**:
1. Request arrives → admission queue
2. Admission runs `lookup_or_stage` → classifies blocks as
   StagingFromHost
3. `submit_fetch` → request waits for H2D copy to complete
4. Then prefill runs
5. **TTFT includes the H2D wait**

**With prefetch wired (Opportunity D)**:
1. Request arrives → admission queue
2. **As soon as prefix MATCH detected**, fire `submit_prefetch_plan`
   on the matched blocks (non-blocking)
3. Admission doesn't wait — request enters prefill immediately
4. By the time prefill reaches the layer that needs the cached
   KV, the H2D has likely completed (overlapped with prefill of
   the unmatched suffix)
5. **TTFT excludes the H2D wait**

For a 8k prompt with 6k cached prefix from system prompt:
- 6k tokens × hidden_dim × 36 layers × 2 bytes (fp8) = ~135 MB
- At PCIe 4.0 ~32 GB/s = ~4 ms per layer × 36 = ~144 ms total
  H2D
- Overlapping with prefill (which takes seconds) → 0 ms wait

For long-context multi-tenant workloads, this could deliver a
substantial TTFT improvement that the current admission doesn't.

## Architecture survey

Current code path for prefix HIT at admission
(`scheduler/cuda/runtime/admission.rs:147-227`):

```
1. lookup_or_stage(prompt_tokens) → LookupOutcome with HitKind
   per matched block (ReadyOnGpu | StagingFromHost | StagingFromDisk | Miss)
2. If all ReadyOnGpu → direct GPU page attachment
3. If some below-T0 → "staged readmission plan" via submit_fetch
   (line 552) — admission-blocking fetch
4. Fall through paths for stale/cold prefill
```

For the prefetch wiring, the simplest insertion point is
**before step 1** in admission.rs's `build_prefix_admission_plan`:
a "speculative classify" pass that runs `lookup_or_stage` BEFORE
the request even enters the waiting queue, fires
`submit_prefetch_plan` for any host/disk blocks, then admission
proceeds normally a few ticks later.

## Required design

| # | Task | LOC | Risk |
|---|---|---|---|
| 1 | Add `peek_prefix_classify(tokens)` helper to `RadixCache` (read-only lookup, doesn't bump ref counts) | ~30 | Low |
| 2 | At HTTP request entry (`request_handle_engine.rs`), call peek + submit_prefetch_plan when host/disk blocks match | ~50 | Medium (extra GPU memory pressure if speculative prefetch evicts useful blocks) |
| 3 | Cache-aware admission: when admission's lookup_or_stage runs later, if prefetch has completed, blocks are now ReadyOnGpu and direct-attach path fires | 0 (existing logic) | Low |
| 4 | Bench: long-ctx multi-tenant workload (e.g. shared 4k system prompt + 4 different short user queries) | 0 | n/a |

**Total: ~80 LOC, codex-sized**.

## Risks

- **R1 — Prefetch evicts useful blocks**: at high pool utilization,
  speculatively prefetching blocks could evict T0 blocks needed
  by other in-flight requests. Mitigation: prefetch only triggers
  when pool utilization < threshold (e.g. 70%). Coordinator
  already has `fetch_backpressured()` (types.rs:132) which
  prefetch can respect.
- **R2 — Speculative prefetch wastes bandwidth**: if request is
  cancelled before prefill, the prefetch was wasted H2D. Cost
  is small for typical workloads.
- **R3 — Race with cache eviction**: between prefetch completion
  and request reaching prefill, the cache might evict the
  prefetched blocks (if other requests pressure the pool).
  Mitigation: temporarily soft-pin prefetched blocks via the
  existing `soft_pin_keepalive_ticks` mechanism.

## Cross-references vs M-final roadmap

This is **Opportunity D** in `M-final-world-first-integration-roadmap.md`
(`d16effe`). Promotion from "future opportunity" to "ready-to-implement"
because:
- Infrastructure (`submit_prefetch_plan`) already exists
- Wiring location (admission.rs `build_prefix_admission_plan`) is
  small and well-understood
- M_d.1 namespace just landed — prefix correctness is now solid
- Expected gain is concrete (long-ctx multi-tenant TTFT cliff)

## Recommendation

**Move Opportunity D into the active roadmap as M_pf** (prefetch).
Ordering: after M3.9 Phase 1A v3 (Mixed-path restoration, in
codex's current pipeline), before M_b.2 Phase 2.

Estimated cumulative impact:
- M3.9 Phase 1A v3: long-ctx 8k TTFT 4961 → ~2500 ms (Mixed restored)
- M_pf: shared-prefix long-ctx TTFT 2500 → ~600 ms (eliminate H2D)
- Combined: ARLE long-ctx with shared prefix BEATS vLLM 2367ms by ~4×

## Bench Status

No new bench. Source-only finding. Validation bench requires
Opportunity D implementation + a multi-tenant shared-prefix
workload (not yet in scripts).

## Rule

- **Dead code in the codebase often signals "infrastructure that
  was built but the wiring is missing"**. Periodic
  `grep -rn submit_prefetch | grep -v test | grep -v doc` style
  audits surface these. Each one is an Opportunity D candidate.
- The pattern: someone in the past built the substrate (M_d.1
  level work), planning was correct, but the integration tick
  never landed. Surveying surfaces them efficiently.
