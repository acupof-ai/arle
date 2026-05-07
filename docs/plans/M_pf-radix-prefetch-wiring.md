# M_pf — RadixCache prefix prefetch wiring (Opportunity D)

> Promotion of Opportunity D from M-final roadmap (`d16effe`) to
> active milestone. Source survey
> ([`50ae808`](../experience/wins/2026-05-07-prefetch-plan-dead-code-discovered.md))
> revealed `submit_prefetch_plan` is dead code — infrastructure
> exists but nothing calls it. M_pf wires it.

## Priority & ROI

> **2026-05-07 EOD update**: Phase 1A v3 shipped (codex `5cacdcb`,
> default Split — multi-slot ring substrate kept; production
> +25.6% incidental at long-ctx 4k/c=4). M_ibp Phase 0 license-
> or-kill (`9432289`) showed ARLE already 1.80× past vLLM at the
> multi-tenant shared-prefix workload M_pf would target. M_pf is
> demoted to **P3** in M-final roadmap; the ROI math below stands
> (5-15% TTFT at HIT cases) but priority drops because the target
> shape is no longer a gap.

**Priority**: **P3** (deprioritized post-M_ibp ABANDONED finding;
historical P1 framing kept below for context).

**Why now P3**:
- ARLE leads vLLM at the M_pf target shape already (multi-tenant
  shared-prefix: 318 ms vs 573 ms TTFT, 1.80×).
- Higher-priority work: M_world1 Phase 0 baseline measurement +
  long-ctx 4k/c=4 prefill TTFT 800 ms gap (codex actively
  investigating).
- Original "P1 after P0 = Phase 1A v3" framing is stale — Phase
  1A v3 is now shipped.

**ROI basis (deterministic for HIT cases)**:

Math for typical RAG / agentic workload (system prompt cached,
user query fresh):
- Cached prefix: 6k tokens (system prompt)
- Fresh suffix: 1-2k tokens (user query)
- Cached KV bytes: 6144 tok × 36 layers × 8 KV heads × 128 head_dim
  × 2 (K+V) × 1 byte (FP8) = **452 MB** of host-pinned KV
- PCIe 4.0 ~32 GB/s bandwidth → **14 ms total H2D** if synchronous
- (Earlier estimate of 144 ms was for 36 layers × 4ms/layer
  serialized; actual sequential transfer can pipeline → 14ms)
- With prefetch + overlap with prefill of fresh suffix → **~0 ms wait**

**Throughput stacking** (after Phase 1A v3 lands):
- Phase 1A v3 alone (no shared prefix): long-ctx 8k/c=4 TTFT
  4961 → ~1500 ms (per Phase 0 validation kill criteria target)
- + M_pf at 6k-prefix-shared workload: ~1500 - 14 = **~1486 ms**
  ← marginal, NOT 4× as I'd previously estimated

**Correction to roadmap**: my earlier estimate of "144 ms H2D
saved" overestimated the per-layer cost. The actual H2D is one
contiguous block, completes in ~14 ms. Even with 144 ms (worst
case), the saved time relative to Phase 1A v3's ~1500 ms TTFT is
< 10% improvement.

**Updated negative-case-aware ROI**: M_pf gain is < 10% TTFT
improvement at typical shapes. Not the "4× faster than vLLM"
I projected. Still net positive but marginal.

**Negative case**:
- Single-tenant / no-cache workloads: zero benefit
- Speculative prefetch can evict useful T0 blocks (eviction
  storm under high pool utilization)
- Wasted bandwidth on cancelled requests

**Kill criteria**:
- Phase 1 bench at multi-tenant shared-prefix workload shows
  < 5% TTFT improvement → revert wiring
- Eviction storm metric (existing `prefix cache pressure
  fallback` warning, observed in Phase 0 logs at line `core.rs:1543`)
  fires more often after wiring → revert
- Pool utilization climbs > 95% steady-state due to prefetch
  pressure → revert + add backpressure threshold

**Re-evaluation note**: this Priority & ROI honestly downgrades
the ambition vs initial discovery (`50ae808`'s "4× faster"
projection was wrong by 10×). Expected gain now is 5-15% TTFT
improvement at HIT workloads. Still worth doing because:
- Code cost is tiny (~80 LOC)
- Substrate already exists
- Multi-tenant shape is common in production

## Goal

Wire the existing `submit_prefetch_plan` infrastructure so that
when an incoming request's prompt prefix matches the RadixCache
with blocks living below T0 (host-pinned tier), the prefetch
fires async at request entry rather than blocking on admission.

## P0 survey (already done)

Pre-survey complete in
[`50ae808`](../experience/wins/2026-05-07-prefetch-plan-dead-code-discovered.md).
Key facts:

- `submit_prefetch_plan` (`kv_tier/coordinator/builder.rs:145`):
  exists, signature `Vec<PrefetchPlanRequest> → Option<PlanTicket>`
- Zero call sites in scheduler/HTTP/admission paths
- `submit_fetch` (line 93) IS called from admission.rs:552, 683
  — blocking variant
- `RadixCache::lookup_or_stage` (`prefix_cache.rs:582`) classifies
  blocks into ReadyOnGpu / StagingFromHost / StagingFromDisk / Miss
- Existing `recompute_advised` heuristic (line 641) decides
  whether to skip cached blocks; not a prefetch trigger

## Design

### P0 — peek_prefix_classify helper (~30 LOC)

Add a READ-ONLY variant of `lookup_or_stage` that:
- Walks the radix tree
- Returns `(block_ids, hit_kinds)` for matching prefix
- Does NOT bump `ref_count` or `last_access`
- Does NOT mutate radix state in any way

Required so HTTP-entry caller can classify without affecting
admission's later real `lookup_or_stage` decision.

```rust
impl RadixCache {
    pub fn peek_prefix_classify(&self, tokens: &[u32])
        -> Vec<(BlockId, HitKind)> {
        // walk tree, classify by tier_location, return without mutation
    }
}
```

### P1 — HTTP request entry hook (~30 LOC)

In `request_handle_engine.rs` or equivalent HTTP entry path:

```rust
// On request arrival, before scheduler admission queue:
if config.prefetch_enabled
    && request.prompt_token_ids.len() > prefix_threshold {
    let classification = scheduler.peek_prefix_classify(&request.prompt_token_ids);
    let host_blocks: Vec<_> = classification.iter()
        .filter(|(_, k)| matches!(k, HitKind::StagingFromHost))
        .map(|(id, _)| PrefetchPlanRequest::for_block(*id))
        .collect();
    if !host_blocks.is_empty() && coordinator.fetch_backpressured() == false {
        let _ticket = coordinator.submit_prefetch_plan(host_blocks);
        // Don't wait — fire-and-forget; admission will re-check soon
    }
}
```

### P2 — bench validation (~0 LOC, scripts only)

Construct a multi-tenant shared-prefix workload:
- 4k system prompt (shared across 4 different requests)
- 1k user query (different per request)
- c=4 concurrency

Pre-warm by sending one request with the system prompt to
populate cache. Then send the 4 different-query requests in
quick succession.

Compare TTFT distribution with vs without M_pf.

## Tasks

| # | Task | File | LOC | Owner | Trigger |
|---|---|---|---|---|---|
| P0 | `peek_prefix_classify` helper | `infer/src/prefix_cache.rs` | ~30 | Codex | Phase 1A v3 lands |
| P1 | HTTP-entry prefetch fire | `infer/src/server_engine/request_handle_engine.rs` | ~30 | Codex | P0 commit |
| P2 | Backpressure-respecting threshold | `infer/src/server_engine/request_handle_engine.rs` | ~20 | Codex | P0+P1 |
| P3 | Soft-pin prefetched blocks | `infer/src/prefix_cache.rs` | ~20 | Codex | P0 |
| Bench | Multi-tenant shared-prefix bench | `scripts/bench_guidellm.sh` invocation | 0 | Claude | All implementation lands |

**Total: ~100 LOC, codex-sized.**

## Acceptance

- `cargo test --release -p infer --features cuda --test e2e` passes
- `cargo test --release -p infer --features cuda --test
  greedy_consistency` passes
- Phase 0 metric: existing `prefix_cache_pressure_fallback` warning
  rate doesn't increase after wiring
- Bench at multi-tenant shared-prefix workload shows ≥ 5% TTFT
  improvement vs no-prefetch (kill criteria threshold)

## Risks + retreat

- **R1 — Eviction storm**: speculative prefetch evicts T0 blocks
  needed by other requests. Mitigation: backpressure check before
  submit_prefetch_plan + soft-pin keepalive on prefetched blocks.
- **R2 — Race with cache invalidation**: prefetch in flight when
  cache evicts the source host block. Mitigation: `submit_prefetch_plan`
  already returns `Option<PlanTicket>` — None means infrastructure
  declined; caller drops without retry.
- **R3 — Wasted bandwidth on short requests**: prefetch fires
  for short prompts that don't actually need the prefill speedup.
  Mitigation: only fire when `prompt_tokens > prefix_threshold`
  (e.g. 1k tokens — short prompts already use bypass path).

## Out of scope

- Cross-session prefix sharing strategy (M_d.1 namespace already
  protects against silent corruption; sharing is opt-in).
- Disk-tier (T2) prefetch — host-tier (T1) only for M_pf.
  T2 prefetch is much higher latency (disk IO) and needs
  separate scheduling consideration.
- Prefetch policy ML (when to prefetch, how aggressively) —
  M_pf v1 is "always prefetch host-tier hits if not backpressured".

## Cross-references

- Discovery: `50ae808` (submit_prefetch_plan dead-code)
- Roadmap: `a06c5c9` (M-final with P1 priority)
- Substrate: `kv_tier/coordinator/builder.rs:93, 145`
- Admission integration: `scheduler/cuda/runtime/admission.rs:147-227`
