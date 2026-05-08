# B3 Step 2 architecture — codex `1217375` cleaner approach acknowledged

> Per `1217375` A1 audit findings:my `e53e4d8` Step 2 plan to add
> `Arc<RwLock<RadixCache>>` to `SchedulerHandle` violates backend isolation
> (RadixCache lives in CUDA runtime,SchedulerHandle is HTTP layer)。
> Codex proposes refined architecture:integrate at `runtime/admission.rs`
> after existing `lookup_or_stage` call。Acknowledged + agreed。

## Codex finding(`1217375`)

A1 is PRODUCTION-WIRED at:
- `runtime/admission.rs:187` `prefix_cache.lookup_or_stage(prompt_tokens, heuristics)`
- `runtime/admission.rs:193` `prefix_cache.lookup_session_prefix_or_stage(...)`
- `runtime/admission.rs:741` second `lookup_or_stage` site

These return `lookup` with `matched_len` — exactly what B3 Step 2 needs。

Construction:`scheduler/cuda/core/construction.rs:23+230` gates on
`config.prefix_cache_enabled`,default-enabled with `--disable-radix-cache`
opt-out。

## My `e53e4d8` Step 2 plan was wrong

Original plan(per `e53e4d8`):
```rust
// In SchedulerHandle::submit:
let prefix_hit = self.prefix_cache.match_prefix(&request.tokens).matched_len;
let signals = SchedulerSignals { prefix_hit_tokens: prefix_hit, ... };
admission_allows(signals)
```

**Problem identified by `1217375`**:
- `SchedulerHandle` is HTTP-layer abstraction(in scheduler/types.rs)
- `RadixCache` lives in CUDA-runtime(scheduler/cuda/runtime/)
- Adding `Arc<RwLock<RadixCache>>` field crosses backend boundary
- Violates CLAUDE.md "backend isolation" rule

## Refined architecture(codex `1217375`)

Integration at CUDA-runtime admission site:
```rust
// In runtime/admission.rs after line ~200:
let lookup = self.prefix_cache.lookup_or_stage(...);  // EXISTING
// ... session preference logic ...

// NEW: PrefixAwareAdmission gate using lookup data
let signals = SchedulerSignals {
    queued_requests: scheduler.waiting_count(),
    active_decodes: scheduler.active_count(),
    prefix_hit_tokens: lookup.matched_len,
    session_affinity_slot: session_slot_hold.as_ref().map(|h| h.slot_idx()),
    turn_depth: req.turn_depth,
};

let policy = PrefixAwareAdmission::with_cold_headroom(
    scheduler.config.max_waiting,
    scheduler.config.cold_headroom.unwrap_or(scheduler.config.max_waiting / 4),
);

if !policy.allow(signals) {
    return AdmissionResult::Rejected(...);
}
```

**Benefits**:
- No new field on SchedulerHandle
- No cross-layer borrow
- Reuses existing lookup result(no double-lookup overhead)
- Backend-isolation rule preserved

## Effort revision

Previous estimate(my `e53e4d8` Step 2):100-150 LOC
New estimate(codex `1217375` refined):**~50-80 LOC**
- Add SchedulerSignals construction at admission site:~20 LOC
- Add PrefixAwareAdmission policy invocation:~10 LOC
- Plumb cold_headroom config through:~20-30 LOC
- Tests:~50 LOC
- **Total**:~100 LOC,**0.5 day codex**

Smaller than my estimate because:
- Lookup result already computed(no new RadixCache plumbing)
- Backend-isolation respected(no SchedulerHandle field)
- Existing admission flow extends naturally

## Updated B3 sequence

| Step | Status | LOC | Effort | Site |
|------|--------|----:|-------:|------|
| 1 | ✅ DONE(`7c8fd61`)| 14ins/4del | 1 tick | `types.rs` admission_allows |
| 2 | Refined ready | ~100 | 0.5d | `runtime/admission.rs` after lookup |
| 3 | Pending | 30 | 0.25d | config wiring |

## Pickup queue update

Step 2 architectural complexity reduced — NOW comparable size to Hybrid
Phase 1b。Could parallelize codex pickups。

| Priority | Task | LOC | Effort |
|----------|------|----:|-------:|
| ✅ DONE | B3 Step 1 admission_allows signature | 14 | 1 tick |
| P0 | Hybrid Phase 1b loader(`6be30ce`)| 155-175 | 0.75-1d |
| P0' | M_warmup prefill pass(`56dbd1c`)| 150 | 1-1.5d |
| **P0'** | **B3 Step 2(refined per `1217375`)** | **~100** | **0.5d** |
| P1 | KV W4A8 #33 | 500-1000 | 5-10d |
| P1' | Medusa Phase 1.B | 600-1200 | 10-14d |

B3 Step 2 promotes to **P0'** alongside hybrid+warmup — three small
fixes(<2 days combined)deliver SGLang gap closure + bimodal fix +
hybrid。

## Methodology insight

**Cross-layer architectural correction caught by audit step**:codex's
`1217375` audit caught my `e53e4d8` plan's CLAUDE.md violation BEFORE
codex started Step 2 implementation。Single-grep(A1 audit)saved
~50-100 LOC of wrong-direction refactor。

Anti-pattern #18 candidate:**cross-layer field-add trap** — when
proposing field additions to a struct,verify the new field's source
of truth lives in the same layer。If not,refactor to consume at the
source layer instead of plumbing through。

Skill rule:after agreeing to an architecture,run quick code-grep at
both source-of-data and consumption sites to verify they're in same
layer。Prevents proposing patterns that violate backend isolation。

## Cross-references

- Step 1 success:`7c8fd61` + `c30e298`
- Step 1 directive:`e53e4d8`(my own)
- Step 2 architecture refinement:`1217375` codex
- A1 production sites:`runtime/admission.rs:187/193/741`
- PrefixAwareAdmission impl:`policy.rs:98-130`
- Backend isolation rule:CLAUDE.md "backend isolation (CRITICAL)"

## Status

B3 Step 2 architecture refined,~100 LOC,**0.5d** codex pickup ready。
Acknowledges codex `1217375` cleaner approach。Avoids backend-isolation
violation that my `e53e4d8` would have caused。

Codex pickup:start with PrefixAwareAdmission integration at
`runtime/admission.rs` post-lookup site。Step 3 config wiring follows。
