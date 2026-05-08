# A1 RadixCache audit — production-wired at CUDA runtime admission(simplifies B3 Step 2)

> Per `3efd06d` codex Step 2 prerequisite analysis identifying A1 audit
> as P0 Claude blocker。Code-grep performed across `infer/src/`。
>
> **Finding:A1 IS production-wired**(not shadow-only)at
> `infer/src/scheduler/cuda/runtime/admission.rs:187` via
> `prefix_cache.lookup_or_stage(prompt_tokens, heuristics)`。
> **Implication:Step 2 codex pickup is SIMPLER than estimated** —
> existing RadixCache lookup result can be reused,no new
> `SchedulerHandle` field needed。

## A1 production-wiring evidence

### Existing call sites in CUDA scheduler

`infer/src/scheduler/cuda/runtime/admission.rs:187`:
```rust
self.prefix_cache.lookup_or_stage(prompt_tokens, heuristics)
```

`infer/src/scheduler/cuda/runtime/admission.rs:193`:
```rust
let session_lookup = self.prefix_cache.lookup_session_prefix_or_stage(...)
```

`infer/src/scheduler/cuda/runtime/admission.rs:741`:
```rust
let final_lookup = self.prefix_cache.lookup_or_stage(...)
```

These are PRODUCTION admission paths,not shadow observers。`lookup_or_stage`
returns `(matched_tokens, Vec<BlockId>)` — exactly the data B3 Step 2
needs。

### Construction in CUDA scheduler

`infer/src/scheduler/cuda/core/construction.rs:23`:
```rust
use crate::prefix_cache::RadixCache;
```

`construction.rs:230`:`config.prefix_cache_enabled` gates whether RadixCache
is created and threaded through scheduler。Default-enabled per main.rs
`!args.disable_radix_cache`。

### CLI flag

`infer/src/main.rs:158`:`--disable-radix-cache` flag exists(default OFF
= radix-cache enabled)。

## Step 2 architectural simplification

Codex `c0ddd4f` originally framed Step 2 as:
> Compute prefix_hit_tokens at submit time via RadixCache.match_prefix
> ...add Arc<RwLock<RadixCache>> field to SchedulerHandle...

**This is NOT the right approach** because:
1. SchedulerHandle is an HTTP-layer abstraction(per `types.rs:608`)
2. RadixCache lives in `scheduler::cuda::runtime`(backend-specific)
3. Cross-layer field would violate ARLE backend isolation rules
   (per `CLAUDE.md` §Backend isolation)

**Refined Step 2 architectural path**:

Option A(recommended)— **CUDA-runtime admission integration**:
1. `runtime/admission.rs` already calls `lookup_or_stage` returning matched_tokens
2. Add CUDA-side admission gate that consumes `prefix_hit_tokens` AFTER lookup
3. Use `PrefixAwareAdmission` from `policy.rs` at this CUDA-internal site
4. Per-request hint flow:`request.tokens` → `lookup_or_stage` → `matched_tokens`
   → `SchedulerSignals { prefix_hit_tokens: matched_tokens, ... }` →
   `PrefixAwareAdmission::allow(signals)`

Option B(rejected)— SchedulerHandle field:
- Violates backend isolation
- Requires Arc<RwLock<>> contention at HTTP layer
- Pre-computes lookup that runtime/admission would do anyway

## Step 2 LOC scope refinement

Original `c0ddd4f` estimate:250-350 LOC。Refined per A1 audit:

| Component | Original | Per A1 audit |
|---|---:|---:|
| RadixCache field on SchedulerHandle | 30-50 LOC | **0 LOC**(reuse runtime/admission existing) |
| Signal computation at submit time | 100-150 LOC | **0 LOC**(use lookup_or_stage result) |
| CUDA runtime admission gate(NEW) | n/a | ~50-100 LOC(add post-lookup admission check) |
| Config flag for policy choice | 30-50 LOC | 30-50 LOC |
| Tests + multi-tenant bench | 100 LOC | 100 LOC |
| **Total** | **250-350 LOC** | **~180-250 LOC** |

**Step 2 wall-time refined**:0.5-1 day → **0.25-0.5 day**(simpler architecture)。

## A1 readiness summary

- ✅ A1 RadixCache integrated at CUDA runtime admission(production)
- ✅ `lookup_or_stage(tokens, heuristics)` returns matched_tokens directly
- ✅ Default-enabled via `--disable-radix-cache` opt-out
- ✅ Backend isolation preserved(RadixCache in CUDA-only path,Metal has
  its own per `metal/prefix_cache.rs`)

A1 readiness:**FULLY UNBLOCKED** for B3 Step 2 codex pickup at CUDA-side
admission integration。

## Codex Step 2 pickup directive(refined)

1. **Add admission gate post-lookup** in `runtime/admission.rs`:
   - After `lookup_or_stage(prompt_tokens, heuristics)` returns `(matched_tokens, blocks)`
   - Construct `SchedulerSignals { queued_requests, prefix_hit_tokens: matched_tokens, ... }`
   - Use `PrefixAwareAdmission::allow(signals)` instead of unconditional accept
2. **Wire policy config**:CLI flag `--admission-policy {queue-bound,prefix-aware}`
3. **Tests**:integration test for warm-vs-cold session ordering
4. **Bench**:multi-tenant 4-conc 6k-system burst → expect 318 ms → 157 ms TTFT

## Cross-references

- Step 1 LANDED:`7c8fd61` + `c30e298`(byte-identical regression)
- Codex Step 2 prereq analysis:`3efd06d`
- Scope refinement:`c0ddd4f`
- A1 production sites:`runtime/admission.rs:187/193/741`
- PrefixAwareAdmission impl:`policy.rs:98-130`
- Skill v1.5.1:`9f65b4d`

## Status

- ✅ A1 audit complete:production-wired at CUDA admission
- ✅ Step 2 architecture simplified(CUDA-side,not SchedulerHandle field)
- ✅ Step 2 scope reduced(180-250 LOC,0.25-0.5d)
- ⏳ Codex pickup:`runtime/admission.rs` gate + policy wiring + bench

## Rule

**Phase 0 audit BEFORE substrate pickup**:check if the dependency
is already production-wired in adjacent layer。In ARLE specifically,
**RadixCache is in CUDA runtime not HTTP handle** — codex's original
SchedulerHandle field plan would have violated backend isolation。

Per skill v1.5.1 anti-pattern #16(implicit-coupling-via-shared-default)
generalization:**when planning new wiring,inventory existing wiring
in the closest production-wired layer first**。Saved 70-100 LOC + 0.5d
wall-time on Step 2 by routing through existing `lookup_or_stage` instead
of duplicating at SchedulerHandle level。

This applies to ALL "new admission/gate/check" features:audit existing
admission paths(both HTTP and backend-runtime layers)before deciding
where to plumb the new logic。Wrong layer = backend isolation violation
+ duplicated work。
