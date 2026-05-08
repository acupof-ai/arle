# PrefixAwareAdmission impl ALREADY EXISTS — `policy.rs:98-130` complete,scope is wiring not policy

> Per `a1965ab` SGLang multi-tenant 2× gap finding,investigated
> `infer/src/scheduler/policy.rs` for what's actually missing。
> **Found**:`PrefixAwareAdmission` struct + impl + `with_cold_headroom`
> constructor are FULLY IMPLEMENTED(policy.rs:98-130)。
>
> **Real blocker**:`SchedulerHandle::admission_allows`(`types.rs:628-637`)
> hardcodes `QueueBoundAdmission` AND passes `SchedulerSignals::queue_state(queued, 0)`
> — discarding prefix_hit_tokens entirely。**Fix scope refines from
> ~350 LOC(per codex `aeff965`)to ~150-250 LOC**(signal computation
> + plug-in wiring,not new policy)。

## What ALREADY exists

`infer/src/scheduler/policy.rs:98-130`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixAwareAdmission {
    pub hard_cap: usize,
    pub cold_soft_cap: usize,
}

impl PrefixAwareAdmission {
    pub fn with_cold_headroom(hard_cap: usize, cold_headroom: usize) -> Self {
        let cold_soft_cap = hard_cap.saturating_sub(cold_headroom);
        Self { hard_cap, cold_soft_cap }
    }
}

impl AdmissionPolicy for PrefixAwareAdmission {
    fn allow(&self, signals: SchedulerSignals) -> bool {
        if signals.queued_requests >= self.hard_cap {
            return false;
        }
        let effective_cold_cap = self.cold_soft_cap.min(self.hard_cap);
        if signals.is_cold_request() && signals.queued_requests >= effective_cold_cap {
            return false;
        }
        true
    }
}
```

Logic correctness:
- Cold requests rejected when queue exceeds soft cap → protects warm sessions
- Hard cap matches `QueueBoundAdmission` behavior
- `is_cold_request()` returns true if `prefix_hit_tokens == 0 && session_affinity_slot.is_none() && turn_depth == 0`

## What's MISSING — admission signal pipeline

`infer/src/scheduler/types.rs:628-637`:
```rust
fn admission_allows(&self, queued_requests: usize) -> bool {
    if self.max_waiting == 0 {
        return true;
    }
    QueueBoundAdmission {
        max_queued_requests: self.max_waiting,
    }
    .allow(SchedulerSignals::queue_state(queued_requests, 0))  // ← passes 0 prefix_hit
}
```

Two gaps:
1. **Hardcoded `QueueBoundAdmission`** — should pick policy based on config
2. **`queue_state(queued, 0)`** — discards prefix_hit_tokens entirely

`SchedulerSignals::queue_state(queued, 0)` constructs signals with
`prefix_hit_tokens: 0` always。Policy can't differentiate warm/cold
because signal is always cold。

## Scope correction

Codex `aeff965` Phase 1 entry:
> P1 — B3 PrefixAwareAdmission
> ~350 LOC(200 policy + 50 wiring + 100 tests)

Refined per this code-grep:
| Component | Codex estimate | Actual |
|---|---:|---:|
| Policy struct + impl | 200 LOC | **0 LOC**(already exists `policy.rs:98-130`)|
| Wiring(SchedulerHandle policy injection)| 50 LOC | **30-50 LOC**(swap + signal injection)|
| **Signal pipeline** | not estimated | **~100-150 LOC**(prefix-hit-tokens lookup at admission time)|
| Tests | 100 LOC | 100 LOC |
| **Total** | **350 LOC** | **~250-350 LOC** |

So total is similar but **breakdown different**。Most work is in:
- Computing `prefix_hit_tokens` at admission time(needs RadixCache.match_prefix lookup)
- Computing `session_affinity_slot`(may exist already in Scheduler)
- Threading these through to admission_allows

## Implementation strategy

### Step 1 — Refactor `admission_allows` to take rich signals

```rust
// Before: only queue depth
fn admission_allows(&self, queued_requests: usize) -> bool { ... }

// After: full signals
fn admission_allows(&self, signals: SchedulerSignals) -> bool { ... }
```

Caller(s)of `admission_allows` need updating to construct signals。

### Step 2 — Compute prefix_hit_tokens at request submission

Caller has access to incoming request tokens。Look up RadixCache:
```rust
let prefix_hit = self.radix_cache.match_prefix(&request.tokens);
let signals = SchedulerSignals {
    queued_requests: ...,
    prefix_hit_tokens: prefix_hit,
    session_affinity_slot: existing_session_slot(&request.session_id),
    turn_depth: ...,
};
```

### Step 3 — Pick policy via config

```rust
let policy: Box<dyn AdmissionPolicy> = match config.admission_policy {
    AdmissionPolicyKind::QueueBound => Box::new(QueueBoundAdmission { ... }),
    AdmissionPolicyKind::PrefixAware => Box::new(
        PrefixAwareAdmission::with_cold_headroom(self.max_waiting, 4)
    ),
};
policy.allow(signals)
```

### Step 4 — Tests + bench

- Unit tests:warm session admitted under cold burst
- Integration test:multi-tenant 4-conc 6k-system reproduces SGLang gap
- Bench:M_world1 P0.2 multi-tenant shape with PrefixAware vs QueueBound

## Empirical magnitude(per `a1965ab`)

If wiring lands:
- 6k system × 4 sessions cold = 24k tokens prefill
- With prefix sharing: 6k cold + 100q × 4 = 6.4k tokens
- Speedup:**3.75× prefill** → real bench measured **2.03×** for multi-tenant TTFT

Closes -50% TTFT gap on multi-tenant axis(157 ms target per SGLang baseline)。

## Status

- ✅ `PrefixAwareAdmission` impl complete(`policy.rs:98-130`)
- ❌ Wiring missing(`types.rs:628-637`)
- 🔧 Codex pickup:Step 1-3 wiring(150-250 LOC)+ Step 4 tests + bench

**Codex priority**:still P1 per `aeff965` ranking — ~250-350 LOC total
substrate work,closes 2× multi-tenant TTFT gap at production deployment。

## Cross-references

- `a1965ab` SGLang gap finding
- `aeff965` final state anchor lists this as P1
- Policy:`infer/src/scheduler/policy.rs:98-130`(impl ready)
- Wiring gap:`infer/src/scheduler/types.rs:628-637`
- M_world1 P0.2 multi-tenant baseline:`m_world1-p0-sglang-baseline-extended`
- Skill v1.5.1:`9f65b4d`

## Rule

**Before estimating substrate scope,grep for partial implementations
that may already exist**。Codex `aeff965` estimated 350 LOC assuming
PrefixAwareAdmission needed from scratch — actual policy is ALREADY 30
LOC complete。Wiring is the real work。

Per skill methodology:**Phase 0 reconnaissance always inventories
existing partial implementations**。Saves estimate accuracy and
prevents duplicate work。This applies to ALL "new feature" scoping —
type-system fields,trait impls,unused branches often exist already
from prior plans that didn't fully land。

For ARLE specifically:`policy.rs` has `PrefixAwareAdmission` ready;
the gap is `types.rs` admission signal pipeline — codex Step 1-3
focuses there。
