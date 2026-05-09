# M_B3 PrefixAwareAdmission Step 1 — admission_allows signature change

> Per `c0ddd4f` finding:`PrefixAwareAdmission` impl ALREADY EXISTS at
> `policy.rs:98-130`(33 LOC,verified)。Real blocker is wiring at
> `types.rs:628-637 admission_allows` which drops per-request signals。
>
> This directive provides the smallest concrete first step for codex
> pickup:**change `admission_allows` signature to accept full
> `SchedulerSignals`**(~5-10 LOC + caller migration)。

## Verified existing state

`infer/src/scheduler/policy.rs:98-130` PrefixAwareAdmission:
- Struct with `hard_cap` + `cold_soft_cap` fields
- `with_cold_headroom(hard_cap, cold_headroom)` constructor
- `allow(signals: SchedulerSignals)` impl using `signals.is_cold_request()`

`infer/src/scheduler/types.rs:628-637` admission_allows(BUG):
```rust
fn admission_allows(&self, queued_requests: usize) -> bool {
    if self.max_waiting == 0 {
        return true;
    }
    QueueBoundAdmission {
        max_queued_requests: self.max_waiting,
    }
    .allow(SchedulerSignals::queue_state(queued_requests, 0))
    //                       ^^^^^^^^^^^ drops prefix_hit_tokens etc
}
```

## Step 1 — Change admission_allows signature(smallest first step)

Smallest concrete diff(~5-10 LOC):
```rust
// BEFORE
fn admission_allows(&self, queued_requests: usize) -> bool {
    ...
    .allow(SchedulerSignals::queue_state(queued_requests, 0))
}

// AFTER
fn admission_allows(&self, signals: SchedulerSignals) -> bool {
    if self.max_waiting == 0 {
        return true;
    }
    QueueBoundAdmission {
        max_queued_requests: self.max_waiting,
    }
    .allow(signals)
}
```

Then update all callers to pass full `SchedulerSignals`(not just `queued`):
- Find callers via `grep -rn 'admission_allows' infer/src/`
- Each caller needs to construct `SchedulerSignals` with at least `queued_requests` populated
- For Step 1,leave `prefix_hit_tokens=0` defaults at all callers — same behavior as before
- Build + tests pass identically to today

## Why this Step 1 first

Per CLAUDE.md "no half-states":
- Don't change to PrefixAwareAdmission yet(needs RadixCache wiring)
- Don't add cold_soft_cap config yet
- Just refactor signature so future steps can populate signals

This is the smallest verification step that unblocks the rest of B3
without adding any new behavior。Build + existing tests pass = LICENSE。

## Step 2 preview(after Step 1 lands)

Once admission_allows takes SchedulerSignals,Step 2 adds RadixCache
lookup at request submission to populate `prefix_hit_tokens`:
```rust
// In SchedulerHandle::submit or similar:
let prefix_hit = if let Some(cache) = &self.prefix_cache {
    cache.match_prefix(&request.tokens).matched_len
} else {
    0
};
let signals = SchedulerSignals {
    queued_requests: current_queue,
    active_decodes: current_decodes,
    prefix_hit_tokens: prefix_hit,
    session_affinity_slot: request.session_affinity,
    turn_depth: request.turn_depth,
};
admission_allows(signals)
```

~50-100 LOC depending on RadixCache integration patterns。

## Step 3 — Switch to PrefixAwareAdmission

After Steps 1-2:
```rust
// Replace QueueBoundAdmission with PrefixAwareAdmission
fn admission_allows(&self, signals: SchedulerSignals) -> bool {
    if self.max_waiting == 0 {
        return true;
    }
    let cold_headroom = self.cold_headroom_config.unwrap_or(self.max_waiting / 4);
    PrefixAwareAdmission::with_cold_headroom(self.max_waiting, cold_headroom)
        .allow(signals)
}
```

~10 LOC + config plumbing for `cold_headroom_config`(another ~30 LOC)。

## Updated total scope(corrected vs `a1965ab`)

| Phase | Original `a1965ab` | Corrected `c0ddd4f` + this |
|-------|-------------------:|---------------------------:|
| Policy struct + impl | 200 LOC | **0 LOC**(exists) |
| Wiring(signature + callers)| 50 LOC | 30-50 LOC(Step 1) |
| Signal pipeline(RadixCache lookup)| not estimated | 100-150 LOC(Step 2)|
| Switch to PrefixAware policy | n/a | 10-30 LOC(Step 3)|
| Tests | 100 LOC | 100 LOC |
| **Total** | **~350 LOC** | **240-330 LOC**(similar magnitude) |
| Wall time | 2-3 days | **2-3 days**(unchanged) |

Same total magnitude,different breakdown。Step 1 is smallest single
unit at 30-50 LOC — could land in 0.5 day codex。

## Validation

After Step 1:
- `cargo build --release -p infer --features cuda`
- `cargo test --release -p infer --features cuda`
- Existing benches should be IDENTICAL behavior(no policy change yet)

After Step 2 + 3:
- Re-run M_world1 multi-tenant burst bench
- Expected:ARLE 318 ms → ~157 ms(matching SGLang)= **-50% TTFT**
- Bench artifact:`docs/experience/wins/2026-05-XX-b3-prefix-aware-multitenant.md`

## KILL criteria

- **Step 1**:if signature change breaks callers in unexpected ways(rare)
  → revert,investigate
- **Step 2**:if RadixCache integration causes overhead > 5% TTFT in
  prefix-miss case → make signal computation lazy / cache result
- **Step 3**:if cold_soft_cap rejects too aggressively at low load →
  tune `cold_headroom` default(e.g. min(4, max_waiting/4))

## Cross-references

- `c0ddd4f` empirical finding(PrefixAwareAdmission exists)
- `a1965ab` original B3 SGLang gap analysis(scope was off by struct existence)
- `policy.rs:98-130` PrefixAwareAdmission impl
- `types.rs:628-637` admission_allows wiring bug
- `M_world1-p0-sglang-baseline-extended` empirical 2.03× gap

## Status

**SUPERSEDED 2026-05-10 by `5e902da` survey** — all three steps
(signature change, signal pipeline, default policy switch) are LANDED
in main. Original 200-400 LOC scope collapsed to a bench-validation
A/B run.

See `docs/research/2026-05-10-36-prefix-aware-admission-substrate-complete-bench-pending.md`
for:
- Step 1 commit attribution (`7c8fd61`) + current code locations
- Step 2 substrate location (`admission.rs:409-435 prefix_aware_admission_allows_plan` +
  `admission.rs:437-458 prefix_aware_fail_open_candidate`)
- Step 3 CLI wiring (`main.rs:124,702` + `types.rs:478-493 enum + parse`)
- Concrete bench command pair for QueueBound vs PrefixAware A/B
- Open question on gate-trigger evidence (low-pressure benches don't
  exercise `cold_soft_cap`)
- Pickup brief for next bench window

Plan body sections below remain useful for KILL-gate framing, SGLang
gap context, and the per-step KILL criteria. Implementation plan is
moot.

### Original status (pre-supersession, retained for history)

Step 1 directive ready for codex pickup(0.5 day,30-50 LOC)。
Steps 2-3 follow sequentially。Total B3 work 2-3 days codex,closes
SGLang multi-tenant 2× gap = -50% TTFT on multi-tenant axis。

Codex pickup queue priority:
- P0 Hybrid Phase 1b
- P0' M_warmup prefill pass
- **P1 B3 Step 1**(this directive — 0.5 day,unblocks Steps 2-3)
- P1 #33 KV W4A8
- P1' Medusa Phase 1.B(corrected to 10-14 days per `d0db904`)

If user wants quick axis 1 wins,B3 Step 1 is faster than Hybrid Phase
1b(0.5d vs 0.75-1d)— could be tackled in parallel pair。
