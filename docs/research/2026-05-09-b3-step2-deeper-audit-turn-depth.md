# B3 Step 2 deeper audit — turn_depth handling clarification

> Per skill v1.6.0 anti-pattern #18 Phase 0 substrate audit + my `c097b2b`
> architecture refinement,deeper-audited admission.rs site to verify ALL
> SchedulerSignals fields can be populated。**Found gap**:`IncomingRequest`
> doesn't carry `turn_depth`。Brief clarifies handling for Step 2 simplicity。

## SchedulerSignals construction at runtime/admission.rs(verified)

After lookup at `runtime/admission.rs:180-200`:

| SchedulerSignals field | Source | Status |
|------------------------|--------|--------|
| queued_requests | scheduler.waiting_count() | ✓ accessible |
| active_decodes | scheduler.active_count() | ✓ accessible |
| prefix_hit_tokens | `lookup.matched_len`(line 187/193) | ✓ ready |
| session_affinity_slot | `session_slot_hold.as_ref().map(|h| h.slot_idx())` | ✓ ready |
| **turn_depth** | **NOT IN IncomingRequest**(verified line 567-594) | ❌ **gap** |

## Why turn_depth gap doesn't block Step 2

Per `policy.rs:55-58` `is_cold_request()`:
```rust
self.prefix_hit_tokens == 0
    && self.session_affinity_slot.is_none()
    && self.turn_depth == 0
```

→ Cold = ALL THREE conditions met。

For B3 admission gate behavior:
- If `prefix_hit_tokens > 0` → not cold(regardless of turn_depth)→ unaffected by gap
- If `session_affinity_slot.is_some()` → not cold(regardless of turn_depth)→ unaffected by gap
- If both above are 0/None → likely a fresh/cold first-turn request → setting `turn_depth = 0` correctly classifies as cold

→ **`turn_depth = 0` default is safe semantically** for Step 2。

## Step 2 simplification

Step 2 implementation can use:
```rust
let signals = SchedulerSignals {
    queued_requests: scheduler.waiting_count(),
    active_decodes: scheduler.active_count(),
    prefix_hit_tokens: lookup.matched_len,
    session_affinity_slot: session_slot_hold.as_ref().map(|h| h.slot_idx()),
    turn_depth: 0,  // Step 4 future: track per-session turn count
};
```

This is functionally equivalent for admission purposes。Future Step 4
could add per-session turn counter at scheduler state for finer tuning
(e.g.,turn_depth-based prioritization)。

## Updated Step 2 LOC estimate

Per `c097b2b` 100 LOC + this turn_depth handling:
- Add `turn_depth: 0` constant — 0 extra LOC vs prior estimate
- Document semantics in code comment — 5 LOC
- **Step 2 total stays ~100 LOC**(unchanged)

## Phase 0 substrate audit pattern empirical(skill v1.6.0 #18)

This deeper audit caught:
- ✅ A1 production-wired(`1217375`)
- ✅ Lookup result has matched_len(`c097b2b`)
- ✅ session_slot_hold accessible
- ⚠ turn_depth NOT in IncomingRequest(this brief)→ but doesn't block,
  set to 0 default

→ All 5 SchedulerSignals fields can be populated。Step 2 unblocked。

**Anti-pattern #18 in action**:by auditing all sources before committing
to LOC estimate,caught the turn_depth gap proactively。Without audit,
codex might have started implementation,encountered the gap mid-impl,
and faced "stuck mid-feature" decision。

## Updated B3 sequence

| Step | Status | LOC | Effort | Site |
|------|--------|----:|-------:|------|
| 1 | ✅ DONE(`7c8fd61`)| 14ins/4del | 1 tick | types.rs admission_allows |
| 2 | Refined ready | ~100 | **0.5d** | runtime/admission.rs post-lookup |
| 3 | Pending | 30 | 0.25d | config wiring |
| 4(future) | Optional | ~50 | 0.25d | turn_depth scheduler tracking |

Step 4 is purely additive — Step 2 produces correct admission behavior
without it。

## Methodology refinement

Per skill v1.6.0 anti-pattern #18:Phase 0 substrate audit must check ALL
fields the new code constructs,not just the primary one。`prefix_hit_tokens`
was the obvious one;turn_depth was a less-obvious gap that could have
been missed。

**Audit checklist refinement**:
- (1) Is the primary dependency wired?
- (2) Where exactly?
- (3) What does it return?
- (4) **Are ALL fields/signals the new code constructs available at that site?**
  ↑ NEW per this brief

Anti-pattern #18 sub-clause:**audit all signal-construction fields,
not just the primary dependency**。Prevents "almost-ready" mid-impl
discoveries that block progress。

## Cross-references

- Step 2 architecture:`c097b2b`
- A1 production-wired audit:`1217375`
- Skill v1.6.0:`125f795`
- IncomingRequest:`types.rs:567-594`
- Admission site:`runtime/admission.rs:180-200`
- SchedulerSignals:`policy.rs:21-41`
- is_cold_request:`policy.rs:55-58`

## Status

B3 Step 2 fully audited + scoped。`turn_depth = 0` default safe for
Step 2 admission behavior。Step 4(future)can add per-session turn
counter if finer tuning needed。

Codex pickup:Step 2 ready as-is per `c097b2b` + this turn_depth note。
~100 LOC,0.5d,no surprises mid-impl。
