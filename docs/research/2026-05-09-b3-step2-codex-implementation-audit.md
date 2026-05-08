# B3 Step 2 — codex mid-flight implementation audit(`12m 12s in`)

> Per pickup queue P0.1 dispatch(2026-05-09 ~01:00),codex picked up
> B3 Step 2 PrefixAwareAdmission CUDA-runtime integration。This entry
> audits codex's mid-flight 5-file diff against the dispatch directive
> and pickup queue acceptance criteria。
>
> **Verdict**:codex matches directive on all major points + EXCEEDS
> with senior-quality defensive design(fail-open guard against
> admission deadlock that I did not specify in the brief)。

## Codex modifications(uncommitted at audit time)

```
infer/src/main.rs                             |  18 +++-
infer/src/scheduler.rs                        |   4 +-
infer/src/scheduler/cuda/core/construction.rs |   6 +-
infer/src/scheduler/cuda/runtime/admission.rs |  58 ++++++++++++-
infer/src/scheduler/types.rs                  | 115 +++++++++++++++++++++++++-
5 files changed, 193 insertions(+), 8 deletions(-)
```

193 LOC ~~ 100 LOC dispatch directive estimate × 2(includes config plumbing
that was in the directive but not LOC-counted)。Within scope。

## Match against dispatch directive

| Dispatch directive item | Codex impl | Status |
|------|------|--------|
| Construct `SchedulerSignals` after `lookup_or_stage` returns | `prefix_aware_admission_allows_plan` reads `plan.lookup.matched_len` etc. | ✅ |
| `PrefixAwareAdmission::with_cold_headroom(max_waiting, cold_headroom)` | `admission_policy_allows` dispatches on `SchedulerAdmissionPolicy` enum | ✅(refined — wraps in `admission_policy_allows` for clean QueueBound default) |
| Gate at admission site | Wired into `collect_admission_candidates` per-request loop | ✅ |
| CLI flag `--admission-policy {queue-bound,prefix-aware}` | `main.rs` Args + `parse()` helper | ✅ |
| `--cold-headroom N` | `main.rs` Args | ✅ |
| `cold_headroom` default `max_waiting / 4` | `Option::unwrap_or(max_waiting/4)` in dispatcher | ✅ |
| Backward-compat preservation | `SchedulerAdmissionPolicy::QueueBound` is `#[default]` + queue-bound short-circuits to `true` in policy gate | ✅ |
| Backend isolation | All RadixCache access stays in `scheduler/cuda/runtime/admission.rs` per refined architecture | ✅ |

## Codex EXCEEDS directive — fail-open guard against admission deadlock

The directive did not specify deadlock protection。Codex
identified the risk independently and added:

```rust
// admission.rs (codex added)
if candidates.is_empty()
    && let Some(candidate) = policy_deferred.pop_front()
{
    candidates.push(candidate);
}

while let Some(mut candidate) = policy_deferred.pop_front() {
    self.release_admission_plan(&candidate.plan);
    candidate.incoming.prompt_tokens = Some(candidate.prompt_tokens);
    insert_waiting_request_by_priority(
        &mut self.waiting,
        candidate.incoming,
        WaitingInsertBias::AfterEqual,
    );
}
```

**Why this matters**:without fail-open,if ALL scanned waiting
requests are cold AND soft cap is reached,`candidates` returns
empty → admission stalls → starvation。

Codex's fix:if soft cap is hit but `candidates.is_empty()`,let one
deferred candidate through。Otherwise re-insert deferred candidates
to waiting queue with `AfterEqual` bias(preserves FIFO for same
priority)。

This is **senior-quality defensive design**。Standard "admit
implementation per spec" approach would have shipped without the
guard,producing a latent deadlock that would surface in
multi-tenant cold-burst workloads。

## Minor observations

### `turn_depth: 0` hardcode is intentional + safe

```rust
self.config.admission_policy_allows(SchedulerSignals {
    queued_requests,
    active_decodes: self.running_batch.len(),
    prefix_hit_tokens: plan.lookup.matched_len,
    session_affinity_slot: plan.session_slot_hold.as_ref().map(...),
    turn_depth: 0,  // hardcoded
})
```

`PrefixAwareAdmission::allow()` impl in `policy.rs:98-130` does NOT
consume `turn_depth` — only reads `signals.queued_requests` and
`signals.is_cold_request()`(which depends on `prefix_hit_tokens` and
`session_affinity_slot`)。Hardcoding to 0 is safe at consumption
site for now。

If future PrefixAware variants want to use turn_depth(e.g. to bias
multi-turn sessions further),the field is already plumbed — only
the admission site needs to populate it from `req.turn_depth`。

### `session_affinity_slot` mapping

```rust
session_affinity_slot: plan
    .session_slot_hold
    .as_ref()
    .map(|_| plan.reusable.map(|(slot_idx, _, _)| slot_idx).unwrap_or(0)),
```

Codex mapped `SessionSlotHold` presence(an Option of "we have a
session prefix retained")to a concrete `slot_idx`。Codex's comment
acknowledges:

> SessionSlotHold currently identifies a retained session prefix,
> not a concrete slot. AdmissionPolicy only reads Option-ness.

So the actual `slot_idx` value(0 fallback)doesn't matter to
PrefixAwareAdmission;only `session_affinity_slot.is_some()` is read。

This is a clean implementation choice — preserves type signature
without over-engineering a slot-mapping that isn't consumed。

## Build status at audit time

- ✅ `cargo fmt --all --check` passed
- 🟡 `cargo check --release -p infer --features cuda` running 12m 12s
  (TileLang AOT + native CUDA recompile,5-15 min normal range)
- ⏳ Tests + bench pending after build success

## Acceptance criteria progress

| Criterion | Status |
|------|------|
| cargo test --release passes | Pending(awaiting build) |
| cargo clippy --all-targets -- -D warnings clean | Pending |
| Byte-identical regression for queue-bound default | Pending(should pass given QueueBound short-circuit) |
| Bench wins entry | Pending(after build success) |
| Multi-tenant TTFT improvement σ < 5% | Pending(needs bench run) |

## Management insight — Claude oversight value during codex execution

**Mid-flight read-only audit catches drift early without disrupting codex**:

1. Diff inspected via `git diff` without modifying codex's working tree
2. Code-grep verified hardcoded values are intentional(not bugs)
3. Architectural decisions(fail-open guard,session_slot_hold mapping)
   verified to match the senior intent of the dispatch directive
4. Acceptance criteria status tracked in advance of codex's own review

This pattern — Claude as **manager-auditor** during codex execution —
extracts value from "Working" cycles without idle waiting OR
interrupting codex。Captures `12m 12s` of substrate-level intent
that codex doesn't naturally write down(the `// fail-open ... so
admission cannot deadlock` comment is the only on-source artifact
of the design decision)。

## Cross-references

- Dispatch directive: pickup queue P0.1 in `docs/plans/codex-pickup-queue-2026-05-09.md`
- B3 Step 1 LANDED: `7c8fd61` + `c30e298`(byte-identical regression)
- A1 audit: `1217375`
- Architecture refinement: `c097b2b` + `637701b`
- Skill v1.7.0 anti-pattern #19(this audit motivated): `c768b70`
- PrefixAwareAdmission impl: `infer/src/scheduler/policy.rs:98-130`

## Status

Codex `12m 12s` into B3 Step 2 implementation。**193 LOC diff matches
+ exceeds dispatch directive**。Audit ready for next-tick verification
when build completes + tests run + bench produces wins entry。

## Rule

**Mid-flight read-only audit during codex execution is high-value
manager work**:catches drift,verifies intentional choices(hardcodes,
fallbacks,defensive patches)without disrupting codex's flow。Capture
in research entry so the architectural reasoning persists past the
git commit's message field。

For ARLE specifically:codex's defensive patches(fail-open guards,
re-insert ordering biases)are often the most-load-bearing parts of
multi-tenant scheduler diffs and rarely have explicit test coverage
in the same PR。Audit them in the research entry so future
regression-bisects know what to look for。
