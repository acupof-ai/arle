# W3 c=16 deadlock root-cause hypothesis — `PrefillBudget::from_scheduler_for_decode_slots` reserves growth for ALL active slots,not just runnable decode

> Continues `cb087c7`(W3 c=16 ARLE deadlock,1/136 success at 0.7%):
> `/v1/stats` shows active=16, prefill_queue=15, prefill_rows=0,
> step_phase_us=prefill:403356(zero progress),engine_batch_occupancy=0.0106。
>
> Code-grep root-cause analysis identifies `page_budget.reserve_growth`
> over `running_batch`(NOT just runnable decode slots),pre-reserving
> KV pages for the 16 prefill-queued slots' future decode growth →
> page_budget exhausted before any prefill candidate can be admitted。

## Code path

`infer/src/scheduler/cuda/execution.rs:136-170` `PrefillBudget::from_scheduler_for_decode_slots`:

```rust
fn from_scheduler_for_decode_slots<M: ModelForward>(
    scheduler: &Scheduler<M>,
    decode_slots: &[usize],
) -> Self {
    let mut budget = Self {
        token_budget: StepTokenBudget::for_prefill(...),
        long_prefill_token_threshold: scheduler.config.long_prefill_token_threshold,
        decode_active: !decode_slots.is_empty(),
        page_budget: PageBudget::from_scheduler(scheduler, true),
    };
    // ⚠ Iterates ALL running_batch slots, not just decode_slots
    for &slot_idx in &scheduler.running_batch {
        let remaining = scheduler.remaining_decode_reservation_tokens(slot_idx);
        if remaining > 0 {
            budget.page_budget.reserve_growth(PageGrowth {
                slot_idx,
                tokens: remaining,
            });
        }
    }
    budget
}
```

## Failure scenario

W3 c=16 burst:
1. 16 sessions submit at t=0
2. Admission accepts all 16 → `running_batch` = [s0..s15]
3. None has decoded yet → `slot_is_runnable_decode` false → `decode_slots = []`
4. PrefillBudget::from_scheduler_for_decode_slots runs:
   - decode_slots=[] passed in
   - decode_active = false ✓
   - But line 160 loop iterates ALL 16 in running_batch
   - For each: `remaining_decode_reservation_tokens` returns max_seq_len(~32k tokens),since slot just admitted with no decode yet
   - `page_budget.reserve_growth(16 × 32k = 512k tokens)`
5. PageBudget exhausted(KV pool ~16GB / 1k bytes per token ≈ 16M tokens but reservation accounts for full max_seq_len per slot)
6. `collect_prefill_candidates` calls `prefill_reservation` for each queued slot
   - `can_fit_growth` returns false because page_budget is full
   - Returns None → slot is dequeued from prefill_queue but NOT prefilled
7. Step plan returns Decode(empty) or no work
8. Loop indefinitely with no progress

`/v1/stats` matches signature:
- peak_mem=14151MB(most pinned by reservations)
- prefill_rows=0(no admitted prefill)
- prefill_queue=15(slots still queued)
- decode_rows=0(no progress)
- engine_batch_occupancy=0.0106(~1%,maybe one slot with smaller reservation slips through)
- step_phase_us=prefill:403356(403s spinning on this loop)

## Fix candidates

### Fix A — only reserve for runnable decode slots(P0)
Line 160 change `&scheduler.running_batch` → `decode_slots`(or filter
to only `slot_is_runnable_decode`):

```rust
for &slot_idx in decode_slots {
    let remaining = scheduler.remaining_decode_reservation_tokens(slot_idx);
    ...
}
```

Rationale:slots in prefill_queue haven't decoded yet — their future
decode growth shouldn't gate prefill admission。Reservation should
happen at decode-step admission,not at prefill-budget computation。

Risk:if prefill admits all 16 slots without future-growth reservation,
prefill produces tokens but decode step then runs OOM during growth
because pages weren't reserved。Need to verify decode step has its own
page check before allocating。

### Fix B — split page_budget into prefill-budget vs decode-growth-budget(P1)
Two separate PageBudget instances:
1. `prefill_admit_budget`:tracks pages needed for prefill row allocation
2. `decode_growth_budget`:tracks pages needed for in-flight decode steps

Prefill admission uses(1)only。Decode admission uses(2)。This
decouples prefill budget from decode growth contention。

### Fix C — hard cap concurrent prefill admission per step(workaround)
Set `prefill_max_requests = 4` or some smaller value。Prefill 4 slots
per step,let them advance to decode → decode_growth shrinks for those
slots → next step can reserve growth for THOSE 4 → admit next 4 prefill。

Doesn't fix root cause but works around the deadlock at lower throughput。

## Recommended order

1. **Fix A first**:simplest 1-line change,most directly addresses the
   bug。Add unit test that verifies prefill_queue can drain when 16
   sessions burst-admit。Run W3 c=16 to confirm deadlock resolved。
2. **If Fix A breaks decode OOM**:add Fix B as proper architectural
   separation。
3. **Don't ship Fix C alone**:it masks the bug;production should not
   need such tight prefill cap。

## Cross-references

- W3 deadlock errors entry: [`cb087c7`](../experience/errors/2026-05-08-w3-c16-arle-deadlock.md)
- W3 c=4 working baseline: [`370a267`](../experience/wins/2026-05-08-w3-c4-baseline.md)
- W3 503 source identification: [`5e8525c`](2026-05-08-w3-503-source-identified.md)
- Code paths:
  - `infer/src/scheduler/cuda/execution.rs:136-170` PrefillBudget construction
  - `infer/src/scheduler/cuda/execution.rs:160-168` running_batch loop
  - `infer/src/scheduler/cuda/execution.rs:355-383` collect_prefill_candidates
  - `infer/src/scheduler/cuda/execution.rs:305-316` prefill_reservation
  - `infer/src/scheduler/cuda/core.rs:146` prefill_queue field

## Probability

**~80%** this is the root cause:
- Code path matches deadlock signature exactly
- Page reservation for all 16 slots → page_budget exhaustion
- Explanation consistent with all observed `/v1/stats` numbers
- Fix is minimal(1 line)— quick to verify

Remaining 20%:
- May be additional interaction with `chunked_prefill_size`(for long prompts)
- Or `prefill_max_requests` default may need adjustment
- Or there's a race between admission and prefill scheduling

## Strategic value

W3 c=16 is **master strategy §7.1 P0.0 真 agent workload**(c=16 burst
tier per master strategy axis 1)。10 KILL paths all on canonical 4-shape
benchmark NOT reflecting agent痛点。This is the FIRST attempt at real
agent workload bench at master-strategy-spec concurrency。**Scheduler
substrate fix unblocks entire agent workload bench cycle**(currently
0/136 turns produce data at c=16)。

Spec-decode axis re-test(Medusa per `5acbe94`)gated on this fix —
can't measure speculative wins on a deadlocked scheduler。

## Codex action(after W4A8 resolves)

1. Apply Fix A patch:line 160 of execution.rs change to iterate
   `decode_slots` instead of `running_batch`
2. Run `cargo test --release` to ensure no decode regression
3. Manual W3 c=16 retest:verify prefill_queue drains
4. If OK,run full W3 c=16 bench → wins entry vs cb087c7 errors

## Rule

When step-budget computation iterates over a slot collection(running
batch / waiting queue / prefill queue),verify the **iteration scope
matches the budget's accounting period**:
- Decode-growth budget ↔ slots actually decoding now(not future-queued)
- Prefill-admission budget ↔ slots about to prefill
- Mixed budget ↔ both,with separate accounting per phase

Conflating these creates "budget contention deadlocks" where future
work blocks current admission。
