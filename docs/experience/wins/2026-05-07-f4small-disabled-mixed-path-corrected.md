# 2026-05-07 · F4-Small Mixed-disable corrected analysis — race IS real (token-loss form)

## Context

Refines [`26b7f86`](2026-05-07-f4small-disabled-mixed-path.md)'s
"1-line fix candidate" hypothesis. Source-archaeology of
`batch_decode.rs::start_greedy_readback_async` showed the race
F4-Small was protecting against IS real, but takes the form of
**silent token loss**, not corruption.

## What the source actually does

`start_greedy_readback_async` (`batch_decode.rs:624-661`):

```rust
pub(crate) fn start_greedy_readback_async(...) -> Result<()> {
    if self.async_readback_in_flight {
        return Ok(());                          // ← THE GUARD
    }
    let ids_src = self.argmax_out.slice(0..batch_size);
    let mut ids_dst = self.async_argmax_gpu.slice_mut(0..batch_size);
    ctx.stream.memcpy_dtod(...)?;               // D2D snapshot
    ...
    ctx.copy_stream.memcpy_dtoh(...)?;          // D2H to host
    self.async_readback_event.record(&ctx.copy_stream)?;
    self.async_readback_in_flight = true;
    self.async_readback_batch_size = batch_size;
    Ok(())
}
```

The early-return when `in_flight = true` MEANS:

- Old readback's snapshot (`async_argmax_gpu`) is preserved (good)
- **NEW launch's `argmax_out` is NOT snapshotted** (bug)
- Next tick: deferred (old) finishes, returns old tokens
  correctly. Sets `in_flight = false`.
- Pending (new) gets to `finish_greedy_readback`. Now
  `in_flight = false`, so line 664 returns `Ok(None)`.
- New pending → re-deferred. But there's still no snapshot.
- Next next tick: same dance. Tokens **never snapshotted, never
  readback** = **silent token loss**.

## Why F4-Small added the `deferred_decode_emit.is_none()` precondition

It's NOT just defensive — it's load-bearing. Without it, the
in_flight guard at line 629 silently drops tokens whenever a new
launch fires while deferred readback is outstanding.

Path-sensitive:
- High-conc steady-state: low chance of deferred outstanding
  (event ready quickly, decode-only path is fast). Mixed path
  rarely fires (no prefill candidates). Bug doesn't manifest.
- Long-ctx steady-state: deferred outstanding much more often
  (slow steps mean events lag); prefill candidates always present
  → would hit Mixed path → would lose tokens on EVERY mixed step.
  F4-Small's precondition correctly forces Split fallback to
  avoid this.

## The actual fix

Options A, B, C from `26b7f86` re-evaluated:

### A) Double-buffer (or FIFO N-buffer) async_argmax_gpu/host ✅

Allocate N buffer slots (2 minimum, more if pipeline deeper).
Per-launch round-robin slot selection. Each in-flight readback
references its own slot via slot_idx in the pending state.

LOC: ~50-100 in `BatchDecodeBuffers` + sample_batch_greedy_launch
+ start_greedy_readback_async + finish_greedy_readback. Need to
extend pending state to carry slot_idx.

Bonus: would also remove the `in_flight` guard entirely, since
slots are independent.

Expected gain:
- Long-ctx 4k/c=4 TTFT: 4961ms → ~2500ms (close vLLM's 2367ms)
  by restoring Mixed path
- High-conc 1k/256/c=64: unchanged (was already Mixed/Decode-only)

### B) Drain deferred before new launch — discarded
Re-introduces sync. Defeats F4-Small's gain.

### C) Remove precondition (1-line) — DISCARDED
Would silently lose tokens. Was a tempting hypothesis from prior
analysis but source-archaeology shows it would corrupt the
contract.

## Recommended Phase 1 path (revised)

**M3.9 Phase 1A v3** (the only viable path):

1. Multi-slot async readback buffers in `BatchDecodeBuffers`:
   - Replace single `async_argmax_gpu` / `async_logprobs_gpu` /
     `async_argmax_host` / `async_logprobs_host` /
     `async_readback_event` with `Vec` (size = config or 2-4 slots)
   - Add `next_async_slot: usize` round-robin counter
2. Carry `async_slot_idx: usize` in `PendingDecode` /
   `deferred_decode_emit` state
3. `start_greedy_readback_async`: pick next free slot
   (round-robin, assert not in flight); fail loud if all slots
   busy (which would mean serial launches falling behind GPU
   completion)
4. `finish_greedy_readback`: free the slot when complete
5. **Remove** `deferred_decode_emit.is_none() &&` precondition in
   `plan_step`
6. Bench long-ctx 4k/c=4 + greedy_consistency + e2e

LOC: ~80-150 (codex-sized work). Codex review essential.

## What this means for codex's M3.9 Phase 0

Codex is implementing instrumentation right now (10 files staged).
The instrumentation will confirm the Mixed/Split routing in
practice — counter values will show `ok_true_count` for Mixed
vs Split fallback rate. Useful as evidence even though this
analysis already located the cause.

After Phase 0 commits + this corrected analysis is read by codex,
Phase 1A v3 (multi-slot async readback) is the natural next
brief.

## Process rule (one more)

- **"1-line fix" hypothesis from blame analysis must be verified
  against the protected invariant**, not just "the line was
  added by recent commit". 26b7f86 jumped to "remove the
  precondition"; 30 more minutes of source archaeology (this
  entry) showed it's protecting against silent token loss.
  Removing without fixing the invariant is a regression.
- Pattern: blame → guard → understand the invariant →
  fix the invariant (not just remove the guard).

## Bench Status

No new bench. Source-only work in this entry.

## Cross-references

- Discovery commit: `26b7f86`
- F4-Small commit: `2a534c4`
- Source: `infer/src/model/qwen3/batch_decode.rs:624-700`
  (start_greedy_readback_async + finish_greedy_readback)
- Plan_step: `infer/src/scheduler/cuda/execution.rs:393-414`
- M3.9 plan: `63af21f` (to revise with this new direction)
