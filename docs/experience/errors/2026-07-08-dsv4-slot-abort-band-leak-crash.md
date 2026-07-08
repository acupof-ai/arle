# DSv4 slot-abort doesn't release FlashMLA fixed band — next occupant crashes all worker ranks

## Context

Discovered 2026-07-08 while verifying the FlashMLA per-layer KV budget fix
(`3ebc763f9`) on the H20 pod via `needle_gate.py` at a deliberately tight
admission boundary (see
`docs/experience/wins/2026-07-08-dsv4-flashmla-budget-needle-gate-pass.md`).
**Not a defect in `3ebc763f9`** — that fix made the tight `num_slots ∈ {1,2}`
configuration reachable for the first time; pre-fix, those `max_seq_len`
values were rejected outright at startup
(`2026-07-06-dsv4-flashmla-budget-reconciliation-verified.md`), so a live
server in this regime never previously existed to expose this path.

## Root Cause (partial — not yet fixed)

At `num_slots ∈ {1,2}` (TP=4, `max_total_tokens` 32768/20000), the scheduler's
admission-reject path (`crates/infer-core/src/lib.rs:1268`, the 2026-07-05
"reject-not-hang" fix for a single prompt that structurally exceeds pool
capacity) aborts the oversized request cleanly — but the DSv4 slot's FlashMLA
fixed-band reservation (`HostPagedKvPool::alloc_fixed_band`, drawn FULL and
up-front per slot) is **not released** as part of that abort.

The next request assigned to the same slot then calls
`alloc_fixed_band` expecting a fresh full band, finds the shortfall left by
the un-released reservation ("slot 0 needs 130, free 2" / "needs 81, free
4"), and `bail!`s — which crashes **all 4 worker ranks** (a `bail!` inside
the DSv4 forward path is not scoped to a single rank's request handling; it
propagates to a full-process abort under this engine's collective-lockstep
model).

Reproduced identically at `num_slots=1` (`max_total_tokens=32768`) and
`num_slots=2` (`max_total_tokens=20000`). Did **not** reproduce at
`num_slots=129` (`max_total_tokens=4096`) — there the same oversized prompt
hit a different, graceful (0-token-completion) branch instead; the crash is
specific to the reject-path interacting with a tight-slot-count band
reservation, not a general property of oversized prompts.

## Fix

Not yet designed or implemented. Candidate directions (untried):
- Release the slot's FlashMLA fixed-band reservation as part of the
  reject-path's cleanup (`infer-core/src/lib.rs:1268` region), symmetric with
  however a normal `finish_slot`/`free_slot` already releases it.
- Alternatively, make `alloc_fixed_band` itself detect and clear a stale
  reservation from a previously-aborted occupant before allocating, rather
  than assuming the slot always arrives in a `free` state.

## Rule

Tightening an admission/budget calculation to be "exact, zero slack" (no
regression by itself) can make previously-unreachable low-`num_slots`
configurations reachable for the first time — and previously-latent bugs in
code paths that only ever ran with headroom can surface as production
crashes at exactly that new boundary. Before shipping a budget-tightening
change, exercise the admission-reject/abort lifecycle at the tightest
`num_slots` the change newly permits, not just the arithmetic correctness of
the budget itself.

Before Route A (`docs/plans/2026-07-08-dsv4-route-a-page-granular-prefix-reuse.md`)
adds more per-layer pools onto this same slot lifecycle (steps 4-6), this
should be root-caused and fixed — a page-addressable pool's own reservation
could leak the same way on an aborted request.
