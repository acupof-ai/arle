# 2026-08-17 — Prefix match length had no cross-rank min-reduce; tier promote failure desynced TP

## Context

The 2026-08-16 engine architecture audit (5D parallelism) flagged a live TP
desync window in the prefix-attach path. Under host-tier KV caching, a single
rank's tier-promote failure truncates that rank's prefix match while peers keep
the full length. The divergent `prefill_start_pos` then desyncs the TP
collectives' shapes (hang or wrong results). Tracked as issue #214; a
prerequisite for CP T3 sharded-KV (plan `docs/plans/2026-08-16-cp-ideal-state.md`
§3.2).

The divergence is rank-local by construction: `materialize_prefix_blocks`
(`infer-core/src/prefix.rs:709`) promotes demoted blocks into freshly allocated
pages, and a promote failure (tier capacity, disk error, allocation failure)
returns only the leading resident run. Peers with healthy tiers promote the
full match. No cross-rank alignment of the resulting `matched_len` existed —
`lookup_prefix_for_attach` returned the rank-local value directly.

The whole-slot tier path already had this alignment (tp_min in
`executor/qwen35.rs:495-570`); the page-tier path missed it.

## Root cause

`lookup_prefix_for_attach` (`infer-core/src/prefix.rs:662`) computed the match
length from rank-local promotion results with no cross-rank reduce. The
admission capacity check (`planner.rs:147`) reduces *capacity* via
`tp_sync_min`, but the *match length* — which sets `prefill_start_pos` and thus
the collective shape — was never reduced.

## Fix

Route the tier-path match length through `executor.tp_sync_min` before return:

- `lookup_prefix_for_attach` now returns `Result<PrefixMatch>`; the tier branch
  calls `self.executor.tp_sync_min(local_len)?` and truncates `block_ids` to the
  aligned length. The truncated tail pages (promoted and retained in
  `materialize_prefix_blocks`) are released via `kv.release_pages` +
  `executor_release_prefix_pages` so they return to evictable; the slot
  re-prefills the tail through the standard chunked path.
- The reduce is gated on `kv_tier_capacity() > 0`, which is rank-symmetric
  (configured capacity, not free capacity), so every rank issues exactly one
  reduce per tier-path attach — the symmetric-call discipline `tp_sync_min`
  requires. The no-tier branch is unchanged (its match is rank-identical under
  lockstep).
- Both call sites (`lib.rs:1682` admission, `planner.rs:403` restore recompute)
  propagate the `Result` with `?`.

Single-rank backends: `tp_sync_min` returns `local` unchanged, so the truncation
branch is unreachable — zero behavior change off the multi-rank tier path.

## Gate

Correctness gate is multi-rank by nature (the truncation only fires when one
rank's match is shorter): needle ladder ×3 at TP=2 with the host tier enabled,
plus a fault-injection run (single-rank promote failure forced) confirming
`prefill_start_pos` stays identical across ranks. **Pending-remote on the H20
pod** (Mac cannot run TP); tracked on issue #214. Local verification:
`cargo check -p infer-core`, `cargo clippy -p infer-core -D warnings`, and the
Mac CUDA typecheck (`CUDARC_CUDA_VERSION=12080 cargo check -p infer-api
--no-default-features --features cuda,no-cuda --lib`) all pass.

## Rule

Any rank-local quantity that sets a collective's shape — prefix match length,
sequence length, accepted-token count — goes through a cross-rank min-reduce
before the collective is issued. Capacity was already reduced; match length was
the gap. The reduce gate must be rank-symmetric (configured capacity, not
free/runtime state) or the call counts themselves desync.
