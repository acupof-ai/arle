# Stop exporting page-tier copy/wait metrics that no backend can populate — 2026-09-12

> Status: Shipped behind Prometheus omission; values remain in `/v1/stats` and
> JSONL. Unit gate only; no device behavior involved.

## Context

The observability field audit found five `KvSystemMetrics` counters exported as
Prometheus series that can never be nonzero:
`demote_mset_copy_bytes`, `demote_mset_copy_ms`, `promote_mget_copy_bytes`,
`promote_mget_copy_ms`, `fetch_wait_ms`. Their only writers are guarded by
`charge_copy = !kv_tier_transfer_is_zero_copy()` inside the page-tier
demote/promote path, which itself runs only when
`kv_tier_capacity_pages() > 0`. No `KvPageTier` implementation satisfies both
conditions — CUDA pins capacity to 0 (`infer-cuda/src/lib.rs:835`), and the
only capacity-bearing tier, Metal, hard-codes zero-copy
(`infer-metal/src/executor.rs:595`).

## What changed

The five fields are kept (a future copying page tier would populate them) with
a struct annotation stating why each is zero and which backend would set it.
Their Prometheus series are omitted: an absent series reports an honest gap,
where a zero would read as "this happened zero times." They remain in the
`/v1/stats` JSON and JSONL snapshot, where `0` is the true value of a counter
that cannot run. Live sibling series (`demote_mset_count_total`,
`promote_mget_count_total`) are unchanged, so omission is selective.

## Rule

A metric no supported configuration can produce must be absent from the
measurement surface, not emitted as a confident zero. Keep the field when a
future implementation could populate it, annotate the invariant, and omit only
the live-facing series rather than deleting the instrumentation.
