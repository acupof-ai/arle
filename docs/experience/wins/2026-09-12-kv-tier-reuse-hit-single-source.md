# KV tier reuse-hit metrics read from one source, not decision plus store read — 2026-09-12

> Status: Pending remote. Unit gates pass locally (infer-core, backend-neutral
> fake tier). The number-moving configuration is Metal with `--kv-disk`, which
> needs a Mac device run; the production CUDA backend is structurally
> unaffected. Not run in this change.

## Context

`KvSystemMetrics.reuse_hit_host_demoted` and `reuse_hit_disk` were assembled in
`Engine::kv_system_metrics` by adding two counters that measure two different
things:

- a page-tier **reuse decision**, incremented once per demoted block in
  `prefix.rs` while classifying each `PrefixBlock` before promotion;
- a tier-store **read**, returned by the backend `kv_tier_read_hits()`,
  incremented when the promote path actually read the key from the host/disk
  store.

`kv_system_metrics` summed them, so one block that was both decided-reusable and
read from the store was reported roughly twice. The sets are not identical — a
block whose tier location is unknown still counts as a host decision but need
not produce a store read, and store reads also occur outside prefix reuse — so
the fold conflated two quantities rather than cleanly doubling one.

## What changed

`reuse_hit_host_demoted`/`reuse_hit_disk` now come from one source: the tier
store's own read counters (`kv_tier_read_hits()`), assigned, not added. The
engine-side per-block decision increments were removed (a write-only counter
once the fold became an assignment). The resident-page hit count in the same
classify loop is unchanged.

CUDA is untouched by the number: its `KvPageTier` capacity is pinned to 0, so
the page-promote decision path never runs there; CUDA feeds the read view from
its slot/sidecar store and has always reported store reads only. The only
configuration whose value moves is Metal `--kv-disk`, the arm that was wrong.

Dashboard note: on Metal the `kv_system_reuse_hit_host_demoted_total` /
`..._disk_total` series step down after this change (toward roughly half in the
case where every decision also produced a read). That is the fix; the prior
value over-counted. CUDA dashboards do not move.

## Verification

Unit gate (`crates/infer-core/src/prefix.rs`, fake `KvPageTier`, no device):

- capacity 0, 3 host + 2 disk store reads: metric is 3/2 and unchanged across
  repeated `kv_system_metrics()` polls (the CUDA shape; guards against any
  per-poll re-add).
- capacity > 0, one demoted block the store reports as one host read: metric is
  1. With the previous decision-plus-store fold this test fails at 2
  (confirmed by temporarily restoring both original hunks: left 2, right 1).

Pending remote check (Mac, Metal): serve with `--kv-disk`, drive prefix reuse
from the host and disk tiers, and confirm `reuse_hit_host_demoted` /
`reuse_hit_disk` equal the observed reuse counts rather than twice them.

## Rule

A reuse metric names one event. Count either the reuse decision or the backing
store read from a single source; adding both reports one event twice and breaks
when the two sets diverge. Prefer the source that is non-zero on the production
backend, so the fix lands on the arm that was wrong and leaves the others
unchanged.
