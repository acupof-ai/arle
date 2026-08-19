# Cross-rank stats aggregation — /v1/stats and /metrics report group-level truth

> Status: Shipped (e2e pending-remote)

## Context

#209: under TP, `send_stats_query` only queried rank 0. Per-rank `kv_free_pages`,
prefix-cache hits, and throughput can diverge (prefix cache is per-rank state).
`/metrics` reported the rank-0 view, not the group-level truth.

## What Changed

- `send_stats_query` now broadcasts to ALL connected ranks (was rank-0-only).
- `stats_sinks` changed from `oneshot::Sender<WireStats>` to an mpsc collector;
  the coordinator collects `worker_count` responses then unregisters.
- `aggregate_wire_stats`: rank 0's counters (identical across ranks under TP)
  + min across ranks for KV gauges (`kv_free_pages`, `prefix_cached_pages`,
  `kv_tier_resident_blocks`, `kv_system_*_pages`).
- `query_stats` and the `/v1/stats` handler both use the shared
  `collect_wire_stats` + `aggregate_wire_stats` path.

## Verification

- 3 unit tests on the aggregate function (rank-0 counters, min gauges,
  single-rank identity, empty default).
- `cargo test -p infer-server` 9/9 pass; clippy clean.
- Multi-rank e2e (TP serve, check /metrics divergence) is pending-remote.

## Rule

Under TP, counters are rank-0 truth (all ranks serve the same requests);
gauges that reflect per-rank KV state are min across ranks (group capacity =
min).
