# /metrics reads cached stats snapshot — non-blocking Prometheus scrape — CUDA, 2026-08-20

> Status: Pending remote verification (Mac cannot run CUDA serve).

## Context

The stats timeout fix (`c7eb23420`) raised the `/metrics` query timeout from 2s
to 30s as a bandaid: workers under load (long prefill, busy decode) routinely
missed the 2s window, and the old code discarded the entire snapshot on any
rank timeout. The 30s timeout made `/metrics` correct but could block a
Prometheus scrape for up to 30s when workers were stalled.

## What Worked

The background observer (`spawn_coordinator_observe`) already polls
`query_stats_all(5s)` every 10s but discarded the result after writing it to
disk. The fix routes that snapshot into a shared cache:

1. `DpCoordinator` gains `cached_stats: Arc<RwLock<Option<CounterSnapshot>>>`.
2. The observer closure writes the snapshot to the cache after each poll.
3. `/metrics` reads the cache (instant, no RPC). On cold start (cache empty,
   observer disabled or first poll not yet completed) it falls back to a 5s
   direct query — the original pre-bandaid behavior.

Cache staleness is bounded by the 10s poll interval + 5s query timeout (~15s
worst case), well within Prometheus scrape tolerance.

## Result

Pending remote verification. Expected: `/metrics` returns instantly under load
instead of blocking up to 30s.

## Rule

A metrics endpoint must never block on the system it measures. Reuse the
existing background poll; a stale snapshot is always better than a blocked
scrape.
