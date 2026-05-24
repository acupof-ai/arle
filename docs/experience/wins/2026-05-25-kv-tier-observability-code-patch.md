# T4a KV-Tier Observability Code Patch

## Context

T4 split into:

- T4a: code-only observability patch, CPU-only, no SERVE bench.
- T4b: >=4k-token SERVE baseline after P5 PID 28950 finishes.

The goal is to add missing metrics before any `PrefetchPolicy::Timeout` or
policy tuning work. This entry records the audit and will be updated by the
implementation and test commits.

## Audit

Grep scope:

```bash
rg -n "record_tier_|record_prefix_lookup_detail|record_prefix_lookup_prefetch_queued|set_kv_coordinator|set_tier_wait_seconds|observe_h2d_latency_us|observe_d2h_latency_us|demote_block_to_host|spill_host_blocks_if_pressured|fallback_to_cold_prefill|fetch_backpressured|recompute_advised|host_pool_usage_fraction|host_pool_spill_target_bytes" infer/src/ -g '*.rs'
```

Verdict table:

| Field family | Current surface | Callsite status | T4a verdict |
| --- | --- | --- | --- |
| Per-tier hit rate T0/T1/T2/T3 | `EngineTelemetry.kv_tier_hit_rates`; T0 from `prefix_hit_rate`, T1/T2/T3 from staged source counters | already-wired via `record_tier_fetch_plan()` in admission and `snapshot_engine_telemetry()` | reuse; no duplicate field |
| Prefix lookup flags | `PrefixLookupDetail` stores `ready_on_gpu`, `direct_gpu_attach`, `staged`, `prefetch`, `recompute` as last-lookup gauges | already-wired in `record_prefix_lookup_detail()` and prefetch queue callback | reuse; add counter only for recompute-advised fallback |
| Coordinator queue gauges | `QueueControlStats` and `ServerMetrics::set_kv_coordinator()` expose capacity, fetch/store depth, waiters, backpressure, store totals | already-wired from scheduler loop | reuse; add specific queue-saturated fallback counter |
| Tier source counters | `record_tier_fetch_plan()`, `record_tier_fetch_promoted()`, `record_tier_fetch_fallback()` | already-wired for staged plan, promotion, and generic cold fallback | reuse; generic fallback is too broad for T4a |
| Oldest fetch/store wait | `set_tier_wait_seconds()` gauge | already-wired from scheduler loop | keep; add completed fetch-wait histogram for p50/p99 |
| T0->T1 demote latency + bytes | none | missing-field and missing-callback in `demote_block_to_host()` | add |
| T1->T2/T3 store latency + bytes | queue totals exist, but no completed latency or bytes | missing-field and missing-callback in store completion path | add |
| Queue-saturated fallback count | only generic `tier_fetch_fallback_total` | missing-field; fallback branches exist at fetch backpressure and submit-full cases | add |
| Recompute-advised fallback count | last lookup gauge only | missing-field; `lookup.recompute_advised` is passed through `PrefixLookupDetail` | add |
| Host pool high/low pressure ticks | host usage helpers exist, but no counters | missing-field and missing-callback in scheduler loop | add |

## What Worked

- Reused existing tier hit-rate projection instead of adding duplicate T0/T1/T2/T3 fields.
- Added code-only metrics for the gaps found by the audit: completed demote/store latency, completed staged-readmission fetch wait, specific fallback counters, and host-pool pressure ticks.
- Wired callbacks only at existing state-transition points:
  `demote_block_to_host()`, staged fetch queue fallback branches, store completion, readmission completion, and the scheduler loop pressure tick.
- Left T4b's SERVE bench deferred while P5 PID 28950 is running.

## Metric Semantics

- `tier_demote_to_host_latency_us`: completed T0 GPU to T1 host-pinned demote latency.
- `tier_demote_to_host_bytes_total`: bytes successfully demoted from T0 to T1.
- `tier_store_latency_us`: completed T1 to T2/T3 coordinator store latency.
- `tier_store_bytes_total`: bytes successfully stored from T1 to T2/T3.
- `tier_readmission_fetch_wait_us`: completed staged-readmission wait from fetch submit to ready adoption, used for p50/p99.
- `tier_fetch_queue_saturated_fallback_total`: staged prefix fell back to cold prefill because the fetch queue was saturated or rejected submit.
- `tier_recompute_advised_fallback_total`: prefix lookup found cached blocks but advised recompute instead of reuse.
- `host_pool_high_pressure_ticks_total`: scheduler ticks where T1 usage is at or above the configured high watermark.
- `host_pool_low_pressure_ticks_total`: scheduler ticks where T1 usage is at or below the configured low watermark.

## Verification

```bash
cargo check -p infer --no-default-features --features no-cuda
```

- Exit 0.
- Full `cargo test -p infer` remains in the test commit. T4a does not run
  SERVE bench; T4b owns the >=4k-token workload.

## Rule

Add observability before policy changes. If a future KV policy patch changes
admission, prefetch, demotion, or spill behavior, its wins/errors entry must
cite these counters or explain why the workload did not exercise the tier path.
