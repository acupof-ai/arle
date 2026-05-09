# Service Trace Summary

- Poll interval: `1000ms`
- Samples: `57` (ok: `57`, failed: `0`)
- Peak waiting: `3`
- Peak active: `4`
- Peak running_batch: `4`
- Peak prefill_queue: `3`
- Plan labels: `idle=16121`, `decode=1509`, `prefill=26`, `split=2`, `mixed=0`
- Peak kv_util: `97.4%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/57`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Trace Distributions

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 3 | 3 |
| kv_util | 24.7% | 50.0% | 75.3% | 97.3% | 97.4% |

## Token Counters

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 0 | 0 | 0 | 0 |
| prefill_tokens | 0 | 0 | 0 | 0 | 0 |
| tokens_out | 8 | 2056 | 4104 | 5367 | 5367 |

## Before

```text
requests=1 active=0 waiting=0 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=0 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=8 step_last=0.0ms step_p50=1.0ms step_phase_us=adm:0,prefill:0,decode:1,emit:0,total:1,cleanup:3,loop_total:4 plan_label=idle:6470,decode:7,prefill:1,split:0,mixed:0 prefill_path=ok_true:0,ok_false:0 spec=draft:0,verified:0,accepted:0,empty_sparse_views:0,accept_rate:0.0%,step_latency_count:0 tier_fetch_wait=0.0ms tier_store_wait=0.0ms kv_util=0.0% prefix_hit_rate=0.0% active_mem=13935.6MB peak_mem=13935.6MB cache_mem=0.0MB queue_p50=1.0ms active_ttft_p50=500.0ms ttft_p50=500.0ms ttft_p99=500.0ms service_p50=150.0ms tpot_p50=20.0ms metal_decode=batch:0/0,scalar:0,fallback:0,qwen35_packed:0/0 prefix_skip_rate=0.0% prefix_request_hit_rate=0.0% prefix_request_skip_rate=0.0% session_affinity_hit=0 session_affinity_miss=0 session_slot_pressure_evictions_hard=0 matched_prefix_tokens=0 resume_prefill_tokens=2 kv_fetch_q=0/16 kv_fetch_waiters=0 kv_store_q=0/16 kv_store=sub:0,done:0,fail:0,rej:0 kv_bp=fetch:0,store:0 engine_ttft_us=500000.0 engine_itl_p50_us=20000.0 engine_itl_p99_us=20000.0 engine_queue_depth=0 engine_active_requests=0 engine_batch_occupancy=0.0000 engine_timestamp_ms=1778314546640 engine_kv_tier_hit_T0=0.0000
```

## After

```text
requests=29 active=0 waiting=0 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=0 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=5367 step_last=0.1ms step_p50=1.0ms step_phase_us=adm:89,prefill:0,decode:1,emit:0,total:90,cleanup:13460,loop_total:13543 plan_label=idle:16121,decode:1509,prefill:26,split:2,mixed:0 prefill_path=ok_true:0,ok_false:0 spec=draft:0,verified:0,accepted:0,empty_sparse_views:0,accept_rate:0.0%,step_latency_count:0 tier_fetch_wait=0.0ms tier_store_wait=0.0ms kv_util=50.0% prefix_hit_rate=0.0% active_mem=14671.7MB peak_mem=14959.6MB cache_mem=0.0MB queue_p50=1.0ms active_ttft_p50=2000.0ms ttft_p50=2000.0ms ttft_p99=2000.0ms service_p50=10000.0ms tpot_p50=25.0ms metal_decode=batch:0/0,scalar:0,fallback:0,qwen35_packed:0/0 prefix_skip_rate=0.0% prefix_request_hit_rate=0.0% prefix_request_skip_rate=0.0% session_affinity_hit=0 session_affinity_miss=0 session_slot_pressure_evictions_hard=0 matched_prefix_tokens=0 resume_prefill_tokens=4097 kv_fetch_q=0/16 kv_fetch_waiters=0 kv_store_q=0/16 kv_store=sub:0,done:0,fail:0,rej:0 kv_bp=fetch:0,store:0 engine_ttft_us=2000000.0 engine_itl_p50_us=25000.0 engine_itl_p99_us=30000.0 engine_queue_depth=0 engine_active_requests=0 engine_batch_occupancy=0.5000 engine_timestamp_ms=1778314604407 engine_kv_tier_hit_T0=0.0000
```
