# Service Trace Summary

- Poll interval: `1000ms`
- Samples: `135` (ok: `135`, failed: `0`)
- Peak waiting: `0`
- Peak active: `4`
- Peak running_batch: `4`
- Peak prefill_queue: `3`
- Plan labels: `idle=63186`, `decode=3577`, `prefill=61`, `split=0`, `mixed=0`
- Peak kv_util: `100.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/135`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Trace Distributions

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 0 | 0 |
| kv_util | 82.2% | 99.9% | 100.0% | 100.0% | 100.0% |

## Token Counters

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 0 | 0 | 0 | 0 |
| prefill_tokens | 0 | 0 | 0 | 0 | 0 |
| tokens_out | 2056 | 6152 | 10248 | 14345 | 14345 |

## Before

```text
requests=1 active=0 waiting=0 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=0 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=8 step_last=0.0ms step_p50=1.0ms step_phase_us=adm:0,prefill:0,decode:1,emit:0,total:1,cleanup:2,loop_total:4 plan_label=idle:8881,decode:7,prefill:1,split:0,mixed:0 prefill_path=ok_true:0,ok_false:0 spec=draft:0,verified:0,accepted:0,empty_sparse_views:0,accept_rate:0.0%,step_latency_count:0 tier_fetch_wait=0.0ms tier_store_wait=0.0ms kv_util=0.0% prefix_hit_rate=0.0% active_mem=13829.6MB peak_mem=13829.6MB cache_mem=0.0MB queue_p50=1.0ms active_ttft_p50=300.0ms ttft_p50=300.0ms ttft_p99=300.0ms service_p50=100.0ms tpot_p50=15.0ms metal_decode=batch:0/0,scalar:0,fallback:0,qwen35_packed:0/0 prefix_skip_rate=0.0% prefix_request_hit_rate=0.0% prefix_request_skip_rate=0.0% session_affinity_hit=0 session_affinity_miss=0 session_slot_pressure_evictions_hard=0 matched_prefix_tokens=0 resume_prefill_tokens=2 kv_fetch_q=0/16 kv_fetch_waiters=0 kv_store_q=0/16 kv_store=sub:0,done:0,fail:0,rej:0 kv_bp=fetch:0,store:0 engine_ttft_us=300000.0 engine_itl_p50_us=15000.0 engine_itl_p99_us=15000.0 engine_queue_depth=0 engine_active_requests=0 engine_batch_occupancy=0.0000 engine_timestamp_ms=1778303017605 engine_kv_tier_hit_T0=0.0000
```

## After

```text
requests=61 active=0 waiting=0 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=0 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=14345 step_last=0.1ms step_p50=1.0ms step_phase_us=adm:31,prefill:2645,decode:0,emit:0,total:2676,cleanup:5,loop_total:141 plan_label=idle:63186,decode:3577,prefill:61,split:0,mixed:0 prefill_path=ok_true:0,ok_false:0 spec=draft:0,verified:0,accepted:0,empty_sparse_views:0,accept_rate:0.0%,step_latency_count:0 tier_fetch_wait=0.0ms tier_store_wait=0.0ms kv_util=61.1% prefix_hit_rate=0.0% active_mem=14629.7MB peak_mem=15632.8MB cache_mem=0.0MB queue_p50=1.0ms active_ttft_p50=2000.0ms ttft_p50=2000.0ms ttft_p99=5000.0ms service_p50=10000.0ms tpot_p50=30.0ms metal_decode=batch:0/0,scalar:0,fallback:0,qwen35_packed:0/0 prefix_skip_rate=0.0% prefix_request_hit_rate=0.0% prefix_request_skip_rate=0.0% session_affinity_hit=0 session_affinity_miss=0 session_slot_pressure_evictions_hard=0 matched_prefix_tokens=0 resume_prefill_tokens=4097 kv_fetch_q=0/16 kv_fetch_waiters=0 kv_store_q=0/16 kv_store=sub:0,done:0,fail:0,rej:0 kv_bp=fetch:0,store:0 engine_ttft_us=2000000.0 engine_itl_p50_us=30000.0 engine_itl_p99_us=30000.0 engine_queue_depth=0 engine_active_requests=0 engine_batch_occupancy=0.6115 engine_timestamp_ms=1778303154333 engine_kv_tier_hit_T0=0.0000
```
