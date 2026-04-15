# 2026-04-15 · Metal runtime observability metrics

## Context

Metal `metal_serve` already had a live scheduler runtime, but the HTTP metrics
surface was still effectively disconnected from the runtime. `/metrics` and
`/v1/stats` only exposed the generic queue / KV / latency skeleton, and the
Metal path did not publish real runtime values for:

- active / waiting requests
- completed-request TTFT / TPOT / E2E histograms
- MLX allocator memory (`active`, `peak`, `cache`)
- a future-proof `prefix_hit_rate` gauge

That made it hard to explain Apple-side regressions without jumping straight to
Xcode captures or MLX internals.

This tranche wires the metrics object into both:

- the live Metal scheduler runtime
- the legacy serial runtime path used by Metal DFlash

It also binds the MLX allocator memory APIs in `mlx-sys`.

Environment:

- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve -- --model-path mlx-community/Qwen3-0.6B-4bit --port 8016 --warmup 0`

## What Worked

### `/metrics` and `/v1/stats` now surface real Metal runtime values

After server startup, idle stats showed real MLX allocator numbers instead of a
blank / static metrics object:

```text
requests=0 active=0 waiting=0 tokens_out=0 kv_util=0.0% prefix_hit_rate=0.0% active_mem=220.5MB peak_mem=315.0MB cache_mem=94.5MB ttft_p50=— ttft_p99=— tpot_p50=—
```

The matching Prometheus lines were present too:

```text
infer_prefix_hit_rate{model="Qwen3-0.6B-4bit",} 0.0000
infer_requests_waiting{model="Qwen3-0.6B-4bit",} 0
infer_kv_gpu_utilization{model="Qwen3-0.6B-4bit",} 0.0000
infer_memory_active_bytes{model="Qwen3-0.6B-4bit",} 231215604
infer_memory_peak_bytes{model="Qwen3-0.6B-4bit",} 330306124
infer_memory_cache_bytes{model="Qwen3-0.6B-4bit",} 99090520
```

### Request-completion metrics now update on the Metal path

One real completion request against `Qwen3-0.6B-4bit` updated the server-side
metrics correctly:

```text
requests=1 active=0 waiting=0 tokens_out=8 kv_util=0.0% prefix_hit_rate=0.0% active_mem=679.6MB peak_mem=736.9MB cache_mem=104.5MB ttft_p50=0.1ms ttft_p99=0.1ms tpot_p50=5.0ms
```

And the Prometheus counters / histograms moved:

```text
infer_requests_total{model="Qwen3-0.6B-4bit",} 1
infer_tokens_generated_total{model="Qwen3-0.6B-4bit",} 8
infer_requests_active{model="Qwen3-0.6B-4bit",} 0
infer_requests_waiting{model="Qwen3-0.6B-4bit",} 0
infer_ttft_seconds_count{model="Qwen3-0.6B-4bit",} 1
infer_e2e_seconds_count{model="Qwen3-0.6B-4bit",} 1
```

### This closes the runtime/HTTP metrics disconnect, but not Metal reuse

Important boundary:

- the observability surface is now real
- `prefix_hit_rate` is intentionally still `0` on today's live path because
  `M0.3` shared-prefix reuse is not wired yet
- KV utilization is still narrow because the live runtime does not yet own a
  scheduler-level shared KV pool

So this tranche is a genuine `M0.4` surface landing, but `M0.4` as a full
milestone still depends on `M0.3` making prefix reuse non-zero on real traffic.

## Rule

For Metal serving, adding metrics names is not enough. The runtime has to own
the values.

When `/metrics` and `/v1/stats` are backed by a detached default metrics object,
bench debugging will misclassify serving regressions as "scheduler shape" or
"unified memory" without evidence.

The minimum acceptable Metal observability surface is:

- runtime-backed queue depth
- runtime-backed request completion histograms
- MLX allocator memory (`active`, `peak`, `cache`)
- a prefix-reuse metric that can turn non-zero as soon as `M0.3` lands
