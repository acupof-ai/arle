# 2026-04-15 · Metal Qwen3 same-length batch decode

## Context

`metal_serve` had already moved off the old serial `BackendRuntimeHandle`, but
`DecodeBatch` was still only a scheduler shape: the runtime loop iterated the
batch one request at a time. That kept the serving profile queue-shaped under
concurrency even after `M0.2b`.

This tranche lands the first real cross-request GPU decode path for Metal:

- the live runtime now tries a Qwen3 batch decode path before falling back to
  per-request decode
- eligibility is intentionally narrow: same model, same decode phase, and the
  same `cache_len` for every request in the batch
- Qwen3.5 still falls back to sequential decode inside the live runtime

Implementation highlights:

- `MetalRequestState::decode_batch(...)` now exposes a batched Qwen3 decode
  entrypoint for the runtime
- the batched path runs one shared MLX graph for embed/QKV/attention/MLP across
  the whole decode batch, then samples one token per row
- `metal::runtime` now removes active requests from the map, tries the batched
  path, then reinserts or finalizes requests explicitly
- `mlx-sys` / `metal::mlx` grew dynamic-RoPE and masked-SDPA bridge hooks for
  future heterogeneous-length work, but the shipped serving path intentionally
  stays on the simpler same-length batch route

Environment:

- Commit under test: post-`feat(metal): batch same-length qwen3 decode steps`
- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve --bin metal_bench`
- Artifacts:
  - `/tmp/metal-batch-2026-04-15/qwen3-direct-20-256.json`
  - `/tmp/metal-batch-2026-04-15/qwen3-serve-focused-small-v2.json`
  - `/tmp/metal-batch-2026-04-15/qwen35-quick.json`

## What Worked

### The runtime now has a real cross-request decode path for Qwen3

This is no longer just scheduler bookkeeping. Under same-length decode batches,
`metal_serve` now executes one shared Qwen3 MLX graph for the whole batch.

The serving path stays conservative on purpose:

- same-length Qwen3 decode batches use the new batched GPU path
- heterogeneous decode lengths fall back to the old per-request path
- Qwen3.5 remains sequential until there is a real batched step path for the
  hybrid full-attention + linear-attention model

That means this tranche is a real `M0.2` serving step, but not the full
throughput exit yet.

### Qwen3 direct sanity stayed healthy

`mlx-community/Qwen3-4B-bf16`, `prompt=20`, `generation=256`, `warmup=1`, `runs=3`

| Variant | Prompt TPS | Gen TPS | Repo E2E TPS | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 2026-04-15 baseline | `237.2` | `24.5` | `24.3` | `84.4 ms` |
| Current tree | `260.8` | `28.2` | `28.0` | `76.7 ms` |
| Delta | `+10.0%` | `+15.0%` | `+14.9%` | `-9.1%` |

This is only a sanity check, not the serving milestone exit, but it confirms
the Qwen3 direct path did not regress while the runtime work landed.

### Qwen3 focused serving improved, but did not break out of the plateau

Focused `metal_serve` check on Qwen3 with a decode-heavy `256`-token output:

| Variant | C | Throughput | TTFT p50 | ITL p50 |
| --- | ---: | ---: | ---: | ---: |
| 2026-04-15 baseline | `1` | `23.29 tok/s` | `1514 ms` | `36.98 ms` |
| Current tree | `1` | `24.73 tok/s` | `1061 ms` | `35.34 ms` |
| Delta | `1` | `+6.2%` | `-29.9%` | `-4.4%` |
| 2026-04-15 baseline | `4` | `23.30 tok/s` | `3559 ms` | `147.32 ms` |
| Current tree | `4` | `25.39 tok/s` | `2716 ms` | `141.84 ms` |
| Delta | `4` | `+9.0%` | `-23.7%` | `-3.7%` |

Interpretation:

- this tranche produced a measurable Qwen3 serving win
- the runtime is no longer *purely* queue-shaped on Qwen3
- but the improvement is still modest, not a step-function

The remaining bottleneck is clear: the batched path only fires for same-length
Qwen3 decode batches and still pays per-request KV gather/update work inside
each layer. It helps, but it is not yet the full `vLLM-style` serving shape.

### Qwen3.5 quick sweep stayed flat

`mlx-community/Qwen3.5-4B-MLX-4bit`, quick HTTP sweep:

| Config | 2026-04-15 rerun | Current tree | Delta |
| --- | ---: | ---: | ---: |
| `512/256 C=1` throughput | `65.2 tok/s` | `66.5 tok/s` | `+2.0%` |
| `512/256 C=1` TTFT p50 | `521 ms` | `512 ms` | `-1.7%` |
| `512/256 C=4` throughput | `65.5 tok/s` | `66.4 tok/s` | `+1.4%` |
| `512/256 C=4` TTFT p50 | `1742 ms` | `1737 ms` | `-0.3%` |

This is the important boundary:

- the new tranche does **not** materially change the current Qwen3.5 serving
  shape
- the fallback still works and did not regress
- `M0.2` throughput exit is therefore still blocked for the real mixed-model
  serving target

## Raw Data

### Qwen3 direct sanity

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":28.21936068391886,"p50":28.18075567946435,"p99":28.417679070275952},"load_ms":2152.163709,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":6081.625,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":260.8297921247369,"p50":260.5346909170794,"p99":261.7198680345613},"quantization":"bf16","repo_e2e_tps":{"mean":27.982830739598764,"p50":27.94567304657089,"p99":28.177565588097483},"timed_runs":3,"total_time_ms":{"mean":9148.716167,"p50":9160.631042,"p99":9200.275542},"ttft_ms":{"mean":76.67882,"p50":76.765209,"p99":76.853667},"warmup_runs":1}
```

### Qwen3 focused `metal_serve`

```json
{
  "variant": "qwen3-batch-current-v2",
  "results": [
    {
      "concurrency": 1,
      "requests": 2,
      "throughput_tps": 24.731238378602285,
      "ttft_p50_ms": 1060.7525415034615,
      "itl_p50_ms": 35.338228997716215,
      "avg_output_tokens": 256
    },
    {
      "concurrency": 4,
      "requests": 4,
      "throughput_tps": 25.385961046032385,
      "ttft_p50_ms": 2715.7896459975746,
      "itl_p50_ms": 141.840937496454,
      "avg_output_tokens": 256
    }
  ]
}
```

### Qwen3.5 quick sweep

```json
{
  "label": "metal-qwen35-post-qwen3-batch",
  "configs": [
    {
      "input_len": 512,
      "output_len": 256,
      "concurrency": 1,
      "throughput": 66.48213946819193,
      "ttft_p50": 511.55996322631836,
      "itl_p50": 13.01717758178711
    },
    {
      "input_len": 512,
      "output_len": 256,
      "concurrency": 4,
      "throughput": 66.42399832737067,
      "ttft_p50": 1736.8240356445312,
      "itl_p50": 52.10590362548828
    }
  ]
}
```

## Rule

When Metal serving has no native paged attention, the first viable cross-request
decode step is to batch the parts that are genuinely shared and keep the scope
tight:

- batch only requests that already have the same decode length
- keep Qwen3 and Qwen3.5 separate until both have a defensible batched step path
- treat a modest Qwen3 gain as proof that the runtime shape is improving, not
  as proof that `M0.2` is complete

The real throughput exit still requires the next tranche:

- heterogeneous-length Qwen3 decode batching, or
- a real batched Qwen3.5 decode path, or
- ideally both.
