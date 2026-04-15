# 2026-04-15 · Metal Qwen3.5 same-length batch decode

## Context

After `M0.2c`, the live Metal runtime had a real cross-request decode batch
path only for Qwen3. Qwen3.5 still ran one request at a time inside the live
scheduler runtime, which kept the main Apple serving target effectively flat on
the HTTP quick sweep.

This tranche extends the compiled Qwen3.5 step path so same-length decode
batches can run through one batched `B x S` forward in the C++/MLX bridge
instead of always falling back to sequential decode.

Scope kept deliberately narrow:

- only Qwen3.5 request-state decode batches
- only decode (`S=1`), not prompt prefill
- only same `cache_len` and same `kv_capacity`
- still no heterogeneous-length decode batch

Environment:

- Commit under test: post-`feat(metal): batch same-length qwen35 decode steps`
- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve --bin metal_bench`
- Artifacts:
  - `/tmp/metal-qwen35-batch-2026-04-15/qwen35-direct-128-128.json`
  - `/tmp/metal-qwen35-batch-2026-04-15/qwen35-quick.json`

## What Worked

### Qwen3.5 now has a real batched decode entry in the live runtime

This is no longer "scheduler says DecodeBatch, runtime loops requests one by
one" for the happy-path Qwen3.5 case.

The new path:

- generalizes the compiled Qwen3.5 bridge from fixed `1 x S` execution to
  batched `B x S`
- adds a batched decode FFI entrypoint for same-length Qwen3.5 request-state
  batches
- keeps the runtime contract narrow enough that correctness is still easy to
  reason about

The important boundary is still explicit:

- same-length Qwen3.5 decode batches now use a batched compiled-model step
- variable-length decode batches still fall back
- this path still rebuilds per-step batched state by concatenating request-local
  KV / recurrent tensors, then slices the outputs back apart

### Direct Qwen3.5 sanity improved slightly

`mlx-community/Qwen3.5-4B-MLX-4bit`, `prompt=128`, `generation=128`, `warmup=1`, `runs=3`

| Variant | Prompt TPS | Gen TPS | Repo E2E TPS | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 2026-04-15 rerun baseline | `719.7` | `82.0` | `73.6` | `178.3 ms` |
| Current tree | `751.0` | `84.2` | `75.7` | `170.5 ms` |
| Delta | `+4.3%` | `+2.7%` | `+2.9%` | `-4.4%` |

That is a healthy sanity result, but it is not the serving exit.

### HTTP quick sweep stayed effectively flat

`mlx-community/Qwen3.5-4B-MLX-4bit`, quick `metal_serve` sweep:

| Config | 2026-04-15 post-`M0.2c` | Current tree | Delta |
| --- | ---: | ---: | ---: |
| `512/256 C=1` throughput | `66.5 tok/s` | `66.3 tok/s` | `-0.3%` |
| `512/256 C=1` TTFT p50 | `512 ms` | `542 ms` | `+5.9%` |
| `512/256 C=4` throughput | `66.4 tok/s` | `66.2 tok/s` | `-0.3%` |
| `512/256 C=4` TTFT p50 | `1737 ms` | `1757 ms` | `+1.2%` |

Interpretation:

- the same-length Qwen3.5 batch path compiles, runs, and does not regress the
  direct path
- but it does **not** yet produce a clear serving-side throughput step
- the likely reason is architectural, not correctness: each decode step still
  pays request-state concat/split overhead for KV and recurrent tensors, which
  cancels most of the batched forward win

So this tranche proves the C++ bridge can execute batched Qwen3.5 decode, but
it does **not** close `M0.2`.

## Raw Data

### Direct `metal_bench`

```json
{"avg_tokens":128,"generation_tokens_requested":128,"generation_tps":{"mean":84.22047874956714,"p50":84.20319775694891,"p99":84.26519702124057},"load_ms":534.3312500000001,"model":"mlx-community/Qwen3.5-4B-MLX-4bit","peak_rss_mb":2520.109375,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":750.9780141076168,"p50":753.5521432503566,"p99":754.6867625129469},"quantization":"4-bit","repo_e2e_tps":{"mean":75.72755017702525,"p50":75.7316759367701,"p99":75.8015140316079},"timed_runs":3,"total_time_ms":{"mean":1690.2710560000003,"p50":1690.177834,"p99":1692.014709},"ttft_ms":{"mean":170.45045833333333,"p50":169.862167,"p99":171.882417},"warmup_runs":1}
```

### `metal_serve` quick sweep

```text
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err |   Wall
--------------------------------------------------------------------------------------------
  128 |   128 |  1 |     71.1 t/s |     165ms |     200ms |   12.8ms |   12.8ms |   0 |  14.4s
  128 |   512 |  1 |     75.6 t/s |     166ms |     166ms |   12.9ms |   12.9ms |   0 |  54.2s
  512 |   256 |  1 |     66.3 t/s |     542ms |     544ms |   13.0ms |   13.0ms |   0 |  30.9s
 1024 |   256 |  1 |     58.8 t/s |    1004ms |    1009ms |   13.1ms |   13.1ms |   0 |  34.9s
 2048 |   256 |  1 |     47.5 t/s |    2001ms |    2014ms |   13.2ms |   13.2ms |   0 |  43.1s
  512 |   256 |  4 |     66.2 t/s |    1757ms |    2483ms |   52.2ms |   52.2ms |   0 |  30.9s
```

## Rule

For Metal Qwen3.5, "batched decode exists" is not the same thing as "serving
throughput improved".

If the runtime still has to rebuild batched state from request-local KV /
recurrent tensors on every decode step, the bridge-level batched forward may
only show up as a direct-path gain while HTTP throughput stays flat.

That means the next Qwen3.5 serving tranche should not be another small bridge
micro-optimization. It should attack the real remaining cost:

- persistent batched state across decode steps, or
- a scheduler/runtime layout that stops concat/split from dominating each step,
  before
- heterogeneous-length decode batching.
