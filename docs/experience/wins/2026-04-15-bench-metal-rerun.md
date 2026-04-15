# 2026-04-15 · Metal bench rerun snapshot

## Context

After `f7b3b84` (`docs(metal): clarify kv quant support scope`), reran the
current Metal benchmark pair on `main` to confirm the serving shape had not
drifted while clarifying the support matrix and roadmap language.

Environment:

- Commit: `f7b3b84`
- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve --bin metal_bench`
- Artifacts:
  - `/tmp/metal-bench-2026-04-15-rerun/qwen35-128-128.json`
  - `/tmp/metal-bench-2026-04-15-rerun/metal-serve-quick.json`

## What Worked

The rerun stayed aligned with the post-`M0.2b` numbers:

- direct `metal_bench` remained in the same range as the last live-runtime
  validation
- `metal_serve` still showed the improved concurrency latency shape from the
  live scheduler runtime
- aggregate throughput at `512 / 256, C=4` remained roughly flat against the
  best sanity rerun, which confirms the next real gain still depends on batched
  decode rather than another docs or single-request tuning pass

### Direct-path rerun vs `M0.2b`

`mlx-community/Qwen3.5-4B-MLX-4bit`, `128 / 128`, `warmup=1`, `runs=3`

| Metric | `M0.2b` local | rerun | Delta |
| --- | ---: | ---: | ---: |
| prompt TPS | `718.6` | `719.7` | `+0.1%` |
| generation TPS | `83.5` | `82.0` | `-1.7%` |
| repo E2E TPS | `74.8` | `73.6` | `-1.6%` |
| TTFT | `178.1 ms` | `178.3 ms` | `+0.1%` |

### HTTP quick sweep rerun vs previous `M0.2b` quick sweep

| Config | previous `M0.2b` quick sweep | rerun | Delta |
| --- | ---: | ---: | ---: |
| `512 / 256, C=1` throughput | `62.8 tok/s` | `65.2 tok/s` | `+3.8%` |
| `512 / 256, C=1` TTFT p50 | `521 ms` | `521 ms` | `+0.0%` |
| `512 / 256, C=4` throughput | `58.7 tok/s` | `65.5 tok/s` | `+11.6%` |
| `512 / 256, C=4` TTFT p50 | `1826 ms` | `1742 ms` | `-4.6%` |

Interpretation:

- the rerun matches the earlier sanity result much more closely than the first
  `M0.2b` quick sweep
- the live scheduler runtime remains materially better than the old serial TTFT
  shape under concurrency
- the remaining gap is still architectural: there is no new evidence that
  cross-request batched decode has landed

## Raw Data

### Direct `metal_bench`

```json
{"avg_tokens":128,"generation_tokens_requested":128,"generation_tps":{"mean":82.03476974469251,"p50":82.30782790044064,"p99":83.32873110270467},"load_ms":1001.557459,"model":"mlx-community/Qwen3.5-4B-MLX-4bit","peak_rss_mb":2334.0,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":719.694559661958,"p50":744.517365396408,"p99":746.718882229038},"quantization":"4-bit","repo_e2e_tps":{"mean":73.6197710242552,"p50":73.27692603038832,"p99":74.96333456922689},"timed_runs":3,"total_time_ms":{"mean":1738.9741386666665,"p50":1746.798166,"p99":1762.622833},"ttft_ms":{"mean":178.33352766666667,"p50":171.923458,"p99":191.660541},"warmup_runs":1}
```

### `metal_serve` quick sweep

```text
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err |   Wall
--------------------------------------------------------------------------------------------
  128 |   128 |  1 |     67.0 t/s |     170ms |     212ms |   13.0ms |   14.2ms |   0 |  15.3s
  128 |   512 |  1 |     71.8 t/s |     168ms |     466ms |   13.2ms |   13.6ms |   0 |  57.0s
  512 |   256 |  1 |     65.2 t/s |     521ms |     524ms |   13.1ms |   13.7ms |   0 |  31.4s
 1024 |   256 |  1 |     57.5 t/s |    1049ms |    1049ms |   13.2ms |   13.4ms |   0 |  35.6s
 2048 |   256 |  1 |     47.1 t/s |    1996ms |    2026ms |   13.3ms |   13.4ms |   0 |  43.4s
  512 |   256 |  4 |     65.5 t/s |    1742ms |    2463ms |   52.6ms |   52.8ms |   0 |  31.3s
```

## Rule

When the latest Metal benchmark rerun lands within noise of the previous
validation, treat that as a routing signal:

- do not reopen direct-path tuning just because one sweep came out slightly
  higher or lower
- use reruns to confirm stability, then put effort back into the next missing
  serving primitive, which is still batched decode
