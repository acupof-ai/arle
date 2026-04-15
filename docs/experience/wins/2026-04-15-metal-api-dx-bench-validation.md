# 2026-04-15 · Metal API/DX tranche bench validation

## Context

User asked to validate the Metal API/DX tranche with real benchmark data before
continuing the scheduler work.

This session had two goals:

1. verify that the user-facing Metal API/DX tranche did not regress the direct
   Metal execution path
2. capture a real `metal_serve` concurrency snapshot so the next serving work
   is judged against data instead of intuition

Environment:

- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a) (`Darwin 25.3.0`)
- Branch: `main`
- Direct bench build:
  `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_serve`
- Direct bench binaries:
  `/tmp/agent-infer-bench-target-current/release/metal_bench`
  `/tmp/agent-infer-bench-target-current/release/metal_serve`

Benchmark hygiene note:

- The first attempt was to benchmark a detached worktree at commit `0ef3a3d`.
- That isolated build failed because in-flight tiered-KV serde changes were not
  yet present there (`BlockLocation` / `SessionId` serialization).
- The final numbers below therefore come from the current integrated tree on
  2026-04-15, not from a clean detached build of `0ef3a3d`.
- The saved HTTP sweep JSON still carries the old label
  `metal-api-dx-0ef3a3d`; that label is stale and should not be treated as the
  actual commit under test.

## What Worked

Two things became clear.

First, the direct Metal path is still healthy:

- `Qwen3.5-4B-MLX-4bit` at `128 / 128` stayed in the same range as the earlier
  direct-path tuning snapshots
- `Qwen3-4B-bf16` baseline and DFlash both stayed in the same range as the
  2026-04-14 DFlash validation

Second, the serving bottleneck is still architecture, not single-request decode:

- `metal_serve` aggregate throughput stayed almost flat once concurrency rose
- TTFT scaled roughly with queue depth
- the current Metal server is still a serialized request queue with streaming on
  top, not a real batching scheduler

That means the roadmap correction was right: `M0.2` is still the main blocker,
and further single-request-only work should not be mistaken for serving
progress.

## Direct Bench Delta

### `Qwen3.5-4B-MLX-4bit` direct path

Reference snapshot:

- [2026-04-13-qwen35-metal-cpp-path-tuning.md](2026-04-13-qwen35-metal-cpp-path-tuning.md)

Workload:

- model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- prompt / generation: `128 / 128`
- warmup / runs: `1 / 2`

| Metric | 2026-04-13 reference | 2026-04-15 validation | Delta |
| --- | ---: | ---: | ---: |
| prompt TPS | `641.5` | `732.9` | `+14.2%` |
| generation TPS | `81.3` | `80.2` | `-1.3%` |
| repo E2E TPS | `72.17` | `72.30` | `+0.2%` |
| TTFT | `199.5 ms` | `174.7 ms` | `-12.4%` |

Interpretation:

- repo-level throughput is effectively flat
- generation throughput stayed within noise
- direct execution did not regress in a way that would block the API/DX tranche

### `Qwen3-4B-bf16` direct path

Reference snapshot:

- [2026-04-14-metal-dflash-qwen3.md](2026-04-14-metal-dflash-qwen3.md)

Workload:

- model: `mlx-community/Qwen3-4B-bf16`
- prompt / generation: `20 / 256`
- warmup / runs: `1 / 3`

| Variant | Metric | 2026-04-14 reference | 2026-04-15 validation | Delta |
| --- | --- | ---: | ---: | ---: |
| baseline | prompt TPS | `249.3` | `247.5` | `-0.7%` |
| baseline | generation TPS | `25.9` | `26.4` | `+1.9%` |
| baseline | repo E2E TPS | `25.7` | `26.2` | `+1.9%` |
| baseline | TTFT | `80.2 ms` | `81.0 ms` | `+0.9%` |
| DFlash | prompt TPS | `236.3` | `243.6` | `+3.1%` |
| DFlash | generation TPS | `152.0` | `154.5` | `+1.7%` |
| DFlash | repo E2E TPS | `144.8` | `147.2` | `+1.7%` |
| DFlash | TTFT | `84.6 ms` | `82.1 ms` | `-3.0%` |

Interpretation:

- no meaningful regression in the target-only path
- DFlash stayed in the same strong generation-heavy regime

## Raw Data

### Direct `metal_bench`

`mlx-community/Qwen3.5-4B-MLX-4bit`, `128 / 128`

```json
{"avg_tokens":128,"generation_tokens_requested":128,"generation_tps":{"mean":80.2208377772112,"p50":79.9924736081488,"p99":80.4492019462736},"load_ms":858.45775,"model":"mlx-community/Qwen3.5-4B-MLX-4bit","peak_rss_mb":2385.3125,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":732.9179522607242,"p50":716.6012360285223,"p99":749.2346684929261},"quantization":"4-bit","repo_e2e_tps":{"mean":72.30253343749574,"p50":72.27589571401803,"p99":72.32917116097344},"timed_runs":2,"total_time_ms":{"mean":1770.3393130000002,"p50":1769.6870840000001,"p99":1770.9915420000002},"ttft_ms":{"mean":174.7309795,"p50":170.841,"p99":178.620959},"warmup_runs":1}
```

`mlx-community/Qwen3-4B-bf16`, baseline `20 / 256`

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":26.395256632646994,"p50":26.276516057422384,"p99":26.830059400348173},"load_ms":4009.885834,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":4072.75,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":247.46096740908652,"p50":252.48328036781157,"p99":256.3828661740535},"quantization":"bf16","repo_e2e_tps":{"mean":26.176514745725587,"p50":26.064594004022215,"p99":26.59136934860753},"timed_runs":3,"total_time_ms":{"mean":9781.066361,"p50":9821.752833,"p99":9894.262458},"ttft_ms":{"mean":80.95615266666665,"p50":79.213166,"p99":85.646959},"warmup_runs":1}
```

`mlx-community/Qwen3-4B-bf16`, DFlash `20 / 256`

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":154.53498055094303,"p50":154.6672154485027,"p99":155.70872041178038},"load_ms":3489.691333,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":7889.046875,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":243.59194280524366,"p50":243.8039511477833,"p99":244.81339863730742},"quantization":"bf16","repo_e2e_tps":{"mean":147.23720228056987,"p50":147.31633235990097,"p99":148.33783403745605},"timed_runs":3,"total_time_ms":{"mean":1738.760833,"p50":1737.757083,"p99":1752.735083},"ttft_ms":{"mean":82.10618066666667,"p50":82.033125,"p99":82.590542},"warmup_runs":1}
```

### `metal_serve` HTTP sweep

Server command:

```bash
/tmp/agent-infer-bench-target-current/release/metal_serve \
  --model-path mlx-community/Qwen3.5-4B-MLX-4bit \
  --port 8010 \
  --warmup 1 \
  --warmup-max-new-tokens 8
```

Sweep command:

```bash
python3 scripts/bench_throughput_sweep.py \
  --url http://127.0.0.1:8010 \
  --label metal-api-dx-0ef3a3d \
  --save /tmp/metal-bench-2026-04-15/metal-serve-throughput-sweep.json
```

Raw table:

```text
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err |   Wall
--------------------------------------------------------------------------------------------
  128 |    64 |  1 |     64.8 t/s |     187ms |     220ms |   12.6ms |   12.7ms |   0 |   7.9s
  128 |   128 |  1 |     72.9 t/s |     185ms |     190ms |   12.3ms |   12.5ms |   0 |  14.1s
  128 |   256 |  1 |     75.4 t/s |     187ms |     190ms |   12.6ms |   12.7ms |   0 |  27.2s
  128 |   512 |  1 |     76.6 t/s |     188ms |     199ms |   12.7ms |   12.8ms |   0 |  53.4s
  512 |   128 |  1 |     57.9 t/s |     545ms |     627ms |   12.6ms |   14.4ms |   0 |  17.7s
  512 |   256 |  1 |     66.6 t/s |     586ms |     655ms |   12.4ms |   13.2ms |   0 |  30.7s
  512 |   512 |  1 |     65.5 t/s |     714ms |    1054ms |   13.8ms |   16.4ms |   0 |  62.6s
 1024 |   128 |  1 |     39.0 t/s |    1517ms |    1863ms |   13.7ms |   15.4ms |   0 |  26.3s
 1024 |   256 |  1 |     54.2 t/s |    1355ms |    1784ms |   12.7ms |   13.5ms |   0 |  37.8s
 1024 |   512 |  1 |     53.2 t/s |    1836ms |    2323ms |   14.5ms |   17.3ms |   0 |  76.9s
 2048 |   256 |  1 |     39.9 t/s |    2943ms |    3353ms |   13.0ms |   13.5ms |   0 |  51.3s
  512 |   256 |  2 |     62.5 t/s |    4540ms |    5267ms |   12.5ms |   12.9ms |   0 |  32.8s
  512 |   256 |  4 |     63.1 t/s |    8669ms |   13164ms |   12.5ms |   12.6ms |   0 |  32.5s
  128 |   128 |  2 |     67.3 t/s |    2045ms |    2451ms |   12.3ms |   15.3ms |   0 |  15.2s
  128 |   128 |  4 |     69.1 t/s |    3799ms |    5787ms |   12.4ms |   12.9ms |   0 |  14.8s
  128 |   256 |  8 |     68.4 t/s |   14625ms |   26720ms |   12.4ms |   13.5ms |   0 |  59.9s
  512 |   256 |  8 |     61.9 t/s |   16416ms |   30796ms |   12.5ms |   16.3ms |   0 |  66.2s
  128 |   256 | 16 |     73.0 t/s |   27485ms |   54128ms |   12.5ms |   14.1ms |   0 | 112.2s
  512 |   256 | 16 |     63.8 t/s |   32645ms |   61609ms |   12.5ms |   13.6ms |   0 | 128.4s
  128 |   256 | 32 |     69.8 t/s |   59108ms |  116193ms |   12.8ms |   17.4ms |   0 | 234.7s
  512 |   256 | 32 |     61.9 t/s |   60402ms |  119506ms |   12.8ms |   15.9ms |   5 | 243.9s
  128 |   256 | 64 |     69.8 t/s |   61747ms |  117975ms |   12.7ms |   15.2ms |  62 | 242.2s
```

## Rule

For Metal in April 2026:

- direct `metal_bench` still matters, but it is no longer the serving roadmap
  driver
- any change that leaves direct benchmarks healthy but still produces
  throughput-flat `metal_serve` curves under concurrency has not solved the real
  serving problem
- `M0.2` should only count as shipped once the HTTP sweep shape changes, not
  once the code merely stops importing `BackendRuntimeHandle`
