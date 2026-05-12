# Qwen3.5 FP8 KV Decode x4 Loads - 2026-05-12

## Goal

- Optimize the live Qwen3.5 FP8 KV decode-attention operator and regression-gate
  the serving path with explicit FP8 KV cache enabled.

## Hypothesis

- Loading four aligned FP8 E4M3 values at a time with `__nv_fp8x4_e4m3` should
  reduce scalar FP8 load/convert overhead in `decode_attention_fp8_partial_kernel`
  for Qwen3.5 `head_dim=256`, without changing the softmax order or KV format.

## Command

Component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/decode_attention_fp8_qwen35 --quiet
```

Server:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 5120 \
  --kv-cache-dtype fp8
```

Canonical sweep attempted:

```bash
PATH=$PWD/.venv/bin:$PATH \
scripts/bench_guidellm.sh qwen35-fp8-kv-decode-x4-load \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B
```

Fixed-c regression control:

```bash
PATH=$PWD/.venv/bin:$PATH \
scripts/bench_guidellm.sh qwen35-fp8-kv-decode-x4-load-c1 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1 --max-seconds 60 --warmup 10
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3.5-4B` served from `infer/models/Qwen3.5-4B`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit under test:** working tree based on `590d0b1`; this entry is
  committed with the code delta
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`
- **Server launch:** command above with `--kv-cache-dtype fp8`
- **Scheduling envelope:** `max_num_batched_tokens=16384`,
  `chunked_prefill_size=2048`, `max_prefill_tokens=16384`,
  `mem_fraction_static=0.85`, `max_slots=8`
- **KV pool:** `81552` max tokens, `5097` pages, `page_size=16`,
  `4 kv_heads x 256 head_dim`, `kv_dim=1024`, `format=FP8E4M3`

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh qwen35-fp8-kv-decode-x4-load`

## Results - Component A/B

Shape: batch `4`, seq_len `4096`, Q heads `16`, KV heads `4`, head_dim `256`,
page_size `16`, FP8 E4M3 K/V plus f32 per-token/head scales.

| metric | baseline scalar FP8 loads | x4 FP8 loads | delta |
|---|---:|---:|---:|
| latency median | 100.51 us | 82.755 us | -17.67% |
| latency interval | 100.40-100.56 us | 82.590-82.864 us | separated |
| throughput median | 667.69 Gelem/s | 810.94 Gelem/s | +21.46% |

Control experiment killed before landing:

| treatment | result |
|---|---:|
| INT8-style shared-memory `cp.async` prefetch ported to FP8 | 134.35 us, +33.7% regression |

See
[`2026-05-12-fp8-kv-decode-shared-prefetch-kill.md`](../errors/2026-05-12-fp8-kv-decode-shared-prefetch-kill.md).

## Results - Fixed c=1 Guidellm

The canonical sweep failed before completion because the Qwen3.5 server hit a
stack overflow under the sweep workload. The fixed c=1 control completed with
0 request errors.

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 1965.2 | 1.0 | 1965.2 | 1966.7 | 22.85 | 15.24 | 0.04 | 15.24 | 15.29 | 15.29 | 15.29 | 5.85 | 5.87 | 1 | 45.44 | 772.73 | 787.89 | 36873 | 2304 | 0.18 |

## Results - Service-Side Metrics

Fixed c=1 service trace:

| metric | value |
|---|---:|
| samples | 72 ok / 0 failed |
| peak waiting | 0 |
| peak active | 1 |
| peak running_batch | 1 |
| peak prefill_queue | 1 |
| plan labels | `idle=12`, `decode=2558`, `prefill=34`, `split=0`, `mixed=0` |
| peak kv_util | 55.2% |
| prefix hit / skip rate | 0.0% / 0.0% |
| kv fetch waiters | 0/72 |
| kv store queue samples | 0/0 |

Canonical failed sweep trace before server exit:

| metric | value |
|---|---:|
| samples | 215 ok / 54 failed |
| peak waiting | 504 |
| peak active | 8 |
| peak running_batch | 8 |
| peak prefill_queue | 8 |
| plan labels | `idle=16`, `decode=6085`, `prefill=86`, `split=60`, `mixed=0` |
| peak kv_util | 65.6% |

## Results - Request Accounting

Fixed c=1:

| metric | value |
|---|---:|
| completed input tokens | 36,873 |
| incomplete input tokens | 4,096 |
| request errors | 0 |
| completed output tokens | 2,304 |
| incomplete output tokens | 1 |

## Problems

- Canonical guidellm sweep is **not** a pass. The server terminated with:
  `thread '<unknown>' ... has overflowed its stack; fatal runtime error: stack overflow`.
  This matches the earlier Qwen3.5 sweep failure mode and prevents claiming a
  full sweep regression pass for this tranche.
- The fixed c=1 control is only a low-concurrency regression gate. It does not
  prove high-concurrency stability.
- The component win is operator-level evidence. Wall-clock request metrics at
  c=1 are dominated by prefill and full model decode, so request-level deltas
  should not be attributed to this micro-kernel.

## Learnings

- FP8 KV decode benefits from aligned vector loads/conversions: x4 loads cut
  the Qwen3.5-shaped component kernel by `17.67%`.
- INT8 and FP8 KV decode have different bottleneck profiles. Shared-memory
  page staging helps the INT8 implementation but regresses FP8 by `33.7%` on
  this shape.
- Keep component A/B and serving regression framing separate: the component
  A/B licenses the kernel change, while serving smoke/fixed-c guards gross
  runtime regressions.

## Delta vs Baseline

- **Component baseline:** same checkout before the
  `decode_attention_fp8_partial_kernel` x4-load patch.
- **Serving baseline:** no prior committed fixed-c Qwen3.5 FP8 decode snapshot
  for this exact operator change; use this entry as the baseline for future
  FP8 decode work.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| component latency median | 100.51 us | 82.755 us | -17.67% |
| component throughput median | 667.69 Gelem/s | 810.94 Gelem/s | +21.46% |
| fixed-c1 request errors | first run | 0 | n/a |
| fixed-c1 ITL p50 | first run | 15.24 ms | n/a |

## Artefacts

- Canonical failed raw dir: `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load/`
- Canonical failed service trace summary:
  `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load/service_stats_trace_summary.md`
- Fixed c=1 raw: `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load-c1/benchmarks.json`
- Fixed c=1 CSV: `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load-c1/benchmarks.csv`
- Fixed c=1 HTML: `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load-c1/benchmarks.html`
- Fixed c=1 service trace summary:
  `bench-output/2026-05-12-qwen35-fp8-kv-decode-x4-load-c1/service_stats_trace_summary.md`
- Fixed c=1 server log:
  `bench-output/server-logs/2026-05-12-qwen35-fp8-kv-decode-x4-load-c1-server.log`

## Notes

- Code changed: `decode_attention_fp8_partial_kernel` now loads K/V in aligned
  `__nv_fp8x4_e4m3` groups and converts each group to `float4`.
- Correctness smoke: `/v1/completions` on `"The cat sat on the mat."` returned
  non-empty coherent text and `completion_tokens=8`.
- Follow-up: fix or isolate the Qwen3.5 canonical sweep stack overflow before
  using high-concurrency sweep results as a stability gate for future FP8 KV
  decode work.
