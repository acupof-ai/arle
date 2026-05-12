# Qwen3.5 FP8 KV scatter packed x4 store - 2026-05-12

## Goal

- Continue the Qwen3.5 FP8 KV operator pass by optimizing the live
  contiguous-to-paged FP8 scatter quantization kernel without changing the
  FP8 E4M3 format, per-token/head scale semantics, or runtime dispatch.

## Hypothesis

- After the earlier scatter x4 win, each thread already owns four head-dim
  values but still emits four scalar `__nv_fp8_e4m3` stores. For the live
  Qwen3.5 shape (`head_dim=256`, `kv_dim=1024`), destination offsets are
  4-byte aligned. Packing the four scaled values into one `__nv_fp8x4_e4m3`
  store should reduce store/conversion overhead while keeping the existing
  scalar fallback for odd/tail shapes.

## Command

Component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_scatter_kv_fp8_qwen35 --quiet
```

Correctness:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  kv_quant::tests::fp8_scatter_quantized_kv_roundtrips_representable_values -- --nocapture
```

Fixed-c serving regression:

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

PATH=$PWD/.venv/bin:$PATH \
scripts/bench_guidellm.sh qwen35-fp8-kv-scatter-packed-x4-c1 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1 --max-seconds 60 --warmup 10
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3.5-4B`, served from `infer/models/Qwen3.5-4B`
- **Operator:** `quantize_scatter_kv_fp8_range_cuda`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, SM89, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
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

## Params

| Param | Value |
|---|---:|
| token_count | 2048 |
| max_seq_len | 4096 |
| num_kv_heads | 4 |
| head_dim | 256 |
| kv_dim | 1024 |
| source layout | contiguous BF16 HND |
| destination layout | paged FP8 E4M3 NHD + f32 scales |

## Results - Component A/B

Only the final store path changed. The reduction, scale formula
`max(abs(row)) / 448.0`, source/destination layout, and scalar fallback are
unchanged.

| Arm | Criterion time | Throughput | Delta vs scalar-store x4 |
|---|---:|---:|---:|
| x4 scalar-store baseline | `13.452-13.498 us`, point `13.481 us` | point `155.56 Gelem/s` | baseline |
| packed x4 store first run | `11.713-11.758 us`, point `11.732 us` | point `178.76 Gelem/s` | `-12.97%` |
| packed x4 store final rerun | `11.718-11.734 us`, point `11.727 us` | point `178.83 Gelem/s` | `-13.01%` |

## Results - Correctness

```text
test kv_quant::tests::fp8_scatter_quantized_kv_roundtrips_representable_values ... ok
```

The existing test covers even and odd head dims, so the aligned packed path and
the scalar tail fallback both remain exercised.

## Results - Fixed c=1 Guidellm

This is an exploration-mode fixed-concurrency regression gate, not a canonical
sweep publication. The run completed with 0 request errors.

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 1963.1 | 4.2 | 1964.7 | 1965.6 | 22.84 | 15.24 | 0.04 | 15.24 | 15.29 | 15.29 | 15.29 | 5.85 | 5.87 | 1 | 45.45 | 772.87 | 788.05 | 36873 | 2304 | 0.18 |

## Results - Service-Side Metrics

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

## Results - Request Accounting

| metric | value |
|---|---:|
| completed input tokens | 36,873 |
| incomplete input tokens | 4,096 |
| request errors | 0 |
| completed output tokens | 2,304 |
| incomplete output tokens | 1 |

## Problems

- The fixed-c run is only a low-concurrency regression gate. It does not prove
  high-concurrency Qwen3.5 FP8 sweep stability.
- The component win is real and isolated, but full request TTFT remains
  dominated by the broader Qwen3.5 prefill path. Do not project the `-13.01%`
  scatter-kernel delta to request-level latency.
- Prior evidence still stands: this packed-store result does not reopen the
  single-token `quantize_paged_kv_fp8` grouping kill or the K/V fusion
  no-license decision.

## Learnings

- For Qwen3.5 long-prefill FP8 KV scatter, the best measured point is now:
  four dim values per thread plus one aligned `__nv_fp8x4_e4m3` store.
- The tail scalar fallback is still needed for odd `head_dim` tests and for
  future non-Qwen3.5 shapes.
- Store width and thread grouping are separate variables. The earlier x4
  grouping win licensed this follow-up; this entry only licenses packed stores
  inside that already-kept x4 grouping.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| component latency median | 13.481 us | 11.727 us | -13.01% |
| component throughput median | 155.56 Gelem/s | 178.83 Gelem/s | +14.96% |
| fixed-c1 TTFT p50 vs scatter x4 snapshot | 1956.4 ms | 1964.7 ms | +0.42% |
| fixed-c1 ITL p50 vs scatter x4 snapshot | 15.24 ms | 15.24 ms | 0.00% |
| fixed-c1 request errors | 0 | 0 | unchanged |

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-packed-x4-c1/benchmarks.json`
- CSV: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-packed-x4-c1/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-packed-x4-c1/benchmarks.html`
- Service trace summary:
  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-packed-x4-c1/service_stats_trace_summary.md`

