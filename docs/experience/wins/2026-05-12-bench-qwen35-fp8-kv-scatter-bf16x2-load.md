# Qwen3.5 FP8 KV scatter bf16x2 source loads - 2026-05-12

## Goal

- Continue the Qwen3.5 FP8 KV operator pass by testing whether the live
  contiguous-to-paged FP8 scatter quantization kernel benefits from vectorized
  BF16 source loads after the prior x4 thread grouping and packed FP8 store
  wins.

## Hypothesis

- `quantize_scatter_kv_fp8_kernel` already processes four head-dim values per
  thread and writes aligned FP8 x4 stores on the Qwen3.5 shape. The source side
  still used four scalar BF16 loads. For `head_dim=256`, `kv_dim=1024`, and
  `d=threadIdx.x * 4`, the source offsets are aligned, so two
  `__nv_bfloat162` loads should slightly reduce load/convert overhead without
  changing the reduction, scale formula, destination layout, or scalar fallback.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/quantize_scatter_kv_fp8_qwen35 --quiet
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  kv_quant::tests::fp8_scatter_quantized_kv_roundtrips_representable_values -- --nocapture
```

Fixed-c serving regression:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo build --release -p infer --features cuda --bin infer

./target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 5120 \
  --kv-cache-dtype fp8

PATH=$PWD/.venv/bin:$PATH \
scripts/bench_guidellm.sh qwen35-fp8-kv-scatter-bf16x2-load-c1 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1 --max-seconds 60 --warmup 10
```

## Environment

- Backend: CUDA
- Model: `Qwen/Qwen3.5-4B`, served from `infer/models/Qwen3.5-4B`
- Operator: `quantize_scatter_kv_fp8_range_cuda`
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, SM89, 16376 MiB VRAM
- Driver / CUDA: 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- Feature set: `cargo build --release -p infer --features cuda --bin infer`
- Non-default flags / env vars: `CUDARC_CUDA_VERSION=13010`,
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`
- Server launch: command above with `--kv-cache-dtype fp8`
- Scheduling envelope: `max_num_batched_tokens=16384`,
  `chunked_prefill_size=2048`, `max_prefill_tokens=16384`,
  `mem_fraction_static=0.85`, `max_slots=8`
- KV pool: `81552` max tokens, `5097` pages, `page_size=16`,
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

Only the BF16 source load path changed. The reduction, scale formula
`max(abs(row)) / 448.0`, destination layout, FP8 packed store, and scalar
fallback are unchanged.

| Arm | Criterion time | Throughput | Delta vs packed-store baseline |
|---|---:|---:|---:|
| current packed-store baseline | `11.725-11.746 us`, point `11.733 us` | point `178.74 Gelem/s` | baseline |
| bf16x2 source loads first run | `11.704-11.709 us`, point `11.706 us` | point `179.15 Gelem/s` | `-0.23%` |
| bf16x2 source loads final rerun | `11.706-11.711 us`, point `11.708 us` | point `179.12 Gelem/s` | `-0.21%` |

## Results - Correctness

```text
test kv_quant::tests::fp8_scatter_quantized_kv_roundtrips_representable_values ... ok
```

The test covers both even and odd head dims, so the aligned bf16x2 path and the
scalar fallback remain exercised.

## Results - Fixed c=1 Guidellm

This is an exploration-mode fixed-concurrency regression gate, not a canonical
sweep publication. The run completed with 0 request errors.

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 1959.8 | 5.3 | 1961.4 | 1968.4 | 22.84 | 15.24 | 0.04 | 15.24 | 15.3 | 15.3 | 15.3 | 5.85 | 5.87 | 1 | 45.47 | 773.22 | 788.47 | 36873 | 2304 | 0.18 |

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

- The component win is real but tiny. Treat this as a local kernel cleanup with
  measured positive direction, not as a request-level performance win.
- The fixed-c run is only a low-concurrency regression gate. It does not prove
  high-concurrency Qwen3.5 FP8 sweep stability.
- Previous FP8 KV decisions still stand: this does not reopen the single-token
  `quantize_paged_kv_fp8` grouping kill or the K/V fusion no-license decision.

## Learnings

- After x4 grouping and packed FP8 stores, source-side BF16 vector loads still
  save about `0.02-0.03 us` on the Qwen3.5-shaped scatter bench.
- This is the end of the obvious scatter-load/store width axis for sm_89:
  larger x8 grouping was already killed, and the remaining win here is below
  1% at component level.
- Keep the scalar fallback. It protects odd or unaligned future shapes while
  the aligned Qwen3.5 path uses the faster bf16x2 loads.

## Delta vs Baseline

| metric | baseline | now | delta |
|---|---:|---:|---:|
| component latency median | 11.733 us | 11.708 us | -0.21% |
| component throughput median | 178.74 Gelem/s | 179.12 Gelem/s | +0.21% |
| fixed-c1 TTFT p50 vs packed-store snapshot | 1964.7 ms | 1961.4 ms | -0.17% |
| fixed-c1 ITL p50 vs packed-store snapshot | 15.24 ms | 15.24 ms | 0.00% |
| fixed-c1 request errors | 0 | 0 | unchanged |

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-bf16x2-load-c1/benchmarks.json`
- CSV: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-bf16x2-load-c1/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-bf16x2-load-c1/benchmarks.html`
- Service trace summary:
  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-bf16x2-load-c1/service_stats_trace_summary.md`
