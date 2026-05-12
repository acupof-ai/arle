# Qwen3.5 FP8 KV scatter x4 quantize - 2026-05-12

## Goal

- Optimize the live Qwen3.5 FP8 KV prefill/contiguous-to-paged write path
  without changing the FP8 E4M3 KV format, scale formula, or runtime dispatch.

## Hypothesis

- `quantize_scatter_kv_fp8_kernel` processes a long prefill migration row with
  one thread per head-dim element. For Qwen3.5 `head_dim=256`, grouping several
  dim values per thread may reduce per-token/head reduction overhead while
  retaining enough parallelism for the BF16->FP8 conversion work.

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
scripts/bench_guidellm.sh qwen35-fp8-kv-scatter-x4-c1 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1 --max-seconds 60 --warmup 10
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3.5-4B`, served from `infer/models/Qwen3.5-4B`
- **Operator:** `quantize_scatter_kv_fp8_range_cuda`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit under test:** working tree based on `48965fd`; this entry is
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

All rows use the same Qwen3.5-shaped component bench. Only the scatter kernel's
thread-to-dim grouping changed.

| Candidate | Status | Criterion time | Throughput | Delta vs baseline |
|---|---|---:|---:|---:|
| baseline 1 dim/thread | pass | 20.327-20.368 us, point 20.342 us | 103.10 Gelem/s | baseline |
| 2 dims/thread | pass | 14.288-14.298 us, point 14.292 us | 146.74 Gelem/s | -29.74% |
| 4 dims/thread | **kept** | 13.453-13.511 us, point 13.476 us | 155.62 Gelem/s | **-33.75%** |
| 8 dims/thread | kill | 19.403-19.416 us, point 19.409 us | 108.05 Gelem/s | -4.59% vs baseline, worse than x4 |

Final rerun on the kept worktree:

| metric | value |
|---|---:|
| time interval | 13.467-13.494 us |
| time point | 13.478 us |
| throughput interval | 155.41-155.73 Gelem/s |
| throughput point | 155.60 Gelem/s |
| delta vs baseline | -33.74% |

## Results - Fixed c=1 Guidellm

This is an exploration-mode fixed-concurrency regression gate, not a canonical
sweep publication. The run completed with 0 request errors.

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 1959.1 | 5.2 | 1956.4 | 1966.1 | 22.83 | 15.24 | 0.04 | 15.24 | 15.3 | 15.3 | 15.3 | 5.85 | 5.86 | 1 | 45.48 | 773.33 | 788.56 | 36873 | 2304 | 0.18 |

## Results - Service-Side Metrics

| metric | value |
|---|---:|
| samples | 72 ok / 0 failed |
| peak waiting | 0 |
| peak active | 1 |
| peak running_batch | 1 |
| peak prefill_queue | 1 |
| plan labels | `idle=13`, `decode=2565`, `prefill=35`, `split=0`, `mixed=0` |
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

- The fixed-c run is a low-concurrency regression gate only. It does not prove
  high-concurrency Qwen3.5 FP8 sweep stability.
- The component win is for the contiguous-to-paged FP8 scatter quantize kernel.
  Full request TTFT remains dominated by the full Qwen3.5 prefill path, so the
  `-33.74%` component delta must not be projected directly to wall-clock.
- The 8-dim grouping demonstrates the limit of this axis: reducing to one warp
  per token/head loses too much parallelism for the per-element FP8 conversion.

## Learnings

- For long prefill FP8 KV scatter on Qwen3.5 `head_dim=256`, four dim values
  per thread is the measured stable point on sm_89.
- This does not transfer to the single-token `quantize_paged_kv_fp8` path:
  that kernel's 2-dim grouping was noise and 4-dim grouping regressed in
  [`2026-05-12-fp8-kv-quantize-thread-grouping-kill.md`](../errors/2026-05-12-fp8-kv-quantize-thread-grouping-kill.md).
- Shape matters: prefill scatter has enough token rows for reduction-work
  savings to show up; single-token decode writes are dominated by fixed costs.

## Delta vs Baseline

- **Component baseline:** same checkout before the `quantize_scatter_kv_fp8`
  x4 grouping patch.
- **Serving baseline:** use
  [`2026-05-12-bench-guidellm-qwen35-fp8-kv-decode-x4-load.md`](2026-05-12-bench-guidellm-qwen35-fp8-kv-decode-x4-load.md)
  as the nearest fixed-c1 Qwen3.5 FP8 serving snapshot.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| component latency median | 20.342 us | 13.478 us | -33.74% |
| component throughput median | 103.10 Gelem/s | 155.60 Gelem/s | +50.92% |
| fixed-c1 TTFT p50 | 1965.2 ms | 1956.4 ms | -0.45% |
| fixed-c1 ITL p50 | 15.24 ms | 15.24 ms | 0.00% |
| fixed-c1 request errors | 0 | 0 | unchanged |

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-x4-c1/benchmarks.json`
- CSV: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-x4-c1/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-x4-c1/benchmarks.html`
- Service trace summary:
  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-scatter-x4-c1/service_stats_trace_summary.md`
- Smoke response: `/tmp/arle-fp8-scatter-smoke.json`

## Notes

- Code changed: `quantize_scatter_kv_fp8_kernel` now handles four head-dim
  values per thread and launches `ceil(head_dim / 4)` rounded to a full warp.
- Correctness guard: `fp8_scatter_quantized_kv_roundtrips_representable_values`
  covers even and odd head dims (`8`, `7`) to catch pair/group tail mistakes.
- The single-token `quantize_paged_kv_fp8_kernel` was intentionally left
  unchanged after its thread-grouping A/B failed.
