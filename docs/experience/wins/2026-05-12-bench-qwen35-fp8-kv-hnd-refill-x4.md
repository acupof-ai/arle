# Qwen3.5 FP8 KV HND refill x4 loads - 2026-05-12

## Goal

- Continue the Qwen3.5 FP8 KV-cache operator pass by optimizing the durable
  FP8 E4M3 NHD to BF16 HND refill kernel used before paged prefill consumes
  historical prefix rows.

## Hypothesis

- The existing refill kernel already uses pairwise FP8 loads and BF16x2 stores.
  For the live Qwen3.5 shape (`head_dim=256`, `kv_dim=1024`), source and
  destination offsets are 4-byte aligned. Loading four FP8 values per thread
  and writing two BF16x2 pairs should reduce per-row loop and launch-thread
  overhead without changing the FP8 format or HND/NHD layout.

## Command

Component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dequantize_paged_kv_fp8_to_hnd --quiet
```

Correctness:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p cuda-kernels --features cuda \
  kv_quant::tests::hnd_refill_quantized_kv_matches_reference_values -- --nocapture
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
scripts/bench_guidellm.sh qwen35-fp8-kv-hnd-refill-x4-c1 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 1 --max-seconds 60 --warmup 10
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3.5-4B`, served from `infer/models/Qwen3.5-4B`
- **Operator:** `dequantize_paged_kv_fp8_to_hnd_cuda`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit under test:** working tree based on `d4c8f3c`; this entry is
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
| total_tokens | 1024 |
| num_kv_heads | 4 |
| head_dim | 256 |
| kv_dim | 1024 |
| source layout | paged FP8 E4M3 NHD + f32 scales |
| destination layout | BF16 HND work buffer |

## Results - Component A/B

All rows use the same Qwen3.5-shaped Criterion bench. Only the FP8 HND refill
thread grouping and vector width changed.

| Candidate | Status | Criterion time | Throughput | Delta vs x2 baseline |
|---|---|---:|---:|---:|
| x2 pairwise baseline | pass | 9.0698-9.0844 us, point 9.0772 us | 115.52 Gelem/s | baseline |
| x4 FP8 load + 2x BF16x2 store | **kept** | 8.1906-8.1987 us, point 8.1951 us | 127.95 Gelem/s | **-9.72%** |
| x8 two-FP8x4 loads | kill | 8.2016-8.2160 us, point 8.2065 us | 127.77 Gelem/s | -9.59%, worse than x4 |

The first x4 pass measured `8.1506-8.1978 us`, point `8.1673 us`; the kept
final rerun above is the committed-worktree result used for the decision.

## Results - Correctness

The existing refill reference test passed after the x4 change and after
reverting the killed x8 candidate:

```text
test kv_quant::tests::hnd_refill_quantized_kv_matches_reference_values ... ok
```

That test covers both `head_dim=8` and `head_dim=7`, so the aligned x4 live
path and the odd/tail scalar fallback are both exercised.

## Results - Fixed c=1 Guidellm

This is an exploration-mode fixed-concurrency regression gate, not a canonical
sweep publication. The run completed with 0 request errors.

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 1962.7 | 4.7 | 1964.8 | 1966.4 | 22.84 | 15.24 | 0.04 | 15.24 | 15.29 | 15.29 | 15.29 | 5.85 | 5.87 | 1 | 45.46 | 772.97 | 788.16 | 36873 | 2304 | 0.18 |

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

- The component win is real but small. It must not be projected directly to
  wall-clock TTFT or ITL because full requests are dominated by model prefill,
  attention, and scheduling work outside this refill kernel.
- The fixed-c run is a low-concurrency regression gate only. It does not prove
  high-concurrency Qwen3.5 FP8 sweep stability.
- This change only applies to the FP8 refill kernel. The INT8 refill path stays
  at the prior pairwise implementation.

## Learnings

- For live Qwen3.5 FP8 refill, x4 is the measured stable point: it cuts the
  component kernel by `9.72%` over the pairwise baseline.
- x8 is not better on sm_89 for this shape. The extra per-thread work reduces
  parallelism enough to lose the tiny advantage over x4.
- Keep the scalar fallback: it is not on the live aligned path, but it is what
  makes odd `head_dim` and unaligned rows testable and safe.

## Delta vs Baseline

- **Component baseline:** same checkout before this x4 patch, after the prior
  pairwise refill optimization.
- **Serving baseline:** nearest fixed-c Qwen3.5 FP8 snapshots are
  `2026-05-12-bench-qwen35-fp8-kv-scatter-x4.md` and
  `2026-05-12-bench-guidellm-qwen35-fp8-kv-decode-x4-load.md`.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| component latency median | 9.0772 us | 8.1951 us | -9.72% |
| component throughput median | 115.52 Gelem/s | 127.95 Gelem/s | +10.76% |
| fixed-c1 TTFT p50 vs scatter snapshot | 1956.4 ms | 1964.8 ms | +0.43% |
| fixed-c1 ITL p50 vs scatter snapshot | 15.24 ms | 15.24 ms | 0.00% |
| fixed-c1 request errors | 0 | 0 | unchanged |

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-x4-c1/benchmarks.json`
- CSV: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-x4-c1/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-x4-c1/benchmarks.html`
- Service trace summary:
  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-x4-c1/service_stats_trace_summary.md`

## Notes

- Code changed: `dequantize_paged_kv_fp8_to_hnd_kernel` now processes four
  head-dim values per thread when source and destination are 4-byte aligned.
- The aligned path uses one `__nv_fp8x4_e4m3` read and two `__nv_bfloat162`
  stores. The fallback writes up to four scalar BF16 values for tails or
  unaligned rows.
- Smoke response for `"The capital of France is"` returned non-empty text and
  usage `prompt_tokens=5`, `completion_tokens=8`, `total_tokens=13`.
