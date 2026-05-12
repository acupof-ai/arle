# Qwen3.5 FP8 KV HND refill pairwise stores - 2026-05-12

## Goal

- Regression-gate the FP8 KV HND refill optimization on Qwen3.5 FP8-KV and record the controlled component A/B for the quantized KV refill operator.

## Hypothesis

- Replacing per-element BF16 refill stores with aligned pairwise FP8/INT8 loads and BF16x2 stores should reduce refill-kernel latency for the live Qwen3.5 `head_dim=256` shape, while preserving runtime stability under the canonical 4096-in/256-out guidellm sweep.

## Command

```bash
PATH=$PWD/.venv/bin:$PATH scripts/bench_guidellm.sh \
  qwen35-fp8-kv-hnd-refill-pairwise-final \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B
```

Component A/B command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dequantize_paged_kv_fp8_to_hnd --quiet
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-4B
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM, driver 595.71.05, CUDA 13.2.78
- **Commit:** working tree based on `9150f36`; this entry is committed with the code delta
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`, `TORCH_CUDA_ARCH_LIST=8.9`
- **Server launch:** `./target/release/infer --model-path infer/models/Qwen3.5-4B --port 8000 --num-slots 8 --max-seq-len 5120 --kv-cache-dtype fp8 --mem-fraction-static 0.85`
- **Scheduling envelope:** `max_num_batched_tokens=16384`, `chunked_prefill_size=2048`, `max_prefill_tokens=16384`, `mem_fraction_static=0.85`, `max_slots=8`
- **KV pool:** `81552` max tokens, `5097` pages, `page_size=16`, `4 kv_heads x 256 head_dim`, `kv_dim=1024`, `format=FP8E4M3`

## Canonical params (resolved by wrapper)

- `--profile sweep`
- `--data prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Workload: `default`
- Wrapper: `scripts/bench_guidellm.sh qwen35-fp8-kv-hnd-refill-pairwise-final`

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sync | 1983.4 | 72.9 | 1955.7 | 2201.5 | 22.93 | 15.24 | 0.04 | 15.23 | 15.31 | 15.31 | 15.31 | 5.87 | 6.08 | 1 | 45.3 | 770.33 | 778.82 | 40970 | 2560 | 0.167 |
| throughput | 20498.9 | 6405.3 | 15765.5 | 32785.9 | 112.56 | 32.61 | 14.58 | 26.69 | 71.87 | 71.87 | 71.87 | 28.82 | 51.11 | 9 | 86.41 | 1469.27 | 2173.44 | 69649 | 4352 | 0.267 |
| 0.17916647659516105r/s | 2047.9 | 85.7 | 2022.3 | 2304.5 | 30.97 | 23.06 | 0.05 | 23.04 | 23.15 | 23.15 | 23.15 | 7.93 | 8.16 | 1 | 45.8 | 778.86 | 819.62 | 40970 | 2560 | 0.167 |
| 0.19166628652365542r/s | 2034.5 | 47.5 | 2020.1 | 2184.2 | 31.2 | 23.35 | 0.13 | 23.31 | 23.62 | 23.62 | 23.62 | 7.99 | 8.09 | 1 | 48.57 | 825.88 | 866.12 | 45067 | 2816 | 0.167 |
| 0.2041660964521498r/s | 2031.3 | 39.6 | 2020.4 | 2153.9 | 31.34 | 23.5 | 0.16 | 23.47 | 23.79 | 23.79 | 23.79 | 8.02 | 8.1 | 2 | 51.25 | 871.43 | 921.95 | 45067 | 2816 | 0.183 |
| 0.21666590638064417r/s | 2029.2 | 37.1 | 2017.3 | 2150.7 | 31.51 | 23.68 | 0.17 | 23.66 | 23.97 | 23.97 | 23.97 | 8.07 | 8.13 | 2 | 54.19 | 921.41 | 970.6 | 49164 | 3072 | 0.2 |
| 0.22916571630913854r/s | 2034.5 | 44.5 | 2021.9 | 2187.1 | 31.66 | 23.8 | 0.17 | 23.79 | 24.12 | 24.12 | 24.12 | 8.11 | 8.18 | 2 | 57.09 | 970.8 | 1019.93 | 53261 | 3328 | 0.2 |
| 0.24166552623763293r/s | 2032.6 | 40.7 | 2025.9 | 2176.9 | 31.75 | 23.9 | 0.19 | 23.9 | 24.24 | 24.24 | 24.24 | 8.13 | 8.21 | 2 | 59.96 | 1019.51 | 1068.64 | 57358 | 3584 | 0.217 |
| 0.2541653361661273r/s | 2030.8 | 30.6 | 2021.4 | 2133.9 | 39.51 | 31.7 | 0.23 | 31.66 | 31.98 | 31.98 | 31.98 | 10.12 | 10.18 | 2 | 60.3 | 1025.29 | 1130.25 | 53261 | 3328 | 0.217 |
| 0.26666514609462166r/s | 2033.1 | 35.8 | 2027.8 | 2160.3 | 39.6 | 31.78 | 0.22 | 31.76 | 32.05 | 32.05 | 32.05 | 10.14 | 10.2 | 3 | 63.2 | 1074.57 | 1179.87 | 57358 | 3584 | 0.233 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `666` (ok: `666`, failed: `0`)
- Peak waiting: `504`
- Peak active: `8`
- Peak running_batch: `8`
- Peak prefill_queue: `8`
- Plan labels: `idle=23`, `decode=20055`, `prefill=196`, `split=241`, `mixed=0`
- Peak kv_util: `85.6%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/666`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 504 | 504 |
| kv_util | 48.3% | 62.7% | 70.5% | 83.1% | 85.6% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 1 | 1 | 2 | 8 | 8 |
| prefill_tokens | 0 | 0 | 2048 | 2048 | 2048 |
| tokens_out | 9552 | 17096 | 25331 | 34592 | 35171 |

## Results — component A/B

Shape: `num_kv_heads=4`, `head_dim=256`, `total_tokens=1024`, `kv_dim=1024`, HND output, FP8E4M3 input bytes, per-token/head FP32 scales.

| metric | baseline scalar refill | pairwise final | delta |
|---|---:|---:|---:|
| latency median | 10.173 us | 9.0893 us | -10.65% |
| throughput median | 103.07 Gelem/s | 115.36 Gelem/s | +11.92% |

The final kernel keeps an aligned vector path for live even-head-dim shapes and falls back to a two-element scalar path when `src_offset` or `dst_offset` is not pair-aligned.


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 8 |
| peak waiting | 504 |
| peak prefill_queue | 8 |
| peak running_batch | 8 |
| peak kv_util | 85.6% |
| `prefix_hit_rate` | 0.0% peak / 0.0% q75 |
| `prefix_skip_rate` | 0.0% peak |
| `kv_fetch_q` | 0/0 samples >0 |
| `kv_fetch_waiters` | 0/666 samples >0 |
| `kv_store_q` | 0/0 samples >0 |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` |
| `kv_bp` | `fetch:0,store:0` |
| tier wait peaks | fetch n/a, store n/a |

## Results — request accounting

| metric | value |
|---|---:|
| successful requests | 125 |
| incomplete requests | 524 |
| errored requests | 0 |
| completed input tokens | 512125 |
| incomplete input tokens | 2146304 |
| completed output tokens | 32000 |
| incomplete output tokens | 3131 |

## Problems

- The throughput arm is intentionally oversaturated: `waiting` peaks at `504`, `incomplete requests=524`, and the wrapper warns `ITL p99/p50 = 2.69`. This is a saturation/regression gate, not a latency target.
- This does not prove broad FP8-KV numerical correctness. Prior Qwen3 FP8-KV tier-1 numerical drift remains a separate blocker; this tranche only validates the HND refill ABI/value mapping with a direct CUDA unit test and the Qwen3.5 serving smoke/regression run.
- FP4 KV is not a live ARLE CUDA KV-cache mode (`fp4` is still rejected by the CLI). It remains design-gated by the W4/FP4 KV plan and was not implemented here.

## Learnings

- Pairwise refill is a real component win for live Qwen3.5 FP8-KV refill: `-10.65%` kernel latency after the final alignment-safe fallback.
- The safe generic form needs two scalar writes when a pair is not aligned; a one-element scalar fallback would silently skip `d + 1` for odd `head_dim` rows.
- Use component A/B for attribution and guidellm for regression. The full sweep is dominated by prefill/decode scheduling, so it should not be used to attribute the operator-level speedup.

## Δ vs baseline

- **Baseline:** no prior canonical guidellm snapshot for this exact label; component baseline was measured immediately before the kernel change with the same Criterion bench.

| metric | baseline | now | Δ% |
|---|---|---|---|
| FP8 HND refill latency median | 10.173 us | 9.0893 us | -10.65% |
| FP8 HND refill throughput median | 103.07 Gelem/s | 115.36 Gelem/s | +11.92% |
| canonical sync TTFT p50 | first run | 1955.7 ms | n/a |
| canonical throughput out tok/s | first run | 86.41 | n/a |

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/benchmarks.json`
- CSV:  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/benchmarks.html`
- Service trace (before): `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_before.txt`
- Service trace (during): `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_trace.jsonl`
- Service trace (after):  `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_after.txt`
- Service trace (summary): `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_trace_summary.md`

## Notes

- Code changed since baseline: `dequantize_paged_kv_{fp8,int8}_to_hnd` now processes two head-dim elements per thread when pair-aligned and stores with `__nv_bfloat162`; odd/unaligned pairs use a scalar two-write fallback.
- Correctness guard: `cargo test --release -p cuda-kernels --features cuda kv_quant::tests::hnd_refill_quantized_kv_matches_reference_values -- --nocapture` covers both `head_dim=8` and `head_dim=7`.
- Smoke: `/v1/completions` with prompt `"The capital of France is"` returned non-empty text and usage `prompt_tokens=5`, `completion_tokens=8`, `total_tokens=13`.
- Follow-ups: FP4 KV should stay out of runtime until the `M_quant-kv-w4a8` phase-0 smoke licenses an actual KV format and accuracy envelope.

## Service Trace

- Poll interval: `1000ms`
- Before: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_before.txt`
- During: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_trace.jsonl`
- After: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_after.txt`
- Summary: `/home/ckl/projects/arle/bench-output/2026-05-12-qwen35-fp8-kv-hnd-refill-pairwise-final/service_stats_trace_summary.md`
