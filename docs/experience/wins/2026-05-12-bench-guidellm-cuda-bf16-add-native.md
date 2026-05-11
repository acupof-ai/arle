# CUDA BF16 Add Native Intrinsic — guidellm sweep, cuda-bf16-add-native, 2026-05-12

## Goal

Optimization + regression gate for replacing the CUDA BF16 elementwise add
vector path with native BF16 add intrinsics.

## Hypothesis

Using `__hadd2_rn` for aligned BF16x4 lanes and `__hadd_rn` for scalar tail
should reduce instruction count in `add_cuda` while preserving BF16
round-to-nearest output. Full Qwen3-4B serving should remain within normal
sweep variance because elementwise add is a small helper kernel, not the
dominant decode or prefill cost.

## Command

Server:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 5120
```

Correctness smoke:

```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","prompt":"The cat sat on the mat.","max_tokens":8,"temperature":0}'
```

Benchmark:

```bash
PATH=$PWD/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-bf16-add-native \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Component A/B:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- ops_cuda/add_batch --quiet
```

## Environment

- **Backend:** CUDA
- **Model:** `infer/models/Qwen3-4B`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit under test:** working tree based on `36e3113`
- **Feature set:** `cargo bench -p infer --features cuda`; server binary from `target/release/infer`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`
- **Scheduling envelope:** `max_num_batched_tokens=16384 | 16384,
  chunked_prefill_size=2048 | 2048, max_prefill_tokens=16384 | 16384,
  mem_fraction_static=0.85 | 0.85, max_slots=8 | (n/a -- SGLang has no fixed cap)`

## Results — Component A/B

`ops_cuda/add_batch/64` is the isolated kernel-level evidence.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Criterion point estimate | 11.532 us | 9.4085 us | -18.41% |
| Criterion interval | 11.069-12.218 us | 9.3899-9.4456 us | improved |
| Throughput point estimate | 5.6831 Gelem/s | 6.9656 Gelem/s | +22.57% |

Correctness:

| Check | Result |
|---|---|
| `cargo test --release -p infer --features cuda ops::tests::test_add_batch_tail -- --nocapture` | PASS |
| HTTP smoke prompt `"The cat sat on the mat."` | PASS, coherent text: `" The mat was red. The cat was"` |

## Results — Guidellm Sweep

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync | 549.4 | 87.2 | 525.8 | 860.1 | 17.65 | 15.57 | 0 | 15.57 | 15.57 | 15.57 | 15.57 | 4.52 | 4.83 | 1 | 57.42 | 976.4 | 981.4 | 57358 | 3584 | 0.217 |
| throughput | 15990.9 | 13667.8 | 13794.4 | 45016.2 | 86.02 | 23.65 | 0.04 | 23.64 | 23.65 | 23.91 | 23.91 | 22.02 | 51.05 | 24 | 220.73 | 3753.25 | 3961.58 | 196656 | 12288 | 0.800 |
| 0.28958333333333336r/s | 581.8 | 29.6 | 574.8 | 697.1 | 20.65 | 18.45 | 0.14 | 18.44 | 18.63 | 18.63 | 18.63 | 5.29 | 5.33 | 1 | 72.70 | 1236.26 | 1262.88 | 69649 | 4352 | 0.267 |
| 0.36250000000000004r/s | 587.8 | 55.5 | 575.9 | 834.2 | 21.28 | 19.06 | 0.19 | 19.06 | 19.25 | 19.27 | 19.27 | 5.45 | 5.50 | 2 | 89.92 | 1528.98 | 1566.36 | 86037 | 5376 | 0.333 |
| 0.43541666666666673r/s | 589.9 | 31.5 | 585.7 | 738.0 | 23.91 | 21.70 | 0.25 | 21.69 | 21.85 | 21.85 | 21.85 | 6.12 | 6.16 | 3 | 105.46 | 1793.27 | 1866.30 | 98328 | 6144 | 0.400 |
| 0.5083333333333334r/s | 591.7 | 22.8 | 586.6 | 705.9 | 26.55 | 24.34 | 0.34 | 24.48 | 24.51 | 24.51 | 24.51 | 6.80 | 6.85 | 3 | 121.06 | 2058.45 | 2164.31 | 114716 | 7168 | 0.467 |
| 0.58125r/s | 595.2 | 41.6 | 587.3 | 820.6 | 29.13 | 26.91 | 0.85 | 27.18 | 27.19 | 27.21 | 27.21 | 7.46 | 7.53 | 4 | 136.09 | 2314.12 | 2471.76 | 127007 | 7936 | 0.517 |
| 0.6541666666666668r/s | 597.7 | 23.2 | 594.7 | 724.4 | 32.02 | 29.80 | 1.13 | 30.23 | 30.24 | 30.25 | 30.25 | 8.20 | 8.32 | 5 | 149.97 | 2550.12 | 2768.08 | 139298 | 8704 | 0.567 |
| 0.7270833333333334r/s | 598.4 | 19.4 | 598.2 | 702.4 | 35.64 | 33.43 | 1.34 | 33.94 | 33.95 | 33.95 | 33.95 | 9.12 | 9.27 | 6 | 163.66 | 2782.84 | 3065.71 | 155686 | 9728 | 0.617 |
| 0.8000000000000002r/s | 700.6 | 102.5 | 680.4 | 887.0 | 39.33 | 36.74 | 1.39 | 37.26 | 37.27 | 37.27 | 37.27 | 10.07 | 10.39 | 8 | 175.41 | 2982.68 | 3353.15 | 163880 | 10240 | 0.667 |

## Results — Service-Side Metrics

| metric | value |
|---|---:|
| service trace samples | 646 ok / 0 failed |
| peak active | 8 |
| peak waiting | 504 |
| peak running_batch | 8 |
| peak prefill_queue | 7 |
| plan labels | `idle=44509`, `decode=23332`, `prefill=347`, `split=514`, `mixed=0` |
| peak kv_util | 100.0% |
| prefix hit rate | 0.0% |
| prefix skip rate | 0.0% |
| kv fetch waiters | 0/646 |
| kv store queue samples | 0/0 |

## Results — Request Accounting

| metric | value |
|---|---:|
| completed input tokens | 1,208,615 |
| incomplete input tokens | 2,232,320 |
| completed output tokens | 75,520 |
| incomplete output tokens | 4,033 |
| request errors | 0 |

## Problems

- The guidellm sweep intentionally drove saturation. The throughput and high
  constant-rate arms reached `peak waiting=504` and `peak kv_util=100%`, so
  they are regression/saturation evidence, not attribution evidence for the
  elementwise add kernel.
- The run was executed before committing the diff, so the commit field records
  the base SHA plus working-tree delta. The code under benchmark is exactly
  the diff in this tranche.

## Learnings

- For BF16 elementwise add, native BF16 intrinsics are materially faster than
  manually widening every lane to FP32 and converting back, while preserving
  the existing BF16 round-to-nearest contract.
- Component microbench framing is the right ground truth for this change.
  Full-request guidellm is still required as a regression gate, but the
  wall-clock service sweep is dominated by model GEMMs, attention, and KV
  pressure rather than a helper add kernel.

## Delta vs Baseline

- **Component baseline:** same checkout before the `elementwise_basic.cu`
  patch.
- **Guidellm baseline:** first canonical sweep for this exact helper-kernel
  change; do not compare to W4/W4A8 entries.

| metric | baseline | now | Delta |
|---|---:|---:|---:|
| `ops_cuda/add_batch/64` point estimate | 11.532 us | 9.4085 us | -18.41% |
| `ops_cuda/add_batch/64` throughput | 5.6831 Gelem/s | 6.9656 Gelem/s | +22.57% |
| guidellm sync TTFT p50 | first run | 525.8 ms | n/a |
| guidellm sync ITL p50 | first run | 15.57 ms | n/a |
| guidellm saturation out tok/s | first run | 220.73 | n/a |

## Artefacts

- Raw: `bench-output/2026-05-12-cuda-bf16-add-native/benchmarks.json`
- CSV: `bench-output/2026-05-12-cuda-bf16-add-native/benchmarks.csv`
- HTML: `bench-output/2026-05-12-cuda-bf16-add-native/benchmarks.html`
- Headline: `bench-output/2026-05-12-cuda-bf16-add-native/headline_table.md`
- Service trace before: `bench-output/2026-05-12-cuda-bf16-add-native/service_stats_before.txt`
- Service trace during: `bench-output/2026-05-12-cuda-bf16-add-native/service_stats_trace.jsonl`
- Service trace after: `bench-output/2026-05-12-cuda-bf16-add-native/service_stats_after.txt`
- Service trace summary: `bench-output/2026-05-12-cuda-bf16-add-native/service_stats_trace_summary.md`

## Notes

- Changed code: `crates/cuda-kernels/csrc/misc/elementwise_basic.cu`
- Follow-up: if future ncu shows add still visible in decode/prefill traces,
  inspect generated SASS for `add.rn.bf16x2` versus compiler expansion.
