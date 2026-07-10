# DSv4 FP4 Grouped GEMV Pair-Load Win

## Context

Phase 3 P3.9 optimized `dsv4_fp4_grouped_gemv_batch_kernel` in
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`.

The kernel is used by the DSv4 grouped expert path through
`dsv4_run_grouped_block_scaled_gemv`. Trace artifacts from the DSv4 DeepEP
decode runs already show it as a hot kernel, but before `5536430` there was no
single-output grouped FP4 microbench, only the grouped pair benchmark.

## What Worked

The treatment changes the grouped FP4 kernel inner loop from one nibble per
iteration to one packed byte per iteration:

- load one packed FP4 byte
- decode low and high nibbles
- load both adjacent BF16 activations
- accumulate both products before the next loop iteration

Launch shape, ABI, expert pointer layout, route offsets, counts, and output
layout are unchanged.

## Evidence

Setup baseline added in `5536430`:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_grouped_gemv/ --save-baseline p3_9_setup_baseline
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_fp4_grouped_gemv/ --baseline p3_9_setup_baseline
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_t4_e4_512x1024` | `16.264 us` | `13.189 us` | `-18.912%` | `0.00` | LICENSE |
| `dsv4_mini_t64_e4_512x1024` | `150.18 us` | `104.71 us` | `-30.211%` | `0.00` | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_fp4_grouped_gemv -- --nocapture
```

Result:
`test_dsv4_fp4_grouped_gemv ... ok` and
`test_dsv4_fp4_grouped_gemv_pair ... ok`.

Compile gate:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check -p infer --features cuda
```

Result: PASS with pre-existing DSv4 warnings.

## Tradeoffs

- License strength: both local SM89 grouped FP4 shapes cross the 3% gate by a
  wide margin.
- Scope: this changes only the single-output grouped FP4 kernel. The grouped
  FP4 pair kernel already used pair-loads and is unchanged except for shared
  test helper reuse.
- Numerical behavior: accumulation order now pairs adjacent FP4 nibbles in one
  loop iteration. The new direct grouped test compares against a CPU FP4
  decode/dot reference.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.

## Rule

For `dsv4_fp4_grouped_gemv_batch_kernel`, process packed FP4 bytes as pairs.
Do not infer this from grouped pair behavior; keep the single-output grouped
microbench as its own gate.
