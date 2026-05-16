# P3.2 DSv4 FP4 batch B1 persistent kernel deferred

## Context

Phase 3 P3.2 A6 evaluated whether launch reduction or a persistent-kernel
shape should be attempted for `dsv4_fp4_gemv_batch_kernel`, the B=1 raw path
behind `dsv4_fp4_gemv_batch_cuda`, after A2 pair-load landed.

## Formula Prediction

Hypothesis before measurement:

- The normal Criterion component bench reports about 8-10 us for the local B=1
  FP4 raw path after A2.
- If kernel launch is the dominant separable cost, a persistent worker could
  theoretically recover a large fraction of the per-call latency.
- This is only licenseable if launch cost is separable from the benchmark
  harness. A persistent worker changes dispatch and lifetime semantics, so it
  is not a local source-level CUDA kernel tweak.

## Root Cause

A6 is not a clean local operator axis under the current bench harness. The
steady nsys run shows launch API time is measurable, but the component bench
also synchronizes before and after every iteration. A persistent-kernel
treatment would need a harness or runtime-lifecycle change to isolate launch
removal from synchronization overhead, violating the single-variable rule.

Source evidence:

- `infer/benches/ops/common/mod.rs` `iter_sync` calls `ctx.sync()` before each
  iteration and after each iteration.
- `infer/benches/ops/ops_cuda_bench.rs` calls `iter_sync` for
  `ops_cuda/dsv4_fp4_gemv_batch_b1`.

## Evidence

Steady nsys command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_2_a6_dsv4_fp4_b1_hidden_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_fp4_gemv_batch_b1/dsv4_mini_hidden_1024x1024 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats command:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
  --format csv --force-export=true \
  /tmp/p3_2_a6_dsv4_fp4_b1_hidden_steady.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `87547` |
| Criterion under nsys point | `11.771 us` |
| `cudaLaunchKernel` avg / median | `3.3418 us` / `3.2500 us` |
| `cuStreamSynchronize` calls | `175144` |
| `cuStreamSynchronize` avg / median | `4.7538 us` / `3.2550 us` |
| Kernel avg / median | `6.6341 us` / `4.3530 us` |
| Kernel launch+queue+kernel avg / median | `11.5382 us` / `9.1040 us` |

Launch API is measurable, but the sync calls are part of the measurement loop
and larger in aggregate. Removing launch without changing the sync framing
cannot be licensed as a local operator axis.

## Fix

No runtime patch was made. A6 is deferred until there is an async or
request-level benchmark that can isolate launch reduction from mandatory
pre/post synchronization.

## Tradeoff

- LOC complexity: persistent worker would add non-local state, lifecycle, and
  shutdown semantics.
- SM89 specificity: nsys evidence is local to RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unknown until a concrete worker design exists.
- Register budget: unknown until a concrete worker design exists.
- CUDA Graph compatibility: persistent workers and graph capture need a
  separate lifecycle analysis.
- Generality across batch sizes: current evidence covers only B=1 hidden
  shape; B>1 tiled path was not touched.
- Numerical correctness margin: not evaluated because no treatment was run.

## Rule

Do not implement a persistent kernel as a local P3.2 A6 tweak. First create an
async or request-level measurement that isolates launch removal from the
component bench's per-iteration synchronization.
