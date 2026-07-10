# P3.1 DSv4 FP8 batch B1 persistent kernel deferred

## Context

Phase 3 P3.1 A6 evaluated whether launch reduction or a persistent-kernel
shape should be attempted for `dsv4_fp8_gemv_batch_kernel`, the B=1 raw path
behind `dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- The normal Criterion component bench reports about 8-10 us for the local B=1
  raw path.
- If the kernel is strongly launch-bound, replacing per-call kernel launch
  with a persistent worker could theoretically recover a large fraction of the
  per-call time.
- This is only licenseable if the measured launch component is both large and
  separable from the benchmark harness. A persistent worker is not a local
  source-level CUDA kernel tweak; it changes dispatch/lifetime semantics and
  needs a different correctness and lifecycle gate.

## Root Cause

A6 is not a clean local operator axis under the current bench harness. The
steady nsys run shows kernel launch API time is non-trivial, but the component
bench also synchronizes before and after every iteration. That means a
persistent-kernel treatment would not isolate launch removal from per-iteration
sync overhead unless the benchmark harness is changed at the same time. That
would violate the single-variable Phase 3 rule.

Source evidence:

- `infer/benches/ops/common/mod.rs` `iter_sync` calls `ctx.sync()` before each
  iteration and after each iteration.
- `infer/benches/ops/ops_cuda_bench.rs` calls `iter_sync` for
  `ops_cuda/dsv4_fp8_gemv_batch_b1`.

## Evidence

Initial `nsys --profile-time` attempt was rejected as evidence because it ran
Criterion test mode and captured only one kernel launch per shape. The steady
run used the compiled Criterion binary with explicit `--bench`:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_1_a6_dsv4_b1_hidden_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_fp8_gemv_batch_b1/dsv4_mini_hidden_1024x1024 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats command:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
  --format csv --force-export=true \
  /tmp/p3_1_a6_dsv4_b1_hidden_steady.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `83092` |
| Criterion under nsys point | `11.983 us` |
| `cudaLaunchKernel` avg / median | `3.3046 us` / `3.2200 us` |
| `cuStreamSynchronize` calls | `166234` |
| `cuStreamSynchronize` avg / median | `5.2563 us` / `3.1000 us` |
| Kernel avg / median | `7.4662 us` / `4.5760 us` |
| Kernel launch+queue+kernel avg / median | `12.4025 us` / `9.2770 us` |

The launch API is measurable, but the sync calls are part of the measurement
loop and are larger in aggregate. A persistent-kernel patch plus a harness
change would change at least two variables.

## Fix

No runtime patch was made. A6 is deferred until there is a request-level or
async component benchmark that can isolate launch reduction without mandatory
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

Do not implement a persistent kernel as a local P3.1 A6 tweak. First create an
async or request-level measurement that isolates launch removal from the
component bench's per-iteration synchronization.
