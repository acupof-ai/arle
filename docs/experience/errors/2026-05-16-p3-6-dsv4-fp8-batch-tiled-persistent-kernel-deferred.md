# P3.6 DSv4 FP8 batch tiled persistent kernel deferred

## Context

Phase 3 P3.6 A6 evaluated whether launch reduction or a persistent-kernel
shape should be attempted for `dsv4_fp8_gemv_batch_tiled_kernel`, the B>1 path
behind `dsv4_fp8_gemv_batch_cuda`.

## Formula Prediction

Hypothesis before trace:

- After A1, the local B=4 MoE tiled path is about 15-16 us in normal Criterion.
- If launch API time is the dominant separable cost, a persistent worker could
  theoretically recover a meaningful fraction of per-call latency.
- This is only licenseable if launch cost is both large and separable from the
  benchmark harness. Persistent workers change dispatch, lifetime, and
  shutdown semantics.

## Root Cause

A6 is not licensed as a local operator patch. nsys shows launch API time is
measurable, but kernel body time is still the larger median component. The
component bench also synchronizes before and after every iteration, so a
persistent-kernel treatment would change launch behavior and measurement
framing together.

Source evidence:

- `infer/benches/ops/common/mod.rs` `iter_sync` calls `ctx.sync()` before each
  iteration and after each iteration.
- `infer/benches/ops/ops_cuda_bench.rs` calls `iter_sync` for
  `ops_cuda/dsv4_fp8_gemv_batch`.

## Evidence

The bench binary was rebuilt after reverting launch-shape experiments:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda --no-run
```

Steady nsys command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_6_a6_dsv4_fp8_tiled_moe_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_fp8_gemv_batch/dsv4_mini_moe_512x1024 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats commands:

```bash
nsys stats --force-export=true --report cuda_api_sum --format csv \
  /tmp/p3_6_a6_dsv4_fp8_tiled_moe_steady.nsys-rep
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  /tmp/p3_6_a6_dsv4_fp8_tiled_moe_steady.nsys-rep
nsys stats --force-export=true --report cuda_kern_exec_sum --format csv \
  /tmp/p3_6_a6_dsv4_fp8_tiled_moe_steady.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `74622` |
| Criterion under nsys point | `17.615 us` |
| `cudaLaunchKernel` avg / median | `3.3630 us` / `3.2900 us` |
| `cuStreamSynchronize` calls | `149294` |
| `cuStreamSynchronize` avg / median | `7.8415 us` / `4.8850 us` |
| Kernel avg / median | `12.8210 us` / `10.1760 us` |
| Kernel launch+queue+kernel avg / median | `17.6680 us` / `14.8930 us` |

Kernel median is about 3.1x launch median. The launch API is visible, but not a
clean dominant local fix under the current synchronized component benchmark.

## Fix

No runtime patch was made. A6 is deferred until there is a request-level or
async component benchmark that isolates launch reduction without mandatory
pre/post synchronization.

## Rule

Do not implement a persistent worker for `dsv4_fp8_gemv_batch_tiled_kernel` as
a Phase 3 local kernel-memory axis. Optimize tiled-kernel body redundancy first;
launch reduction needs a separate runtime/harness design.
