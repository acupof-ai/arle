# P3.3 DSv4 Route Persistent Kernel Deferred

## Context

Phase 3 P3.3 A6 evaluated whether launch reduction or a persistent-kernel
shape should be attempted for `dsv4_route_kernel`, the route-select primitive
behind `dsv4_route_cuda`.

The local route microbench covers the DeepSeek V4 1B-style learned-bias
sqrtsoftplus path:

- decode t1/e16/top2
- batch t64/e16/top2

## Formula Prediction

Hypothesis before measurement:

- The normal Criterion component bench reports about 9.4 us for t1 and 9.6 us
  for t64, so most per-call cost may be launch/sync framing rather than the
  router's 16-expert loop.
- A persistent kernel could theoretically remove launch cost.
- This is only licenseable if launch cost is separable from the component bench
  synchronization. Persistent route selection changes dispatch lifetime and is
  not a local source-level kernel tweak.

## Root Cause

A6 is not a clean local operator axis under the current Criterion harness.
`infer/benches/ops/common/mod.rs` `iter_sync` synchronizes before and after
every iteration. A persistent-kernel treatment would remove launch while also
requiring a new lifecycle/harness framing, so the result could not be attributed
to route kernel code alone.

## Evidence

Steady nsys command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_3_a6_dsv4_route_t1_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_route/dsv4_mini_decode_t1_e16_top2 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats commands:

```bash
nsys stats --report cuda_api_sum --format csv \
  /tmp/p3_3_a6_dsv4_route_t1_steady.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  /tmp/p3_3_a6_dsv4_route_t1_steady.nsys-rep
nsys stats --report cuda_kern_exec_sum --format csv --force-export=true \
  /tmp/p3_3_a6_dsv4_route_t1_steady.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `94972` |
| Criterion under nsys point | `11.786 us` |
| `cudaLaunchKernel` avg / median | `3.3573 us` / `3.2800 us` |
| `cuStreamSynchronize` calls | `189994` |
| `cuStreamSynchronize` avg / median | `4.2652 us` / `2.8950 us` |
| Kernel avg / median | `5.7119 us` / `4.3520 us` |
| Kernel launch+queue+kernel avg / median | `10.5719 us` / `9.0630 us` |

Launch API is measurable, but the component loop synchronizes twice per
iteration. Removing launch without changing the sync framing is not an
isolated local operator experiment.

## Fix

No runtime patch was made. A6 is deferred until there is an async/request-level
route benchmark or CUDA graph/persistent-worker plan that isolates launch
removal from per-iteration synchronization.

## Tradeoff

- LOC complexity: persistent route selection would add non-local lifecycle
  state and shutdown semantics.
- SM89 specificity: nsys evidence is local to RTX 4070 Ti SUPER / SM89.
- Generality: evidence covers the 1B E16/top2 path, not the full E256/top6
  production shape.
- Correctness: not evaluated because no treatment was run.

## Rule

Do not implement persistent DSv4 route selection as a P3.3 local kernel tweak.
First create a benchmark framing where launch removal can be isolated from
mandatory synchronization.
