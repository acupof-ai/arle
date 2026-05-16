# P3.5 DSv4 FP4 grouped pair persistent kernel deferred

## Context

Phase 3 P3.5 A6 evaluated whether launch reduction or a persistent-kernel
shape should be attempted for `dsv4_fp4_grouped_gemv_pair_batch_kernel`, the
grouped FP4 paired-output GEMV path behind
`dsv4_fp4_grouped_gemv_pair_batch_cuda`.

## Formula Prediction

Hypothesis before edit:

- The local t4 component bench reports about 18 us after the A2 packed-byte
  pair-load win.
- If the small t4 shape is strongly launch-bound, replacing per-call kernel
  launch with a persistent worker could theoretically recover a meaningful
  fraction of per-call time.
- This is only licenseable if launch cost is the binding separable component.
  A persistent worker changes dispatch, lifetime, and shutdown semantics, so it
  is not a local in-kernel memory-access tweak.

## Root Cause

nsys does not license a persistent-worker treatment for this tranche. Launch
API time is measurable, but the kernel body is still the larger median
component, and the component bench synchronizes before and after every
iteration. A persistent-kernel treatment would change launch behavior and
measurement framing at the same time, violating the single-variable Phase 3
rule.

Source evidence:

- `infer/benches/ops/common/mod.rs` `iter_sync` calls `ctx.sync()` before each
  iteration and after each iteration.
- `infer/benches/ops/ops_cuda_bench.rs` calls `iter_sync` for
  `ops_cuda/dsv4_fp4_grouped_gemv_pair`.

## Evidence

Steady nsys command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_5_a6_dsv4_grouped_pair_t4_steady \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_fp4_grouped_gemv_pair/dsv4_mini_t4_e4_512x1024 \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats commands:

```bash
nsys stats --force-export=true --report cuda_api_sum --format csv \
  /tmp/p3_5_a6_dsv4_grouped_pair_t4_steady.nsys-rep
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  /tmp/p3_5_a6_dsv4_grouped_pair_t4_steady.nsys-rep
nsys stats --force-export=true --report cuda_kern_exec_sum --format csv \
  /tmp/p3_5_a6_dsv4_grouped_pair_t4_steady.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `45203` |
| Criterion under nsys point | `20.506 us` |
| `cudaLaunchKernel` avg / median | `3.3935 us` / `3.3190 us` |
| `cuStreamSynchronize` calls | `90454` |
| `cuStreamSynchronize` avg / median | `10.4181 us` / `4.4245 us` |
| Kernel avg / median | `18.0922 us` / `13.0560 us` |
| Kernel launch+queue+kernel avg / median | `23.0063 us` / `17.8260 us` |

Kernel median is about 3.9x launch median. Launch reduction alone is not the
dominant local operator fix, and the sync framing remains a confounder.

## Fix

No runtime patch was made. A6 is deferred until there is a request-level or
async component benchmark that can isolate launch reduction without mandatory
pre/post synchronization.

## Tradeoff

- LOC complexity: persistent grouped GEMV would add non-local worker state,
  queueing, lifecycle, and shutdown semantics.
- SM89 specificity: nsys evidence is local to RTX 4070 Ti SUPER / SM89.
- Shared memory budget: unknown until a concrete worker design exists.
- Register budget: unknown until a concrete worker design exists.
- CUDA Graph compatibility: persistent workers and graph capture need a
  separate lifecycle analysis.
- Generality across route counts: current evidence covers the small t4 shape;
  t64 was not used for the launch-bound decision because kernel body work is
  larger there.
- Numerical correctness margin: not evaluated because no treatment was run.

## Rule

Do not implement a persistent kernel as a local P3.5 A6 tweak. First create an
async or request-level measurement that isolates launch removal from the
component bench's per-iteration synchronization.
