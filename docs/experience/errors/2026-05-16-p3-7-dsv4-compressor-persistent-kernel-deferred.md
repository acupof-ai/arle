# P3.7 DSv4 compressor persistent kernel deferred

## Context

Phase 3 P3.7 A6 checked whether a persistent-kernel or launch-reduction
treatment should follow the `dsv4_compressor_update_kernel` body wins.

The trace used the post-A3 r96 HCA pending-only shape because it is the
strongest remaining local compressor update case after
`67b1db9 perf(cuda): skip stale dsv4 compressor pending copies`.

## Formula Prediction

Hypothesis before trace:

- After A3, r96 pending update is about 6-7 us in normal Criterion.
- If launch API time dominates, a persistent worker could be material.
- Persistent workers change dispatch/lifetime/shutdown semantics and are not a
  local kernel-body patch, so this needs launch dominance and a request-level
  design before implementation.

## Root Cause

A6 is deferred. nsys shows launch API time is material, but the synchronized
component benchmark also pays mandatory pre/post `cuStreamSynchronize` around
every iteration. A persistent treatment would change dispatch semantics while
leaving the benchmark synchronization framing unresolved.

Source evidence:

- `infer/benches/ops/common/mod.rs` `iter_sync` calls `ctx.sync()` before each
  iteration and after each iteration.
- `infer/benches/ops/ops_cuda_bench.rs` benchmarks
  `ops_cuda/dsv4_compressor_update` through `iter_sync`.

## Evidence

Steady nsys command:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o /tmp/p3_7_a6_dsv4_compressor_hca_pending \
  target/release/deps/ops_bench-d80e79bd3e0cee50 \
  --bench ops_cuda/dsv4_compressor_update/dsv4_mini_hca_pending_r96_h64_rope \
  --exact --sample-size 10 --noplot --discard-baseline
```

Stats command:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv \
  --force-export=true \
  /tmp/p3_7_a6_dsv4_compressor_hca_pending.nsys-rep
```

Summary:

| Metric | Value |
|---|---:|
| Kernel launches | `93762` |
| Criterion under nsys point | `12.118 us` |
| `cudaLaunchKernel` avg / median | `3.4658 us` / `3.3900 us` |
| `cuStreamSynchronize` calls | `187574` |
| `cuStreamSynchronize` avg / median | `4.4088 us` / `3.1750 us` |
| Kernel avg / median | `4.8920 us` / `4.0000 us` |

Kernel median is only about 1.18x launch median, so launch overhead is visible.
However, the component harness sync cost is similarly large, and a persistent
worker cannot be licensed as a scoped `dsv4_compressor_update_kernel` edit.

## Fix

No runtime patch was made. A6 is deferred until there is an async component or
request-level benchmark that isolates launch reduction without mandatory
per-iteration synchronization.

## Rule

Do not implement a persistent worker for DSv4 compressor update as a local
kernel-memory axis. The remaining launch cost needs a separate runtime/harness
tranche, not more complexity inside `dsv4_compressor_update_kernel`.
