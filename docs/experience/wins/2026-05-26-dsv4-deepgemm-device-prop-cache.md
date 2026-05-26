# DSv4 DeepGEMM device-property cache

## Goal

Remove the repeated `cudaGetDeviceProperties` call from the DSv4 DeepGEMM
native bridge decode path after the default DeepEP + DeepGEMM nsys trace showed
`cudaGetDeviceProperties_v2_v12000` as the largest CUDA runtime API bucket.

This is a narrow API-sync cleanup, not the A3 completion gate. A3 still needs
DtoH <= 50 and a wall-clock win on the final 32K / 1.5K, c=8, qps=8 SLO frame.

## Hypothesis

DeepGEMM grouped GEMM calls run repeatedly on long-lived scheduler threads.
Caching `cudaDeviceProp` in thread-local storage keyed by current device should
preserve multi-GPU correctness while eliminating per-GEMM device property
queries.

## Params

| Item | Value |
|---|---|
| GPU | 8x H20 |
| CUDA | 12.9 toolchain, sm_90 cubins |
| Runtime | ARLE DSv4 CUDA, 8 workers, `--num-slots 1` |
| KV | FP8 |
| Request | short prompt, `max_tokens=32` |
| Backend | DeepEP-style dispatch/combine + required DeepGEMM validation mode |
| Build | `cargo build --release -p infer --features cuda,nccl --bin infer` |
| Local check | `CUDARC_CUDA_VERSION=12090 cargo check -p infer --no-default-features --features cuda,no-cuda` |

Model path and host identifiers are intentionally omitted.

## Results

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Non-nsys smoke wall-clock | 12.1466 s | 8.2378 s | -32.2% |
| nsys profile request wall-clock | 8.0900 s | 5.4238 s | -33.0% |
| Decode range p50 | 210.033 ms | 56.802 ms | -73.0% |
| Decode wave max | 362.614 ms | 251.243 ms | -30.7% |
| `cudaGetDeviceProperties_v2_v12000` calls | 12,802 | absent from top APIs | removed from hot frame |
| `cudaGetDeviceProperties_v2_v12000` per-rank-range time | 96.559 ms | absent from top APIs | removed from hot frame |
| `cuMemcpyDtoHAsync_v2` calls | 10,682 | 10,697 | unchanged |
| `cuMemcpyDtoHAsync_v2` per-rank-range time | 85.585 ms | 24.724 ms | -71.1% |

Greedy output stayed stable for the profiled request:

```text
4062  | 0.0000  | 0.0000  | 0.0000  | 0.0000  |
```

## Problems

- This does not remove the D2H sites. D2H call count stays at roughly 10.7k in
  the filtered nsys capture, so A3 remains open.
- The first decode wave still has a large cold-start tail (`251.243 ms`). The
  p50 wave is much better, but final SLO evidence still needs the 32K / 1.5K
  c=8/qps=8 framing.
- NCCL and allocator/API churn remain visible after the device-property query
  is gone.

## Artefacts

- Before default DeepEP + DeepGEMM nsys:
  `/sgl-workspace/bench-artifacts/dsv4-default-deepep-nsys-max32-20260526`
- Before default DeepEP + DeepGEMM smoke:
  `/sgl-workspace/bench-artifacts/dsv4-default-deepep-smoke-max32-20260526`
- After build:
  `/sgl-workspace/bench-artifacts/dsv4-deepgemm-prop-cache-build-20260526`
- After smoke:
  `/sgl-workspace/bench-artifacts/dsv4-deepgemm-prop-cache-smoke-max32-20260526`
- After nsys:
  `/sgl-workspace/bench-artifacts/dsv4-deepgemm-prop-cache-nsys-max32-20260526`

## Learnings

DeepGEMM native bridge host-side helpers must obey the same per-thread device
cache rule as TileLang SM dispatch. A single runtime-property query in a tight
GEMM launch path can dominate nsys wall-clock attribution and materially hurt
short decode wall-clock.

## Rule

Do not call `cudaGetDeviceProperties` inside a per-layer/per-expert launch
loop. Cache immutable device properties per thread and invalidate only when the
current CUDA device ordinal changes.
