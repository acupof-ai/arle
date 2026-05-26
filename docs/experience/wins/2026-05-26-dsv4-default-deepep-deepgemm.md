# DSv4 default DeepEP + DeepGEMM integration gate

## Goal

Make DSv4 default to the DeepEP-style dispatch/combine route and the DeepGEMM
auto local expert path, then verify the fail-fast validation path with real
decode (`max_tokens=32`).

This is an integration gate, not a performance claim. The DSv4 SLO frame stays:
input 32K / output 1.5K, H20 qps 8 at concurrency 8, TTFT <= 5000 ms,
TPOT <= 30 ms. The current target baseline remains TTFT 4800 ms, TPOT 18 ms,
total throughput 8402.

## Hypothesis

Defaulting `ARLE_DSV4_MOE_BACKEND` to DeepEP should exercise the existing
multi-rank dispatch/combine path. Defaulting the expert backend to
`deepgemm-auto` should build the FP8 expert cache and use DeepGEMM when the
native bridge is available, while preserving an explicit
`ARLE_DSV4_MOE_BACKEND=allreduce` escape hatch.

## Params

| Item | Value |
|---|---|
| GPU | 8x H20 |
| CUDA | 12.9 toolchain, sm_90 cubins |
| Runtime | ARLE DSv4 CUDA, 8 workers, `--num-slots 1` |
| KV | FP8 |
| Request | short prompt, `max_tokens=32` |
| Runtime defaults | `ARLE_DSV4_MOE_BACKEND=deepep`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm-auto` |
| Validation defaults | `ARLE_DSV4_MOE_BACKEND=deepep`, `ARLE_DSV4_EXPERT_BACKEND=deepgemm` via toolchain defaults |
| Build | `cargo build --release -p infer --features cuda,nccl --bin infer` |
| Local check | `CUDARC_CUDA_VERSION=12090 cargo check -p infer --no-default-features --features cuda,no-cuda` |

Model path and host identifiers are intentionally omitted.

## Results

### Build and smoke

| Gate | Result |
|---|---|
| Local shell check | pass: `bash -n scripts/dsv4_toolchain.sh scripts/profile_dsv4_single_decode_nsys.sh` |
| Local CUDA/no-cuda typecheck | pass with existing warnings |
| Remote env-check | pass: DeepGEMM root resolved, NCCL found, default env prints `deepep` + `deepgemm` |
| Remote release build | pass in 6m50s |
| Remote smoke | pass, `elapsed_s=12.1466`, `prompt_tokens=17`, `completion_tokens=32` |

Smoke output:

```text
4062  | 0.0000  | 0.0000  | 0.0000  | 0.0000  |
```

### nsys decode trace

Single profile request, filtered to `step_decode_kernel_launch` NVTX ranges:

| Metric | Value |
|---|---:|
| Profile request wall-clock | 8.0900 s |
| Decode waves | 31 |
| Decode ranges | 248 |
| Decode wave p50 | 210.033 ms |
| Decode wave max | 362.614 ms |
| D2H activity | 10,700 calls, 1,365,136 B |
| H2D activity | 47,180 calls, 1,313,964 B |

Top remaining costs in the filtered trace:

| Rank | Item | Per-rank-range time |
|---:|---|---:|
| 1 | `cudaGetDeviceProperties_v2_v12000` | 96.559 ms |
| 2 | `cuMemcpyDtoHAsync_v2` | 85.585 ms |
| 3 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 68.691 ms |
| 4 | `dsv4_fp8_gemv_batch_kernel` | 10.801 ms |
| 5 | `cudaLaunchKernel_v7000` | 9.118 ms |

DeepGEMM kernels are present in the trace, but their filtered per-rank-range
time is below the NCCL/D2H/API-sync costs:

| Kernel group | Per-rank-range time |
|---|---:|
| DeepGEMM w13 FP8 grouped GEMM | 0.819 ms |
| DeepGEMM w2 FP8 grouped GEMM | 0.522 ms |

## Problems

- This does not reverse the 2026-05-26 DeepGEMM optimization KILL. Required
  DeepGEMM had already failed wall-clock and byte-identical gates; this entry
  only verifies that the user-licensed default integration path builds and
  runs decode.
- The default path is still using the DeepEP-style NCCL fallback, not native
  DeepEP low-latency `internode_ll` kernels. The nsys evidence says the next
  real wall-clock lever is native DeepEP LL plus device-count combine, not more
  tuning around the current NCCL fallback.
- D2H and `cudaGetDeviceProperties` are still large in the decode frame, so A3
  remains open.

## Artefacts

- Remote env-check:
  `dsv4-default-deepep-envcheck-20260526-2.log`
- Remote build:
  `dsv4-default-deepep-build-20260526`
- Remote smoke:
  `dsv4-default-deepep-smoke-max32-20260526`
- Remote nsys:
  `dsv4-default-deepep-nsys-max32-20260526`

## Learnings

Default integration is now usable enough to keep DeepEP/DeepGEMM on the main
DSv4 validation path. Performance work should pivot to a native DeepEP
low-latency C ABI around `internode_ll::dispatch/combine`; the current
NCCL-based DeepEP-style path is evidence, not the final architecture.

## Rule

A default-path flip needs decode evidence with `max_tokens>=32`, build/runtime
logs, and an nsys cross-check. Do not claim performance from a default flip
unless wall-clock and correctness gates pass against an isolated baseline.
