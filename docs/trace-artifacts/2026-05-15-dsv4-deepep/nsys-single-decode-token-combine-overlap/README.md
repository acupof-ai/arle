# DSv4 Combine Overlap Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run enables the opt-in return-side MoE combine overlap
experiment:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_COMBINE_REDUCE_SCATTER=1
ARLE_DSV4_COMBINE_OVERLAP=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

Prompt:

```text
Compute 137 + 269. Answer with the number only.
```

The profiled request returned `406` with `prompt_tokens=17` and
`completion_tokens=1`.

## Result

`summary.json` filters the trace to the `step_decode_kernel_launch` NVTX
ranges. There are 8 rank-local ranges for one decode wave:

| Metric | Value |
| --- | ---: |
| Decode wave wall time | 104.359 ms |
| Rank-range min | 103.888 ms |
| Rank-range p50 | 104.144 ms |
| Rank-range max | 104.253 ms |

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 18.918 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.475 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 11.099 ms |
| 4 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 10.900 ms |
| 5 | `dsv4_hybrid_attention_kernel` | 7.396 ms |
| 6 | `dsv4_route_kernel` | 5.661 ms |
| 7 | `dsv4_mhc_params_kernel` | 5.502 ms |
| 8 | `ncclDevKernel_SendRecv` | 4.564 ms |
| 9 | `dsv4_csa_select_kernel` | 4.130 ms |

Top CUDA runtime API time per rank-range:

| Rank | Runtime API | Time | Calls |
| ---: | --- | ---: | ---: |
| 1 | `cudaLaunchKernel_v7000` | 28.595 ms | 16,177 |
| 2 | `cuMemAllocAsync` | 9.485 ms | 6,760 |
| 3 | `cuMemcpyDtoHAsync_v2` | 8.115 ms | 346 |
| 4 | `cuMemFreeAsync` | 5.724 ms | 3,048 |
| 5 | `cuMemsetD8Async` | 5.638 ms | 3,640 |
| 6 | `cudaEventRecord_v3020` | 3.104 ms | 2,760 |
| 7 | `cudaStreamGetCaptureInfo_v2_v11030` | 2.803 ms | 4,512 |
| 8 | `cudaStreamWaitEvent_v3020` | 2.357 ms | 2,064 |
| 9 | `cuMemcpyHtoDAsync_v2` | 2.005 ms | 1,040 |
| 10 | `cuLaunchKernelEx` | 1.775 ms | 1,032 |
| 11 | `cuEventRecord` | 0.974 ms | 688 |
| 12 | `cuStreamWaitEvent` | 0.856 ms | 688 |
| 13 | `cuEventCreate` | 0.531 ms | 688 |

## Diagnosis

This is a negative trace. The overlap communicator reduces the visible
reduce-scatter kernel time from the current reference's 20.549 ms to
18.918 ms per rank-range, but the total single-token decode wave regresses
from 94.841 ms to 104.359 ms. The extra cross-stream event traffic, second
communicator ordering, and worse all-reduce timing dominate the small
reduce-scatter improvement.

The default remains off. The useful code from this experiment is the opt-in
communication stream and routed-output fence plumbing; the next performance
step still needs true grouped GEMM/DeepGEMM plus coarser DeepEP overlap, not
per-layer event churn around the existing B=1 reduce-scatter.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
