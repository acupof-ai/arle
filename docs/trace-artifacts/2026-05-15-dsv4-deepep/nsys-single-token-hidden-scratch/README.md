# DSv4 Hidden Scratch Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. The run uses the default DeepEP path with FP8 KV cache and
incremental KV enabled, then profiles a streaming `max_tokens=2` request:

```text
prompt: 请写两个汉字，要求意象偏城市夜景。
output: 霓虹
```

This trace validates per-layer incremental hidden scratch reuse for the DSv4
attention and FFN HyperConnection pre-projection and RMSNorm temporary buffers.
It removes four one-token temporary allocations per layer/rank from the decode
path without changing the generated output.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 135.390 ms |
| Per-rank decode range p50 | 135.104 ms |
| Request wall time | 1.375 s |
| Returned text | `霓虹` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 23.603 ms | 15088 |
| `cuMemAllocAsync` | 18.332 ms | 7760 |
| `cuMemcpyDtoHAsync_v2` | 17.130 ms | 887 |
| `cuMemFreeAsync` | 13.588 ms | 6048 |
| `cuMemsetD8Async` | 13.176 ms | 8838 |
| `cudaEventRecord_v3020` | 5.017 ms | 4136 |
| `cudaStreamWaitEvent_v3020` | 4.138 ms | 3440 |
| `cuLaunchKernelEx` | 3.145 ms | 1720 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 26.092 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 14.319 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.471 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.855 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.296 ms | 328 |
| `dsv4_route_kernel` | 5.660 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.500 ms | 688 |
| `dsv4_csa_select_kernel` | 4.133 ms | 168 |

## Comparison

Compared with [`../nsys-single-token-segment-input/`](../nsys-single-token-segment-input/):

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 145.104 ms | 135.390 ms |
| Per-rank decode range p50 | 144.623 ms | 135.104 ms |
| `cuMemAllocAsync` calls | 9136 | 7760 |
| `cuMemFreeAsync` calls | 7424 | 6048 |
| `cuMemsetD8Async` calls | 10214 | 8838 |
| `cuMemAllocAsync` time per rank range | 30.884 ms | 18.332 ms |
| `cuMemFreeAsync` time per rank range | 23.027 ms | 13.588 ms |

The exact removed call count is 1,376 for alloc/free/memset, matching four
hidden temporaries across 43 layers and 8 ranks. This confirms allocator
lifetime cleanup is useful, but the remaining single-token bottleneck is still
not sampler or KV lookup. The ranked costs are now launch/runtime overhead,
D2H route readback, NCCL SendRecv/AllReduce, and local expert FP8/FP4 GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
