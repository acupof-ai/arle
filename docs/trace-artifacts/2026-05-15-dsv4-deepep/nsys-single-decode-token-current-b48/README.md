# DSv4 Single Decode Token Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20 at commit `b48a363d`. The service ran under
`nsys profile` with CUDA profiler API start/stop and the shipped DSv4 DeepEP
path:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The profiled request used streaming `max_tokens=2` for prompt
`用两个字形容彩虹。` and returned `霓彩`. This is intentional: a
`max_tokens=1` request finishes from the prefill step in the current scheduler
and does not create a real `step_decode_kernel_launch` range. The tables below
filter only the single `step_decode_kernel_launch` wave created by the second
token.

## Result

| Metric | Value |
| --- | ---: |
| Prompt tokens | 9 |
| Completion tokens | 2 |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 125.497 ms |
| Per-rank decode range p50 | 125.219 ms |
| Profiled request wall time | 1.120 s |
| Returned text | `霓彩` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 23.829 ms | 7765 |
| `cudaLaunchKernel_v7000` | 21.963 ms | 15722 |
| `cuMemFreeAsync` | 18.119 ms | 6048 |
| `cuMemsetD8Async` | 11.855 ms | 8789 |
| `cuMemcpyDtoHAsync_v2` | 11.497 ms | 344 |
| `cudaEventRecord_v3020` | 2.647 ms | 3448 |
| `cudaStreamWaitEvent_v3020` | 2.075 ms | 2752 |
| `cuMemcpyDtoDAsync_v2` | 1.969 ms | 1048 |

Top CUDA kernels launched from the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 25.481 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 15.278 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.479 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.850 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 6.954 ms | 328 |
| `dsv4_route_kernel` | 5.660 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.501 ms | 688 |
| `dsv4_csa_select_kernel` | 3.962 ms | 168 |

Memcpy GPU-duration inside the same decode-token window is small:

| Direction | Time per rank range | Calls | Bytes |
| --- | ---: | ---: | ---: |
| Device-to-Device | 0.169 ms | 1048 | 20078784 |
| Device-to-Host | 0.104 ms | 344 | 44032 |
| Host-to-Device | 0.103 ms | 1040 | 56448 |

## Bottleneck

The single-token decode step is not sampler-bound and the KV cache path is not
the dominant cost in this trace. Attention is visible, but it is smaller than
the communication, expert GEMV, and host/runtime overhead stack.

The current slow stack is:

1. NCCL exchange: `SendRecv` plus BF16 all-reduce is about 40.8 ms per rank
   range.
2. CUDA allocator/runtime churn: alloc/free alone is about 41.9 ms per rank
   range, with another 11.9 ms in memset API work.
3. Kernel launch overhead: about 22.0 ms per rank range across 15722 launches.
4. Local expert math: FP8 and FP4 GEMV together are about 22.3 ms per rank
   range.
5. Route/MHC/attention kernels: route plus MHC plus attention are about
   18.1 ms per rank range.
6. D2H API overhead remains visible at 344 calls / 11.5 ms per rank range, but
   the actual Device-to-Host copy work is only 44 KiB total and 0.104 ms per
   rank range, so the pain is call/synchronization overhead rather than payload
   size.

This confirms the next performance work should target DeepEP-style
dispatch/combine overlap and fewer communication launches, a real grouped
GEMM/DeepGEMM expert path, and scratch/CUDA Graph cleanup for alloc/free,
memset, and launch churn.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
