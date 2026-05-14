# DSv4 Expert-Grouped GEMV Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. This run enables the opt-in expert-wise grouped GEMV path, warms one real `max_tokens=2` decode request, then profiles a second `max_tokens=2` request.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_GROUPED_EXPERTS=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The profiled request returned `霓彩`.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 145.693 ms |
| Per-rank decode range p50 | 145.378 ms |
| HTTP request elapsed | 1569.814 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemcpyDtoHAsync_v2` | 41.074 ms | 344 |
| `cudaLaunchKernel_v7000` | 26.603 ms | 15942 |
| `cuMemAllocAsync` | 15.235 ms | 8464 |
| `cuMemFreeAsync` | 13.898 ms | 6051 |
| `cuMemsetD8Async` | 5.567 ms | 3650 |
| `cuMemcpyHtoDAsync_v2` | 2.977 ms | 1658 |
| `ncclDevKernel_SendRecv` | 58.049 ms | 680 |
| `dsv4_fp4_grouped_gemv_pair_batch_kernel` | 23.162 ms | 203 |
| `dsv4_fp4_grouped_gemv_batch_kernel` | 11.428 ms | 203 |
| `dsv4_fp8_gemv_batch_kernel` | 11.353 ms | 2896 |
| `dsv4_hybrid_attention_kernel` | 6.955 ms | 328 |
| `dsv4_route_kernel` | 5.658 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.501 ms | 688 |

This is a negative trace. Expert-wise grouped GEMV reduces some expert-loop kernel count, but it regresses the steady-state decode wave versus the default warm decode trace. The grouped kernels are still GEMV-shaped and add host scheduling/H2D pressure for active expert metadata; they are not a substitute for a real grouped GEMM/DeepGEMM MoE path with DeepEP overlap.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
