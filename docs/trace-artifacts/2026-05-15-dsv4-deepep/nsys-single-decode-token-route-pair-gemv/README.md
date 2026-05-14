# DSv4 Route Pair GEMV Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. The service used DSv4 DeepEP, FP8 KV, fused BF16 dispatch payload, and the opt-in route-grouped expert path with pair route GEMV:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The profiled non-streaming request used `max_tokens=2` for prompt `用两个字形容彩虹。` and returned `霓彩`.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 117.894 ms |
| Per-rank decode range p50 | 116.687 ms |
| HTTP request elapsed | 519.837 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 33.662 ms | 11888 |
| `cudaLaunchKernel_v7000` | 21.741 ms | 15471 |
| `cuMemFreeAsync` | 15.486 ms | 6048 |
| `cuMemsetD8Async` | 6.220 ms | 4328 |
| `cuMemcpyHtoDAsync_v2` | 4.666 ms | 2760 |
| `cudaEventRecord_v3020` | 2.347 ms | 2760 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 1.925 ms | 4512 |
| `cudaStreamWaitEvent_v3020` | 1.818 ms | 2064 |
| `ncclDevKernel_SendRecv` | 50.338 ms | 560 |
| `dsv4_fp4_route_gemv_pair_batch_kernel` | 19.616 ms | 280 |
| `dsv4_fp4_route_gemv_batch_kernel` | 10.487 ms | 280 |
| `dsv4_fp8_gemv_batch_kernel` | 9.408 ms | 2400 |
| `dsv4_hybrid_attention_kernel` | 5.605 ms | 264 |
| `dsv4_route_kernel` | 4.555 ms | 280 |
| `dsv4_mhc_params_kernel` | 4.541 ms | 568 |
| `dsv4_csa_select_kernel` | 3.204 ms | 136 |

This trace confirms the current single-token decode path is still dominated by NCCL exchange/reduction, CUDA launch overhead, async allocation/free, and expert GEMV. The route-pair GEMV path removes part of the route-local D2H scheduling issue, but it is not enough to make decode fast because each layer still launches many small kernels and collectives.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
