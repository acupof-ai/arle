# DSv4 Default Warm Decode Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. Unlike the earlier one-token captures, this run first sends a `max_tokens=2` decode warmup request, then profiles a second `max_tokens=2` request. This isolates steady-state decode from first-decode scratch allocation.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
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
| Decode wave wall time | 128.130 ms |
| Per-rank decode range p50 | 127.853 ms |
| HTTP request elapsed | 520.744 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 30.559 ms | 16416 |
| `cuMemAllocAsync` | 16.802 ms | 8453 |
| `cuMemcpyDtoHAsync_v2` | 16.470 ms | 347 |
| `cuMemFreeAsync` | 13.801 ms | 6048 |
| `cuMemsetD8Async` | 5.858 ms | 3645 |
| `cudaEventRecord_v3020` | 3.276 ms | 2760 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 2.702 ms | 4512 |
| `cudaStreamWaitEvent_v3020` | 2.388 ms | 2064 |
| `ncclDevKernel_SendRecv` | 24.403 ms | 688 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 21.258 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.471 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.848 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 6.950 ms | 328 |
| `dsv4_route_kernel` | 5.658 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.501 ms | 688 |
| `dsv4_csa_select_kernel` | 3.967 ms | 168 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float,` | 1.078 ms | 1504 |
| `dsv4_compressor_update_kernel` | 0.587 ms | 496 |

The decode warmup reduces first-use allocation noise but does not change the core diagnosis: steady-state single-token decode remains dominated by NCCL SendRecv/AllReduce, launch overhead, async allocation/free, and local expert FP8/FP4 GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
