# DSv4 Stream Recycle Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after adding incremental stream scratch recycling. The run first sends a `max_tokens=2` decode warmup request, then profiles a second `max_tokens=2` request.

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

| Metric | Default warm decode | Stream recycle |
| --- | ---: | ---: |
| Decode wave wall time | 128.130 ms | 111.798 ms |
| Per-rank decode range p50 | 127.853 ms | 109.991 ms |
| `cuMemAllocAsync` | 8,453 calls / 16.802 ms | 7,757 calls / 12.574 ms |
| `cuMemFreeAsync` | 6,048 calls / 13.801 ms | 5,352 calls / 11.096 ms |
| `cuMemcpyDtoHAsync_v2` | 347 calls / 16.470 ms | 344 calls / 12.225 ms |
| `cudaLaunchKernel_v7000` | 16,416 calls / 30.559 ms | 16,417 calls / 27.876 ms |

Top stream-recycle decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 27.876 ms | 16,417 |
| `cuMemAllocAsync` | 12.574 ms | 7,757 |
| `cuMemcpyDtoHAsync_v2` | 12.225 ms | 344 |
| `cuMemFreeAsync` | 11.096 ms | 5,352 |
| `ncclDevKernel_SendRecv` | 22.892 ms | 688 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 14.362 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.477 ms | 2,920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.848 ms | 774 |

The trace confirms stream recycling removes some allocator/free overhead and improves the isolated decode wave by about 12.7%, but the remaining steady-state stack is still NCCL SendRecv/AllReduce, local expert FP8/FP4 GEMV, launch overhead, D2H synchronization, and residual allocator churn. The next real performance target remains grouped GEMM/DeepGEMM plus DeepEP-style overlap.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
