# DSv4 Compressor Projection Scratch Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after reusing the GPU compressor update `kv_raw` and `score_raw` projection buffers. The run first sends a `max_tokens=2` decode warmup request, then profiles a second `max_tokens=2` request.

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

| Metric | Stream recycle | Compressor projection scratch |
| --- | ---: | ---: |
| Decode wave wall time | 111.798 ms | 121.550 ms |
| Per-rank decode range p50 | 109.991 ms | 121.272 ms |
| `cuMemAllocAsync` | 7,757 calls / 12.574 ms | 6,765 calls / 11.417 ms |
| `cuMemFreeAsync` | 5,352 calls / 11.096 ms | 4,360 calls / 8.537 ms |
| `cuMemcpyDtoHAsync_v2` | 344 calls / 12.225 ms | 344 calls / 21.125 ms |
| `cudaLaunchKernel_v7000` | 16,417 calls / 27.876 ms | 16,417 calls / 29.912 ms |

Top compressor-scratch decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 29.912 ms | 16,417 |
| `cuMemcpyDtoHAsync_v2` | 21.125 ms | 344 |
| `cuMemAllocAsync` | 11.417 ms | 6,765 |
| `cuMemFreeAsync` | 8.537 ms | 4,360 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 24.773 ms | 344 |
| `ncclDevKernel_SendRecv` | 23.971 ms | 688 |
| `dsv4_fp8_gemv_batch_kernel` | 11.483 ms | 2,920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.849 ms | 774 |

The scratch reuse removes 992 alloc/free pairs from the warmed decode window, but this capture is not a wall-time win: D2H and NCCL timing noise dominate the wave. The useful conclusion is narrower allocator pressure; the main performance target remains D2H elimination, NCCL/DeepEP overlap, and true grouped GEMM/DeepGEMM for local experts.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
