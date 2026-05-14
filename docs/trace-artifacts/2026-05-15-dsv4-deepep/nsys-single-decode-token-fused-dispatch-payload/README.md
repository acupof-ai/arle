# DSv4 Fused Dispatch Payload Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20. The service used the default DSv4 DeepEP path with FP8 KV
and the B=1 decode fused dispatch payload path:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The profiled streaming request used `max_tokens=2` for prompt
`用两个字形容彩虹。` and returned `霓彩`.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 118.985 ms |
| Per-rank decode range p50 | 118.789 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 30.069 ms | 16410 |
| `ncclDevKernel_SendRecv` | 24.692 ms | 688 |
| `cuMemAllocAsync` | 14.945 ms | 8453 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 13.803 ms | 344 |
| `cuMemFreeAsync` | 12.776 ms | 6048 |
| `dsv4_fp8_gemv_batch_kernel` | 11.478 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.850 ms | 774 |
| `cuMemcpyDtoHAsync_v2` | 10.633 ms | 344 |
| `cuMemsetD8Async` | 5.925 ms | 3645 |

The fused BF16 payload appends the 3xI32 route metadata as raw 16-bit words
behind each hidden row and exchanges hidden+metadata with one BF16 grouped
send/recv. This reduces decode-window SendRecv launches from 1,032 to 688
without using the slower U8 NCCL path. The extra payload pack/unpack kernels
are visible but small:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `dsv4_pack_dispatch_payload_kernel` | 0.116 ms | 344 |
| `dsv4_unpack_dispatch_payload_kernel` | 0.102 ms | 344 |

Compared with `nsys-single-decode-token-uninit`, this keeps the SendRecv launch
count lower at 688 instead of 1,032, but the latest single-token wave is
118.985 ms. The fresh trace shows the wall time is still dominated by NCCL
exchange/reduction, launch overhead, async allocation/free, D2H, and local
expert FP8/FP4 GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
