# DSv4 Uninitialized Full-Write Scratch Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20. The service used the default DSv4 DeepEP path with FP8 KV
and replaced selected full-write temporary `HiddenStates` allocations with
uninitialized buffers:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
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
| Decode wave wall time | 112.724 ms |
| Per-rank decode range p50 | 112.458 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 24.104 ms | 1032 |
| `cudaLaunchKernel_v7000` | 23.074 ms | 15728 |
| `cuMemAllocAsync` | 18.129 ms | 7765 |
| `cuMemFreeAsync` | 17.427 ms | 6048 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 14.666 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.474 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.842 ms | 774 |
| `cuMemcpyDtoHAsync_v2` | 10.891 ms | 345 |
| `cuMemsetD8Async` | 4.180 ms | 2957 |

Compared with `nsys-single-decode-token-current-b48`, the selected
uninitialized full-write allocations reduce decode-window `cuMemsetD8Async`
from 8,789 calls / 11.855 ms per rank range to 2,957 calls / 4.180 ms per rank
range. The single decode wave moves from 125.497 ms to 112.724 ms.

## Conclusion

This is a real positive cleanup, but it does not change the main ranking:
NCCL SendRecv/AllReduce, launch overhead, async allocation/free, and local
expert FP8/FP4 GEMV remain ahead of attention and sampler work. The next
performance work should still target fewer DeepEP communication launches,
scratch reuse or graph capture for allocation/free, and real grouped
GEMM/DeepGEMM for local experts.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
