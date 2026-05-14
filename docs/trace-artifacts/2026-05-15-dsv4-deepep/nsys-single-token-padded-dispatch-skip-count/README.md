# DSv4 Padded Dispatch Skip-Count Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run uses the shipped B=1 decode route: fixed `ep_world * topk`
padded dispatch and no send-rank zero/count kernel in that padded path.

```text
prompt: 用两个字形容彩虹。
output: 霓彩
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 123.955 ms |
| Per-rank decode range p50 | 122.908 ms |
| Request wall time | 1.225 s |
| Returned text | `霓彩` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 25.317 ms | 15378 |
| `cuMemAllocAsync` | 16.275 ms | 7765 |
| `cuMemsetD8Async` | 13.809 ms | 8789 |
| `cuMemFreeAsync` | 13.396 ms | 6048 |
| `cuMemcpyDtoHAsync_v2` | 8.761 ms | 344 |
| `cudaEventRecord_v3020` | 3.533 ms | 3448 |
| `cudaStreamWaitEvent_v3020` | 2.774 ms | 2752 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 2.656 ms | 4856 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 25.211 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 13.796 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.479 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.840 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 6.954 ms | 328 |
| `dsv4_route_kernel` | 5.658 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.499 ms | 688 |
| `dsv4_csa_select_kernel` | 3.961 ms | 168 |

## Comparison

Compared with [`../nsys-single-token-allgather-counts/`](../nsys-single-token-allgather-counts/):

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 129.768 ms | 123.955 ms |
| Per-rank decode range p50 | 129.550 ms | 122.908 ms |
| `cuMemcpyDtoHAsync_v2` calls | 543 | 344 |
| 256-byte all-rank count D2H calls | 344 | 0 |
| 128-byte local-count D2H calls | 199 | 344 |
| `ncclDevKernel_AllGather` calls | 344 | 0 |

The path trades a small amount of padded dispatch/combine payload for removing
the count AllGather and its host readback. On this 8xH20 single-token decode
trace, the trade is net positive by 5.813 ms wall versus the prior default
path. The remaining concrete bottleneck is no longer the all-rank count matrix;
it is NCCL SendRecv/AllReduce, CUDA launch/runtime/allocator churn, the
local-count D2H readback, and local expert FP8/FP4 GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
