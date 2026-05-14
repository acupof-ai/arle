# DSv4 Padded Peer Combine Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run keeps the shipped B=1 padded dispatch path and adds a
return-side combine optimization: each expert rank sums valid padded route rows
into one BF16 row per origin peer before the combine send/recv.

```text
prompt: 用两个字形容彩虹。
output: 霓彩
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 112.133 ms |
| Per-rank decode range p50 | 111.420 ms |
| Request wall time | 1.219 s |
| Returned text | `霓彩` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 22.695 ms | 15722 |
| `cuMemAllocAsync` | 15.740 ms | 7765 |
| `cuMemFreeAsync` | 13.071 ms | 6048 |
| `cuMemsetD8Async` | 11.802 ms | 8789 |
| `cuMemcpyDtoHAsync_v2` | 8.355 ms | 344 |
| `cudaEventRecord_v3020` | 2.882 ms | 3448 |
| `cudaStreamWaitEvent_v3020` | 2.199 ms | 2752 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 2.191 ms | 4856 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 23.329 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 12.494 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.486 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.847 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 6.954 ms | 328 |
| `dsv4_route_kernel` | 5.659 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.500 ms | 688 |
| `dsv4_csa_select_kernel` | 3.961 ms | 168 |

## Comparison

Compared with [`../nsys-single-token-padded-dispatch-skip-count/`](../nsys-single-token-padded-dispatch-skip-count/):

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 123.955 ms | 112.133 ms |
| Per-rank decode range p50 | 122.908 ms | 111.420 ms |
| `ncclDevKernel_SendRecv` time | 25.211 ms | 23.329 ms |
| `cudaLaunchKernel_v7000` time | 25.317 ms | 22.695 ms |
| `cuMemcpyDtoHAsync_v2` time | 8.761 ms | 8.355 ms |
| `dsv4_fp8_gemv_batch_kernel` time | 11.479 ms | 11.486 ms |
| `dsv4_fp4_gemv_batch_tiled_kernel` time | 10.840 ms | 10.847 ms |

The return exchange still launches one grouped NCCL send/recv per layer, but
the BF16 rows returned to each origin rank are now pre-summed per origin peer
instead of returned as padded `topk` route rows. This reduces combine bandwidth
without changing the dispatch metadata path. The remaining concrete bottlenecks
are still NCCL SendRecv/AllReduce, local expert FP8/FP4 GEMV, and CUDA
launch/runtime plus allocator/memset/free churn.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
