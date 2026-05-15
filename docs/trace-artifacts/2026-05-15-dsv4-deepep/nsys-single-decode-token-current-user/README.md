# DSv4 Current Single Decode Token Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This is a fresh user-requested `nsys` rerun of the current default
DeepEP path with incremental KV, fused dispatch payload, FP8 KV cache, and
default-on reduce-scatter combine.

Prompt:

```text
Compute 137 + 269. Answer with the number only.
```

The profiled request returned `406` with `prompt_tokens=17` and
`completion_tokens=1`.

## Result

`summary.json` filters the trace to the `step_decode_kernel_launch` NVTX
ranges. There are 8 rank-local ranges for one decode wave:

| Metric | Value |
| --- | ---: |
| Decode wave wall time | 94.841 ms |
| Rank-range min | 94.417 ms |
| Rank-range p50 | 94.637 ms |
| Rank-range max | 94.764 ms |

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 20.549 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.471 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 11.103 ms |
| 4 | `dsv4_hybrid_attention_kernel` | 7.396 ms |
| 5 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 6.184 ms |
| 6 | `dsv4_route_kernel` | 5.661 ms |
| 7 | `dsv4_mhc_params_kernel` | 5.501 ms |
| 8 | `dsv4_csa_select_kernel` | 4.140 ms |
| 9 | `ncclDevKernel_SendRecv` | 3.815 ms |

Top CUDA runtime API time per rank-range:

| Rank | Runtime API | Time | Calls |
| ---: | --- | ---: | ---: |
| 1 | `cudaLaunchKernel_v7000` | 27.918 ms | 16,176 |
| 2 | `cuMemAllocAsync` | 7.991 ms | 6,760 |
| 3 | `cuMemcpyDtoHAsync_v2` | 7.635 ms | 344 |
| 4 | `cuMemsetD8Async` | 5.671 ms | 3,640 |
| 5 | `cuMemFreeAsync` | 4.511 ms | 3,048 |
| 6 | `cudaEventRecord_v3020` | 3.003 ms | 2,760 |
| 7 | `cudaStreamGetCaptureInfo_v2_v11030` | 2.678 ms | 4,512 |
| 8 | `cudaStreamWaitEvent_v3020` | 2.293 ms | 2,064 |

## Diagnosis

The single-token slowdown is not sampler-bound. The largest exposed cost is
still MoE communication and expert execution: reduce-scatter combine costs
20.549 ms per rank-range, local FP8 and FP4 expert GEMV cost 22.573 ms
combined, and residual all-reduce/send-recv remains visible. Attention, MHC,
route, and CSA add another material block.

Host-side overhead is also too high for one token: the decode range still has
16,176 kernel launches, 6,760 async allocations, 3,048 async frees, 3,640
memsets, and 344 D2H copies. The next high-impact work remains true grouped
GEMM/DeepGEMM for local experts, DeepEP-style communication overlap, CUDA
Graph or persistent scheduling for launch overhead, and removing remaining
per-token allocation/D2H synchronization.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
