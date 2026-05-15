# DSv4 Combine Overlap Disabled Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run uses the same binary as the combine-overlap experiment but
forces the overlap path off:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_COMBINE_REDUCE_SCATTER=1
ARLE_DSV4_COMBINE_OVERLAP=0
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

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
| Decode wave wall time | 107.562 ms |
| Rank-range min | 107.199 ms |
| Rank-range p50 | 107.414 ms |
| Rank-range max | 107.562 ms |

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 20.022 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.475 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 11.104 ms |
| 4 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 9.955 ms |
| 5 | `dsv4_hybrid_attention_kernel` | 7.396 ms |
| 6 | `dsv4_route_kernel` | 5.661 ms |
| 7 | `dsv4_mhc_params_kernel` | 5.503 ms |
| 8 | `ncclDevKernel_SendRecv` | 5.317 ms |
| 9 | `dsv4_csa_select_kernel` | 4.131 ms |

Top CUDA runtime API time per rank-range:

| Rank | Runtime API | Time | Calls |
| ---: | --- | ---: | ---: |
| 1 | `cudaLaunchKernel_v7000` | 32.920 ms | 16,174 |
| 2 | `cuMemAllocAsync` | 8.511 ms | 6,760 |
| 3 | `cuMemcpyDtoHAsync_v2` | 7.099 ms | 344 |
| 4 | `cuMemsetD8Async` | 6.441 ms | 3,640 |
| 5 | `cuMemFreeAsync` | 4.290 ms | 3,048 |
| 6 | `cudaEventRecord_v3020` | 3.812 ms | 2,760 |
| 7 | `cudaStreamGetCaptureInfo_v2_v11030` | 3.332 ms | 4,512 |
| 8 | `cudaStreamWaitEvent_v3020` | 2.865 ms | 2,064 |

## Diagnosis

This is the same binary with `ARLE_DSV4_COMBINE_OVERLAP=0`, so it validates
that the new overlap plumbing can stay dormant by default. This individual
Nsight capture is slower than the earlier current-reference trace
(107.562 ms versus 94.841 ms), but the matching trace-off HTTP `decode64`
smoke in `bench-combine-overlap-disabled/` still reaches the baseline
12.05 post-first tok/s. Treat this capture as a variance/control artifact,
not evidence that the default path throughput regressed.

The slow stack remains MoE reduce-scatter combine, local expert FP8/FP4 GEMV,
all-reduce/send-recv, attention/MHC/route kernels, and high per-token
launch/alloc/free/D2H runtime overhead.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
