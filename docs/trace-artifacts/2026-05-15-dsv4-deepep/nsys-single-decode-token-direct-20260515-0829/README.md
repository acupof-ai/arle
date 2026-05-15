# DSv4 Single Decode Token Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. The run used the current DSv4 DeepEP path with incremental KV, fused dispatch payload, FP8 KV cache, and `--deepseek-distributed-layers 43`.

Prompt:

```text
Compute 137 + 269. Answer with the number only.
```

The profiled request returned `406` with `prompt_tokens=17` and `completion_tokens=1`.

## Result

`summary.json` filters the trace to the `step_decode_kernel_launch` NVTX ranges. There are 8 rank-local ranges for one decode wave:

| Metric | Value |
| --- | ---: |
| Decode wave wall time | 97.071 ms |
| Rank-range min | 96.565 ms |
| Rank-range p50 | 96.849 ms |
| Rank-range max | 97.050 ms |

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_SendRecv` | 23.163 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.476 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 10.851 ms |
| 4 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 7.505 ms |
| 5 | `dsv4_hybrid_attention_kernel` | 7.399 ms |
| 6 | `dsv4_route_kernel` | 5.660 ms |
| 7 | `dsv4_mhc_params_kernel` | 5.501 ms |
| 8 | `dsv4_csa_select_kernel` | 4.138 ms |

Top CUDA runtime API time per rank-range:

| Rank | Runtime API | Time | Calls |
| ---: | --- | ---: | ---: |
| 1 | `cudaLaunchKernel_v7000` | 27.388 ms | 16415 |
| 2 | `cuMemAllocAsync` | 10.258 ms | 6760 |
| 3 | `cuMemcpyDtoHAsync_v2` | 8.398 ms | 344 |
| 4 | `cuMemFreeAsync` | 5.651 ms | 3048 |
| 5 | `cuMemsetD8Async` | 5.155 ms | 3640 |
| 6 | `cudaEventRecord_v3020` | 2.765 ms | 2760 |
| 7 | `cudaStreamGetCaptureInfo_v2_v11030` | 2.546 ms | 4512 |
| 8 | `cudaStreamWaitEvent_v3020` | 2.073 ms | 2064 |

## Diagnosis

The single-token decode slowdown is not sampler-bound. The main exposed costs are:

1. DeepEP return/dispatch exchange shape: `ncclDevKernel_SendRecv` is the largest single kernel bucket.
2. Local expert execution: current decode still launches many per-expert GEMV kernels instead of a true grouped GEMM/DeepGEMM path.
3. Cross-rank reductions: residual `AllReduce` remains material in the decode wave.
4. Attention/MHC/route kernels are non-trivial but below the MoE communication plus expert GEMV cost.
5. Host-side runtime overhead is still large: many launches, async alloc/free, D2H readbacks, memsets, and event waits show up inside the decode range.

Next optimization targets should therefore stay focused on DeepEP-style dispatch/combine consolidation, grouped GEMM/DeepGEMM for local experts, scratch reuse to remove per-token alloc/free, and D2H/launch reduction via CUDA Graph or persistent decode scheduling.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
