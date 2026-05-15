# DSv4 Attention Projection Scratch Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run validates per-layer incremental attention projection
scratch reuse for `c_q`, `c_q_normed`, `q_raw`, `kv_raw`, and `kv_normed`.

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
| Decode wave wall time | 90.946 ms |
| Rank-range min | 88.205 ms |
| Rank-range p50 | 89.644 ms |
| Rank-range max | 90.897 ms |

Compared with the current-user baseline trace:

| Metric | Baseline | This run |
| --- | ---: | ---: |
| Decode wave wall time | 94.841 ms | 90.946 ms |
| `cuMemAllocAsync` calls | 6,760 | 5,040 |
| `cuMemAllocAsync` time/rank-range | 7.991 ms | 7.096 ms |
| `cuMemFreeAsync` calls | 3,048 | 1,328 |
| `cuMemFreeAsync` time/rank-range | 4.511 ms | 2.052 ms |

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 20.302 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.476 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 11.107 ms |
| 4 | `dsv4_hybrid_attention_kernel` | 7.393 ms |
| 5 | `dsv4_route_kernel` | 5.660 ms |
| 6 | `dsv4_mhc_params_kernel` | 5.501 ms |
| 7 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 5.081 ms |
| 8 | `dsv4_csa_select_kernel` | 4.125 ms |
| 9 | `ncclDevKernel_SendRecv` | 2.091 ms |

## Diagnosis

This is a small positive trace. Reusing the attention projection intermediates
does not change the dominant MoE/expert kernels, but it removes 1,720 async
allocations and 1,720 async frees from the single-token decode range. The
remaining slow stack is still reduce-scatter combine, local FP8/FP4 expert
GEMV, attention/MHC/route kernels, D2H readbacks, and launch overhead.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
