# DSv4 Reduce-Scatter Combine Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. This profiles one real generated token with `ARLE_DSV4_COMBINE_REDUCE_SCATTER=1`.

Prompt:

```text
Compute 137 + 269. Answer with the number only.
```

The profiled request returned `406` with `prompt_tokens=17` and `completion_tokens=1`.

## Result

`summary.json` filters the trace to the `step_decode_kernel_launch` NVTX ranges. There are 8 rank-local ranges for one decode wave:

| Metric | Value |
| --- | ---: |
| Decode wave wall time | 94.923 ms |
| Rank-range min | 93.548 ms |
| Rank-range p50 | 93.832 ms |
| Rank-range max | 94.749 ms |

Compared with `nsys-single-decode-token-direct-20260515-0829/`, the isolated single-token wave moves from 97.071 ms to 94.923 ms. The return-side combine exchange is now visible as `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` at 20.443 ms per rank-range, while the remaining `ncclDevKernel_SendRecv` bucket drops to 3.259 ms.

Top kernel time per rank-range:

| Rank | Kernel | Time |
| ---: | --- | ---: |
| 1 | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 20.443 ms |
| 2 | `dsv4_fp8_gemv_batch_kernel` | 11.470 ms |
| 3 | `dsv4_fp4_gemv_batch_tiled_kernel` | 11.107 ms |
| 4 | `dsv4_hybrid_attention_kernel` | 7.396 ms |
| 5 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 6.942 ms |
| 6 | `dsv4_route_kernel` | 5.659 ms |
| 7 | `dsv4_mhc_params_kernel` | 5.499 ms |
| 8 | `dsv4_csa_select_kernel` | 4.139 ms |
| 9 | `ncclDevKernel_SendRecv` | 3.259 ms |

## Diagnosis

The change reduces the return-side combine P2P launch count and removes the separate padded peer-sum output combine kernel, but the total gain is small because the new reduce-scatter still costs about 20 ms per rank-range. The next high-impact targets remain local expert grouped GEMM/DeepGEMM, DeepEP overlap, CUDA Graph or persistent scheduling for launch overhead, scratch reuse for alloc/free, and D2H readback reduction.

Artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `summary.json`
- `decode-only-kernel-top.csv`
- `decode-only-runtime-api-top.csv`
- `stats_cuda_api_sum.csv`
- `stats_cuda_gpu_kern_sum.csv`
- `server.log.gz`
