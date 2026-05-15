# DSv4 Single Decode Token With NCCL LL128

Date: 2026-05-15

This capture reruns the current default DSv4 DeepEP decode path on the remote
8xH20 host with `NCCL_PROTO=LL128` set in the profiling environment. It uses
the real `/root/DeepSeek-V4-Flash` checkpoint, FP8 KV cache, incremental KV,
fused dispatch payload, and default-on reduce-scatter combine.

The profiled arithmetic request returned the exact result `406` with
`prompt_tokens=17` and `completion_tokens=1`.

## Result

`NCCL_PROTO=LL128` is not a win for this single-token decode shape:

- Current default reference:
  `nsys-single-decode-token-current-user`, 94.841 ms decode wave.
- This LL128 run:
  94.936 ms decode wave.

The top kernel buckets per rank-range are still:

| Bucket | Time |
| --- | ---: |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 21.371 ms |
| `dsv4_fp8_gemv_batch_kernel` | 11.471 ms |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 11.222 ms |
| `dsv4_hybrid_attention_kernel` | 7.397 ms |
| `ncclDevKernel_AllReduce_Sum_bf16_TREE_LL` | 6.410 ms |
| `dsv4_route_kernel` | 5.658 ms |
| `dsv4_mhc_params_kernel` | 5.500 ms |
| `dsv4_csa_select_kernel` | 4.128 ms |
| `ncclDevKernel_SendRecv` | 3.271 ms |

The top runtime buckets per rank-range are:

| Runtime API | Calls | Time |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 16,142 | 27.905 ms |
| `cuMemAllocAsync` | 6,760 | 8.572 ms |
| `cuMemcpyDtoHAsync_v2` | 344 | 7.709 ms |
| `cuMemsetD8Async` | 3,640 | 5.542 ms |
| `cuMemFreeAsync` | 3,048 | 4.893 ms |

## Interpretation

This rules out a simple NCCL protocol switch as the next default-path fix. The
single-token slow stack remains return-side reduce-scatter combine, local
FP8/FP4 expert GEMV, attention/route/MHC kernels, and host/runtime launch plus
allocation/readback churn. The next real performance work should target
DeepEP-style communication overlap and replacing the per-expert GEMV path with
true grouped GEMM/DeepGEMM where the hardware and weight format support it.

