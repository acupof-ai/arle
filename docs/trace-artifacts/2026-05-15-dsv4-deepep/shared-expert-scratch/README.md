# DSv4 Shared Expert Scratch Cleanup

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This change moves the per-layer shared expert decode path onto
reusable MoE scratch and adds an in-place BF16 add kernel so the shared expert
output is accumulated into the routed MoE output without allocating a separate
add result.

The change also restores the CUDA-facing `argmax_batch_readback_into` re-export
required by Qwen3.5 CUDA code, which was exposed by the remote CUDA/NCCL build.

## Functional Smoke

Trace-off DeepEP serving kept grouped experts disabled to match the default
path.

| Case | Prompt tokens | Completion tokens | Latency | Completion tok/s | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup | 13 | 1 | 0.448 s | 2.23 | `2` |
| `37*29` | 15 | 12 | 1.263 s | 9.50 | `37 × 29 = 1073。  \n解释：` |
| `58+67` | 15 | 12 | 1.267 s | 9.47 | `58 + 67 等于 **125**。\n\n**简短` |
| writing | 16 | 10 | 1.103 s | 9.07 | `**智联万物，芯动未来。**` |

## Single-Token Nsys

The profiled streaming request used `max_tokens=2`, returned `霓灯`, and
captured one `step_decode_kernel_launch` wave across 8 rank threads.

| Metric | Before shared scratch | Shared expert scratch |
| --- | ---: | ---: |
| Decode wave wall time | 158.439 ms | 140.111 ms |
| Per-rank decode range p50 | 158.161 ms | 139.860 ms |
| `cuMemAllocAsync` calls | 9136 | 7416 |
| `cuMemFreeAsync` calls | 9144 | 7424 |
| `cuMemsetD8Async` calls | 10182 | 8462 |
| `cudaLaunchKernel_v7000` calls | 15056 | 15056 |

Top decode runtime APIs after this cleanup:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 28.145 ms | 7416 |
| `cuMemFreeAsync` | 24.518 ms | 7424 |
| `cuMemcpyDtoHAsync_v2` | 19.747 ms | 871 |
| `cudaLaunchKernel_v7000` | 18.054 ms | 15056 |
| `cuMemsetD8Async` | 9.770 ms | 8462 |

Top decode kernels after this cleanup:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 28.677 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 17.639 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.471 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.877 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.357 ms | 328 |

## Bottleneck

The shared expert cleanup removes a real slice of allocator and memset churn,
but the remaining single-token cost is still dominated by runtime allocation,
D2H routing readbacks, NCCL exchange/reduction, and local expert GEMV. Attention
is present and still below those costs. The next production target is fewer host
route readbacks plus graph/lifetime cleanup, then the DeepEP grouped
GEMM/DeepGEMM path.

The scratch is decode-only. Prefill keeps the non-cached temporary path so long
prompts do not retain prompt-sized shared expert buffers across layers.

Raw trace files are committed here as compressed artifacts:

- `nsys/trace.nsys-rep.gz`
- `nsys/trace.sqlite.gz`
