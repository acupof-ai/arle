# DSv4 Incremental Attention Scratch Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after reusing incremental attention scratch buffers. The run first sends a `max_tokens=2` warmup request, then profiles a second `max_tokens=2` request. The profiled request returned `406` and produced one `step_decode_kernel_launch` wave across 8 rank threads.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Metric | Compressor projection scratch | Attention scratch |
| --- | ---: | ---: |
| Decode wave wall time | 121.550 ms | 97.042 ms |
| Per-rank decode range p50 | 121.272 ms | 96.793 ms |
| `cuMemAllocAsync` | 6,765 calls / 11.417 ms | 6,760 calls / 9.696 ms |
| `cuMemFreeAsync` | 4,360 calls / 8.537 ms | 3,048 calls / 4.909 ms |
| `cuMemsetD8Async` | 3,645 calls / 6.188 ms | 3,640 calls / 5.381 ms |
| `cuMemcpyDtoHAsync_v2` | 344 calls / 21.125 ms | 344 calls / 8.111 ms |
| `cudaLaunchKernel_v7000` | 16,417 calls / 29.912 ms | 16,415 calls / 27.987 ms |

Top attention-scratch decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 27.987 ms | 16,415 |
| `cuMemAllocAsync` | 9.696 ms | 6,760 |
| `cuMemcpyDtoHAsync_v2` | 8.111 ms | 344 |
| `cuMemFreeAsync` | 4.909 ms | 3,048 |
| `ncclDevKernel_SendRecv` | 23.410 ms | 688 |
| `dsv4_fp8_gemv_batch_kernel` | 11.480 ms | 2,920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.849 ms | 774 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 7.499 ms | 344 |
| `dsv4_hybrid_attention_kernel` | 7.398 ms | 328 |
| `dsv4_route_kernel` | 5.662 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.501 ms | 688 |

The single-token bottleneck is not sampler-side. The remaining decode wave is dominated by communication and per-layer launch/runtime overhead: NCCL SendRecv/AllReduce, route-count D2H synchronization, many small kernel launches, local FP8/FP4 expert GEMV, and attention/MHC kernels. Attention scratch is deliberately gated to B=1 decode so prefill does not retain prompt-sized prepared-Q/K buffers; in the warmed decode window this keeps allocation count effectively flat but cuts 1,312 free calls. The main performance target remains eliminating host readbacks, reducing launch count with CUDA Graph/PDL, and replacing per-route GEMV with real grouped GEMM/DeepGEMM plus DeepEP overlap.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
