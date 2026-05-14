# DSv4 Local Expert Segment-Input Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run rebuilt the current local source in a clean remote source
tree (`/root/arle-local-expert-segment`) and launched the HTTP server under
`nsys profile` with CUDA profiler API start/stop.

The profiled request used streaming `max_tokens=2`:

```text
prompt: 请写两个汉字，要求意象偏城市夜景。
output: 霓虹
```

This trace validates the default DeepEP path after the local expert fallback
learned to run DSv4 block-scaled `w1`/`w3` GEMV directly from the packed
`expert_hidden` segment. The old per-active-expert D2D copy into
`scratch.input` remains as a fallback for unsupported weight formats.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 145.104 ms |
| Per-rank decode range p50 | 144.623 ms |
| Request wall time | 1.164 s |
| Returned text | `霓虹` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 30.884 ms | 9136 |
| `cuMemFreeAsync` | 23.027 ms | 7424 |
| `cuMemcpyDtoHAsync_v2` | 20.332 ms | 887 |
| `cudaLaunchKernel_v7000` | 17.785 ms | 15088 |
| `cuMemsetD8Async` | 11.903 ms | 10214 |
| `cudaEventRecord_v3020` | 3.972 ms | 4136 |
| `cudaStreamWaitEvent_v3020` | 3.566 ms | 3440 |
| `cuMemcpyDtoDAsync_v2` | 1.240 ms | 613 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 25.491 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 19.499 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.466 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.846 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.299 ms | 328 |
| `dsv4_route_kernel` | 5.659 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.499 ms | 688 |
| `dsv4_csa_select_kernel` | 4.130 ms | 168 |

## Comparison

Compared with [`../nsys-single-token-live/`](../nsys-single-token-live/), the
same prompt and output show:

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 146.448 ms | 145.104 ms |
| Per-rank decode range p50 | 146.161 ms | 144.623 ms |
| `cuMemcpyDtoDAsync_v2` time per rank range | 1.795 ms | 1.240 ms |
| `cuMemcpyDtoDAsync_v2` calls | 871 | 613 |

The optimization removes part of the expected D2D scratch traffic, but it is
not the main bottleneck. The remaining ranked costs are still allocator/runtime
churn, D2H route readback, NCCL SendRecv/AllReduce, and per-expert FP8/FP4 GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`

