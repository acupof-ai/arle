# DSv4 Live Single-Token Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run rebuilt the current local source in a clean remote source
tree (`/root/arle-nsys-token-current`) and launched the HTTP server under
`nsys profile` with CUDA profiler API start/stop.

The profiled request used streaming `max_tokens=2`:

```text
prompt: 请写两个汉字，要求意象偏城市夜景。
output: 霓虹
```

Nsight captured one `step_decode_kernel_launch` wave across the 8 rank threads.
The tables below are filtered to CUDA runtime calls inside those rank-local
decode ranges, and kernels are matched by the CUDA launch correlation IDs from
the same ranges.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 146.448 ms |
| Per-rank decode range p50 | 146.161 ms |
| Request wall time | 1.124 s |
| Returned text | `霓虹` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 27.806 ms | 9136 |
| `cuMemFreeAsync` | 22.110 ms | 7424 |
| `cudaLaunchKernel_v7000` | 19.803 ms | 15088 |
| `cuMemcpyDtoHAsync_v2` | 18.782 ms | 887 |
| `cuMemsetD8Async` | 13.157 ms | 10214 |
| `cudaEventRecord_v3020` | 4.299 ms | 4136 |
| `cudaStreamWaitEvent_v3020` | 3.822 ms | 3440 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 25.784 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 17.787 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.470 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.885 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.296 ms | 328 |
| `dsv4_route_kernel` | 5.660 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.500 ms | 688 |
| `dsv4_csa_select_kernel` | 4.132 ms | 168 |
| `ncclDevKernel_AllGather_RING_LL` | 3.656 ms | 344 |

## Bottleneck

The single-token decode path is not sampler-bound. KV and attention kernels are
present, but attention is smaller than runtime overhead, NCCL communication, and
MoE expert GEMV work.

Concrete priority from this trace:

1. remove remaining per-token allocation, free, memset, and launch churn;
2. eliminate or batch D2H routing readbacks;
3. reduce DeepEP-style dispatch/combine communication count and overlap it;
4. replace per-expert FP8/FP4 GEMV with true grouped GEMM or DeepGEMM.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`

