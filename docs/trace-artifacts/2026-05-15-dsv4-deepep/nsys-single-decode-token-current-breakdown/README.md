# DSv4 Current Single Decode Token Nsight Breakdown

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This reruns the current default DeepEP decode path under Nsight
Systems with CUDA profiler API start/stop and filters the summary to the single
`step_decode_kernel_launch` wave for the second token of a `max_tokens=2`
request.

```text
prompt: Compute 137 + 269. Answer with the number only.
output: 406
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 105.205 ms |
| Per-rank decode range p50 | 105.070 ms |
| Profile request wall time | 1.164 s |
| Returned text | `406` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 34.159 ms | 16,177 |
| `cuMemcpyDtoHAsync_v2` | 7.306 ms | 347 |
| `cuMemsetD8Async` | 6.932 ms | 3,640 |
| `cuMemAllocAsync` | 6.897 ms | 5,040 |
| `cudaEventRecord_v3020` | 3.988 ms | 2,760 |
| `cudaStreamGetCaptureInfo_v2_v11030` | 3.393 ms | 4,512 |
| `cudaStreamWaitEvent_v3020` | 2.904 ms | 2,064 |
| `cuMemcpyHtoDAsync_v2` | 2.063 ms | 1,040 |
| `cuLaunchKernelEx` | 2.054 ms | 1,032 |
| `cuMemFreeAsync` | 1.955 ms | 1,328 |

Memcpy activity bytes inside the same decode window:

| Direction | Calls | Bytes | Activity time |
| --- | ---: | ---: | ---: |
| Device-to-Host | 347 | 44,044 B | 0.844 ms |
| Host-to-Device | 1,040 | 56,448 B | 0.821 ms |
| Device-to-Device | 360 | 17,260,736 B | 0.476 ms |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 20.122 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.474 ms | 2,920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 11.109 ms | 795 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 8.978 ms | 344 |
| `dsv4_hybrid_attention_kernel` | 7.394 ms | 328 |
| `dsv4_route_kernel` | 5.660 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.500 ms | 688 |
| `ncclDevKernel_SendRecv` | 4.876 ms | 344 |
| `dsv4_csa_select_kernel` | 4.124 ms | 168 |

## Interpretation

The slow path is not the sampler, and it is not a full-prefill or missing-KV
failure. The isolated second-token decode wave is dominated by:

- MoE return-side combine: reduce-scatter costs 20.122 ms per rank range.
- Local expert compute: split FP8/FP4 per-expert GEMV costs 22.583 ms per rank
  range before scatter/activation overhead.
- Runtime granularity: 16,177 CUDA kernel launches plus allocator, memset,
  event, and stream-wait API work are visible in the decode range.
- Host synchronization: 347 D2H calls remain. The main code path is the
  per-layer local expert count readback used to build host offsets before the
  local expert loop. The actual D2H payload is only 44,044 bytes in the decode
  window, so the visible runtime cost is call/synchronization overhead rather
  than transfer bandwidth.
- Attention is real but secondary at this context size: hybrid attention is
  7.394 ms per rank range, MHC parameter kernels are 5.500 ms, and CSA select is
  4.124 ms.

The next performance target remains the same concrete stack: remove or batch
the local-count host sync, replace per-expert GEMV with real grouped
GEMM/DeepGEMM, reduce CUDA launch/runtime granularity, and overlap or fuse the
DeepEP combine path.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
