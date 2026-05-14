# DSv4 AllGather Count Readback Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run keeps the default DeepEP path and profiles the same
streaming `max_tokens=2` request:

```text
prompt: 请写两个汉字，要求意象偏城市夜景。
output: 霓虹
```

The change moves the default AllGather count exchange before route packing and
uses the single all-rank count readback to derive both send and receive counts.
That removes the separate 32-byte `send_rank_counts` host readback in each
layer/rank. The `ARLE_DSV4_COUNT_EXCHANGE=sendrecv` fallback still uses the old
send-count readback because it does not have the all-rank matrix available.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 129.768 ms |
| Per-rank decode range p50 | 129.550 ms |
| Request wall time | 1.384 s |
| Returned text | `霓虹` |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cudaLaunchKernel_v7000` | 20.913 ms | 15088 |
| `cuMemAllocAsync` | 20.123 ms | 7760 |
| `cuMemcpyDtoHAsync_v2` | 18.106 ms | 543 |
| `cuMemFreeAsync` | 15.645 ms | 6048 |
| `cuMemsetD8Async` | 11.759 ms | 8838 |
| `cudaEventRecord_v3020` | 3.927 ms | 4136 |
| `cudaStreamWaitEvent_v3020` | 3.415 ms | 3440 |
| `cuLaunchKernelEx` | 2.629 ms | 1720 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 26.631 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 16.104 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.469 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.855 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.297 ms | 328 |
| `dsv4_route_kernel` | 5.658 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.501 ms | 688 |
| `dsv4_csa_select_kernel` | 4.138 ms | 168 |

## Comparison

Compared with [`../nsys-single-token-hidden-scratch/`](../nsys-single-token-hidden-scratch/):

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 135.390 ms | 129.768 ms |
| Per-rank decode range p50 | 135.104 ms | 129.550 ms |
| `cuMemcpyDtoHAsync_v2` calls | 887 | 543 |
| 32-byte D2H count calls | 344 | 0 |
| 256-byte D2H count calls | 344 | 344 |
| 128-byte D2H local-count calls | 199 | 199 |

The removed 32-byte readbacks were expensive in CUDA runtime time in the prior
trace: 344 calls accounted for 108.352 ms total runtime API time across the 8
decode ranges. After the change, the remaining D2H cost is concentrated in the
256-byte all-rank count matrix readback: 344 calls, 136.692 ms total runtime
API time. This makes the next concrete target device-side count prefix/sizing
or an exchange path that can avoid host count materialization before dispatch.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
