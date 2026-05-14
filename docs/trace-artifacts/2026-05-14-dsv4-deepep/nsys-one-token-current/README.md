# DSv4 single-token nsys trace

Captured on 2026-05-14 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. The remote source tree was refreshed from local `c89d3457`, rebuilt
with `cargo build --release -p infer --bin infer --features cuda,nccl`, and the
service was launched under `nsys profile` with CUDA profiler API start/stop.

The profiled HTTP request used streaming `max_tokens=2`, `ignore_eos=true`, and
returned two completion tokens: `霓灯`. The Nsight decode filter found one
`step_decode_kernel_launch` wave across 8 rank threads, so the tables below are
for a single generated decode token, normalized per decode range/rank.

## Result

| Metric | Value |
| --- | ---: |
| Prompt tokens | 23 |
| Completion tokens | 2 |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 266.020 ms |
| Per-rank decode range p50 | 265.759 ms |

Top CUDA runtime API time inside the single decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuStreamSynchronize` | 97.863 ms | 21616 |
| `cuMemFreeAsync` | 38.412 ms | 11988 |
| `cuMemAllocAsync` | 23.346 ms | 11980 |
| `cudaLaunchKernel_v7000` | 20.081 ms | 15080 |
| `cuMemsetD8Async` | 16.180 ms | 13050 |
| `cuMemcpyDtoHAsync_v2` | 7.838 ms | 883 |

Top CUDA kernels inside the same decode-token ranges:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 30.801 ms | 1032 |
| `dsv4_fp8_gemv_batch_kernel` | 11.469 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.881 ms | 774 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 8.166 ms | 344 |
| `dsv4_hybrid_attention_kernel` | 7.825 ms | 328 |
| `ncclDevKernel_AllGather_RING_LL` | 6.294 ms | 344 |

Layer trace medians from the same run:

| Phase | p50 | p95 |
| --- | ---: | ---: |
| `ffn_total` | 2.301 ms | 3.757 ms |
| `ffn_deepep_dispatch_combine` | 1.750 ms | 2.709 ms |
| `ffn_deepep_combine` | 0.593 ms | 1.427 ms |
| `ffn_deepep_combine_exchange` | 0.498 ms | 1.303 ms |
| `ffn_deepep_local_experts` | 0.458 ms | 0.923 ms |
| `attn_total` | 1.346 ms | 6.911 ms |

## Bottleneck

The slow single token is not a sampler issue and not a missing-KV issue.
`dsv4_hybrid_attention_kernel` is present, but it is smaller than MoE/NCCL and
host-runtime overhead. The concrete bottleneck stack for B=1 decode is:

1. host-side synchronization and stream waits (`cuStreamSynchronize`);
2. temporary allocation/free churn (`cuMemFreeAsync`/`cuMemAllocAsync`);
3. many small kernel launches and memsets;
4. DeepEP-style return-side send/recv plus remaining all-gather/all-reduce;
5. per-expert FP8/FP4 GEMV work, which still needs real grouped GEMM/DeepGEMM.

Raw `.nsys-rep` and `.sqlite` stay on the remote host under:

`/root/arle-nsys-one-token-c89d3457/docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-one-token-current/`

Committed local artifacts include `summary.json`, decode-only CSVs, nsys stats
text, request profiles, command/env logs, and compressed `server.log.gz`.
