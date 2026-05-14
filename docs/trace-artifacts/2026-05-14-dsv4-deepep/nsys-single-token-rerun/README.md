# DSv4 Nsight Decode Rerun

Date: 2026-05-14

Remote workspace: `/root/arle-perf-main`

Raw remote artifacts:

- `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-single-token-rerun/arle-dsv4-decode-rerun.nsys-rep`
- `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-single-token-rerun/arle-dsv4-decode-rerun.sqlite`

The local tree commits only the small parsed outputs. The raw trace is kept in
the remote repo workspace rather than `/tmp`.

## Run Shape

- DeepEP default MoE path.
- Incremental KV enabled.
- FP8 KV cache.
- `ARLE_DSV4_GROUPED_EXPERTS` unset.
- 8 H20 ranks.
- Warmed HTTP request with `max_tokens=8`.

Profiled request:

| Metric | Value |
| --- | ---: |
| Prompt tokens | 14 |
| Completion tokens | 8 |
| End-to-end latency | 2.705 s |
| Output | `霓灯吻碎江，夜城` |

## Top CUDA Runtime API Cost

Normalized by 8 completion tokens and 8 ranks:

| API | ms/token/rank |
| --- | ---: |
| `cuStreamSynchronize` | 158.833 |
| `cuMemFreeAsync` | 44.297 |
| `cuMemAllocAsync` | 28.656 |
| `cudaLaunchKernel` | 19.916 |
| `cuMemsetD8Async` | 17.388 |
| `cuMemcpyDtoHAsync_v2` | 7.267 |

## Top Kernel Cost

Normalized by 8 completion tokens and 8 ranks:

| Kernel | ms/token/rank |
| --- | ---: |
| `ncclDevKernel_SendRecv` | 37.338 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 22.057 |
| `dsv4_fp8_gemv_batch_kernel` | 10.037 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 7.866 |
| `dsv4_hybrid_attention_kernel` | 7.009 |
| `ncclDevKernel_AllGather_RING_LL` | 5.718 |

## Layer Trace

Last profiled request, `tokens=1` rows only:

| Phase | p50 | p95 |
| --- | ---: | ---: |
| `ffn_total` | 3.580 ms | 4.121 ms |
| `ffn_deepep_dispatch_combine` | 2.751 ms | 3.318 ms |
| `attn_total` | 1.555 ms | 2.232 ms |
| `attn_core` | 1.206 ms | 1.862 ms |
| `ffn_deepep_combine` | 0.797 ms | 1.526 ms |
| `ffn_deepep_local_experts` | 0.526 ms | 0.966 ms |

## Conclusion

The single-token decode bottleneck is not missing KV-cache reads. The attention
kernel is present, but it is smaller than the synchronization, allocation/free,
launch/memset, and small NCCL boundary costs. The next decode optimizations
need to remove dynamic allocations, reduce stream synchronizations, make the
MoE route exchange/combine path graph- or overlap-friendly, and replace the
per-expert GEMV style work with real grouped GEMM/DeepGEMM.
