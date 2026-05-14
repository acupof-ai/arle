# DSv4 Dispatch Scratch Nsight Trace

Date: 2026-05-14

Remote workspace: `/root/arle-perf-dispatch-scratch`

Raw remote artifacts:

- `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-dispatch-scratch/arle-dsv4-dispatch-scratch.nsys-rep`
- `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-dispatch-scratch/arle-dsv4-dispatch-scratch.sqlite`

The local tree commits only parsed summaries and CSVs. The raw trace remains in
the remote repo workspace.

## Run Shape

- DeepEP default MoE path.
- Incremental KV enabled.
- FP8 KV cache.
- `ARLE_DSV4_GROUPED_EXPERTS` unset.
- Per-layer MoE dispatch scratch reuse enabled by default.
- 8 H20 ranks.
- Warmed HTTP request with `max_tokens=8`.

Profiled request:

| Metric | Value |
| --- | ---: |
| Prompt tokens | 14 |
| Completion tokens | 8 |
| End-to-end latency | 2.798 s |
| Output | `霓灯吻碎江，夜城` |

## Runtime API Delta

Normalized by 8 completion tokens and 8 ranks:

| API | Before rerun | Dispatch scratch | Delta |
| --- | ---: | ---: | ---: |
| `cuStreamSynchronize` | 158.833 ms | 160.094 ms | +0.8% |
| `cuMemFreeAsync` | 44.297 ms | 39.001 ms | -12.0% |
| `cuMemAllocAsync` | 28.656 ms | 28.796 ms | +0.5% |
| `cudaLaunchKernel` | 19.916 ms | 23.581 ms | +18.4% |
| `cuMemsetD8Async` | 17.388 ms | 20.469 ms | +17.7% |

The allocation/free call count dropped from 136,825 to 111,531 in the profiled
window. This confirms the scratch reuse removes a measurable part of allocator
churn, while the remaining wall time is still dominated by synchronization and
communication boundaries.

## Layer Trace

Trace-only run without Nsight capture, last profiled 8-token request:

| Phase | p50 | p95 |
| --- | ---: | ---: |
| `ffn_deepep_dispatch_combine` | 1.552 ms | 2.004 ms |
| `ffn_total` | 2.079 ms | 2.557 ms |
| `ffn_deepep_local_experts` | 0.439 ms | 0.836 ms |
| `ffn_deepep_combine_exchange` | 0.485 ms | 1.231 ms |
| `ffn_deepep_dispatch` | 0.064 ms | 0.119 ms |
| `attn_total` | 1.124 ms | 1.543 ms |

## Conclusion

Per-layer dispatch scratch reuse is a real cleanup for the current DeepEP path:
it reduces allocator churn and brings the traced FFN/MoE decode phases back
toward the earlier scratch-reuse baseline. It does not finish the high
performance route. The remaining work is still real grouped GEMM/DeepGEMM for
local experts plus a DeepEP-style combine path that overlaps or reduces the
return-side send/recv boundary.
