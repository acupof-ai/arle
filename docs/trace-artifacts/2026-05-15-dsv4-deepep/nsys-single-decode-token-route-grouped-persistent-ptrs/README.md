# DSv4 Route-Grouped Persistent Pointer Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20 with `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. This run validates moving
grouped expert weight/scale pointer tables from request scratch into
`DeepseekV4MoeBlock` load-time caches, which is also the pointer-table shape
needed by a future raw-pointer DeepGEMM/true grouped GEMM path.

```text
prompt: Compute 137 + 269. Answer with the number only.
output: 406
```

## Result

| Metric | Before | After |
| --- | ---: | ---: |
| Decode wave wall time | 105.808 ms | 94.828 ms |
| Per-rank decode range p50 | 104.476 ms | 93.710 ms |
| Profile request wall time | 3.500 s | 1.210 s |
| `cuMemcpyHtoDAsync_v2` runtime | 5.490 ms / 2,760 calls | 1.380 ms / 696 calls |
| H2D activity | 374,752 B / 1,918 calls | 7,808 B / 440 calls |
| `cuMemAllocAsync` | 12.186 ms / 8,480 calls | 8.193 ms / 6,416 calls |

Top CUDA kernels after the change:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 42.828 ms | 216 |
| `dsv4_fp4_route_gemv_pair_batch_kernel` | 15.121 ms | 216 |
| `dsv4_fp4_route_gemv_batch_kernel` | 8.085 ms | 216 |
| `dsv4_fp8_gemv_batch_kernel` | 7.071 ms | 1,808 |
| `dsv4_hybrid_attention_kernel` | 4.527 ms | 200 |
| `dsv4_mhc_params_kernel` | 3.456 ms | 432 |
| `dsv4_route_kernel` | 3.451 ms | 216 |

## Interpretation

The optimization removes the repeated 256-byte grouped expert pointer-table
copies from the decode window and materially reduces route-grouped overhead.
It does not make route-wise GEMV the default path. The remaining top stack is
still reduce-scatter combine plus route-wise FP4/FP8 GEMV, so the production
target remains true grouped GEMM/DeepGEMM with DeepEP overlap/fusion.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
