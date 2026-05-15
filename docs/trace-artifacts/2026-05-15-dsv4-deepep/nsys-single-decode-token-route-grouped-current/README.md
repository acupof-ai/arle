# DSv4 Current Route-Grouped Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20 with `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. This is the current opt-in
route-wise grouped expert path before moving grouped expert weight pointer
tables to layer-load-time caches.

```text
prompt: Compute 137 + 269. Answer with the number only.
output: 406
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 105.808 ms |
| Per-rank decode range p50 | 104.476 ms |
| Profile request wall time | 3.500 s |
| Returned text | `406` |

Top decode-window costs:

| Bucket | Value |
| --- | ---: |
| `cudaLaunchKernel_v7000` | 30.015 ms / 15,126 calls |
| `cuMemAllocAsync` | 12.186 ms / 8,480 calls |
| `cuMemcpyHtoDAsync_v2` | 5.490 ms / 2,760 calls |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 47.613 ms / 232 calls |
| `dsv4_fp4_route_gemv_pair_batch_kernel` | 16.253 ms / 235 calls |
| `dsv4_fp4_route_gemv_batch_kernel` | 8.700 ms / 235 calls |
| `dsv4_fp8_gemv_batch_kernel` | 7.852 ms / 2,008 calls |

Memcpy activity confirms the route-grouped path removes decode-window D2H, but
it pays many small H2D pointer-table copies:

| Direction | Calls | Bytes | Activity time |
| --- | ---: | ---: | ---: |
| Host-to-Device | 1,918 | 374,752 B | 1.950 ms |
| Device-to-Device | 240 | 11,813,760 B | 0.315 ms |

This path is kept as an opt-in diagnostic. Removing the local-count D2H by
itself is not enough; route-wise GEMV and reduce-scatter timing still dominate.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
