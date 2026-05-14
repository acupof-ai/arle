# DSv4 Route-Grouped Expert Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20. The service used the DSv4 DeepEP path with FP8 KV and
the opt-in route-grouped local expert experiment:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The profiled streaming request used `max_tokens=2` for prompt
`用两个字形容彩虹。` and returned `霓彩`. As with the baseline trace,
`max_tokens=2` is intentional because `max_tokens=1` completes from prefill
and does not create a real `step_decode_kernel_launch` range.

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 145.669 ms |
| Per-rank decode range p50 | 144.992 ms |
| Returned text | `霓彩` |

Top decode-window costs:

| Item | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 57.430 ms | 1008 |
| `dsv4_fp4_route_gemv_batch_kernel` | 35.895 ms | 1008 |
| `cuMemAllocAsync` | 36.081 ms | 11200 |
| `cudaLaunchKernel_v7000` | 24.179 ms | 15127 |
| `cuMemFreeAsync` | 18.028 ms | 6048 |
| `cuMemsetD8Async` | 15.045 ms | 9472 |

## Conclusion

The route-grouped experiment removes the local-count D2H readback from the
top decode-window runtime API list, but it is not a win. The padded B=1 shape
launches route-wise FP4 GEMV across fixed `ep_world * topk` slots, and that
single kernel family costs 35.895 ms per rank range. The decode wave regresses
from the current 125.497 ms baseline to 145.669 ms.

Keep `ARLE_DSV4_ROUTE_GROUPED_EXPERTS` default-off. This confirms that simply
moving metadata device-side is not enough; the next compute path needs real
grouped GEMM/DeepGEMM and DeepEP overlap rather than route-wise GEMV.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
