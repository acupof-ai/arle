# DSv4 Route-Grouped Pair HTTP Comparison

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20. This is a trace-off HTTP comparison between the default fused-dispatch DeepEP decode path and the opt-in route-wise grouped pair GEMV path.

Default:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

Route-grouped pair:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

Throughput below comes from the server `request_trace` entries in `server-trace-summary.json`.

| Case | Default | Route-grouped pair | Output check |
| --- | ---: | ---: | --- |
| `warmup16` | 10.26 tok/s | 6.24 tok/s | same normal Chinese sentence |
| `decode64` | 11.47 tok/s | 6.54 tok/s | normal English paragraph |
| `math` | `410` | `410` | exact arithmetic answer |

The route-grouped pair path remains default-off. It can look competitive in an isolated single-token nsys wave, but trace-off decode throughput regresses sharply once the full HTTP request and repeated decode steps are measured. This confirms the next optimization target is still true grouped GEMM/DeepGEMM plus DeepEP overlap, not the current route-wise grouped GEMV fallback.

Artifacts:

- `summary.json`
- `server-trace-summary.json`
- `default.server.log.gz`
- `route_grouped_pair.server.log.gz`
- `run_cases.py`
