# DSv4 Stream Recycle HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after adding incremental stream scratch recycling.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Case | Status | TTFT | E2E requested tok/s | Output check |
| --- | ---: | ---: | ---: | --- |
| `warmup16` | 200 | 374 ms | 10.29 | normal Chinese text |
| `decode64` | 200 | 463 ms | 11.48 | normal English text |
| `math` | 200 | 395 ms | 33.76 | exact `410` |

Compared with the default trace-off run in `bench-route-grouped-pair-vs-default/`, the `decode64` e2e throughput is effectively unchanged: 11.47 tok/s before, 11.48 tok/s after. Stream recycling helps the isolated nsys decode window, but it does not move the HTTP throughput bottleneck by itself.

Artifacts:

- `summary.json`
- `client.log`
- `server.log.gz`
- `run_cases.py`
