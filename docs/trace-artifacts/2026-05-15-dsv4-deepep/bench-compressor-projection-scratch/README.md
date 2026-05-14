# DSv4 Compressor Projection Scratch HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after reusing the GPU compressor update `kv_raw` and `score_raw` projection buffers.

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
| `warmup16` | 200 | 377 ms | 10.28 | normal Chinese text |
| `decode64` | 200 | 459 ms | 11.47 | normal English text |
| `math` | 200 | 404 ms | 33.16 | exact `410` |

This is effectively flat versus `bench-stream-recycle/` (`decode64` 11.48 tok/s). The change is an allocator-call cleanup, not an HTTP throughput win.

Artifacts:

- `summary.json`
- `client.log`
- `server.log.gz`
- `run_cases.py`
