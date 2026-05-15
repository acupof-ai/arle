# DSv4 Incremental Attention Scratch HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint on 8xH20 after reusing incremental attention scratch buffers for prepared Q/K, local attention output, and the `wo_a` latent projection.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Case | Status | TTFT | Post-first tok/s | Output check |
| --- | ---: | ---: | ---: | --- |
| `warmup16` | 200 | 370 ms | 12.73 | normal Chinese explanation |
| `decode64` | 200 | 732 ms | 12.08 | normal English continuation |
| `math` | 200 | 453 ms | n/a | exact `410` |
| `writing` | 200 | 504 ms | 12.72 | normal Chinese writing |

The trace-off HTTP path is not a strict throughput win; prompt/output shape and metric definitions differ from the prior compressor-scratch smoke. The point of this change is allocator-pressure cleanup in the decode window, validated by the paired Nsight artifact.

Artifacts:

- `summary.json`
- `client.log`
- `run_cases.py`
- `server.log.gz`
