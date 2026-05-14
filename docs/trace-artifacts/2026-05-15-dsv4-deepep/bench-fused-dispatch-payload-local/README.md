# DSv4 Fused Dispatch Payload HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20 after enabling the default BF16 fused dispatch payload for
B=1 DeepEP decode.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Case | TTFT | Total | Post-first decode | Output check |
| --- | ---: | ---: | ---: | --- |
| `warmup16` | 0.384 s | 1.572 s | 12.62 tok/s | normal Chinese sentence |
| `decode64` | 0.450 s | 5.603 s | 12.22 tok/s | normal English paragraph |
| `math` | 0.395 s | 0.481 s | n/a | `410` |

The `decode64` case keeps the same normal content used in earlier DSv4 decode
smokes and improves over the previous uninitialized-scratch default run
recorded at 11.99 post-first tok/s. The arithmetic case confirms the streaming
decoder path still returns the exact answer `410`.

Artifacts:

- `summary.json`
- `server.log.gz`
- `run_decode_case.py`
