# DSv4 MoE Scratch Uninit HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20 without Nsight overhead. This verifies the additional MoE scratch
uninitialized allocation change with multi-token streaming output.

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_COMBINE_REDUCE_SCATTER=1
ARLE_DSV4_COMBINE_OVERLAP=0
ARLE_DSV4_ROUTE_GROUPED_EXPERTS=0
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Case | Status | Output check | Post-first decode |
| --- | ---: | --- | ---: |
| `warmup16` | 200 | normal Chinese explanation | 12.75 tok/s |
| `decode64` | 200 | normal English sequence | 12.06 tok/s |
| `math` | 200 | exact `410` | n/a |
| `writing` | 200 | normal Chinese release note | 12.70 tok/s |

Artifacts:

- `summary.json`
- `client.log`
- `server.log.gz`
- `models.json`
- `command.txt`
