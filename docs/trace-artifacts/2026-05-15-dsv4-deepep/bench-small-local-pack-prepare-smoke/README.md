# DSv4 Small Local Pack Prepare HTTP Smoke

This trace-off HTTP smoke validates the B=1 padded DeepEP local expert prepare
cleanup on the real 8xH20 `/root/DeepSeek-V4-Flash` checkpoint with FP8 KV.

Environment highlights:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_COMBINE_REDUCE_SCATTER=1
ARLE_DSV4_COMBINE_OVERLAP=0
ARLE_DSV4_ROUTE_GROUPED_EXPERTS=0
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

Results:

| Case | Result |
| --- | ---: |
| `warmup16` | 12.79 post-first tok/s |
| `decode64` | 12.05 post-first tok/s |
| `math` | `410` |
| `writing` | Normal Chinese release-note text |

The smoke confirms the fused local prepare kernel keeps multi-token streaming
content correct and keeps decode throughput at the previous default-path level.
