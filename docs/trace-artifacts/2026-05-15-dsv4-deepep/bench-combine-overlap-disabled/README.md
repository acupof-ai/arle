# DSv4 Combine Overlap Disabled HTTP Smoke

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run uses the combine-overlap-capable binary with the overlap
path disabled:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=1
ARLE_DSV4_COMBINE_REDUCE_SCATTER=1
ARLE_DSV4_COMBINE_OVERLAP=0
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

## Result

| Case | Status | Output check | Timing |
| --- | ---: | --- | ---: |
| `decode64` | 200 | normal English sequence | 12.05 post-first tok/s |
| `prefill1k` | 200 | returned `One` | 15.419 s TTFT |
| `prefill4k` | OOM | `mem_fraction_static=0.10` profile exhausted memory | n/a |

This is a partial smoke artifact. The `prefill4k` case hit the known low
static-memory profile limit, so the run was manually stopped after collecting
the successful decode and 1k-prefill records. It is kept here to show that
the default-off overlap binary still matches the current decode throughput
baseline while returning normal text.

Artifacts:

- `client.log`
- `server.log.gz`
- `models.json`
- `summary.json`
