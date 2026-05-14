# DSv4 Pair GEMV Decode-Only Benchmark

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20 after commit `626477b1`. The benchmark compares the
default split expert GEMV path with the gated single-expert gate/up fusion:

```text
ARLE_DSV4_MOE_BACKEND=deepep
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
--kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

The `pairgemv` run additionally sets:

```text
ARLE_DSV4_PAIR_EXPERT_GEMV=1
```

Each service was warmed with a 16-token streaming decode request, then measured
with a 64-token streaming decode request and a short arithmetic correctness
request.

## Result

| Case | TTFT | Total | Post-first decode | Output check |
| --- | ---: | ---: | ---: | --- |
| Default `warmup16` | 0.533 s | 1.767 s | 12.16 tok/s | normal sequence |
| Default `decode64` | 0.718 s | 6.060 s | 11.79 tok/s | normal sequence |
| Default `math` | 0.434 s | 0.517 s | n/a | `410` |
| Pair GEMV `warmup16` | 1.464 s | 3.382 s | 7.82 tok/s | normal sequence |
| Pair GEMV `decode64` | 1.987 s | 10.170 s | 7.70 tok/s | normal sequence |
| Pair GEMV `math` | 1.128 s | 1.254 s | n/a | `410` |

## Conclusion

`ARLE_DSV4_PAIR_EXPERT_GEMV=1` is functionally correct but slower on this B=1
decode shape. The default split GEMV path remains faster, so the pair GEMV
experiment stays opt-in. The compute target remains real grouped GEMM/DeepGEMM
and DeepEP overlap rather than fusing only one local expert's `w1`/`w3` GEMV.

Artifacts:

- `default/summary.json`
- `default/request_trace.log`
- `default/server.log.gz`
- `pairgemv/summary.json`
- `pairgemv/request_trace.log`
- `pairgemv/server.log.gz`
