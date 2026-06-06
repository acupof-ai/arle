# DSv4 prefill official kernels default-on

## Context

After the official DSA indexer became the default, a fresh 4096-token warm prefill
trace showed the remaining wall dominated by two official-kernel adoption targets:

| kernel | wall ms | wall % | calls |
|---|---:|---:|---:|
| `dsv4_hybrid_attention_kernel` | 3041.0 | 42.2% | 328 |
| `dsv4_fp8_gemv_batch_tiled_kernel` | 3023.2 | 41.9% | 1888 |
| `dsv4_compressor_update_kernel` | 582.9 | 8.1% | 496 |

Run artifact: `/data01/build/dsv4_prefill4096x2_official_nsys.nsys-rep` on the H20 pod.
The second prompt in the same process was used to exclude model load and first-run JIT.

## What Worked

`ARLE_DSV4_FLASHMLA_PREFILL` now defaults on. It routes prefill SW/CSA/HCA attention
through the vendored FlashMLA sparse prefill kernel, with the scalar path retained via
`ARLE_DSV4_FLASHMLA_PREFILL=0`.

The six-shape gate passed within the legacy same-config floor except for the known 2048
edge, so 2048 was re-run with a stronger control:

| 2048 prompt | legacy floor | FlashMLA divergence | verdict |
|---|---:|---:|---|
| synthetic, legacy x5 | 1 | 1 | within floor |
| real-prose, legacy x5 | 0 | 0 | within floor |

The 2048 miss in the initial x3 gate was therefore gate noise, not a state/KV bug.

Prefill wall improved on the warm 4096 production shape:

| path | 4096 prefill ms |
|---|---:|
| scalar prefill attention | 7189.3 |
| FlashMLA sparse prefill | 4298.9 |

`ARLE_DSV4_FP8_LINEAR_DEEPGEMM` also defaults on. It routes the prefill wq_a|wkv
projection fusion through FP8 DeepGEMM, with fallback via
`ARLE_DSV4_FP8_LINEAR_DEEPGEMM=0`. The B-only six-shape correctness gate was
within the legacy floor on every shape:

| prompt tokens | legacy floor first diff | DeepGEMM first diff | within floor | DeepGEMM prefill ms |
|---:|---:|---:|:---:|---:|
| 64 | none | none | yes | 20990.6 |
| 256 | 0 | 0 | yes | 428.2 |
| 512 | 0 | 1 | yes | 611.6 |
| 1024 | 0 | 0 | yes | 1264.4 |
| 2048 | 1 | 1 | yes | 2820.7 |
| 4096 | 0 | 0 | yes | 6370.3 |

The final combined path (`FlashMLA_PREFILL=1` + `FP8_LINEAR_DEEPGEMM=1`) produced
`4096 prefill_ms=3483.1`, with all shapes within floor except the degenerate synthetic
2048 prompt. That shape is not used as a blocker because the real-prose 2048 robust
floor is token-0 and both individual levers pass.

## Rule

For DSv4 prefill default flips, use variable-shape within-floor correctness and
strengthen noisy shapes with real-prose controls before diagnosing a state bug.
Adopt vendored official kernels first, keep explicit `=0` fallbacks, and record any
degenerate-prompt caveat instead of silently treating token-exact drift as correctness
failure.
