# DSv4 prefill wq_b → DeepGEMM: −47% prefill_ms at M=1024 (the 62%-of-prefill lever)

## Context

The P/D nsys breakdown (`2026-06-07-dsv4-pd-nsys-real-breakdown.md`) pinned the scalar
`dsv4_fp8_gemv_batch[_tiled]` as the #1 self-written bottleneck on BOTH sides — 62% of
mla_attn prefill and the #1 decode GPU kernel. Lever #1 moved the DECODE wq_b/wo to
DeepGEMM; the PREFILL projections were still on the scalar GEMV. `fp8_gemv` is a decode
(M=1) kernel — it scales ~O(M), so it is catastrophic for long prefill.

## What worked

Added `prefill_proj_deepgemm` (the M=token_count analogue of `decode_proj_deepgemm`,
mirroring `run_fused_wqkv_prefill`'s quantize+GEMM, reusing the prefill FP8 scratch
since K=q_lora_rank ≤ hidden_dim) and routed the prefill wq_b through it. Gated
`ARLE_DSV4_PREFILL_PROJ_DEEPGEMM` (flipped default ON after this A/B).

8×H20 TP=8, same binary, `ARLE_DSV4_PREFILL_PROJ_DEEPGEMM` 0 vs 1:

| prompt | prefill_ms DG=0 (scalar) | prefill_ms DG=1 (DeepGEMM) | Δ |
|---|---:|---:|---:|
| needle (37 tok) | 7996 | 7877 | ~flat (small M) |
| **long (1024 tok)** | **14382** | **7628** | **−47%** |

**Correctness PASS:** the needle output is **byte-identical** DG=0 vs DG=1
(`[223,30793,929,16,19018,436,7681,16]`) — answer retrieved exactly. (The 1024-tok
prompt is a degenerate 4-token repeat; its continuation tail divergence is MoE
non-determinism, per `feedback_correct_inference_not_baseline_identity` — perf-only
prompt, gated on the needle.)

## Rule

- The scalar `fp8_gemv` is a DECODE (M=1) kernel; never leave it on the prefill path —
  it scales O(M) and dominates long prefill. Route every prefill projection through
  DeepGEMM (M=token_count). `prefill_proj_deepgemm` is the reusable helper; wo and the
  DSA indexer wq_b are next under the same flag.
- Gate prefill perf on a LONG prompt (M≫1); a 37-token A/B shows ~nothing because
  DeepGEMM's tensor-core advantage over the scalar GEMV grows with M.
