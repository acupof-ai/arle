# DSv4 decode lever #1a: wq_b projection scalar GEMV → tensor-core DeepGEMM (+2.5%)

## Context

The clean decode nsys breakdown ([`2026-06-07-dsv4-decode-nsys-real-breakdown.md`](2026-06-07-dsv4-decode-nsys-real-breakdown.md))
pinned the #1 decode GPU kernel as the scalar projection GEMV `dsv4_fp8_gemv_batch`
(3.62 ms/token, 18.9% GPU, <1% tensor pipe). In the fastest variant
`flashmla_fused_wqkv`, `fused_wqkv` already moved `wq_a|wkv` to DeepGEMM (+18.4%);
the residual is **wq_b + wo_a + wo_b** (task #36's "剩 3/4 未融"). The in-tree
hand-rolled MMA GEMV (`ARLE_DSV4_FP8_GEMV_MMA`) was **killed** — broken (all-zero)
+ slower ([`2026-06-07-dsv4-mma-fp8-gemv-killed.md`](2026-06-07-dsv4-mma-fp8-gemv-killed.md)) —
so the lever goes through DeepGEMM, the proven `fused_wqkv` path.

## What worked

`wq_b` (the Q LoRA up-projection, M=1 at decode) now routes through tensor-core
DeepGEMM:
- Load: `wq_b_deepgemm = Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(wq_b)` (the
  single-weight DeepGEMM block-scale repack), under the fused-wqkv alloc gate.
- Decode (`run_fused_wqkv_decode`): quantize `c_q_normed` (K=q_lora_rank ≤
  hidden_dim, so the fused FP8 scratch — already consumed by the wq_a|wkv GEMM on
  the same stream — is reused) + `dsv4_deepgemm_fp8_gemm_nt`, gated
  `ARLE_DSV4_DECODE_PROJ_DEEPGEMM` (now default ON).

**A/B (8×H20 TP=8, `flashmla_fused_wqkv`, same binary, two processes, reproduced ×2):**

| `ARLE_DSV4_DECODE_PROJ_DEEPGEMM` | steady tok/s | Δ |
|---|---:|---:|
| `0` (scalar) | 38.19 / 38.32 | — |
| `1` (DeepGEMM wq_b) | 39.17 / 39.28 | **+2.5%** |

**Correctness (37-tok needle, passcode 73914 = `[223,30793,929]`):** both scalar and
DeepGEMM emit `[223, 30793, 929, 16, …]` — the answer retrieved **bit-identically**
(first 4 tokens); DeepGEMM diverges only at idx4 in the free-continuation tail
(legitimate FP8 accumulation numerics, same signature as the batched all-reduce, NOT
a bug). Gate on needle retrieval, not byte-parity.

## Rule

- Decode projection GEMV → DeepGEMM is a per-projection win, not just `fused_wqkv`:
  `from_dsv4_weight` builds a single-weight DeepGEMM cache; reuse the fused FP8 input
  scratch when K ≤ hidden_dim. +2.5% for wq_b alone (1/3 of the 3.62ms residual).
- Next (lever #1b): `wo_a`/`wo_b`. wo_a's K = local_width (4096–8192) can **exceed**
  hidden_dim, so wo needs a **dedicated** decode FP8 scratch in state (the fused
  scratch is sized for hidden_dim) — more than a scratch reuse. Projected ~+5% more.
- License via the needle (determined-answer retrieval) + same-load reproduced A/B,
  never byte-parity (FP8 tensor-core ≠ scalar bit-exact, legitimately).
