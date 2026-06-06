# DSv4 decode — residual GEMV → DeepGEMM fusion (#2 decode lever, 14.4%)

**Date:** 2026-06-06. **Status:** execution-ready design (parallel prep; touches
`attention.rs` → IMPLEMENT after the EAGLE rollback merges). **Goal:** the second
decode kernel bucket after fused-wqkv. The clean decode profile
([`2026-06-06-dsv4-decode-6ms-remaining-levers.md`](2026-06-06-dsv4-decode-6ms-remaining-levers.md))
pins the residual scalar `dsv4_fp8_gemv_batch_kernel` at **14.4%** — this is the
`wo` o-projection (+ the compressor `wkv`/`wgate` GEMVs) that fused-wqkv (#9, the
`wq_a|wkv_a` fusion) left on the scalar path.

## The lever

fused-wqkv proved the pattern at decode (+18.4%): replace a scalar
`dsv4_fp8_gemv_batch_kernel` GEMV with an FP8 DeepGEMM (tensor-core). The residual
14.4% is the SAME scalar kernel on the remaining MLA projections — most of it the
`wo` output projection (`dsv4/linear/wo` nvtx range), a `[local_heads*head_dim →
hidden]` FP8 GEMV at B=1. Route `wo` (and the compressor `wkv`/`wgate` if shapes
allow) through the DeepGEMM FP8 path the fused-wqkv scratch already uses.

## Exact changes

1. **Locate** the `wo` projection call (`dsv4_linear(ctx, &attention.wo_*, ...)`,
   nvtx `dsv4/linear/wo`) on the decode path. It is `WeightFormat::Dsv4Fp8BlockScaled`
   → `dsv4_fp8_gemv_batch_cuda` today.
2. **Add a fused `wo` DeepGEMM** mirroring `run_fused_wqkv_decode`'s DeepGEMM call
   (the FP8 block-scaled GEMM with the act-quantize → grouped-GEMM pattern). At
   B=1 the M=1 grouped GEMM is the same shape family already used for `wqkv_a`.
   Reuse / extend the `Dsv4FusedWqkvDecodeScratch` (or a sibling scratch) for the
   `wo` act-fp8 + scales + output buffers; allocate once per slot.
3. **Gate + opt-out** (`ARLE_DSV4_FUSED_WO_DECODE`, default decided by the A/B) —
   same flag discipline as fused-wqkv.

## Risk & gate

Lower risk — proven pattern (fused-wqkv), same DeepGEMM kernel, same B=1 decode
harness. Gate: (1) token-exact / needle (DeepGEMM vs scalar FP8 float order may
flip near-ties — gate on needle, not strict byte-identity); (2) **same-binary
env-A/B** 64-tok decode, TP=8/EP=8: the 14.4% scalar bucket should shrink toward
the DeepGEMM cost. License on the wall-clock tok/s delta.

## Ordering vs the other levers

After the EAGLE rollback lands (it touches `attention.rs`), the fire-ready
decode/prefill levers are: (a) **this** residual `wo` fusion (14.4%, proven,
lowest-risk decode); (b) prefill fused-wqkv extension
([`2026-06-06-dsv4-prefill-fused-wqkv-extend.md`](2026-06-06-dsv4-prefill-fused-wqkv-extend.md),
22.8% prefill). Both mirror the +18.4% fused-wqkv win. mhc-fuse (12.2%, needs the
TileLang f32-mma fix) and the s_q=K verify (the EAGLE perf tranche, post-rollback)
are higher-risk and come after. B=1 decode is GPU-bound, so each fusion's win is
real compute removed, not overhead — but it must clear a wall-clock A/B, not the
kernel-table % ([[feedback_b1_decode_gpu_bound_overhead_removal_wash]]).
