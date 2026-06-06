# DSv4 prefill perf — extend fused-wqkv (DeepGEMM) to multi-token (#1 prefill lever, 22.8%)

**Date:** 2026-06-06. **Status:** execution-ready design (parallel prep while the
decode EAGLE rollback lands; this touches `attention.rs` so IMPLEMENT after the
rollback merges to avoid conflict). **Goal:** lift the proven decode fused-wqkv
win (+18.4%, `wins/2026-06-06-dsv4-fused-wqkv-decode-default-on.md`) to the prefill
path, where the same scalar `dsv4_fp8_gemv_batch_kernel` Q/KV-LoRA projection is
**22.8% of prefill GPU** (clean 4096-tok profile,
[`2026-06-06-dsv4-decode-6ms-remaining-levers.md`](2026-06-06-dsv4-decode-6ms-remaining-levers.md)).

## The lever

Decode (`token_count==1`) runs `run_fused_wqkv_decode` (`attention.rs:2140`):
fuses `wq_a | wkv` into ONE FP8 DeepGEMM (tensor-core) instead of two scalar
`dsv4_fp8_gemv_batch_cuda` GEMVs. Prefill (`token_count>1`) still takes the scalar
path (`attention.rs:2359+`, the `else` of the `fused_wqkv` gate at `:2347`). The
DeepGEMM FP8 grouped GEMM **already handles multi-token M** (it's the MoE GEMM) —
so the only thing pinning the fusion to decode is the B=1-specific scaffolding.

## Exact changes

1. **Generalize the scratch.** `Dsv4FusedWqkvDecodeScratch` (`attention.rs:286`,
   alloc `:300`) sizes buffers for `seq_len==1`. Add a `max_tokens` capacity (the
   prefill chunk size — reuse the existing prefill token budget) and size `c_q` /
   `c_kv` / the fused output for `[max_tokens, ...]`. Allocate once per slot, reuse.
2. **Generalize `run_fused_wqkv_decode`** (`:2140`, asserts `hidden.seq_len==1` at
   `:2148`): take `token_count`, pass `M=token_count` to the fused `wqkv_a`
   DeepGEMM, and swap `mla_rms_norm_decode_slice` (`:2225`, asserts `seq_len==1`)
   for the batched `mla_rms_norm` (`:2196`, already multi-token) on the `c_q`/`c_kv`
   slices. Rename it `run_fused_wqkv` (decode = the token_count==1 case).
3. **Drop the gate.** `attention.rs:2347` `let fused_wqkv = token_count == 1 &&
   dsv4_fused_wqkv_decode_enabled()?;` → `... = dsv4_fused_wqkv_decode_enabled()?;`
   so prefill chunks take the fused path too. Keep the env opt-out
   (`ARLE_DSV4_FUSED_WQKV_DECODE=0`).
4. **Inverse-RoPE / SW-window update** already loop over `seq_len`
   (`update_bf16_sw_window` takes `start_pos`); confirm the fused prefill path
   feeds them the multi-token `k_prepared` (the scalar prefill path already does).

## Risk & gate

Lower risk than novel work — same DeepGEMM kernel, only M:1→token_count, mirroring
the scalar prefill path's existing multi-token RMSNorm/wq_b tail. Gate:
1. **Token-exact prefill** vs the scalar path (DeepGEMM vs scalar GEMV float order
   may differ on near-ties → gate on needle retrieval + first-token argmax, not
   strict byte-identity, like any FP8-GEMM swap).
2. **prefill_ms A/B** (4096-tok, TP=8/EP=8, same-binary env flip
   `ARLE_DSV4_FUSED_WQKV_DECODE` 1 vs 0): expect the 22.8% scalar-GEMV bucket to
   shrink toward the DeepGEMM tensor-core cost. License on the wall-clock TTFT
   delta, not the kernel-table %.

## Why this is the right prefill #1

The other prefill levers are harder: scalar hybrid attention (23.6%) — FlashMLA-
prefill was KILLED (+36%, prepare-chain overhead); csa-select fused top-k (17.7%)
— novel kernel. The fused-wqkv extension is the **proven** pattern (already
shipping at decode) lifted to multi-token, the lowest-risk prefill win. Do it
first; then re-profile prefill (the 22.8% bucket collapses, re-rank the rest).
