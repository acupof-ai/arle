# INT4 + KIVI two-level K (static × dynamic) + asymmetric [-8,7]

## Context

[`2026-05-27-int4-kv-kivi-poc.md`](2026-05-27-int4-kv-kivi-poc.md) shipped
INT4 + KIVI per-channel K end-to-end on V100 (Qwen3.5-4B audit). PoC
mean_match = 0.094 — coherent generation but immediate step-1 drift vs
BF16. User pushback: "coherent diff text 肯定能正确实现的 实现掉" —
accepting a 0.094 mean_match as "intrinsic 4-bit floor" was premature;
the literature has more levers before reaching for Hadamard.

Two cheap levers stacked into one commit:

1. **Asymmetric INT4 range [-8, 7] with /7.5 scale.** PoC used symmetric
   [-7, 7] / divisor 7. The full nibble is 16 levels (signed two's-comp
   −8..+7), not 15. /7.5 = midpoint of |−8| and |7|, minimizes max abs
   error at symmetric activation distributions. ~7% finer resolution at
   the same clipping rate.

2. **Two-level K scale: per-channel STATIC × per-(token, kv_head)
   DYNAMIC.** PoC quantized K with KIVI per-channel scale only:
   `k_int4 = round(k_bf16 / static[h, d])`. The static scale captures
   channel-wise outlier structure but underfits per-token magnitude.
   Two-level normalizes to the channel first, then takes the
   per-(token, head) absmax of the channel-normalized value and uses
   that as a second scale:
   `ratio = k_bf16 / static[h, d]; dyn[t, h] = max_d(|ratio|) / 7.5;
    k_int4 = round(ratio / dyn[t, h])`. Decode reads back
   `k_bf16 ≈ int4 * static[h, d] * dyn[t, h]`. Effectively grants K
   ~1 extra bit of dynamic range at the same 4-bit storage, which is
   exactly what closes most of the INT4-vs-INT8 quality gap the
   per-channel-only design leaves on the floor at low bit budgets.

V symmetric range bumped to [-8, 7] / 7.5 too, for the same reason.

## Implementation

- `quantize_paged_kv_int4_per_channel_kernel`: block reshaped to
  `(num_kv_heads, batch)` grid with `head_dim` threads. Threads compute
  the per-channel-normalized ratio, warp-reduce to per-(token, head)
  absmax, derive `dyn[t, h]`, write it to `k_dynamic_scales`, then
  stage signed nibbles in shared memory and pack pairs.
- `decode_attention_int4_per_channel_k_partial_kernel`: reads
  `K_dynamic_scales[row * num_kv_heads + kv_head]` per timestep, folds
  into the K dequant during QK: `qk += q_reg[i] * k_int4 *
  k_scale_reg[i] * k_dynamic`.
- `k_dynamic_scales` storage repurposes the existing per-token
  `k_scales` buffer (`pool.k_scales_ptr(layer)`) — already allocated
  for INT4 via `KVFormat::INT4.has_scales() == true` from the PoC. No
  new pool allocation.
- Rust FFI + wrapper + qwen35 prefill/decode dispatch all carry the
  new `k_dynamic_scales` pointer.

## Measurement

Pending-remote. V100 build is currently blocked behind a TileLang fork
substrate issue unrelated to this change: `tilelang-sm70-copy` HEAD
69bc43e2 regresses on `tilelang_batch_prefill_paged_hd128_q32_kv8` with
`Layout infer conflict between m_new and scale_i` in
`tilelang.transform.LayoutInference()`. Rolling back to commit
14489d9d ("GemmFMA: drop redundant pre-staging sync"; the same commit
already in `~/.tilelang/cache/0.1.10_cuda_git14489d9d-x86_64`) requires
a full cmake rebuild of the TileLang C++ + cython bridge. Until that
lands, the audit number can't be regenerated.

Expected: mean_match should rise meaningfully from the PoC's 0.0938.
If two-level K alone reaches ≥0.9, the 2026-05-27 PoC entry's
"intrinsic 4-bit floor" framing becomes wrong and that entry needs
amending. If two-level K stalls in the 0.2–0.5 band, Hadamard rotation
becomes the next lever (TQ4-style; already implemented for sm_80+).

## Rule

When a "PoC works but quality is bad" entry suggests an *intrinsic*
floor, audit the literature's standard quality levers before
documenting the floor as fundamental. KIVI's per-channel-only K is one
of several KV-quant designs; two-level (per-channel × per-token) is a
well-known cheap upgrade. Symmetric vs asymmetric range, group size,
and rotation (Hadamard) are the other free levers. Document
"intrinsic" only after all of them are tried — otherwise the entry
mis-frames a tunable as a wall.
