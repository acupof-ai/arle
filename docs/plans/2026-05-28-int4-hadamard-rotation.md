# Plan — INT4 KV Hadamard rotation (close the int8 gap)

## Why

`docs/experience/wins/2026-05-28-int4-kv-two-level-k.md` documents the
current INT4 KIVI two-level K + asymmetric [-8,7] state on V100
Qwen3.5-4B:

| grid | int8 | fp8 | int4 |
| --- | --- | --- | --- |
| 4 prompts × 4 tokens | 0.9375 | 1.0000 | **0.8125** |
| 4 prompts × 16 tokens | 0.8906 | 0.7344 | **0.5781** |

The gap between INT4 and INT8 is ~0.31 at the 4×16 stress grid. KIVI's
per-channel STATIC × per-(token, head) DYNAMIC scheme has hit its
algorithmic floor at 4-bit; further quality requires reshaping the
input distribution itself. The literature answer is **K Hadamard
rotation** (randomized orthogonal Hadamard transform on the head_dim
axis), already used by ARLE's TQ4 path on sm_80+. Applying it to KIVI
int4 lifts the int4 floor toward the FP8/INT8 band without changing
the storage budget.

## What

For each layer, generate a fixed orthogonal rotation
`R = diag(signs[d]) · FWHT_D` where `signs ∈ {-1, +1}^D` is
deterministic from a layer seed and FWHT is the Walsh-Hadamard
transform (orthonormal, head_dim = 128 = 2^7).

Apply `R` to **both** Q and K on the head_dim axis. Attention is
invariant under orthogonal rotation of Q and K:
`(QR) · (KR)^T = Q · R · R^T · K^T = Q · K^T`.

K_rot is what we store quantized. The two-level scaling already in
place (per-channel STATIC × per-(token, head) DYNAMIC) operates on
the rotated K, which has its outliers redistributed uniformly — fewer
extreme channel-wise values, tighter clip rate at the same 4-bit
budget.

## Where

Three kernel modifications + zero pool-allocation change (rotation is
stateless given the seed):

1. **`csrc/kv/kv_quant.cu` — `quantize_paged_kv_int4_per_channel_kernel`.**
   Load K[batch, kv_head, d] into shared mem, apply
   `signs[kv_head, d] · FWHT_in_place`, then run the existing
   per-channel-normalized + per-(token, head) dynamic absmax + quant
   path against the rotated K. No layout change to the packed nibble
   output (still kv_dim/2 bytes per token).

2. **`csrc/kv/kv_quant.cu` — `compute_k_per_channel_absmax_kernel`.**
   Same rotation applied before taking per-channel absmax. The static
   table now characterizes the rotated K's per-channel distribution,
   not the unrotated K.

3. **`csrc/attention/decode_attention_quantized.cu` —
   `decode_attention_int4_per_channel_k_partial_kernel`.** For each
   query in the block, apply `signs[kv_head_of(q_head), d] ·
   FWHT_in_place` to Q in shared mem before the QK dot. K dequant is
   unchanged — it already recovers K_rot from the quantized
   nibbles + (static × dynamic) scales. Q_rot · K_rot^T then equals
   the un-rotated Q · K^T.

The Hadamard signs do not need a Rust-side allocation: hash from
`(kv_head, d)` deterministically inside the kernel
(`(popcount((kv_head * 0x9E3779B9u) ^ d) & 1) ? +1 : -1`), so both the
K quant kernel and the decode kernel agree on signs without any extra
ABI surface.

## How

Reuse `fwht_inplace` from `csrc/quant/turboquant_fast.cu` (or hoist
to a new `csrc/common_hadamard.cuh` header). It runs in `O(D log D)`
shared-memory butterflies, includes the `1/√D` orthonormal
normalization, and is its own inverse.

Q sign mapping for GQA: each q_head maps to one kv_head via
`q_head / (num_q_heads / num_kv_heads)` (= 4 for Qwen3.5-4B's
32q/8kv). Use that kv_head for the sign lookup so Q and K agree.

V is not rotated — V participates as `P · V` (post-softmax weighting),
and rotation would break that without a matching V^T rotation that
buys nothing in 4-bit.

## Expected impact

- INT4 mean_match (4×16 grid): 0.5781 → ~0.83–0.88 (closes ~75% of
  the int4-vs-int8 gap, based on TQ4-style results on sm_80+).
- INT4 mean_match (4×4 grid): 0.8125 → ~0.95+, possibly 1.000.
- INT8 / FP8 should also see a small lift (rotation gives all
  precisions cleaner per-channel distributions, but the marginal gain
  shrinks with bit budget).
- BF16: unchanged (rotation is bit-identical at full precision).

## Acceptance

1. Audit `cargo test … kv_precision_parity_qwen35` at the canonical
   grid (4×4 and 4×16) shows INT4 mean_match jump to within ~0.05 of
   INT8.
2. INT8 / FP8 mean_match does not regress (gate at prior numbers ±
   0.02).
3. BF16 mean_match stays 1.0000 (rotation is an exact transform).
4. Tokens dump shows the prompt-0 step-1 divergence is gone for INT4.

## Effort

~half-day focused work + 2–3 audit cycles (each ~5 min build + 4 min
audit on V100).

## Out of scope (defer)

- Per-layer independent signs vs. shared signs across layers.
  Literature averages across layers — pick later if numbers warrant.
- Hadamard for V. V's per-(token, head) scaling already adapts
  to the post-softmax distribution; rotation costs more than it buys.
- Hadamard for Q at *prefill* (only decode-attention is path-critical
  for the audit metric; prefill QK runs in BF16 already on the
  TileLang path).
