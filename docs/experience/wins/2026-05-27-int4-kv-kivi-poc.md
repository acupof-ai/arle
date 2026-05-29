# INT4 + KIVI per-channel K — PoC works, 4-bit quality intrinsically below 8-bit

> **Retracted (2026-05-28):** The "intrinsic 4-bit floor" framing in
> this entry — that 0.094 mean_match is a fundamental property of
> 16-level symmetric per-channel KIVI on Qwen3.5 — is **wrong**.
>
> Two cheap orthogonal levers, stacked the same day, lifted INT4
> mean_match from **0.0938 → 0.5781 at the same 4×16 grid** (6×):
>
> 1. Asymmetric INT4 range `[-8, 7] / 7.5` instead of symmetric
>    `[-7, 7] / 7`. Uses the full 16 nibble levels and minimizes the
>    midpoint clipping error.
> 2. Two-level K scale: per-channel **STATIC** × per-(token, kv_head)
>    **DYNAMIC**. The per-(token, head) dynamic absmax of the
>    channel-normalized ratio captures per-token magnitude that the
>    per-channel-only PoC couldn't.
>
> Current state and full audit numbers (incl. 4×4 and 4×16 grids,
> token traces, and the V100 substrate unblock) live in
> [`2026-05-28-int4-kv-two-level-k.md`](2026-05-28-int4-kv-two-level-k.md).
> Read that entry, not this one, for current INT4 KV behavior. This
> PoC stays in the tree only as a historical "what symmetric +
> per-channel-only looked like" reference.
>
> The lesson — documented as the Rule in the two-level entry — is to
> exhaust the literature's standard levers (asymmetric range, group
> size, two-level scaling, Hadamard rotation) before calling any
> quality number "intrinsic". KIVI's per-channel-only K is one of
> several KV-quant designs, not the design.

## Context

After INT8 + FP8 KIVI shipped bit-identical with BF16 on V100 (see
[`2026-05-27-int8-kv-kivi-per-channel-k-fix.md`](2026-05-27-int8-kv-kivi-per-channel-k-fix.md)),
the obvious follow-up axis was 4-bit. ARLE already has TQ4 (4-bit
packed + Hadamard rotation + FP16 group norms) but it doesn't run on
V100's sm_70 wrapper. The empirical question for the 4-bit slot: does
plain INT4 + KIVI per-channel K compete with TQ4-Hadamard?

User asked for a PoC ("2" from the menu of options). Implementation
goal: stand up the full INT4 + KIVI path end-to-end on V100, measure
mean_match vs BF16, decode the actual tokens, document the gap.

## What worked

Direct mirror of the INT8 KIVI implementation, with 4-bit packing
swapped in:

- `KVFormat::INT4` variant + paged_kv.rs gate (allocates
  `k_static_scales` and `int8_attn_workspace` like INT8/FP8).
- Pool storage halved: `max_total_tokens * kv_dim / 2` bytes per
  layer (2 nibbles per byte).
- `quantize_paged_kv_int4_per_channel_kernel` — each thread packs
  two dims into one byte.
- `quantize_paged_kv_single_int4_kernel` — head_dim threads do
  absmax reduction then half the threads do pairwise nibble packing.
- `finalize_k_per_channel_scales_int4_kernel` — divides by 7 (INT4
  symmetric max).
- `decode_attention_int4_per_channel_k_partial_kernel` — unpacks
  nibbles inline during QK and PV via arithmetic shift sign-
  extension. Synchronous loads (no cp.async pipelining — the data-
  dependent unpack would block the pipeline anyway).
- Rust FFI + wrappers + dispatch in qwen35 prefill/decode. qwen3
  paths gain `unreachable!` guards (PoC scope).

V100 Qwen3.5-4B audit added an INT4 case alongside BF16/INT8/FP8/TQ4:

```
bf16  mean_match=1.0000  (reference)
int8  mean_match=1.0000  (KIVI per-channel K — bit-identical)
fp8   mean_match=1.0000  (KIVI per-channel K — bit-identical)
int4  mean_match=0.0938  (KIVI per-channel K — real quant drift)
tq4   sm_70 unsupported  (architectural, pending sm_80)
```

Decoded INT4 tokens for prompt 0 vs BF16 reference:

```
BF16: "\n\nKV caching and attention mask are the most important
       parts.\n\nKV caching is"
INT4: "\n\n<think>\nHere is4\n\n</think>\n\nI am an AI assistant
       who specializes"
```

Coherent text in both. The kernel is producing real outputs, not
garbage; the quantization noise has shifted the greedy trajectory
to a different (also-coherent) continuation. INT4 found the model's
`<think>` prefix; BF16 stayed in completion mode.

## What this proves and disproves

**Proves:**

- The INT4 + KIVI per-channel K pipeline is correct end-to-end:
  packing, quantize kernel, dequant-in-attention, V handling. No
  NaN, no garbage, no architectural fail.
- The same KIVI dispatch scaffolding that shipped for INT8 and
  FP8 generalizes to 4-bit with a one-day port. Memory cost is half
  of INT8 (~25% of BF16) per the pool allocation.

**Disproves:**

- That per-channel K calibration alone is enough at 4-bit. INT4 with
  KIVI per-channel K hits step-1 divergence with mean_match=0.094.
  That's the floor of 16-level symmetric quantization without
  additional outlier handling.

**Confirms (without sm_80 access):**

- The TQ family's choice of Hadamard rotation as the outlier
  treatment for low-bit KV is structurally right. KIVI per-channel
  scaling alone is a single-axis transform; Hadamard rotation
  randomizes the entire (head, dim) distribution before
  quantization, recovering more precision per bit. The 4-bit
  literature (KVQuant, QuaRot, QServe) consistently lands on
  rotation-based methods for this same reason.

## What it does NOT tell us yet

- INT4-KIVI vs TQ4-Hadamard head-to-head on sm_80. INT4 ran on V100
  sm_70 (uses no Tensor Core hardware); TQ4 on V100 hits the
  architectural sm_70 wrapper limit. Both will run on A100 sm_80
  when access returns, and that audit is the actual data point
  needed to decide what 4-bit KV ARLE ships in production.
- Whether INT4 + KIVI + asymmetric quant (offset + scale) would
  recover quality. The PoC uses symmetric; the asymmetric variant
  is a separate one-day port if the head-to-head shows it's worth
  pursuing.
- Whether KIVI-2 (per-channel K + per-token V with 2-bit packing,
  the paper's lowest bit setting) is viable. The PoC's quality
  floor at 4-bit suggests 2-bit pure-symmetric KIVI would be
  drastically worse without additional tricks.

## Rule

**4-bit KV quantization without distribution-randomizing
preprocessing (Hadamard / random projection) loses to BF16
trajectory immediately at step 1.** The PoC's mean_match=0.094 is
the empirical floor for symmetric per-channel 4-bit on Qwen3.5-4B
under greedy decode.

If a future implementation wants to compete with TQ4 in the 4-bit
slot, the addition has to be on the rotation axis (Hadamard /
random Q rotation / random Givens), not finer-grained scaling.
Per-channel K + per-token V + symmetric 4-bit has been audited; that
combination is not enough.

This rule's flip side: at INT8 and FP8 precision, per-channel K
alone *is* enough — proven by the same audit's mean_match=1.0000
for both. The bit-budget below which Hadamard becomes structurally
necessary is somewhere between 4 and 8. The literature suggests 4 is
where it crosses; this PoC's data is consistent.

## Related

- Commit `d36c528f` — INT4 + KIVI PoC implementation.
- [`docs/experience/wins/2026-05-27-int8-kv-kivi-per-channel-k-fix.md`](2026-05-27-int8-kv-kivi-per-channel-k-fix.md)
  — INT8 KIVI fix that the INT4 path direct-mirrors.
- [`docs/experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md`](2026-05-27-v100-kv-precision-parity-qwen35-4b.md)
  — V100 audit infrastructure this PoC reuses.
- [`docs/quantization.md`](../../quantization.md) §1.4 — TQ family
  (TQ2/TQ3/TQ4) Hadamard description.
- KIVI paper: <https://arxiv.org/abs/2402.02750>.
- KVQuant: <https://arxiv.org/abs/2401.18079> — KMeans non-uniform K
  + Hadamard, the deepest study of 4-bit KV correctness so far.
- QServe: <https://arxiv.org/abs/2405.04532> — W4A8 + KV4 in
  production, also Hadamard-rotated.
