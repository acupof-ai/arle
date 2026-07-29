# BF16-resident MoE experts cleared by the grouped-cache concat

## Context

#13 (BF16-resident experts, `set_qwen35_moe_experts_bf16_resident`) shipped so
the OPD rollout student can re-merge LoRA into FP8 MoE experts. First real run
(iso64 distill, all-linear LoRA, Hopper H20) died at **step 3** with #13's own
assertion: `layer 0 mlp.experts.0.gate_proj expert matrix is not resident as a
per-expert BF16 DeviceMatrix`.

## Root Cause

The dequant loop flips `expert_weight_format` → `DenseBf16`, so
`routed_quant = false`. On Hopper (`deepgemm_native_ready`, `!sm120`) that made
`deepgemm_ready = true` → the BF16 grouped DeepGEMM concat ran and **cleared the
per-expert `gate/up/down` Vecs** (`loader.rs` `gate.clear()`). The per-step LoRA
re-merge (`qwen35.rs` `lora_matrix_mut` → `moe.gate[local_idx]`) then found the
Vec empty. Step 3 not step 1 only because the LoRA delta is zero until the
optimizer first moves — the empty-Vec was there from load.

The fix dequantized the experts but immediately re-hid them in the very cache
the re-merge can't reach.

## Fix

Gate the BF16 grouped path on the flag too:
`deepgemm_ready = !routed_quant && deepgemm_native_ready && !sm120 &&
!experts_bf16_resident`. The eager per-expert MoE forward keeps the Vecs mutable
— the same path the sm120 self-disable already uses in production
(commit 227790953).

## Rule

A "resident so it's mutable" flag must suppress **every** downstream path that
consolidates or frees the mutable copy — not just the one that constructs it.
Grep the format-derived branches (`routed_quant`, `*_grouped`, `*.clear()`) for
consumers before declaring a residency fix complete.
