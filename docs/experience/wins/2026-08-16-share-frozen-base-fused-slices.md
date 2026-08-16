# share-frozen-base: alias fused QKV/gate-up slices, not duplicate FP8 base

**Status.** Shipped (verified on H20, 2026-08-16).

## Context

The 27B rubric-opd LoRA sync OOMed on one 96 GB H20 despite `cuMemGetInfo`
reporting ~45 GB free. The root cause was a name mismatch in the
`--share-frozen-base` export: the engine shipped fused `self_attn.qkv_proj` and
`mlp.gate_up_proj` suffixes, while the train student models Q/K/V and gate/up as
separate projections. The loader's `SharedFrozenBaseEntry::matches` found no
hits for q/k/v/gate/up, so the student uploaded its own ~20 GB FP8 base copy
instead of aliasing the engine's resident bytes.

Peak during the sync (with `--lora-merge-fp8` off):
engine FP8 (23 GB) + store FP8 duplicate (23 GB) + store BF16 LoRA (5 GB) +
per-layer BF16 promotion ≈ 51 GB before the first layer, growing past 90 GB as
layers promoted — OOM at layer ~59.

## What worked

1. `frozen_base_fp8_pointers` / `frozen_base_bf16_pointers` now export the
   individual row-slices of fused matrices:
   - `qkv_proj` → `self_attn.{q,k,v}_proj` at row offsets `(0, q_gated)`,
     `(q_gated, kv)`, `(q_gated+kv, kv)`.
   - `gate_up_proj` → `mlp.{gate,up}_proj` at `(0, inter)`, `(inter, inter)`.
   - `in_proj_qkvz` → `linear_attn.{in_proj_qkv,in_proj_z}`.
   - `in_proj_ba` (BF16) → `linear_attn.{in_proj_b,in_proj_a}`.
   `push_row_slice` computes the qweight byte offset (`row_offset * cols`) and
   the scale row offset (`(row_offset / block_m) * scale_cols`); `row_offset`
   is asserted a multiple of `block_m` (FP8 block-scaled invariant).

2. `--lora-merge-fp8` default flipped `false → true`. The per-layer requant
   keeps the dense peak one layer wide instead of accumulating all layers' BF16
   dense (~40 GB). The existing `pristine_fp8` keepalive preserves the
   share-frozen-base alias and idempotent re-merge.

3. Removed a stray QKV debug dump (`/tmp/qkv_pre_conv.bin`) left in
   `qwen35_attention.rs`.

## Expected result

Store FP8 residency drops from 23 GB (duplicate) to 0 (aliased). The LoRA sync
peak becomes: pristine FP8 (23 GB) + merged FP8 (23 GB) + BF16 LoRA (5 GB) +
per-layer transients ≈ 52 GB — well under 96 GB.

## Verified result (H20, 2026-08-16)

27B FP8 student, `all-linear`, `--self-consistency`, 1 round, 4 prompts × 2
samples, writeback-cap=2:

- `--share-frozen-base: borrowing 400 resident FP8 base projections from the
  rollout engine (zero-copy)` — no private FP8 upload.
- Peak VRAM during writeback: **44 919 MiB (44.9 GB)**.
- Pre-sync (after retain_ids + trim): **35 991 MiB**.
- LoRA sync exited 0; store params `fp8=23200 MiB bf16=4895 MiB` (the FP8 is
  the aliased engine base, not a store copy).
- 8/8 accepted, 2 trained, mean_loss=0.0260.

Before the fix the same config OOMed at the LoRA sync (store held its own
23 GB FP8 duplicate, leaving no headroom for the per-layer BF16 promotions).
