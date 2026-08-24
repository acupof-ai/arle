# GSPO math-opd lane — end-to-end smoke on Qwen3.8-27B-NVFP4

Date: 2026-08-24
Scope: `crates/train/src/math_harness.rs`, `crates/cli/src/train_cli/math_opd.rs`,
`crates/infer-cuda/src/qwen35_lora.rs`, `crates/cuda-kernels/src/quant_linear.rs`,
`crates/cli/src/train_cli/opd_engine.rs`, `crates/cli/src/args.rs`

## Context

The GSPO length-compression experiment trains `unsloth/Qwen3.8-27B-NVFP4`
(qwen3_5 arch, mixed NVFP4/FP8, ~23.4 GB) to answer math correctly with shorter
reasoning. The `math-opd` lane reuses the agent-opd GSPO loop with a
boxed-answer grader and a length-shaped reward
(`1 - α·(len-len_min)/(len_max-len_min+ε)` within group, α=0.3). Five smoke
iterations on a single H20 (96 GB) were needed to make the lane run end to end.

## What worked

Three blockers, each fixed at its root:

1. **RoPE clamp (smoke #1).** `cc_total_pages` for the continuous-cache KV pool
   could exceed `rope_len` (262144), so the engine guard rejected the config.
   `opd_engine.rs` now clamps `cc_total_pages` to `rope_cache_len_hint / 16`.

2. **FP8-marlin LoRA promote arm (smoke #2/#3).** The LoRA merge refused
   per-channel FP8 weights that `repack_for_marlin_fp8` had packed: the merge
   path promoted NVFP4 from Marlin tiles but had no FP8 analog. The fix
   composes two existing kernels — `marlin_fp8_to_e4m3` (reproduces the
   checkpoint's own E4M3 bytes, no value transform) then
   `dequantize_fp8_block_scaled_to_bf16` with a 1×K block shape and the held
   per-channel `scale_f32`. No new CUDA code. The helper
   `dequantize_fp8_marlin_to_bf16` is the FP8 analog of
   `dequantize_fp4_marlin_to_bf16`. The Linux build needed a follow-up
   (`f96b946a3`): `RawDevicePtr` has no `len()`, and the E4M3 scratch passes via
   `cache_ptr` — neither visible to Mac clippy (CUDA-type signatures compile
   Linux-only).

3. **attention-qv default (smoke #4).** The lane inherited agent-opd's
   `--lora-target-set all-linear` default. With all-linear, every projection is
   a LoRA target, and the first sync promotes each one to BF16 permanently (the
   merged bytes are the next round's frozen base, so the promotion cannot be
   transient). The footprint (~2× the 23 GB quantized student) does not fit on
   one GPU alongside the model and the 50%-of-free KV pool; the sync OOM'd at
   layer 40. The default is now `attention-qv` — the GSPO-standard set, with a
   promotion footprint of ~5 GB.

## Result (smoke #5, `6bcc7bd92`)

Build PASS (2m49s). Run EXIT=0, 420s, two rounds complete, both LoRA syncs
passed with no OOM. Command: `--task-limit 4 --eval-n 8 --rounds 2
--samples-per-prompt 4 --prompts-per-update 4 --sync every-round`.

- **8 group rows:** passed 4/4 in every group; `capped==0` in 8/8;
  `think_tokens>0` in 8/8.
- **2 update rows:** trajectories 16 then 4; tokens_trained 3766 then 2193;
  `is_ratio_mean` 1.0015 / 1.0011 (finite, ≈1); `clip_frac` 0.470 / 0.686
  (<1); `adv_std` 1.038 / 1.010 (>0).
- **2 eval rows:** base accuracy 0.875 (7/8), round-2 accuracy 0.875 (7/8) —
  no regression on the 8-item held-out set.
- Reward mean rose 0.844 → 0.966 across rounds (the model produced shorter
  correct reasoning on the trained tasks).

One benign workload property: in round 1, 3 of 4 groups had all-four samples
correct with near-identical length, so reward was 1.0 across the group and
`zero_variance=true` — GSPO produces no gradient for zero-variance groups, so
the round-1 update trained only the 4 trajectories of the one mixed group.
This is the expected behavior of the length-shaped reward on easy problems,
not a defect.

## Rule

- A quantized student's LoRA merge must reuse the existing dequant kernels
  (Marlin-tile → E4M3 → block-scaled BF16 for FP8; the FP4 analog already
  existed). Composing them is cheaper and lower-risk than a new kernel, and
  the held per-channel `scale_f32` survives `repack_for_marlin_fp8` for exactly
  this.
- A 27B quantized student cannot run `all-linear` LoRA on one GPU: the merged
  BF16 bytes are the permanent frozen base, so the promotion footprint is
  intrinsic to the target set. `attention-qv` is the GSPO-standard default and
  fits.
- Mac clippy does not compile CUDA-type function bodies in `cuda-kernels`
  (`RawDevicePtr`, `DevicePtr` signatures are Linux-only), so a green local
  run does not prove the Linux build. The pod build is the verification.
