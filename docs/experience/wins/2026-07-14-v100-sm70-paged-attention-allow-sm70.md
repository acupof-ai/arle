# V100 (sm_70) paged-attention kernel — allow_sm70 for HD256 q8_kv2

**Date:** 2026-07-14. **Backend:** CUDA, legacy Volta (sm_70, V100-SXM2-32GB).
**Scope:** `crates/cuda-kernels/kernels.toml` — TileLang AOT paged-attention
kernel eligibility gate.
**Status: pending-remote** — fix compiles clean; end-to-end serve + chat
verify on the 0.8B model to run on the V100 box (sm_70 test lane).

## Context

The V100 new-user flow errored on the first inference step (prefill) at
layer-3 full attention with `CUDA_ERROR_NOT_SUPPORTED`. The 0.8B model
(`Qwen3.5-0.8B`) is **HD256 q8_kv2** (head_dim=256, num_attention_heads=8,
num_key_value_heads=2). Its active forward path is `forward_hidden_staged` →
`full_attention_paged` → `ffi::resolve_paged_attn_v1`, which dispatches to
the TileLang `batch_prefill_paged_hd256_q8_kv2` /
`batch_decode_paged_hd256_q8_kv2` kernels.

Those two kernel rows had `allow_sm70 = false`, so the sm_70 cubin was
never compiled (build.rs filter: `eligible_targets = sm_targets.filter(|sm|
allow_sm70 || !is_legacy_volta_sm(sm))`). At runtime the dispatch wrapper's
`default:` arm returned `cudaErrorNotSupported` — same failure mode as the
BF16 GEMM, but for the attention kernel itself.

## What Worked

Set `allow_sm70 = true` on both `batch_prefill_paged_hd256_q8_kv2` and
`batch_decode_paged_hd256_q8_kv2`. The TileLang sm_70 path already exists
in `batch_prefill_paged_hd256.py` / `batch_decode_paged_hd256.py` (Volta MMA
with FP16 GEMM operands); the sibling `q16_kv4` rows already had
`allow_sm70 = true`, proving the sm_70 target compiles and runs. Updated
the `allow_sm70` header comment to list HD256 q8_kv2 as supported.

## Rule

- **`allow_sm70` must be set for every kernel a legacy-Volta config hits.**
  A `false` row silently drops the sm_70 cubin; the runtime dispatch then
  returns `cudaErrorNotSupported`. The HD256 q8_kv2 config is the default
  0.8B dense model — the first model a new user runs.
- **A sibling `allow_sm70 = true` row is existence proof the target
  builds.** q16_kv4's row licensed flipping q8_kv2 without a fresh TileLang
  port.

Verify (V100 box, pending): same build + serve + chat as the BF16-GEMM entry
(`2026-07-14-v100-sm70-bf16-gemm-fp16cast.md`) — both fixes land together
and clear the same layer-3 prefill step.
