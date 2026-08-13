# w2s base offload round-trip dequantized the FP8 base

> Status: Confirmed

## Context

`arle train w2s` with ThinkingCap-Qwen3.6-27B-FP8 as base/student and four
0.8B aux checkpoints failed at setup on a 95.2 GB H20:

```
error: cuda bf16 htod copy failed: shape=[5120, 17408] len=89128960
  bytes=178257920 err=DriverError(CUDA_ERROR_OUT_OF_MEMORY)
```

A per-phase VRAM ledger showed the base loaded at 27.9 GB. The failing copy was
89.1M elements at 2 bytes each — bf16, not FP8.

## Root Cause

`run_w2s` and `w2s_step` offloaded the 851 base tensors to host before each aux
forward and re-uploaded them afterwards. Two device-handle facts make that
round-trip wrong:

- The base loads as `CudaFp8BlockScaled`; its values exist only in the device
  handle. `offload_to_host` → `ensure_host` → `readback` hits
  `dequantize_fp8_block_scaled_host`, producing an f32 host vector and
  discarding the dtype.
- The re-upload called `upload_frozen_bf16_from_host` unconditionally, so the
  base returned as `CudaBf16` at 54 GB.

So one round-trip doubled the resident base and re-quantized it FP8 → f32 →
bf16. The OOM was the visible half; the silent half was that π_base for the
global KL regularizer no longer matched the checkpoint.

The round-trip existed to make room for "the aux post-RL 27B", per its own
comment. The aux models are 0.8B — 5.6 GB for all four. The premise was false,
so the mechanism had no reason to exist.

## Fix

Deleted the whole round-trip: `TensorStore::upload_frozen_bf16_from_host`,
`Qwen35Model::rope_cache_ids` (which existed only to exempt the RoPE caches from
that path), the two `run_w2s` offload blocks, the `run_w2s` re-upload block, and
the per-step offload/re-upload pair plus the `upload_model_bf16` helper in
`w2s_step`. Net −157 lines. The per-phase VRAM ledger stays.

The base now stays FP8-resident for the whole run. Measured on GPU auto,
95.2 GB total:

| Phase | Used | Free |
|-------|------|------|
| base (FP8) + student | 27.9 GB | 67.4 GB |
| + four 0.8B aux | 33.5 GB | 61.7 GB |

`--steps 2` then reached `RUN_EXIT=0`: `step=0 loss=25.158342 max_prob=0.8188
consistency=0.7372`; step 1 skipped by the confidence gate at max_prob=0.9726.

A second, independent gap surfaced once the aux forward was reached: flashqla had
no AOT instantiation for the 0.8B GDN geometry H=16/Hg=16. Added as five
`kernels.toml` rows (`f77ca2eb5`); the FFI dispatch table is generated from that
file, so no Rust change was needed.

Commits: `62017ec8a` (deletion), `bc96d29ec` (collapse duplicated aux loading),
`f77ca2eb5` (AOT geometry). Earlier in the same chain: `2dfb12140` (RoPE cache
host mirror), `b56274770` (bf16 operands in add/mul/matmul).

## Rule

Host offload assumes the tensor's value lives in the host f32 mirror. For a
quantized device handle it does not, so the round-trip is a lossy re-quantization
and not a memory-management primitive. Before adding an offload path, check the
handle variant; before keeping one, check the number that justified it — the
27B-vs-27B budget here was never real.
