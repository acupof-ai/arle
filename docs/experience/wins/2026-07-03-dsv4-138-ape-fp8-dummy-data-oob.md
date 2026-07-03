# #138 FIXED: DSv4 compressor `ape` read a 1-element FP8 dummy buffer OOB

## Context
DSv4-Flash eager prefill produced NaN/empty output for any recall across
context length 128 (the sliding-window boundary). <128 coherent, >=128 empty.
Deterministic (once the radix prefix cache was salted out). MTP/decode were
clean — only the eager prefill lane broke.

## Root cause (code-proven, not probed)
The compressor UPDATE kernel reads `ape` RAW as bf16 — `compressor.ape.data`
cast to `*const ffi::Half` (attention.rs) — unlike wkv/wgate which route through
the quant-aware `dsv4_linear`. But on an FP8 checkpoint `ape` loads via the
quantized path, where `DeviceMatrix.data` is a **1-ELEMENT DUMMY**
(`alloc_zeros::<bf16>(1)`, tensor.rs `from_dsv4_fp8_block_scaled`; the real FP8
bytes live in `qweight_u8`). The kernel indexed a `[ratio, width]` matrix
(128×512 for HCA) into that 1-element buffer → massive out-of-bounds read of
adjacent pool memory → garbage/NaN in the compressed-key positional encoding →
NaN compressed rows.

Why it gated exactly at ctx-128: HCA (ratio-128) finalizes AND reads its
compressed rows only at pos>=128 (`comp_keys = abs_pos/128`), so <128 never
touched the corrupt ape. Decode/MTP read the FP8 paged pool (correctly packed),
so only the bf16 compressed-staging path (eager prefill) exposed it.

## Fix
Load `ape` dense bf16 via `load_dsv4_block_scaled_dialect`, now taught to
dequant the DSv4 E8M0 `<base>.scale` dialect too (it previously only dequanted
GLM's F32 `weight_scale_inv` and fell DSv4 through to the dummy-data quantized
form). wkv/wgate stay quantized (dsv4_linear reads qweight_u8). `954d9905`.

Pod verify (TP=4 eager, salted needle x3): 74915 3/3 at pt=313 and pt=414
(EMPTY 3/3 pre-fix), coherent prose >128, decode-crossing coherent, 0 NaN/inf.

## Rule
A `DeviceMatrix` that `is_quantized()` has a DUMMY `.data` — the weights are in
`qweight_u8`. Any kernel reading `.data` directly (not via the quant-aware GEMV
path) on a quantized matrix reads a 1-element buffer OOB. When adding a raw
`.data` reader, assert `!is_quantized()` at the call site, or dequant at load.
And: a false hypothesis chain (RoPE-NaN, ape-OOB-by-shape, chunk-carry,
uninit-kv_unified, zero-dilution, acc-overflow) collapses the moment you read
the buffer's ACTUAL storage — `is_quantized()==true` + a raw bf16 read was the
tell; the dummy `.data` was the proof.
