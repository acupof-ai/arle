# DSv4 FP8 model fails to load — W4A16 detection probe rejects native E8M0 scales

## Context

Loading `DeepSeek-V4-Flash-FP8` on CUDA failed with an E8M0 scale rejection
error. The model's FP8 experts use block-scaled E8M0 scales (`.scale` suffix),
which are native to DSv4's quantization scheme. The W4A16 detection probe
(`ec9997cfd`) calls `quant_view_for()` to check whether the first expert is
W4A16, and that call triggered the Qwen safety rejection.

## Root Cause

`reject_dsv4_e8m0_scale_abi()` in `quant_format.rs:593` rejects any weight that
has an E8M0 scale tensor. This guard exists to prevent the Qwen loader from
accepting DSv4's E8M0 scales through the wrong ABI path. But the W4A16
detection probe in `loader.rs` calls `quant_view_for()` on the first expert's
`w1` weight — and DSv4's FP8 experts *do* have E8M0 scales, so the probe
tripped the guard before the loader could even check the format.

The guard is correct for the Qwen loading path (where E8M0 scales would be an
ABI mismatch). The W4A16 detection probe is a DSv4-path concern and should not
be subject to the Qwen guard.

## Fix

`5cc681759` — added `quant_view_for_dsv4()` in `loader.rs` that skips the E8M0
rejection. `quant_view_for()` delegates to a shared inner function with a
`reject_e8m0` flag; the DSv4 W4A16 detection probe calls the `_dsv4` variant.
21 insertions, 5 deletions.

## Rule

A safety guard that protects one loading path must not fire on a different
path that legitimately uses the guarded format. When a probe or detection
routine traverses a model's tensors, it must use the path-appropriate view
function — not the guarded one.
