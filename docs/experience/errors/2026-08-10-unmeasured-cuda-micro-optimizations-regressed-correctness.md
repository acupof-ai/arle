# Unmeasured CUDA micro-optimizations regressed correctness

## Context

`77635e8c5` vectorized `split_qkv`, `split2`, and `silu_mul_fused` and changed
shared activation functions to CUDA fast intrinsics. `a70cf5f7e` cached GEMV
activations in shared memory and hoisted scale-row addressing. Neither change
had an operator reference comparison or a measured serving win. A later
correctness run reported a precision regression, but its raw outputs and error
metrics were not preserved. The numerical cause is unknown.

## Root Cause

The changes entered the runtime before their numerical contracts were stated.
Pure data movement and address-only changes should have required bit-exact
outputs. Pointwise approximations needed a higher-precision reference and an
output-ULP bound. A model-level smoke result could not identify which contract
failed after these changes were combined.

## Fix

`17c60435e` restored `expf` and `logf`. `9a6ca91ac9` restored the scalar split
kernels and the direct GEMV input loads. The restored split file matches the
parent of `77635e8c5`, the eight fast-math files match the same parent, and the
GEMV file matches the parent of `a70cf5f7e`.

Remote verification used an isolated clean tree at exact commit `9a6ca91ac9`
on one H20:

- CUDA release build completed with exit 0;
- product binary SHA-256
  `495e37600c6f3e507e6a18f9229bd09f071c18a4845e9be4bcbb7e1e292b1d63`;
- kernel bundle
  `bundle:ecad6b4459768d07a6576eb0e29fe321e49e4b5165cc0942abd801eaec040c4d`;
- Qwen3.6-27B RAW needle at 2K, 8K, and 16K: 3/3 exact at every length,
  zero misses;
- sampled coherence: 200/200 tokens, zero glued repetition.

Artifacts are under
`/host/fq-fwd-9a6ca91ac9-g0/artifacts/{build.log,serve-baseline.log,needle-baseline.log,temp-baseline.log}`
on the validation host.

## Rule

Classify a kernel optimization before timing it. Data movement is bit-exact.
Pointwise math has an output-ULP bound. Reductions compare reference-error
metrics against the accepted kernel. The exact candidate binary then passes the
model gate. The full contract is
[`docs/bench-and-trace-spec.md` section 4.1](../../bench-and-trace-spec.md#41-numerical-acceptance-for-kernel-optimizations).
