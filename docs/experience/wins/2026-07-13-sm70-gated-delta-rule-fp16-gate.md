# sm_70 fp16-operand gate restores gated_delta_rule on real Volta

> The reorg/DSpark-month hybrid linear-attention GDR kernel fed bf16 into
> Volta's FP16-only `mma.sync m16n16k4`, aborting the sm_70 AOT build at
> TileLang's own `mma_sm70.h` static_assert. Add the fp16-operand gate every
> working attention kernel already has. V100-verified: compiles, numerically
> correct, full binary builds.

## Context

Chasing "CI sm70 lane green" surfaced, after 5 fragile in-CI tvm-ffi-wheel-rebuild
rounds, a *second* wall the wheel failure was masking. Building on the **real
V100** (`ssh v100`, Tesla V100 sm_70) instead of the CI cross-compile lane
(no GPU) gave the definitive answer: the blocker is **code, not CI infra**.

`tools/tilelang/gated_delta_rule.py` (new hybrid linear-attn kernel) had no
sm_70 gate — `ARLE_TILELANG_CUDA_ARCH` marker count 0, while every working
attention kernel has it (`batch_prefill_paged_hd128.py`: 15). Its `T.gemm`s fed
bf16 tiles into `mma.sync m16n16k4`; Volta MMA (GemmMMASm70) only accepts FP16
inputs, so `mma_sm70.h:349` static_assert aborted `gated_delta_rule_prefill_chunk_a`.

## What Worked

Mirror the existing pattern exactly. `_sm70_gemm_gate(dtype, accum_dtype)` →
`gemm_dtype = "float16" if sm_arch < 80 else dtype` + a `to_gemm` cast that
routes bf16→f32→fp16 (direct bf16→fp16 is an ambiguous nvcc conversion). Every
`T.gemm` operand tile in the 4 GEMM kernels (chunk_a / recompute / state / o)
uses `gemm_dtype`; loads cast through `to_gemm`. Accumulators stay f32.

**sm_80+ byte-identical**: `gemm_dtype == dtype` makes the gate a no-op
passthrough, so shipped Hopper/Ampere AST is unchanged.

V100 verification (real Volta, CUDA 12.4, stock TileLang 0.1.11):
- All 7 GDR AOT targets compile — static_assert gone.
- Full `cargo build --release -p arle` EXIT=0, 30 MB binary loads + resolves CUDA 12.4 libs.
- Numerical parity fp16-operand vs f32 reference: rel_err 3e-4..8e-4, cosine≈1
  (1−cos ≤ 2.4e-7, same order as the bf16 baseline), no fp16-range overflow
  across 64-chunk state accumulation. fp16's 10-bit mantissa > bf16's 7-bit →
  operand precision strictly better where the range holds (RMSNorm-bounded q/k/v).

Landed `6d14e1ff0` (byte-identical to the V100-verified diff: +64/−37, one file).

## Rule

- A new TileLang kernel with a `T.gemm` needs the sm_70 fp16-operand gate before
  it can build for Volta — grep `ARLE_TILELANG_CUDA_ARCH` count; 0 = missing.
  Mirror `batch_prefill_paged_hd128.py`, don't hand-roll.
- **A GPU CI *cross-compile* lane (no device) can't validate; when a real box
  exists (V100 for sm_70), verify there** — it revealed the real blocker in one
  build after 5 CI rounds chased a masking wheel-build failure. See
  [[reference_v100_box_access]], [[reference_sm70_tilelang_multiconflict]].
