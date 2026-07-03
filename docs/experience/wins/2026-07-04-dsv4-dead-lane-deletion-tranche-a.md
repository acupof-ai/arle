# DSv4 dead-lane deletion tranche A: −2191 LOC (MMA lane, dead levers, unreachable GEMV arms, zero-caller FFI)

> Status: pending-remote — sm_90 build + ≤2048-ctx needle regression gate ride
> the #146 discriminator pod round. Default paths are byte-identical by
> construction (only dead/default-off code deleted), so the gate is
> confirmation, not an A/B.

## Goal
Delete inferior/dead DSv4 kernel lanes and concluded experimental levers
(directive 2026-07-04: keep the fast lanes, delete the rest).

## What went
- **MMA GEMV lane** (KILLED 2026-06-07: all-zero output, −32% decode):
  quantized_gemv_mma.cu (−330) + env gate `ARLE_DSV4_FP8_GEMV_MMA` + dispatch
  branch + FFI decl + the HIP stub that existed only for it.
- **Concluded #138 A/B levers** (root cause was the ape dummy-data OOB, not a
  sync race — both levers measured no-effect in the #146 ladder too):
  `ARLE_DSV4_DSA_BUILD_SYNC`, `ARLE_DSV4_FLASHMLA_PREFILL_SYNC`,
  `ARLE_DSV4_DEEP_COPY_KEEPALIVE` (its removal flips the one gated call site
  to `new(false)` — behavior byte-identical; the now-inert keepalive machinery
  is scheduled for tranche B).
- **Unreachable non-batch GEMV arms**: `dsv4_fp8_gemv_kernel/_cuda` +
  `dsv4_fp4_gemv_kernel/_cuda` + the quant_linear::gemv Dsv4 arms — every
  `ops::gemv` caller is DenseBf16-restricted (dsv4.rs:6381/6419); nsys 0
  instances.
- **Zero-caller FFI hosts** (each grep-verified tree-wide; shared kernels
  kept): fp8/fp4 pair/grouped/route-pair GEMV family, dsv4_grouped_gemm.cu
  (−408, whole file), `dsv4_prepare_qk_fused_cuda` (kernel kept — live
  `_start_pos_ptr` host shares it), utility kernels
  (zero_bf16/fill_i32/cast_i64/dequantize_fp8_rows).
- **Deferred**: the MoE-route native-allreduce cluster in dsv4_route.cu
  (pending owner confirm) and 5 inventory misjudgments the executor caught
  live callers for (deepgemm swiglu/unpad/block-scaled-cache, grouped
  swiglu/down decode wrappers, the gemm parity test).

## Rule
- A deletion inventory from static search is a hypothesis list — the executor
  re-verifies reachability per symbol before deleting; 5 of ~30 items had
  live callers the inventory missed.
- Deleting an env conjunct can INVERT behavior (`DEEP_COPY_KEEPALIVE` was the
  only thing holding a diagnostic path off) — always resolve the residual
  expression, never just drop the env term.
