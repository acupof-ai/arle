# DSv4 mhc-fuse adopt — BLOCKED: vendored TileLang kernel emits unsupported f32 tensor-core MMA on sm_90a

**Date:** 2026-06-06. **Backend:** CUDA, sm_90a (H20), TileLang AOT.
**Status:** parked. Wiring (FFI + hc.rs + build.rs registration) was written +
Rust-typechecked; the **kernel AOT compile fails**, so it cannot land (the AOT
runs at build time — committing it breaks the CUDA build).

## Context

Adopting the vendored `mhc_pre_big_fuse` TileLang kernel (collapse 3 scalar HC
launches → 2 fused). The adapter (`tools/tilelang/mhc_pre_big_fuse.py`) wraps the
vendored `_mhc_pre_norm_fn_fwd_mul` kernel.

## Root Cause

The AOT nvcc compile of `_mhc_pre_norm_fn_fwd_mul_kernel` fails:

```
tl::mma_sync: unsupported configuration
  AType=kFloat32 BType=kFloat32 CType=kFloat32 M=16 N=8 K=8 TransB=true
2 errors detected in the compilation of tvm_kernels.cu
Hint: bump tilelang (pin in pyproject.toml) OR exclude sm_90
```

There is **no f32×f32→f32 tensor-core MMA instruction** — the f32 tensor-core
path is **TF32**. The vendored kernel calls `round_to_tf32` (so it *intends*
tf32), but the current TileLang's `T.gemm` lowering selects a raw-f32 `mma_sync`
that `mma.h` static-asserts against. The vendored kernel was written for a
TileLang version whose lowering auto-mapped f32-mma → tf32-mma.

## Fix options (for the eventual re-attempt)

1. **Explicit tf32 in the adapter** — make the `T.gemm`/mma operands tf32 (or set
   the gemm policy) so the lowering picks the tf32 MMA. Cleanest, preserves the
   f32-storage/tf32-compute intent. Needs the right TileLang API for our version.
2. **bf16 mma** — feed the mix GEMM bf16 inputs (the `mix_fn/base/scale` weights
   are *stored* bf16 anyway, so this also removes the bf16→f32 mirror the spec
   flagged). Compiles (bf16 mma is supported); precision risk on the mix must be
   needle-verified.
3. **Bump TileLang** (the hint) — **risky**: the existing FlashMLA / DeepGEMM /
   paged AOT kernels compile fine on the current pin; a bump must re-verify *all*
   of them. Not worth it for one kernel.

## Rule

A "vendored kernel, just wire the FFI" adopt is **not** automatically clean — a
TileLang kernel carries a TileLang-version contract. f32 `T.gemm` only works where
the lowering maps it to tf32-mma; on a version that emits raw-f32 mma it dies with
`mma_sync: unsupported configuration`. Before calling a TileLang adopt "clean,"
AOT-compile it on the target SM early. Parked here because the HC win is
uncertain-magnitude (likely overlap-bound per the decode-graph finding) and EAGLE
is the higher-value ready lever — re-attempt with fix option 1 or 2 later.
