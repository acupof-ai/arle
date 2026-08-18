# W4AFP8 -729 crash root cause

## Symptom
CUTLASS W4AFP8 grouped GEMM crashes with -729 (cudaErrorLaunchFailure) on first MoE forward.

## Root cause
The SGLang mixed-input extensions were written for CUTLASS 4.x (commit
57e3cfb4) but backported to CUTLASS 3.7.0. The backport introduced TMA
descriptor lifecycle bugs: the 3.7.0 TMA descriptor handling differs from
4.x in ways beyond just the missing `tma_desc_commit_group`/`tma_desc_wait_group`
wrappers.

The TMA commit/wait PTX fix (e2e0b0401) was necessary — without it, the TMA
hardware uses the initial (1×1) descriptor → out-of-bounds → -729. But it
was not sufficient: the crash persisted because other 3.7.0 vs 4.x TMA
handling differences remain in the backported kernel.

## Fix
Switch to the FlashMLA vendored CUTLASS 4.x (`crates/cuda-kernels/vendor/
flashmla/csrc/cutlass/`, NVIDIA tag 147f5673) and revert the 3.7.0
compatibility code in the SGLang kernel:

1. Remove custom `MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput` — 4.x
   has it in `dispatch_policy.hpp`
2. Use `cutlass::detail::ConversionMode` from 4.x `mixed_input_utils.hpp`
3. Replace inline PTX with `cute::tma_desc_commit_group()`/
   `cute::tma_desc_wait_group()` (identical instructions, but the 4.x header
   pulls in the correct TMA descriptor types and lifecycle)
4. Remove the standalone CUTLASS 3.7.0 download from `pod-build-env.sh` and
   CI workflows — the vendored 4.x is already tracked in the repo

The `ElementScalePacked = Array<BF16, K/128>` modification (SGLang's
interleaved scale layout) is preserved — it is the key SGLang extension
that the official 4.x kernel lacks.

## Key files
- `crates/cuda-kernels/csrc/moe/w4a8/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` — the kernel
- `crates/cuda-kernels/build.rs` — CUTLASS include path (line ~2973)
- `scripts/pod-build-env.sh` — removed CUTLASS 3.7.0 download
- Commit: 453bf60fd
