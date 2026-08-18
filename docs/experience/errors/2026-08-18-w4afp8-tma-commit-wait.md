# W4AFP8 -729 crash root cause

## Symptom
CUTLASS W4AFP8 grouped GEMM crashes with -729 (cudaErrorLaunchFailure) on first MoE forward.

## Root cause
`tensormaps_cp_fence_release` in the mixed-input kernel skipped the
`cp.async.bulk.commit_group` + `cp.async.bulk.wait_group.read 0` calls that
commit TMA descriptor dims/strides updates to the TMA hardware.

Without commit/wait, the TMA hardware uses the initial (1×1) descriptor
created in `to_underlying_arguments`. The first TMA load in the mainloop
reads from a bogus address → out-of-bounds → -729.

SGLang's mixed-input kernel has these calls (inside `elect_one_sync`, before
the `tma_descriptor_cp_fence_release` calls). Our CUTLASS 3.7.0 backport
removed them with a comment claiming "added in 3.8+, not needed" — wrong.

## Fix
Add back the commit/wait in `tensormaps_cp_fence_release`:
```cpp
if (cute::elect_one_sync()) {
  asm volatile("cp.async.bulk.commit_group;");
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(0) : "memory");
}
```

CUTLASS 3.7.0 lacks the `cute::tma_desc_commit_group()` wrapper; use inline PTX.
FlashMLA CUTLASS 4.3.5 defines the same functions with identical PTX.

## Key files
- `crates/cuda-kernels/csrc/moe/w4a8/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` — the fix
- SGLang reference: `/data00/sglang-eic/sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` line 1491-1494
