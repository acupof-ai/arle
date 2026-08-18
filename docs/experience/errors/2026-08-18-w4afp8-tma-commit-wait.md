# W4AFP8 -729 crash + workspace OOM + NCCL shm SIGKILL

## Symptom 1: -729 crash on first MoE forward

CUTLASS W4AFP8 grouped GEMM crashes with -729 (cudaErrorLaunchFailure) on first MoE forward.

### Root cause

The SGLang mixed-input extensions were written for CUTLASS 4.x (commit
57e3cfb4) but backported to CUTLASS 3.7.0. The backport introduced TMA
descriptor lifecycle bugs: the 3.7.0 TMA descriptor handling differs from
4.x in ways beyond just the missing `tma_desc_commit_group`/`tma_desc_wait_group`
wrappers.

The TMA commit/wait PTX fix (e2e0b0401) was necessary — without it, the TMA
hardware uses the initial (1×1) descriptor → out-of-bounds → -729. But it
was not sufficient: the crash persisted because other 3.7.0 vs 4.x TMA
handling differences remain in the backported kernel.

### Fix

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

Commits: 453bf60fd (CUTLASS 4.x switch), e68512944 (builder stage-count fix)

## Symptom 2: 64MB workspace OOM

`DSv4 W4AFP8 workspace alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`
on first MoE forward, despite 1.8GB free VRAM at TP=2 on H20.

### Root cause

The CUTLASS workspace was hardcoded to 64MB. At TP=2 with 95.7GB/GPU used
(weights + KV cache), the pool had no retained block large enough, and the
GPU had insufficient free VRAM for a new 64MB allocation. The oversized
workspace also caused 36s TTFT on 1K prefill (memory pressure).

### Fix

Right-sized to 32MB (actual need: <16MB CUTLASS ws + 17KB metadata at
E=128). Commit: fb0b877d2.

## Symptom 3: silent SIGKILL during weight loading

Process vanishes during weight loading — no error, no panic, no dmesg entry,
no core dump. Reproduced across both builds.

### Root cause

Stale NCCL shared memory files (`/dev/shm/nccl-*`) from crashed runs
interfere with new serve launches. The NCCL init succeeds (self-test passes)
but the process is killed silently during subsequent weight loading.

### Fix

`rm -f /dev/shm/nccl-*` before launching a new serve after a crash.

## Why the 3.7.0-shaped call compiled there and not here

The builder passed one extra explicit template argument, `StageCountType::bytes`,
ahead of `SmemAlignment`. The two trailing `int` parameters swapped order
between versions:

- CUTLASS 3.7.0: `..., class TileShapeMNK, int carveout_bytes, int alignment = 128`
- CUTLASS 4.3.5: `..., class TileShapeMNK, int alignment = 128, int carveout_bytes_`

In 4.3.5 `carveout_bytes_` is the trailing parameter, deduced from the function
argument `StageCountAutoCarveout<carveout_bytes_>`. The extra explicit argument
therefore bound `alignment = StageCountType::bytes` and forced
`carveout_bytes_ = SmemAlignment` (128), so the parameter type became
`StageCountAutoCarveout<128>` while the argument was
`StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>` — not
convertible, hence "no instance of overloaded function matches". Upstream 4.3.5
calls the same helpers with 7/7/5 explicit args
(`cutlass/gemm/collective/builders/sm90_gmma_builder.inl:462-470`), which is what
the fix restores.

The fourth error ("Could not find a mainloop specialization",
`collective_mma_array_mixed_input.hpp:43`) is a cascade: `PipelineStages` is an
error type, so the partial specialization keyed on
`MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ...>` cannot match and
the primary template's `dependent_false` static_assert fires.

## Why this surfaced late

The break sat undetected across several apparently successful builds: `csrc/` is
compiled by a plain recursive glob with no feature or SM-tier gate, so an
incremental build that touched nothing under `moe/w4a8/` simply reused the
cached objects. The logs of those builds mention `w4a8` zero times. Touching an
unrelated `csrc/` file forced the unit to rebuild and the failure appeared at
once.

Diagnostic shortcut: the error line numbers identify the tree. Pre-fix the three
call sites are at 206/215/224; at the fix they are 206/214/222. Seeing 215/224
means the build read the pre-fix file regardless of what the checkout claims.
`grep -c 'StageCountType::bytes' <that .inl>` must return 0.

## Key files
- `crates/cuda-kernels/csrc/moe/w4a8/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` — the kernel
- `crates/cuda-kernels/csrc/moe/w4a8/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl` — stage-count template arg fix
- `crates/cuda-kernels/build.rs` — CUTLASS include path (line ~2973)
- `scripts/pod-build-env.sh` — removed CUTLASS 3.7.0 download
- Commits: 453bf60fd (CUTLASS 4.x switch), e68512944 (builder stage-count fix)
