# TileLang Phase 0 — AOT export blocked on TileLang 0.1.9

## Context

Phase 0 verification on remote L4 (sm_89, CUDA 12.8.93) per
[`docs/plans/tilelang-integration-verification.md`](../../plans/tilelang-integration-verification.md)
§2 (build) was attempted on 2026-04-26 with the latest pip-installable
TileLang (`tilelang>=0.1` → `tilelang 0.1.9`). All four AOT
specializations (`q16/q32/q40/q64 × kv8`) needed to compile from
[`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`](../../../crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py)
to land cubins under `target/release/build/cuda-kernels-*/out/tilelang_aot/`.

**Stack at attempt:**
- HEAD: `802c5fc8` (TileLang 0.1.9 target string fix; see Fix #1 below)
- Backend: cuda
- Hardware: NVIDIA L4 24 GB, sm_89, driver 580.82.07
- Toolchain: rustc 1.95.0, nvcc 12.8.93, zig 0.14.0
- TileLang: 0.1.9 (PyPI, installed via `pip install -e ".[tilelang]"`)
- Build invocation:
  ```
  CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:/tmp/zig-linux-x86_64-0.14.0:$PATH \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs CARGO_HOME=/tmp/cargo-home-local \
    INFER_TRITON_PYTHON=/usr/bin/python3 INFER_CUDA_SM=89 \
    ZIG=/tmp/zig-linux-x86_64-0.14.0/zig \
    LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
    cargo build --release -p infer --no-default-features --features cuda,tilelang-attn
  ```

## Root Cause

The TileLang 0.1.9 lowering pipeline rejects this kernel at the
LayoutInference pass. Two earlier issues were uncovered and fixed
inline; a third deeper layout-inference failure is the actual blocker
and is in TileLang's TVM-based lowering, not in our kernel source.

**Issue 1 — target string format (fixed in `802c5fc8`).**
TileLang ≥ 0.1 dropped the legacy `cuda:<sm>` target alias. The error:
```
ValueError: Target kind "cuda:89" is not defined.
AssertionError: Target cuda:89 is not supported. Supported targets include:
  'auto', 'cuda', 'hip', 'metal', 'llvm', 'webgpu', 'c', 'cutedsl'.
  Pass additional options after the base name, e.g. 'cuda -arch=sm_80'.
```
Source: `crates/cuda-kernels/build.rs::tilelang_target` formatted
`cuda:{sm}` and `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py::parse_target`
required that prefix.

**Issue 2 — Q/K/V vs P dtype mismatch in the second GEMM.**
`gemm_base.py:88` asserts `A.dtype == B.dtype`. The kernel allocates
`p` (the softmax probabilities) as `accum_dtype = "float32"` and then
calls `T.gemm(p, v_tile, acc_o)` against `v_tile` in `bfloat16`. Older
TileLang versions appear to have auto-cast or accepted mixed dtypes
silently; 0.1.9 does not. Standard FlashAttention pattern is to narrow
the f32 softmax output to bf16 right before the P @ V matmul:
```python
p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
T.copy(p, p_bf16)
T.gemm(p_bf16, v_tile, acc_o)
```
This patch was applied locally and unblocks the dtype assert. **Not
committed** because Issue 3 still blocks AOT and the patch should land
together with whatever ultimately makes Phase 0 build green.

**Issue 3 — TVM layout-inference InternalError (BLOCKER).**
With Issues 1 + 2 cleared, `LowerAndLegalize` panics inside
`LayoutInference()`:

```
tvm.error.InternalError: loop_var_to_thread = d // 64 * 64
    + i // 32 * 32
    + i % 8 * 4
    + d % 8 // 2
    contains inner var d
```

Stack pointer:
```
tilelang/engine/phase.py:191  LayoutInference()(mod)
tilelang/3rdparty/tvm/python/tvm/ir/transform.py:167  RunPass(self, mod)
tvm::tl::LayoutInferencer::Substitute(...)
tvm::tl::BufferUseDefCollector::Run/RunInferStep(...)
tvm::tl::GemmNode::InferLayout(...)
tilelang/tileop/gemm/__init__.py:23  gemm_infer_layout
tilelang/tileop/gemm/gemm_mma.py:39  _make_mma_emitter
```

This is TileLang/TVM internals failing to express the GEMM tile→thread
layout for our (Q@K, P@V) pair. It is not a kernel-source bug — the
same kernel pattern is the canonical FlashAttention-2 layout used in
every TileLang attention example. The defect is in 0.1.9's
LayoutInferencer when handling the specific BLOCK_M=64, BLOCK_N=64,
HEAD_DIM=128, NUM_THREADS=128 shape combination on cuda -arch=sm_89.

We have not isolated whether 0.1.9 also fails on sm_90; the H100
verification host can confirm or rule that out separately.

## Fix

**What landed:**
- `802c5fc8 fix(cuda): use TileLang 0.1.9 target string format` — Issue 1.

**What is parked:**
- The dtype-narrowing patch (Issue 2) is correct standalone and matches
  the FlashAttention reference, but landing it without Issue 3 closed
  would leave `--features cuda,tilelang-attn` still broken — a no-op
  half-state per `feedback_no_half_states.md`. Hold until Issue 3 is
  resolved.

**What is required to unblock Phase 0:**
1. **Pin a TileLang version that actually compiles this kernel.** The
   plan §6 already calls for a version pin once the H100 spike runs;
   that pin needs to happen *before* the next AOT attempt, not after.
   Candidate path: bisect 0.1.0 → 0.1.9 to find the last version where
   `LayoutInferencer` accepts our prefill kernel, then pin that in
   `pyproject.toml::[project.optional-dependencies].tilelang`.
2. **Or simplify the kernel to a layout TileLang 0.1.9 accepts.** The
   error mentions an `i % 8 * 4 + d % 8 // 2` expression — likely the
   default TileLang shared-memory swizzle for HD128. Reducing
   BLOCK_M/BLOCK_N or removing `T.use_swizzle(panel_size=8)` may dodge
   the bug at a likely perf cost (re-bench afterwards).
3. **Or upstream the fix to TileLang.** Best long-term but blocks Phase 0
   on a third-party release cycle.

Recommended: option 1 (version pin) is the lowest-risk shortest path.

Per `tilelang-integration.md` §5 risk gate #2, the prescribed action
when AOT export fails on the chosen SM is to revert the Phase 0 commit
trio (`9896d25 76e044b 022e8dd`). I have **not** reverted on this
session because the user explicitly directed continued verification.
Reverting now would discard the FlashInfer-prefill forwarding fix
(`9896d25`) which is independently correct, plus the per-head-config
specialization (`76e044b`) which has no runtime effect when
`tilelang-attn` is off. If the team agrees Phase 0 is parked rather
than closed, leave the trio in place and treat this errors entry as
the recorded blocker.

## Rule

When pulling in a Python-driven kernel toolchain (TileLang, Triton),
**pin the version in `pyproject.toml` at the same commit as the
kernel-source change**. `tilelang>=0.1` is unversioned in practice —
0.1.0 → 0.1.9 spans a ~9-month window with breaking changes to the
target-string format and the LayoutInferencer pass. The Phase 0 plan
called this out as a §6 follow-up; this incident proves the version
pin must precede the AOT attempt, not follow it.

Concretely:
- `pyproject.toml::[project.optional-dependencies].tilelang` ships
  `tilelang>=0.1` today. Before re-attempting Phase 0, change to
  `tilelang==<verified-version>`.
- The verifier runbook §1.4 already records the exact version with
  `python3 -c 'import tilelang; print(tilelang.__version__)'`. The
  output of this command must be cross-referenced against the pin.

## Cross-references

- Plan: [`docs/plans/tilelang-integration.md`](../../plans/tilelang-integration.md) §5 risk gate #2
- Runbook: [`docs/plans/tilelang-integration-verification.md`](../../plans/tilelang-integration-verification.md) §2.2
- Pending stub: [`docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`](../wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md) — stays in place; this errors entry is the recorded blocker, not a closure.
- Phase 0 commit trio: `022e8dd` `76e044b` `9896d25`. Revert path
  available per plan §5/§7 if the team decides to close Phase 0
  formally.
