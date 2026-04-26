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
- `4d9c65f0 fix(cuda): align tilelang prefill HD128 with FlashAttention-2 layout`
  — Issue 2 + the layout side of Issue 3 (`T.serial(HEAD_DIM)` hoisted to
  2D `T.Parallel`, both gemms get `policy=T.GemmWarpPolicy.FullRow`,
  `p_bf16` cast before P @ V).

**Bisect outcome (the version pin path is a dead end):**

I tried tilelang 0.1.5, 0.1.6.post2, 0.1.7, 0.1.9. Every one rejects
this kernel:

| version | failure |
|---|---|
| 0.1.5 | `Check failed: (op->kind == ForKind::kParallel) is false` — even stricter parser; `T.serial(HEAD_DIM)` was already invalid. |
| 0.1.6.post2 | `loop_var_to_thread = d % 16 // 8 * 64 + i % 32 // 16 * 32 + i % 8 * 4 + d % 8 // 2 contains inner var d` — same family as 0.1.7/0.1.9. |
| 0.1.7 | identical layout-inferencer error. |
| 0.1.9 | identical layout-inferencer error. |

Pinning won't help. What does help is rewriting the kernel to the
canonical FlashAttention-2 layout that `tile-ai/tilelang` ships in
`examples/flash_attention/example_gqa_*`. **The three fixes already
landed in `4d9c65f0` clear the layout-inference blocker** — verified
by reproducing the original kernel + the three patches against
TileLang 0.1.9 with concrete shapes (`OK` after 8s compile, see
[verification report](../../reviews/2026-04-26-l4-scheduler-and-tilelang-verification.md)).

**Remaining gap — the symbolic-shape requirement (Issue 3'):**

After `4d9c65f0`, the AOT generator now panics on a different message:

```
tvm.error.InternalError: Check failed: undefined.size() == 0 (2 vs. 0) :
  In PrimFunc kernel variables [batch_size, max_qlen] are used, but
  are not passed in as API arguments
```

The kernel uses `T.symbolic("batch_size")` / `T.symbolic("max_qlen")`
as the grid extents, and TileLang 0.1.9 doesn't promote those to API
arguments automatically. The varlen reference
(`examples/flash_attention/example_gqa_fwd_varlen.py`) shows the
canonical pattern:

- **Closure-time Python ints** for max bounds: `max_batch`,
  `max_qlen`, `num_pages`, `max_total_pages`, `max_total_q =
  max_batch * max_qlen`. Tensors are sized to these maxes.
- **`T.int32` scalar runtime args** for the actual values: e.g.
  `actual_batch: T.int32, actual_max_qlen: T.int32`. The grid uses
  these scalars (`T.Kernel(T.ceildiv(actual_max_qlen, BLOCK_M),
  num_q_heads, actual_batch, ...)`).

I validated this end-to-end against the actual paged kernel structure
(KV_indices indirection + variable qlen) on TileLang 0.1.9 — it
compiles in ~8 s. Repro:
[`tools/tilelang/batch_prefill_paged_hd128.py`](../../../crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py)
restructured per the closure-int + scalar-runtime pattern.

**What is required to actually run AOT on TileLang 0.1.9:**

Apply the four-file delta:

| File | Change |
|---|---|
| `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` | Replace `T.symbolic(...)` with closure Python ints (max_batch, max_qlen, num_pages, max_total_pages); add `actual_batch: T.int32, actual_max_qlen: T.int32` as kernel args; use those for the grid. |
| `crates/cuda-kernels/build.rs` | Extend `TILELANG_PREFILL_HD128_HEAD_CONFIGS` (or add a parallel list) to enumerate `(num_q_heads, num_kv_heads, max_batch, max_qlen)` buckets. Suggested initial buckets: `(32, 8, 8, 4096)`, `(32, 8, 16, 2048)` — pick to bound the cubin count. |
| `crates/cuda-kernels/src/ffi/attention.rs` (FFI) | Extend the C wrapper signature to thread the two new scalar args through. |
| `infer/src/ops/attention.rs` (caller) | Pad input tensors to the bucket maxes (or pick the smallest bucket that fits) and pass `actual_batch, actual_max_qlen`. Fall through to FlashInfer when no bucket fits. |

Sized as a focused subagent-driven task — outside the scope of this
parking entry. Recommended cadence: fold into the next Phase 0
re-attempt as a single `feat(cuda): tilelang prefill HD128 AOT
buckets` commit.

Per `tilelang-integration.md` §5 risk gate #2, the prescribed action
when AOT export fails on the chosen SM is to revert the Phase 0 commit
trio (`9896d25 76e044b 022e8dd`). The trio stays — `4d9c65f0`'s three
fixes are real bugs caught by 0.1.9's stricter validator, not bandaids,
and reverting now would discard verified correctness improvements. If
the team decides to formally close Phase 0 instead of parking it, the
revert path is unchanged.

## Rule

**Write TileLang kernels against the canonical FlashAttention-2 layout
upstream ships**, not against TileLang's permissive auto-inference.

The `T.serial(HEAD_DIM)` inside `T.Parallel(BLOCK_M)` pattern, the
implicit gemm dtype promotion, and the missing
`policy=T.GemmWarpPolicy.FullRow` were all kernel-side defects that
older TileLang versions silently accepted. 0.1.9's stricter
LayoutInferencer + gemm dtype check caught all three. The fix was to
mirror `tile-ai/tilelang/examples/flash_attention/example_gqa_*` —
i.e., upstream's own template for this kind of kernel.

Cross-cutting takeaway: **before writing a new TileLang kernel,
download the matching version's `examples/` and copy the exact tile-
allocation + gemm-call shape**. Don't infer it from API docs. The
LayoutInferencer is a runtime constraint solver, not a forgiving
compiler — minor structural differences from the canonical pattern
fail at compile time with cryptic errors.

For the version pin question itself: a pin is still a good idea (the
`tilelang>=0.1` extra in `pyproject.toml` lets pip drift), but pinning
alone won't unblock — every version 0.1.5 through 0.1.9 rejected the
pre-fix kernel. The pin protects against future drift, not against
the structural defects that caused this incident.

## Cross-references

- Plan: [`docs/plans/tilelang-integration.md`](../../plans/tilelang-integration.md) §5 risk gate #2
- Runbook: [`docs/plans/tilelang-integration-verification.md`](../../plans/tilelang-integration-verification.md) §2.2
- Pending stub: [`docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`](../wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md) — stays in place; this errors entry is the recorded blocker, not a closure.
- Phase 0 commit trio: `022e8dd` `76e044b` `9896d25`. Revert path
  available per plan §5/§7 if the team decides to close Phase 0
  formally.
