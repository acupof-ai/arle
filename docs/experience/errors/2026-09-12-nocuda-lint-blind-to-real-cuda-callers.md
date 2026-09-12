# Real-CUDA build blind spot: no-cuda lint cannot see `cfg(not(no-cuda))` callers

## Context

Two merged visibility refactors — `7683dc256` (autograd op/tensor narrowing)
and its sibling deletions — passed the Mac pre-push lint and CI Lint mirror,
but left the real-CUDA build (`--features cuda,nccl` without `no-cuda`) unable
to compile. Nothing on a Mac compiles that feature set (no nvcc), so the
breakage shipped. A pod `cargo check --workspace --features cuda,nccl
--all-targets` found three stacked faults:

1. **E0599 — deleted method with a live in-crate caller.**
   `TapeDtype::nvrtc_prelude` was deleted as a zero-caller item, but
   `backend_cuda/kernels.rs:674` `concat_sources` still called it under
   `#[cfg(not(feature = "no-cuda"))]`. The no-cuda lint compiles out that whole
   function, so the call site was invisible. Fix: inline the two-arm dtype
   prelude at `concat_sources`.
2. **E0603 ×84 — visibility narrowed past sibling modules.**
   `F32Operand` / `Bf16Operand` were narrowed `pub(super)` → private inside
   `backend_cuda/handle.rs`, but sibling modules of `backend_cuda`
   (`elementwise.rs`, `matmul.rs`, …) name their return types. Fix:
   `pub(in crate::backend_cuda)`, which is the actual scope.
3. **E0283 ×4 — untyped `Iterator::product` inside `assert_eq!`.**
   `test_cuda_lazy_ops.rs:755,1988` compared against
   `shape.iter().product()`; `assert_eq!` supplies no expected type, so the
   product type parameter was ambiguous under the real build. Direct
   `let x: usize = …` sites annotated already. Fix: `.<product::<usize>()`.

## Root cause

The Mac/CI lint command is `cargo clippy … --features cuda,no-cuda`: the
`no-cuda` feature compiles out the actual CUDA code (`#[cfg(not(feature =
"no-cuda"))]`), while still gating the crate as "cuda". It validates feature
plumbing and backend-isolation cfgs, but never the kernels, handle methods, or
cuda-gated tests. A refactor that deletes or hides an item whose only users are
real-CUDA callers sails through. The "zero callers" audit in `7683dc256` was
done under that same feature set, so it found no callers for an item that had
one.

## Fix

- Restore the callers' visibility / inline the deleted prelude / annotate the
  two test expressions (hotfix lane `hotfix-cuda-real-check`, 3 commits).
- Verification: pod `cargo check --workspace --features cuda,nccl --all-targets`
  with no `no-cuda` — `CUDA_CHECK_EXIT=0`.

## Second instance: feature-gated caller in the root crate (Vulkan)

The same blind-spot class broke the Vulkan build the same day.
`infer_vulkan::forward::set_submit_cap` (added in `ff84fd0b6`) was deleted as
a zero-caller item in `9df269d1a` (#344, "gate test-only items with
cfg(test)…"), which collapsed the `AtomicUsize` cap to a private `const`. Its
only caller lives in the **root crate** at `src/backends/vulkan.rs:18`, behind
the root `vulkan` feature. A per-crate `infer-vulkan` clippy never compiles
the root crate, and the other feature sets don't compile `src/backends
/vulkan.rs`, so the missing cross-crate symbol was invisible. Found by the
feature-matrix sweep `cargo check --features vulkan,no-cuda` (E0425); fix
restores the `AtomicUsize` and the `pub fn set_submit_cap`. Same rule: a
"zero callers" deletion must grep the code that is compiled under every
feature the item is visible to, including root-crate feature gates.

## Third instance: example targets, not lib targets (train CUDA examples)

The third case was in `--all-targets` rather than the library. Once the
lib-side real-CUDA lints were clean, `cargo clippy -p train -p cli --features
cuda --all-targets` surfaced nine pre-existing lints in the CUDA train
**examples** (`opd_step_cuda_*`, `qwen36_fp8_lora_*`): `collapsible_if`,
`undocumented_unsafe_blocks` (NVTX FFI), `clippy::exit` on `--help`, and
`unnecessary_sort_by`. The no-cuda lint and even a real-CUDA `--lib` check
compile no examples, so a clean library is exactly what exposed them: they had
been behind both the cfg blind spot and the target-selection gap. Fix was
lint-only (let-chains, safety comments, `sort_by_key(Reverse(..))`, and
`--help` returning `Ok(None)` from the parser instead of `process::exit(0)`).
Rule addition: when the library first compiles clean under real CUDA, the gate
must include `--all-targets`; example and bench targets carry their own
unverified code and fail independently of the lib.

## Fourth instance: test targets (autograd CUDA-only tests)

The fourth case was the remaining target class: integration **tests**. After
the lib, root-crate, and example targets were clean, a pod
`cargo clippy --workspace --all-targets --features cuda,nccl,deepep
-- -D warnings` failed in two autograd `tests/` files that compile only under
real CUDA:

- `test_cuda_marlin_fp4_share.rs:179` — `CudaStream::memcpy_stod`, deprecated
  in cudarc (`-D deprecated`); use `clone_htod`.
- `test_linear_attention.rs:1997,2035` — two `Fn` closures bound `mut`
  (`unused-mut`); their mutable state is local or passed by parameter, not
  captured.

`cargo check` reports none of these — check type-compiles but does not run
clippy lints — which is why the rule-5 `CUDA_CHECK_EXIT` gate let all four
instances through. With this case the blind spot covers the whole target
matrix: lib (#368), the root crate (#370 Vulkan), examples (#375 train), and
tests (this fix). The pod gate therefore has to be
`cargo clippy --workspace --all-targets --features cuda,nccl -- -D warnings`
without `no-cuda`, not `cargo check`.

## Rule

For any crate whose CUDA code is `#[cfg(not(feature = "no-cuda"))]`, the
no-cuda lint is necessary but not sufficient: it proves the stub/feature surface
compiles, not that the CUDA code does. Every change that deletes, hides, or
renames an item in those crates needs a real-CUDA gate (pod, no `no-cuda`)
before merge, and a "zero callers" deletion claim must grep the
`cfg(not(no-cuda))` code, not just the no-cuda-compiled tree. That gate must be
`cargo clippy --workspace --all-targets --features cuda,nccl -- -D warnings`:
`--workspace --all-targets` to cover lib, root-crate, example, bench, and test
targets alike, and `clippy -D warnings` rather than `cargo check`, because
deprecated-item and unused-mut lints (the fourth-instance failures) are
invisible to check. Four separate hotfixes (#368, #370, #375, and the test
case above) established each arm of this matrix.
