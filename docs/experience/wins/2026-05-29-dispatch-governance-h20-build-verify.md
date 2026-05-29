# Dispatch-Governance Refactor — H20 Real-CUDA Build Verification

> Verifies the dispatch-governance landing (analysis →
> [`../../reviews/2026-05-29-gpu-dispatch-governance-analysis.md`], plan →
> [`../../plans/gpu-dispatch-governance.md`], oplib design →
> [`../../plans/backend-operator-library.md`]) on real CUDA. Commits:
> DispatchPolicy ops-layer + loud fallbacks + model-layer migration, codex-fix
> (resettable cache), and round-2 (`oplib::linear` relocation, scheduler-knob,
> explain-dispatch). All on `origin/main` (merge `789b6fea`).

## Goal (type: regression / verification)

Confirm the refactor — which is **behavior-preserving** (env-knob unification,
warn-only fallbacks, a pure-function relocation of linear dispatch selection) —
compiles and links on real `nvcc`, with the cuda-gated pieces that a Mac cannot
typecheck (the `LinearKernelPlan → oplib::linear::LinearKernel` rename across every
launch site) proven on the GPU toolchain.

## Hypothesis

A behavior-preserving refactor links clean on `nvcc` with zero numeric change. The
risk is purely **compile/link** of the cuda-gated rename (Mac `cargo check
--features cuda,no-cuda` typechecks the non-cuda surface but link-fails on FFI
symbols; the rename touches cuda-only match arms). Expected: build PASS, no errors.

## Command

```bash
# pod, origin/main @ 789b6fea, target/ cache reused (incremental)
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda,nccl --bin infer
# supplementary: cuda test harness + pure tests (filters after --)
CUDA_HOME=/usr/local/cuda cargo test --release -p infer --features cuda --lib -- dispatch_policy oplib load_hybrid_w4_marlin
# Mac-side (no nvcc): typecheck + pure tests
CUDARC_CUDA_VERSION=12060 cargo check -p infer --no-default-features --features cuda,no-cuda
cargo test -p infer --lib            # default features
```

## Environment

- 8× NVIDIA H20 (97 GB) pod, CUDA 12.9 (`nvcc` V12.9.86), `cargo` 1.x, `cuda,nccl`
  feature set, `deepep-sys` stub (`ARLE_DEEPEP_DIR` unset — not needed for this path).
- Mac (M-series, no `nvcc`): typecheck via `CUDARC_CUDA_VERSION=12060`; pure-logic
  tests under default features (cuda `cargo test` link-fails on Mac).
- Source: `origin/main` merge `789b6fea`. Pod tree re-synced (`git reset --hard
  origin/main`); `dispatch_fallback:` log strings confirmed present in
  `target/release/infer` (anti-stale-tree precondition, per
  `errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).

## Results

| Check | Result |
|---|---|
| **H20 build, T1+B+C** (`--bin infer`, cuda,nccl) | **PASS** — `Finished release in 12m12s`, 0 errors, 7 pre-existing lib warnings |
| **H20 build, round-2** (oplib + codex-fix + scheduler-knob) | **PASS** — `Finished release in 12m06s`, `BUILD_DONE_rc=0`. Verifies the cuda-gated `LinearKernelPlan→LinearKernel` rename links on `nvcc` |
| **Symbols in binary** (anti-stale) | **PASS** — `strings target/release/infer \| grep dispatch_fallback:` → 2 hits (C's loud-fallback code is in the binary) |
| **Mac default lib suite** | **604 passed, 0 failed, 14 ignored** — incl. `oplib::` full-sweep `plan()`-vs-legacy-oracle equivalence (3) + `dispatch_policy::` parser/token-set (7, incl. bypass presence + r4-only-`"1"`) |
| **Mac cuda,no-cuda typecheck** | clean (no new warnings) |
| **H20 cuda test harness + pure tests** | supplementary run (see Problems) |

The headline property is realized: `oplib::linear::plan()` is backend-neutral, so
its full input-cross-product equivalence vs the legacy resolver is a **CPU unit
test** — selection correctness is provable without a GPU or a bench.

## Problems

- **Loader test model-blocked.** `load_hybrid_w4_marlin_dispatches_to_w4a8_prefill`
  (the test the codex-fix repairs) is cuda-gated and needs
  `models/Qwen3-4B-W4-hybrid-zpfix`, **absent on the pod** → cannot validate the
  cache-reset fix end-to-end on CUDA. The fix's *logic* is verified on Mac
  (`dispatch_policy` tests, by-value-return compile, the guard's `reset_dispatch_policy_cache`
  call on env set/restore). End-to-end loader validation stays **pending** until the
  model is staged.
- **Operator error (mine):** first `cargo test` invocation passed multiple filters
  before `--` (`... --lib dispatch_policy oplib load_hybrid_w4_marlin`) → `error:
  unexpected argument 'oplib'`. Re-run with filters after `--`. (Build was unaffected;
  `BUILD_DONE_rc=0`.)

## Learnings

- **`CUDARC_CUDA_VERSION=<ver>` unlocks Mac CUDA-Rust typecheck** — cudarc's
  `build.rs` panics on missing `nvcc` unless this env short-circuits it. Turns most
  CUDA refactors from "pending-remote everything" into "typecheck locally, pod only
  for link/run". (`reference_cudarc_mac_typecheck` memory.)
- **A `--bin` build already verifies cuda-gated *non-test* renames** — no need for a
  cuda test-harness compile to prove `ops/linear.rs` links; the bin pulls the whole
  lib. Reserve the (expensive) `cargo test --features cuda` compile for when you
  actually need to *run* a cuda-gated test.
- **Backend-neutral `plan()` is the lever** that moves "did dispatch select the right
  kernel?" from a pod bench to a CPU unit test — the core governance win, now
  demonstrated.
- **cuda `cargo test` filters go after `--`** (libtest OR's them); cargo positionals
  before `--` reject the 2nd filter.

## Δ vs baseline

First verification of this refactor — no perf baseline applies (behavior-preserving;
no numeric/throughput change expected or claimed). The pending-remote bench notes in
the constituent commit bodies are discharged for **build+link**; the loader-test
end-to-end and any throughput A/B remain out of scope (behavior-preserving change).
