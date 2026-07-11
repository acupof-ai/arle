# Unified kernel set — one full-build binary serves Qwen AND DSv4

> Verification for `89ea8e7c4` (delete model-family kernel partition) +
> `93dcf4bef` (prebuilt-manifest TARGET fix). Build-shape change: runtime kernel
> dispatch is unchanged per model, so the gate is correctness (both families
> serve, no NOT_SUPPORTED) + the prebuilt round-trip, not a new SLO number.

## SLO-shape probed?  N — build-shape/correctness change; runtime kernels dispatched per model are byte-identical to a prior full build

## Context

The `ARLE_CUDA_KERNEL_SET=dsv4_flash|opd_gdr` sets stubbed the Qwen/GDR TileLang
attention FFI with `CUDA_ERROR_NOT_SUPPORTED` to skip TileLang AOT — trading a
one-time cacheable build cost for a persistent runtime footgun (a binary that
links + loads but crashes on the first Qwen forward). Kernel builds must
differentiate only on physical axes (SM tier, CUDA version), never on model
family. Deleted the partition; every SM target now builds every registry kernel.

## What Worked

Pod (H20 sm_90, nvcc 12.9, `INFER_TILELANG_PYTHON=/host/tilelang-preserve/.venv`):

1. **Full source build, no `ARLE_CUDA_KERNEL_SET`** — `BUILD_EXIT=0`, TileLang
   AOT actually ran (no "AOT skipped" — that path is deleted): "built per-SM
   cubins ... HD64/HD128/HD256 prefill+decode + Qwen3.5 GDR". cargo `Finished in
   ~1m55s` warm.
2. **Qwen3.6-27B-FP8 serves on the full-set binary** — 0 `CUDA_ERROR_NOT_SUPPORTED`,
   FP8 dense DeepGEMM warmed. `The capital of France is` → ` Paris.`
3. **DSv4-Flash-FP8 serves on the SAME binary** (rebuilt `--features cuda,nccl`
   for TP=4, identical kernel bytes, AOT ran) — TP=4 on GPUs 4-7, allreduce MoE +
   deepgemm experts, all ranks engine-ready, 0 regression. `The capital of
   France is` → ` Paris.` Re-verified Qwen on the same nccl binary (` Tokyo.`).
4. **Reproducible prebuilt bundle** — export produced `libkernels_cuda.a`
   (10.77 MB) + `libtilelang_kernels_aot.a` (2.95 MB) + `arle-cuda-kernels.manifest`
   + `manifest.json`, 14 MB total. After the TARGET fix (`93dcf4bef`) a fresh
   consumer build with NO TARGET override consumes it cleanly ("skipping nvcc and
   TileLang AOT"), no manifest panic.
5. **Consumer binary serves Qwen** — `Two plus two equals` → ` four.` The
   released bundle carries the full kernel set.

## Problems

- **Ship-blocker found + fixed (`93dcf4bef`):** the content-addressed manifest
  byte-match (`6cb2c0054`) rejected every exported bundle. `cuda_prebuilt_manifest.sh`
  keyed on `target=$TARGET`; cargo injects `TARGET` into build.rs subprocesses
  but the standalone export script has it unset → producer `target=` vs consumer
  `target=x86_64-unknown-linux-gnu` → 100% byte-mismatch panic. Dropped `TARGET`
  (redundant with `rustc_id`'s `host:` line; CUDA cubins key on SM arch, not the
  Rust triple).
- **Build cost, not runtime cost:** the full AOT (~53s cached / ~251s cold) is
  the tax the partition skipped. It is cacheable and, for deploys, absorbed by
  the prebuilt bundle (consumer skips AOT). Runtime dispatch per model is
  unchanged — DSv4 still routes FlashMLA/DeepGEMM; the extra Qwen TileLang cubins
  are present but uncalled. No serving perf delta vs a prior full build (see
  [`2026-07-11-bench-guidellm-fp8-operator-convergence.md`](2026-07-11-bench-guidellm-fp8-operator-convergence.md)).
- Every pod CUDA build now needs the pinned TileLang venv (full AOT always runs);
  system tilelang 0.1.8 false-passes the sm_90 gate.

## Rule

**Differentiate kernel builds only on physical axes (SM tier, CUDA version),
never on model family.** A model-family build partition that stubs kernels
produces a binary that links and loads but crashes on the first forward — a
runtime footgun far worse than the one-time, cacheable build cost it saves.
Build cost belongs at the build layer (AOT cache / prebuilt bundle), not traded
for a stub-crash binary.
