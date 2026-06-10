# CUDA build chain: bounded parallel nvcc pool + prebuilt-pack producer script

**Status: pending-remote** — build-time deltas need a pod (or Colab) run with
nvcc; this Mac can only typecheck the build.rs change. Runtime kernel
artifacts are byte-identical by construction (same argv per file, same `ar`
queue order), so no inference bench is owed — the pending measurement is
build wall-clock only.

## Context

Commissioned by the 2026-06-10 devops/compile-chain review; implements lever
#6 of [`docs/research/2026-06-05-build-compile-speed-optimization.md`](../../research/2026-06-05-build-compile-speed-optimization.md)
and closes the "prebuilt escape hatch is manual-harvest" gap (§Verified facts).

- `crates/cuda-kernels/build.rs` compiled all ~62 native `.cu` files through
  a **serial** nvcc loop. Any `csrc/**` touch reruns the build script and
  recompiles everything one-by-one.
- `ARLE_CUDA_KERNELS_PREBUILT_DIR` (link-only fast path, skips all nvcc +
  TileLang, ~5s) had a consumer in build.rs but **no producer** — packs were
  harvested by hand from `target/release/build/cuda-kernels-*/out`.

## What changed

1. **Bounded parallel nvcc pool** (`run_nvcc_jobs`, build.rs): args are built
   serially in queue order (so `obj_files`/`ar` ordering — and the FlashMLA
   stub-vs-shim symbol-resolution constraint — are byte-identical to the old
   loop), then jobs run on `min(cores, 8)` scoped threads pulling from an
   atomic index. `ARLE_NVCC_PARALLEL` overrides; `=1` restores serial. A
   worker panic propagates at scope join and fails the build loudly.
   Composes with `ARLE_NVCC_WRAPPER=sccache` (concurrent-safe).
2. **`scripts/export_prebuilt_cuda_kernels.sh <dest> [target-dir] [profile]`**
   — picks the newest complete `cuda-kernels` OUT_DIR, copies
   `libkernels_cuda.a` + `libtilelang_kernels_aot.a` (+ DeepEP sidecar if
   present) and writes a provenance `manifest.json` (HEAD, cuda-kernels tree
   object, dirty flag, nvcc version, SM list) so stale packs are detectable
   (`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).
   Verified against a synthetic target layout (skips incomplete candidates,
   provenance correct).
3. **`make build-cuda` auto-wraps with sccache when installed** (rustc +
   nvcc + TileLang cubins) — same `prefer_sccache` pattern as
   `dsv4_fast_build.sh`, no-op when sccache is absent.

## Pending measurement (pod)

```
# cold + warm, serial vs pool, on 8xH20:
rm -rf target/release/build/cuda-kernels-*
ARLE_NVCC_PARALLEL=1 time cargo build --release --features cuda,nccl  # baseline serial
rm -rf target/release/build/cuda-kernels-*
time cargo build --release --features cuda,nccl                       # pool (default min(cores,8))
```

Expect roughly Nx on the native-`.cu` half of a clean kernel build; TileLang
AOT half is unchanged (already sccache-covered since 2026-06-05).

## Rule

- Parallelize build steps only when output ordering is decided by queue
  order, not completion order — and say so at the site that depends on it.
- A cache/fast-path mechanism is not "done" when the consumer lands; it's
  done when there's a producer, a provenance key, and a staleness story.
