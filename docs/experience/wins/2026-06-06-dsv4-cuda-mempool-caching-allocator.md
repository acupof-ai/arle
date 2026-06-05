# DSv4/CUDA: caching allocator — raise cuMemAllocAsync release threshold to MAX (kill per-step decode alloc churn)

**Date:** 2026-06-06. **Backend:** CUDA. **Scope:** `crates/cuda-kernels/src/tensor.rs`
(`DeviceContext::on_device`). **Status:** landed default-on (opt-out
`ARLE_CUDA_MEMPOOL_RETAIN=0`); decode `tok/s` A/B **pending-remote**. Found by ckl
("怀疑你弄的不对" re memory reuse — correct suspicion).

## Context

cudarc allocates every `CudaSlice` through `cuMemAllocAsync` on the device's default
memory pool. **ARLE never set the pool's release threshold** (grep-confirmed), so it
inherited the CUDA default of **0**: when the pool holds > 0 bytes of freed-but-cached
blocks, the allocator **releases them back to the OS at the next stream/context sync**.
DSv4 decode has a per-step sync (`executor.rs:469`, `dsv4.rs:643/1249`), so every
per-step `HiddenStates::uninit`/`alloc_zeros` in the eager path (48 sites in
`attention.rs`, 10 in `hc.rs`, plus `dsv4.rs`) **re-allocates from the OS each step** —
the same `cuMemAllocAsync` churn #29 fixed, but #29 only covered the MoE path via a
persistent `Dsv4MoeDecodeScratch`; attention/HC/structural allocs still churned.

## What Worked

- **Raise `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` to `u64::MAX`** at `DeviceContext::on_device`
  (cudarc `mem_pool::set_attribute`). The pool now **caches freed blocks and reuses them
  across syncs** instead of returning them to the OS — a true caching allocator, exactly
  what PyTorch's caching allocator + SGLang do. One-time, best-effort (warn on failure).
- `trim_memory_pool()` (`cuMemPoolTrimTo(0)`) still reclaims VRAM explicitly when needed
  (weight offload) — the threshold governs *automatic* release-at-sync, not explicit trim.
- Opt-out `ARLE_CUDA_MEMPOOL_RETAIN=0` restores the old behavior (for the A/B + a
  memory-tight escape hatch).

## Honest read / what to verify (pending-remote)

- **Decode `tok/s` A/B** (`ARLE_CUDA_MEMPOOL_RETAIN` default vs `=0`) on the TP=8/EP=8
  pod — expected win is the residual per-step alloc overhead outside the MoE scratch
  (#29's MoE-only slice was 36.5% of decode `cuMemAllocAsync`; the attention/HC residual
  is the target here).
- **Prefill-peak-held risk:** threshold=MAX means the pool holds up to peak allocation
  and never auto-shrinks. A large prefill (512/2048) raises the held peak; verify it does
  **not** OOM at the production shape (the KV pool + mem budget must leave headroom, as
  SGLang's mem_fraction does). If a shape OOMs, call `trim_memory_pool()` at the
  prefill→decode boundary, or opt out for that shape.

## Rule

The async allocator (`cuMemAllocAsync`) with the **default 0 release threshold is not a
caching allocator** — it returns freed blocks to the OS at every sync, so any per-step
alloc/free in a synced loop churns the OS. For a serving runtime, set the release
threshold to MAX at context init (PyTorch/SGLang do this) and reclaim explicitly via
trim. A persistent-scratch fix for one stage (#29, MoE) leaves the other stages churning
until the *pool itself* caches.
