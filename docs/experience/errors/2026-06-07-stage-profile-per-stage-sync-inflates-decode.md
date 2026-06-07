# DSv4 stage_profile per-stage sync inflates B=1 decode ~1.7×

## Context

Profiling DSv4 decode per-stage with the in-tree `stage_profile` (env
`ARLE_DSV4_STAGE_PROFILE=1`, via `dsv4_resident_ab`). It reported decode =
**48-51 ms/token**, a **~16 ms/token "unaccounted/host gap" (≈32% of wall)**, and
`mla_attn` (12.4ms) as the dominant cost. I drew two conclusions and shipped them
to the user before re-checking:
1. "decode is ~32% host-launch-bound" (→ batched decode is the #1 lever).
2. "`mla_attn` dominates GPU at 12.4ms".

Both were **measurement artifacts**. (Earlier in the same session I had also
twice mis-attributed the "unaccounted" bucket — first as "18ms routed-MoE", then
as "16ms MoE glue" — both falsified by adding stage labels. The user's "数据真实吗"
forced the audit that found the real cause.)

## Root Cause

`crates/infer-cuda/src/stage_profile.rs:70` — `profile()` does
`stop.synchronize()` on **every** labeled stage when active. At B=1 decode, ~24
labels × 43 layers = **~hundreds of device syncs per token**, which serialize the
pipeline and block the host from launching ahead. This:
- **inflates the wall ~1.7×** (the syncs become the wall), and
- inflates `unaccounted_cuda = wall − Σ(stage cuda_ms)` into a fake "host gap".

Internal proof (same run): warmup steps run **before**
`set_dsv4_stage_profile_active(true)` (un-profiled), steady steps **after**
(profiled). Across 3 runs: warmup = **26-27 ms/token**, profiled steady = **48-51
ms/token**. The un-profiled 26ms matches the independent 2026-06-07 win. Clean
nsys (`stage_profile` OFF) confirmed **29 ms/token**, **66% GPU-busy / 34%
launch-gap (~10ms, not 16ms)**, and that the dominant GPU kernels are the
**scalar projection GEMV** (`dsv4_fp8_gemv_batch`, 3.62ms) and the **mHC
Sinkhorn** (`dsv4_mhc_params`, 3.06ms) — the FlashMLA attention math is only
~1.85ms, so the `mla_attn` "stage" was a composite that bundled GEMV + mHC +
overhead. Same family of trap as
[[../../memory/reference_nvtx_range_ending_in_sync_phantom_bottleneck]] — a
profiling sync masquerading as a bottleneck — but per-stage instead of
end-of-step.

## Fix

For decode wall-clock + per-op attribution, use `nsys
--capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx` (no osrt)
on a **`stage_profile`-OFF** run, with `INFER_DSV4_AB_PROFILE_VARIANT` firing
`cudaProfilerStart` after warmup so only the steady window is captured. Read GPU
time from `cuda_gpu_kern_sum` and idle from `window_wall − kern_sum`. See
[`../wins/2026-06-07-dsv4-decode-nsys-real-breakdown.md`](../wins/2026-06-07-dsv4-decode-nsys-real-breakdown.md).

## Rule

`stage_profile`'s per-stage `cuda_ms` (CUDA-event) is trustworthy as **relative
per-op GPU time** — a sync doesn't change a kernel's own duration. Its **wall,
`host_ms`, and `unaccounted_cuda` are NOT** — the per-stage `stop.synchronize()`
serializes B=1 (~hundreds/token) and inflates the wall ~1.7×. Never quote the
stage-profiler's wall or its "host gap" as the real decode latency or a
host-bound verdict; cross-check with the un-profiled warmup ms/token (free, in
the same run) and confirm host-vs-GPU split with nsys, not the syncing profiler.
