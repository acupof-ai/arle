# B-3.3.5 DeepGEMM grouped expert dispatch — JIT compiles but runtime crashes on H20

## Context

After 38bf157b fixed the long-running JIT compile failure
(`-std=c++20` → `-std=c++17` in the ARLE nvcc invocation), DeepGEMM
JIT now compiles successfully on the pod (8 × H20, CUDA 12.2,
gcc 8.3). Trace of progress through this axis today:

1. **B-3.3.5 code wire-in** (commit 67ac6400) — added grouped DeepGEMM
   branch to `forward_native_deepep_routed_gpu`, mirroring the
   `use_deepgemm_experts` path from `forward_deepep_routed_gpu`.
2. **JIT compile fight** (~7 build attempts) — chased ghost cache
   issues (compiler.hpp patch + GCC 11 ccbin + DG_JIT_CPP_STANDARD env
   + multiple cargo cache invalidations) before discovering ARLE's
   own `csrc/gemm/deepgemm_native.cu` had `-std=c++20` hardcoded in
   the nvcc cmdline. That was the actual root cause, not DeepGEMM's
   own compiler.hpp.
3. **Real JIT compile success** (commit 38bf157b) — `-std=c++17` fix
   in deepgemm_native.cu makes nvcc compile cute headers correctly.

The current symptom is a **runtime kernel failure**:

```
2026-05-27T16:41:56  ERROR ... native-deepep combine failed:
  deepep call returned status -2: sync after combine:
  unspecified launch failure
2026-05-27T16:41:56  ERROR ... H2D copy failed: DriverError(
  CUDA_ERROR_LAUNCH_FAILED, "unspecified launch failure")
DeepEP timeout check failed: rank = 4, thread = 6, value = 0
```

Cascade pattern: one of the early DeepGEMM kernel invocations crashes
(presumably illegal memory access or SM 90 sub-feature mismatch), GPU
context enters error state, subsequent DeepEP combine's host-poll
times out waiting for the (poisoned) recv buffers, then every
subsequent H2D / kernel launch fails with sticky launch failure.

## Root Cause (suspected, not yet verified)

Three candidate root causes, ranked by likelihood given evidence:

### Candidate 1: H20 SM 9.0 sub-feature mismatch with DeepGEMM

DeepGEMM is tuned for H100 SM 9.0 with full TMA + cluster + thread-
block features. H20 is the "削减版" of H100 with the same SM 90 ISA
but reduced bandwidth + some SM features may behave differently. The
JIT'd kernel might use a feature path that works on H100 but trips
on H20.

Evidence:
- `unspecified launch failure` is the generic CUDA error for any
  device-side trap (illegal mem, OOB, async copy crash).
- DeepGEMM upstream tests primarily on H100; H20 isn't in their CI.
- Our `groups=32 max_m=8 n=4096 k=4096 scale_stride_m=8`
  parameters are valid shapes for DeepGEMM in principle.

### Candidate 2: B-3.3.5 packed_x / packed_token / packed_weight layout mismatch

Our `forward_native_deepep_routed_gpu` (B-3.3.5 wire-in) passes
`scratch.packed_x` as `expert_hidden`, `scratch.packed_token` as
`expert_route_slot`, `scratch.packed_weight` as `expert_weight` to
`forward_deepgemm_all_dsv4_experts_gpu`. The baseline path
(`forward_deepep_routed_gpu`) constructs these via a different code
path that may have specific alignment / scale-stride guarantees that
our path doesn't replicate.

Evidence:
- The 32 DeepGEMM "FP8 expert cache built" log lines confirm
  weights loaded correctly across 43 layers × 8 ranks.
- But the cache is built at model load; the runtime layout of recv
  tokens (`packed_x`) is constructed by us in B-3.3.5 and may not
  match what `forward_deepgemm_all_dsv4_experts_gpu` expects.

### Candidate 3: Stream / sync ordering issue

DeepGEMM JIT'd kernel runs on some stream; our B-3.3.5 code may not
properly wait on the dispatch buffer to be valid before passing it.
On H20 the race window may surface where it didn't on H100 testing.

## Diagnosis Plan (not pursued in this session)

1. Run `cuda-memcheck` (or `compute-sanitizer`) wrapping a single
   prefill+decode request with EXPERT=deepgemm: should produce the
   exact illegal access line + kernel name. ~30 min on the pod.
2. Isolate DeepGEMM kernel by replacing our packed buffers with
   known-good synthetic inputs: rule out (or confirm) layout mismatch.
3. If H20 SM issue, check DeepGEMM's launch heuristic — may need
   passing `num_sms` override or skipping the cluster path.

## Status

- B-3.3 + B-4: ✅ shipped (+46.5% over NCCL baseline, 15.82 tok/s p50)
- B-3.3.5: ✅ code shipped (67ac6400), ✅ JIT compile fixed (38bf157b),
  ❌ runtime crash on H20 (this errors entry)
- Path forward: ~1 day's work to root-cause + fix; deferred.
- TPOT remains at 15.82 tok/s c=1 baseline; +8-12 ms TPOT estimated
  lift from DeepGEMM not realized.

## Rule

When a vendored compute library's JIT compile passes the first
non-trivial gate, **assume runtime failure is a separate root cause
class** and budget for an independent diagnosis pass (cuda-memcheck +
synthetic-input isolation). Don't chain a JIT debug session into a
runtime debug session without explicit user check-in — they're
different research questions and consume different blocking knowledge
(build-system internals vs CUDA kernel semantics).

The compile-pass victory feels close to the runtime-pass victory but
isn't — and the 30-60 min "just one more try" trap is exactly when
SOLID self-audit fails.

## Refs

- B-3.3.5 wire-in: commit `67ac6400`
- c++17 nvcc fix: commit `38bf157b`
- nsys data showing real bottleneck breakdown: in-session message
  trail (cached_notify_combine 26%, ncclAllReduce 17.6%,
  expert FFN 24.5%, attention 10%)
- SGLang-vs-ARLE detailed implementation gap analysis: in-session
  subagent report from this session.
