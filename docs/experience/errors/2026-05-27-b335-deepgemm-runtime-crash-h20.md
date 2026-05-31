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

## Update 2026-05-31 — ROOT CAUSE found (candidate 2 confirmed by source; 1 & 3 refuted)

Full source trace of both DeepGEMM callers + the FFI kernel + the scratch sizing:

**Candidate 1 (H20 SM) — REFUTED.** The identical JIT kernel
(`dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda`, generic SM90, `num_sms=78` baked in) is
**validated working on the same 8×H20 pod** via the `=deepep` device-count path
(`wins/2026-05-26-dsv4-deepgemm-device-counts.md`, real 7.55 s bench). DeepGEMM is fine on H20.

**Candidate 3 (stream ordering) — REFUTED.** Pack + GEMM both run on `ctx.stream` (ordered).

**Candidate 2 (input-contract divergence) — CONFIRMED.** The native-deepep DeepGEMM branch
(`forward_native_deepep_routed_gpu`, mlp.rs:5272-5298) does NOT replicate the contract of the only
working caller, `forward_deepep_routed_gpu` (mlp.rs:4120-4131), which is **decode-only, ≤256 routes,
padded** (gate `prepared_small_local_pack`, mlp.rs:4006-4007). Two divergences, hit on PREFILL with
large unpadded routes:
1. **`route_capacity = total_local_routes`** (live, variable, unpadded; mlp.rs:5294) vs the working
   path's `total_recv_routes` (fixed padded bound). `route_capacity` → GEMM `max_m` AND scratch
   `capacity_m`/`scale_stride_m = align_to(capacity_m,4)`. On scratch REUSE (state.rs:896, reused when
   `capacity_m` shrinks) the FP8 pack writer (dsv4_deepgemm_ops.cu:44-52) uses the NEW small
   non-128-aligned `scale_stride_m` while the SFA TMA descriptor (deepgemm_native.cu `make_tma_sfa_desc`
   644-659) uses the scratch's allocated stride → disagree → the masked-grouped kernel's TMA tiles for
   upper groups address past the written/allocated region → device-side TMA trap → `CUDA_ERROR_LAUNCH_FAILED`.
2. **`expert_route_slot = scratch.packed_token`** (recv-token index, `alloc_zeros`, never `-1`-init;
   mlp.rs:5290) vs the working path's `-1`-init per-route-unique route-slot (`dsv4_fill_i32_cuda(-1)`
   mlp.rs:4069 + `dsv4_pack_received_experts`). `dsv4_scatter_all_route_slots` (dsv4_route.cu:1248-1266)
   then treats stale `0` as valid recv-token 0 (no sentinel) + plain `=` (the Finding-2 overwrite) — a
   correctness bug riding along.

### Fix plan (mirror the working contract)
1. Pass a **STABLE PADDED `route_capacity`** the scratch is keyed on so `max_m == capacity_m` every call
   (no writer/reader scale-stride divergence). NB `capacity_local_routes` (capacity_recv×topk) is huge →
   OOMs the `num_experts × max_m × hidden` scratch; the right value is a 128-aligned per-call-stable bound,
   OR fix the deepgemm fn to use the SCRATCH's allocated `capacity_m` for the scale-stride (writer + reader)
   instead of the call's `route_capacity`.
2. Build `expert_route_slot` as a `-1`-init per-route-unique route-slot (`dsv4_pack_local_experts_with_slots_cuda`
   + `dsv4_fill_i32_cuda(-1)`). Also fixes the Finding-2 overwrite.

Cheapest crash-clearing test: change (1) alone at mlp.rs:5294; if the launch failure clears, the
scale-stride-on-reuse divergence is confirmed; then `compute-sanitizer --tool memcheck` one prefill to name
the exact trapping load + add (2).

**Validation blocked 2026-05-31**: a parallel-session rebuild left the pod `target-pod/release/infer` WITHOUT
`--features nccl` (boot panics "TP/EP world_size > 1 requires building infer with --features nccl"). A clean
`--features cuda,nccl` rebuild is needed before the discriminator/fix can be pod-tested. Root cause above is
from source (high confidence on the class; medium on the exact trapping line — needs the sanitizer run).

## Update 2026-05-31 (pod-reproduced) — the error is `NOT_SUPPORTED`, NOT an OOB; config-dependent at large max_m

Rebuilt with nccl + ran `EXPERT_BACKEND=deepgemm` `MOE_BACKEND=native-deepep`, short prompt. New evidence
collapses the earlier hypotheses:

```
DeepSeek V4 DeepGEMM w13 GEMM failed: DriverError(CUDA_ERROR_NOT_SUPPORTED, "operation not supported");
groups=32 max_m=88 n=4096 k=4096 scale_stride_m=88 active_experts=32   (also max_m=56,72,96)
abort kind=architectural_deferral
```

- **It is `CUDA_ERROR_NOT_SUPPORTED`, not the b335 "unspecified launch failure"** — the runtime path now CATCHES
  the DeepGEMM failure cleanly (`architectural_deferral`) instead of poisoning the context. So the earlier
  context-cascade symptom is gone; this is the raw kernel-launch rejection.
- **The agent's scale-stride-OOB / illegal-memory hypothesis is REFUTED** — `NOT_SUPPORTED` is a launch-config
  *rejection*, not a memory trap.
- **It fails at `max_m ∈ {56,72,88,96}` (unpadded prefill local routes) and WORKS at `max_m=8`** (the
  decode-padded `=deepep` path, `wins/2026-05-26`). `prop.major==9` passes (deepgemm_native.cu:855), so the
  `NOT_SUPPORTED` comes from a CUDA API at launch: `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)`
  (line 355), the SFA `cuTensorMapEncode` (line 870), or the cluster `cudaLaunchKernelEx` (line 373). The
  `get_best_config` heuristic (line 520) picks the layout by min `num_cycles`, so a larger `max_m` selects a
  different config — `block_m=128` and/or `cluster=2` (line 490-516) — than the `max_m=8` config (`block_m=64`,
  `cluster=1`). The larger config's smem or cluster is what H20 rejects.
- **Note: DeepGEMM prefill (large max_m) was NEVER validated** — the `=deepep` +46% was DECODE (15.82 tok/s,
  small padded max_m). So this is a "deepgemm-config-at-large-m-on-H20" gap, not a regression.

### Probe in flight
Added env `ARLE_DEEPGEMM_CONSERVATIVE_LAYOUT=1` (deepgemm_native.cu:486) that forces `block_m=64` + `cluster=1`.

## Update 2026-05-31 (the REAL root cause) — my builds shipped the STUB; deepgemm was never compiled in

The conservative-layout probe STILL returned `NOT_SUPPORTED` — which is impossible if it were a layout/config
issue (the probe changes the config). That collapsed the puzzle: in `deepgemm_native.cu` the ONLY
`return CUDA_ERROR_NOT_SUPPORTED` is `if (prop.major != 9)` (line 863) — shape-independent — yet H20 IS sm_90
(the JIT even compiles `sm_90a`). The resolution:

**The whole `deepgemm_native.cu` body is `#ifdef ARLE_ENABLE_DEEPGEMM_NATIVE`. When that macro is NOT defined,
`csrc/gemm/deepgemm_bridge_stub.cu:17` compiles instead and returns `CUDA_ERROR_NOT_SUPPORTED` UNCONDITIONALLY.**
`build.rs:1355` only defines `-DARLE_ENABLE_DEEPGEMM_NATIVE=1` when the BUILD-TIME env
`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` (or `_TORCH`) is set. **None of my builds set it** — so every "deepgemm" run
this session linked the STUB. So:
- The `NOT_SUPPORTED` was the stub, NOT a real deepgemm runtime fault.
- The "shape-dependent" `max_m=56/72/88/96` was an illusion: the stub ignores everything; the Rust error string
  just logs the params it *would* have used.
- The conservative-layout probe (e2b3f40e) was inert — the stub ignores the layout.
- The "DeepGEMM FP8 expert cache built" log is a SEPARATE always-compiled weight-pack path, so it fired even with
  the stub — which is exactly what made it look like deepgemm was engaged.
- b335's original "unspecified launch failure" was a build that DID set the env (real deepgemm) — a genuinely
  different, real runtime bug. The agent's candidate-2 (route_capacity / SFA padding) analysis may still apply to
  THAT real path; it was never refuted, only untested here because the stub masked it.

**Fix to even TEST deepgemm: build with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + `ARLE_DEEPGEMM_ROOT` +
`ARLE_DEEPGEMM_LIBRARY_ROOT`** (the `scripts/dsv4_toolchain.sh` build mode sets these; a plain
`cargo build --features cuda,nccl` does NOT). Rebuilt with the env; real-deepgemm test pending. The candidate-2
padding fix is the next thing to validate once the real kernel runs.

## Update 2026-05-31 (real deepgemm confirmed) — REPRODUCES the original "unspecified launch failure"

Rebuilt with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + `ARLE_DEEPGEMM_ROOT`/`_LIBRARY_ROOT` (binary now contains the
real `sm90_fp8_m_grouped_gemm_masked` symbol, not the stub). Re-ran `EXPERT=deepgemm` `MOE_BACKEND=native-deepep`,
short prompt → the requests now fail with **`CUDA_ERROR_LAUNCH_FAILED, "unspecified launch failure"`** (44
sticky-error lines; the GEMM device-trap poisons the context → the next `H2D copy` reports the cascade). This is
EXACTLY the original b335 symptom — so the real deepgemm kernel does crash on the DSv4 expert shapes, and the
earlier `NOT_SUPPORTED` was purely the stub masking it.

So we are now at the GENUINE bug: an **illegal device access inside the real DeepGEMM kernel** (`LAUNCH_FAILED` =
a device-side trap), which is consistent with the agent's **candidate-2** (the native-deepep DeepGEMM branch feeds
unpadded/variable `route_capacity`=`total_local_routes` + a non-`-1` `expert_route_slot`, so on scratch reuse the
FP8-pack scale-stride disagrees with the SFA TMA descriptor stride → OOB). Next:
1. `compute-sanitizer --tool memcheck --target-processes all` (with `ARLE_RENDEZVOUS_TIMEOUT_SECS=900`) on ONE
   short native-deepep+deepgemm request → name the exact OOB kernel + access (writer scale-store vs SFA TMA load).
2. Apply candidate-2: pass a stable padded `route_capacity` (or make the deepgemm fn use the scratch's allocated
   `capacity_m` for the scale-stride, both writer+reader) + build `expert_route_slot` as a `-1`-init per-route-unique
   slot (`dsv4_pack_local_experts_with_slots_cuda`) — which also fixes the Finding-2 overwrite.
3. Rebuild (with the deepgemm env) + re-test; coherent output + no LAUNCH_FAILED confirms it; then bench the
   claimed ~2.5× expert-GEMM win vs the scalar grouped GEMM.

## Update 2026-05-31 (RESOLVED) — root cause was the JIT *failing to compile*, not any kernel/contract bug

candidate-2 is **REFUTED**. The crash was never an IMA, never the Rust calling contract, never the kernel port.
Running the real deepgemm under the new `ARLE_SERVER_WRAP` compute-sanitizer toolchain hook surfaced the actual
error buried in `server.log` (which prior greps for `IMA`/`LAUNCH_FAILED` had skipped):

```
DeepGEMM native bridge failed: NVCC DeepGEMM compile failed:
/usr/include/c++/13/type_traits(2651): error: identifier "requires" is undefined
      requires requires { typename _Op<_Args...>; }
```

The runtime JIT (`compile_with_nvcc`, deepgemm_native.cu) compiled every generated kernel with
`-std=c++17 --compiler-options=...,-fconcepts`. `-fconcepts` makes host gcc-13 define `__cpp_concepts`, which makes
its libstdc++ `<type_traits>` expose the **C++20 `requires`-clause** detection idiom (type_traits:2651). NVCC's
**device** C++17 frontend (cicc/EDG) can't parse `requires`, so the JIT compile died — and the GEMM surfaced that
failure downstream as `CUDA_ERROR_UNKNOWN` / earlier `CUDA_ERROR_LAUNCH_FAILED`. The kernel never ran; the
varying `max_m` (88/96/104/112/144) across layers was just per-chunk route counts, never a shape the kernel choked on.

The `c++17+fconcepts` combo (commit `38bf157b`) worked on the **old** pod (the comment said CUDA 12.2). This pod is
**CUDA 12.9 + gcc 13.3** — 12.9's nvcc fully supports `-std=c++20` device-side, so concepts parse natively and
`-fconcepts` is redundant.

**Fix** (landed): JIT uses `-std=c++20`, drop `-fconcepts`. **Ground-truthed without an infer rebuild** by
recompiling the EXACT failing generated `~/.deep_gemm/tmp/arle-*/kernel.cu` on the pod:
`c++17+fconcepts` → EXIT=1 (type_traits:2651), `c++20`-no-`fconcepts` → EXIT=0 (clean 83 KB cubin).

**e2e VALIDATED** (2026-05-31, rebuilt with `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + `ARLE_DEEPEP_DIR`):
native-deepep + `EXPERT=deepgemm` smoke returns coherent **"The capital of France is Paris."**, `GEMM_FAILURES=0`,
no compile failures, 6 JIT kernels compiled cleanly. First request was 20.6 s because it paid the JIT cold-start (6
serial nvcc compiles) — steady-state needs a warm cache; the perf A/B (deepgemm vs scalar grouped GEMM) warms first.
NB: a build that sets the deepgemm env but **omits `ARLE_DEEPEP_DIR`** ships a `deepep-sys` stub → native-deepep
panics at boot ("built in stub mode"); the full build env needs BOTH.

### Rule
- **A `NOT_SUPPORTED` / "operation not supported" from a vendored, build-flag-gated bridge is "the stub is linked"
  until proven otherwise.** Check the `#ifdef`/build.rs feature gate and confirm the real TU compiled (grep the
  binary / the build log for the real symbol) BEFORE theorizing about kernel internals.
- **A device error code names the *surface*, not the *cause* — grep the WHOLE error chain, including subprocess
  stderr, before theorizing about the device.** `LAUNCH_FAILED`/`UNKNOWN` on a JIT'd kernel sent two sessions down
  IMA → calling-contract → kernel-port rabbit holes; the real cause (`NVCC ... compile failed: ... "requires"`) was
  sitting in `server.log` the whole time, skipped because the greps only matched `IMA`/`LAUNCH_FAILED`. For any
  runtime-JIT path, grep `compile failed`/`error:`/the compiler name first.
- **When a leftover JIT artifact exists, validate the compiler-flag fix on the exact failing TU before rebuilding the
  consumer.** `compile_with_nvcc` leaves `$HOME/.deep_gemm/tmp/arle-<digest>/kernel.cu` on failure (it throws before
  the tmp→cache rename) — recompiling that with candidate flags is a 2-minute experiment vs a 20-minute infer rebuild.

## Refs

- B-3.3.5 wire-in: commit `67ac6400`
- c++17 nvcc fix: commit `38bf157b`
- nsys data showing real bottleneck breakdown: in-session message
  trail (cached_notify_combine 26%, ncclAllReduce 17.6%,
  expert FFN 24.5%, attention 10%)
- SGLang-vs-ARLE detailed implementation gap analysis: in-session
  subagent report from this session.
