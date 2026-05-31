# DSv4 deepgemm FP8 grouped GEMM — unblocked (b335) + prefill A/B vs scalar

## Context

The b335 "deepgemm crash" (CUDA_ERROR_LAUNCH_FAILED / UNKNOWN on every DSv4 expert
GEMM) was root-caused as a **runtime JIT nvcc compile failure**, not a kernel/contract
bug: `-std=c++17 --compiler-options=...,-fconcepts` made gcc-13's libstdc++
`<type_traits>` expose its C++20 `requires` path, which nvcc's c++17 device frontend
can't parse. Fix = `-std=c++20`, drop `-fconcepts` (CUDA 12.9 supports c++20 device-side).
Full root-cause + validation: [`errors/2026-05-27-b335-deepgemm-runtime-crash-h20.md`].

With the GEMM finally **running** (coherent "Paris", GEMM_FAILURES=0), this entry answers
the actual question: **does deepgemm beat the scalar grouped GEMM in prefill?**

## Params / Env

- Pod: 8×H20 (SM90), CUDA 12.9, gcc 13.3. Binary built with
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + `ARLE_DEEPEP_DIR` (BOTH required —
  omitting the latter ships a deepep-sys stub → boot panic).
- Serve: `native-deepep` MoE backend, num_slots=1, max-seq-len 4096,
  mem-fraction-static 0.10, fp8 KV, 43 distributed layers, ARLE_MULTIPROC_SERVE=1.
- **A/B = same binary, same prompts, only `ARLE_DSV4_EXPERT_BACKEND` flipped**
  (`deepgemm` vs `native`=scalar grouped GEMV loop).
- Workload: `max_tokens=1` (isolates prefill = the expert-GEMM lever), median of 5
  timed requests after 2 warm-up requests (pays off the deepgemm JIT cold-start —
  the JIT compiles ~6 kernels once, then cache-hits). Harness:
  `/tmp/dgbench_run.sh` + `dgbench_client.py` (custom curl A/B, not guidellm —
  single-variable expert-GEMM isolation; guidellm not installed on the pod).

## Results — prefill latency (lower = better)

| prefill prompt | deepgemm | native (scalar) | Δ |
|---|---|---|---|
| 545 tok  | **6.89 s** (min 6.82, reproduced 2 runs) | 7.62 s (min 7.62) | **−9.5%** (deepgemm faster) |
| 1089 tok | **TIMEOUT >300 s** ✗ | 16.70 s (min 16.68) ✓ | **deepgemm REGRESSES — fails where scalar serves** |
| 2048 tok | timeout >300 s | timeout >300 s | both fail (shared cliff, not deepgemm) |

deepgemm 545-tok times: `[6.823, 6.823, 6.885, 6.890, 6.924]`
native   545-tok times: `[7.619, 7.620, 7.623, 7.652, 7.714]`
→ clean cluster separation at 545 tok (all deepgemm < all native), so the −9.5% is real.
**BUT deepgemm 1089-tok times out at 300 s while native serves it in 16.7 s** — deepgemm has
a WORSE scaling cliff (breaks at ~1k tokens vs native's ~2k). **Verdict: deepgemm is NOT
production-viable as-is.**

## Root-cause hypothesis for the deepgemm cliff (code-grounded, needs profile to confirm)

`forward_deepgemm_all_dsv4_experts_gpu` (mlp.rs) passes `route_capacity = total_local_routes`
as BOTH `total_local_routes` AND `max_local_routes` → the masked grouped GEMM's `expected_m`
= max_m = the FULL per-rank route total, applied PER GROUP across all 32 experts. So the
kernel's tile grid is sized for `total_routes × 32_experts` when only `total_routes` rows
actually exist (avg ~total/32 per expert) — a ~32× tile-count oversizing that grows with
token count. Most tiles early-exit via `masked_m`, but the persistent-kernel tile scheduling
over 32× too many tiles scales badly → tolerable at 545 tok (max_m~384), pathological at
1089 tok (max_m~768) → >300 s. This is the "candidate-2" padding issue resurfacing — NOT as
the crash cause (that was the JIT flag) but as the PERF cliff. The fix: pass
`max_local_routes = max_e(count_e)` (the real per-expert peak, available as `counts_host` in
the native path), not the total. The non-`_all_` `forward_grouped_dsv4_experts_gpu` already
does this correctly (its `max_local_routes` = `local_counts.max()`).

## Learnings (the honest verdict)

1. **deepgemm as-is is NOT a usable win.** It's −9.5% at 545 tok but TIMES OUT at 1089 tok
   where the scalar path serves in 16.7 s — a worse scaling cliff (see the max_m=total
   root-cause above). Do not enable deepgemm by default. It's only a candidate after the
   `max_local_routes` padding fix, and even then (see #2) the ceiling is small.

2. **Even a perfect expert GEMM is a minority lever.** Flipping ONLY the expert GEMM moved
   e2e prefill by 9.5% at 545 tok. An infinite-speed GEMM caps the e2e prefill win at the
   GEMM's wall-clock fraction; the −9.5% from a finite speedup says the GEMM is a minority
   of prefill (bounded ~1/3 per prior nsys). The hypothesized "~2.5×" was a kernel-microbench
   framing — textbook §0 narrow-window-vs-wall-clock. The remaining ~2/3 (dispatch/combine/
   host-poll/attention) is untouched by deepgemm and is the real lever.

3. **The 2048-tok timeout is NOT deepgemm — native fails identically.** Shared DSv4 prefill
   scaling cliff: native 16.7 s @1024 (61 tok/s) → >300 s @2048 (>18× for 2× tokens =
   superlinear), with "prefix cache pressure: host tier full" WARNs, cut off by the 300 s
   non-streaming handler deadline (handlers.rs:133). Separate scaling bug, blocks long prompts.

4. **The whole DSv4 prefill is ~10× too slow.** 16.7 s for 1024 tokens = 61 tok/s; a tuned
   engine does this in ~1-2 s. The expert GEMM is a ~10% slice of that; the headline problem
   is the overall native-deepep prefill path (host-poll + dispatch/combine) and the
   superlinear scaling. That, not deepgemm, is where "beat SGLang 30%" lives.

5. **JIT cold-start is bounded, not an explosion.** Layout selection buckets many distinct
   `max_m` (88/96/104/112/144/…) into ~6 cached kernels; first request pays ~6 serial nvcc
   compiles (≈the 20.6 s first-request cost), then steady-state. No per-token-count storm.

## Rule

- **A "N× kernel" claim must be discounted by the kernel's wall-clock fraction before it
  becomes a perf license.** deepgemm's expected ~2.5× collapsed to −9.5% e2e because the
  expert GEMM is ~1/3 of prefill and the dominant cost (dispatch/combine/host-poll) is
  untouched. Bench the e2e A/B before claiming the lever.
- **A timeout/cliff that reproduces on BOTH arms of an A/B is a shared substrate bug, not
  the treatment's fault** — the 2048 stall hit deepgemm and scalar identically; don't
  attribute it to deepgemm.

## Refs
- Fix commits: deepgemm_native.cu c++20 (this session), b335 doc.
- Bench harness: `/tmp/dgbench_run.sh`, `/tmp/dgbench_client.py` (pod-local).
- Earlier prefill profiling: native-deepep combine ~52% of FFN, expert FFN ~24.5%.
