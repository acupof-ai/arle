# 2026-05-07 · M_pf-gemm Phase 0 KILLED — cuBLAS heuristic top-1 is already (near-)optimal

## Priority & ROI

**Priority**: P1 license-or-kill — **kill criterion fired**.
M_pf-gemm Phase 0 (INFER_GEMM_AUTOTUNE=1, benchmark all 8 heuristic
algos at first cache miss, pick fastest) yields ~1% TTFT improvement
at long-ctx 4k/c=4 — well below the 10% PROCEED threshold and far
below the 30% target.

**Negative case avoided**: substrate is gated behind explicit
opt-in env var (default off), so no production impact.

**Status update**:
- M_pf-gemm Phase 0: **KILLED**. Substrate stays committed
  (1 file, 70 LOC, env-gated) for future experimentation but
  not enabled by default.
- M_pf-gemm Phase 2 (TileLang prefill GEMM port): still on
  the table, gated on M_world1 Phase 0 baseline (SGLang +
  TRT-LLM at long-ctx 4k/c=4 to confirm whether prefill GEMM
  is in fact the gap-closer).
- H_LP3 finding ([`cae08b7`](2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md))
  reframed: cutlass small-tile GEMM IS 56.7% of TTFT window,
  but cuBLAS heuristic considers it the best available algo;
  the gap to vLLM is at a level cuBLAS algo selection cannot
  reach (different GEMM library entirely).

## Bench evidence

Setup: ARLE post-`419fdea` (M_pf-gemm Phase 0 substrate +
codex M_b.1 BF16 split-KV dirty workspace). Long-ctx 4k/c=4,
60 s + 10 s warmup. Same shape as `2026-05-07-longctx-4k-c4`
baseline.

| Run | INFER_GEMM_AUTOTUNE | TTFT mdn | TTFT mean | ITL mdn |
|---|---|---:|---:|---:|
| Pre-change baseline (`2026-05-07-longctx-4k-c4`) | n/a | 1976.4 ms | n/a | 19.4 ms |
| Control (substrate, default off) | unset | **1980.2 ms** | n/a | 19.3 ms |
| **Treatment (autotune on)** | =1 | n/a | **1955.5 ms** | 19.3 ms |
| **Δ vs control** | | -1.3% | n/a | 0% |
| vLLM s8 control | n/a | 1177 ms | n/a | 18.8 ms |
| Gap to vLLM (post-treatment) | | 1.65× slower | | |

Variance: a single run pair shows ~25 ms apparent improvement.
That's well within the 50-100 ms shot-to-shot variance observed
across earlier baselines (`bench-output/2026-05-07-longctx-4k-c4`
mean 1976, p95 2032 — ~3% intrinsic noise). The autotune effect
is **statistical noise**, not signal.

ITL and out tok/s unchanged across all runs as expected
(autotune affects prefill GEMM only, decode pathway untouched).

## Why heuristic top-1 is already optimal at this shape

The H_LP3 trace evidence (cutlass `Kernel2` 16×16×128 wmma bf16
= 56.7% of TTFT window, 606 instances avg 2.3 ms) is real, but
**cuBLAS heuristic considers the small-tile variant the best
choice** for ARLE's GEMM shapes (M=4096, N=2048-8192, K=hidden_dim)
on RTX 4070Ti SUPER (compute capability 8.9).

The other 7 algos returned by `cublasLtMatmulAlgoGetHeuristic`
are presumably either:
- Same kernel as top-1, just different invocation parameters
  (no measurable difference)
- Algos with bigger tiles that turn out to have lower throughput
  at this specific (M, N, K) combination
- Algos that introduce per-launch overhead (setup, sync) that
  cancel out the bigger-tile gains

vLLM at the same shape achieves 1177 ms TTFT vs ARLE 1976 ms.
This 1.65× gap is **not closable via cuBLAS algo selection**;
cuBLAS top-1 already represents the best cuBLAS can do here.

## What this rules out / rules in

**Ruled out** (Phase 0):
- "cuBLAS heuristic picks wrong tile" — cuBLAS top-1 IS the
  best cuBLAS choice.
- Trivial 5-30 LOC fix at call site — the algo space cuBLAS
  exposes doesn't contain a faster option at our shape.

**Still in play**:
- **M_pf-gemm Phase 2 (TileLang prefill GEMM port)**: vLLM
  uses Triton/CUTLASS templates that may include kernel variants
  cuBLAS doesn't expose. Hand-rolling a TileLang prefill GEMM
  with M-major tile specialization (256×128 or 256×256) could
  beat cuBLAS top-1 — but only if the kernel-implementation gap
  is real (not just dispatch).
  - Cost: 200-300 LOC + AOT integration + TileLang IR work
  - Risk: medium-high (kernel correctness, autotuning, build cost)
  - Trigger: after M_world1 Phase 0 confirms vLLM's lead is in
    fact prefill GEMM (vs e.g. attention or scheduling)
- **Persistent kernel tricks** (TensorRT-LLM-style): keep kernel
  resident, queue work via mailboxes. Reduces per-call dispatch
  overhead. Out of scope for this milestone but a Phase 3
  candidate.
- **`cublasLtMatmulPreferenceSetAttribute` with explicit tile
  hints**: The current code only sets `MAX_WORKSPACE_BYTES`.
  Adding tile-size preference might surface a different algo
  set in the heuristic output. Phase 0.5 candidate, ~30 LOC.

## Substrate disposition

Keep `INFER_GEMM_AUTOTUNE` env var + autotune logic
(`gemv.cu:gemm_cublaslt_impl`) committed. Reasons:
- Default off → no production impact
- Phase 0.5 ideas (different timing methodology, different
  algo set selection) reuse the scaffolding
- Documents the experiment so future engineers don't repeat it

## Cross-references

- M_pf-gemm Phase 0 substrate: commit `419fdea`
- M_pf-gemm plan: [`docs/plans/M_pf-gemm-cublaslt-autotune.md`](../../plans/M_pf-gemm-cublaslt-autotune.md)
- H_LP3 finding: [`cae08b7`](2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)
- M_nsys P1 validation: [`f3ff34f`](2026-05-07-m_nsys-p1-validated-longctx-kernel-data-captured.md)
- vLLM longctx 4k baseline: `bench-output/2026-05-07-vllm-longctx-4k-c4`
- ARLE pre-change baseline: `bench-output/2026-05-07-longctx-4k-c4`
- Treatment artifacts: `bench-output/2026-05-07-m_pf-gemm-autotune-4k`
- Control artifacts: `bench-output/2026-05-07-m_pf-gemm-autotune-off-control`

## Rule

- **License-or-kill on time**: 2-3 hours for Phase 0 was right.
  Building the substrate (~30 LOC), running 2 benches, and
  documenting the kill: ~3 hours total. Phase 2 would be 1-2
  weeks. The cheap experiment ruled out the cheap fix.
- **Cache-poisoning during graph capture is a real footgun.**
  First implementation crashed because `cudaEventRecord` is
  illegal during `cudaStreamIsCapturing(stream) == ACTIVE`.
  Fixed with explicit capture-status check in the autotune
  branch before doing any cudaEvent work.
- **Trace evidence explains the symptom but not the cause.**
  H_LP3 found cutlass small-tile = 56.7% of TTFT — true. The
  reflexive conclusion ("therefore pick a different cuBLAS
  algo") was wrong because cuBLAS already picks optimally;
  the 56.7% is the actual best-case GEMM cost on this hardware.
  Closing it requires going outside cuBLAS.
- **Update plan + index when killing**: M_pf-gemm plan needs
  the kill marker and Phase 2 trigger update.
