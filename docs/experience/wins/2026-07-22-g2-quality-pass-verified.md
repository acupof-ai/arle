# G2 sm_120 CUTLASS grouped FP8 MoE — quality pass VERIFIED + sync-hoist landed — CUDA (RTX PRO 6000), 2026-07-22

> Status: Shipped — VM-verified (Colab RTX PRO 6000, sm_120, CUDA 12.8) on
> `Qwen/Qwen3.6-35B-A3B-FP8`. Closes the pending-remote gate.

## Context

Two things gated on a warm sm_120 session and are now verified end-to-end:
1. The post-ship quality pass on the G2 kernel (commits `112792b59`,
   `6720a1689`, `96c9f002e`, `55458ab06`) — device-scratch mutex/map,
   SFB-layout marker on the cache, `is_sm120` helper, dispatch dedup. Only
   isolated-TU compiled before; never full-built or runtime-gated.
2. The deferred **#2 sync-hoist** (`1312e80d7`): the CUTLASS grouped GEMM D2H'd
   `group_offsets`/`group_counts` + a full `cudaStreamSynchronize` on **every**
   call, twice per MoE layer (w13 + down) with identical geometry — a §0.3
   per-call hot-loop sync.

Shipped baseline: `2026-07-22-bench-sm120-fp8-moe-cutlass-grouped.md`
(c=1 prefill TTFT 84.6s→760ms, 111×, needle exact/DET).

## Phase A — quality pass verified (no regression)

- **Build:** full `cargo build --release --features cuda` on real sm_120
  (`arch=sm_120a`), RC 0, 7m01s. Closes the compile-gate risk — the `.cu`
  scratch-mutex/map changes were only isolated-TU-compiled before.
- **Needle** (`scripts/needle_gate.py`, RAW=1 TEMPLATE=qwen3_nonthink, ×3
  same-config repeats) — verifies the `#3 sfb_n_contiguous` dispatch picks the
  right GEMM:

```
SUMMARY len=115  depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=241  depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=1000 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=2000 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=4000 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=8000 depth=0.00 exact=3 partial=0 miss=0 DET
```

- **Bench** (`bench_throughput.py`, 64 unique ~2751-tok prompts → each a cold
  prefill, 120 s/concurrency, max_tokens 256, seed 20260416). The 8-prompt
  canned pool collapses prefill under prefix cache (465 tok, TTFT 20 ms — a
  smoke shape); a 3k-tok workload is required to exercise the prefill MoE GEMM
  where the 111× win lives.

| c | TTFT p50 (ms) | TTFT mean (ms) | out tok/s | complete | vs baseline TTFT |
|--:|--------------:|---------------:|----------:|---------:|-----------------:|
| 1 | 693.9 | 682.1 | 71.0 | 34/34 | 760 → **no regression** (122×) |
| 8 | 6834.4 | 7056.7 | 253.0 | 120/120 | 6708 → +1.9% (within noise) |

`error=0`, `correctness_failed=0`. The 111× prefill lever holds. Raw:
`bench_A3k.csv`.

## Phase B — sync-hoist landed (`1312e80d7`)

Hoisted the offsets/counts D2H + host geometry to **once per layer** in
`deepgemm_routed_tail` (`dtoh_i32_pair`: two stream-ordered async D2H + a single
`synchronize`), and changed the kernel ABI to take **host-resident** offsets/
counts. Both GEMMs reuse the host slices → second readback gone, one sync/layer
instead of two, the remaining sync moved earlier. Behavior-preserving; the
kernel does no per-call D2H/sync. Files: `fp8_moe_grouped_cutlass_sm120.cu`,
`cuda-kernels/src/{moe.rs,ffi/gemm.rs}`, `infer-cuda/src/moe.rs`.

- **Build:** RC 0, 6m44s (nvcc on the modified `.cu` + Rust).
- **Needle:** exact/partial/miss = 3/0/0 DET at every length 115..8000 —
  byte-identical to Phase A. Correctness preserved.
- **Bench** (same 3k workload, same seed), vs Phase A:

| c | TTFT p50 (ms) | Δ p50 | TTFT mean | Δ mean | out tok/s | complete |
|--:|--------------:|------:|----------:|-------:|----------:|---------:|
| 1 | 684.6 | −1.3% | 671.3 | −1.6% | 72.0 | 34/34 |
| 8 | 6749.5 | −1.2% | 6928.1 | −1.8% | 251.6 | 123/123 |

All four TTFT metrics move the same direction (faster); out tok/s flat;
`error=0`, `correctness_failed=0`. Raw: `bench_B3k.csv`.

**Verdict — KEEP.** The measured ~1.3–1.8% TTFT gain is within the
cross-session drift band (Phase A and B ran on distinct VM instances, rebuilt
binaries — not a matched same-shell side-by-side), so it is a **match /
marginal improvement, not a proven win**. But it never regresses, correctness
is byte-identical, and it removes a genuine §0.3-forbidden redundant per-call
CPU–GPU sync in the hot loop — the architecturally-correct fast-path form. The
task gate (needle passes AND TTFT improves-or-matches) is met.

## Rule

A behavior-preserving hazard/cleanup or hot-loop-sync-removal pass on a shipped
GPU path gates on the same-model needle (exact/DET) + a **prefill-shaped**
non-regression bench — the 8-prompt canned pool is a smoke shape (prefix-cache
collapses prefill to ~20 ms); use ~3k-tok unique prompts to exercise the MoE
GEMM the win lives in. A sub-drift-band delta measured across two VM sessions is
a match, not a win — keep it for correctness/design, not for a perf claim.
