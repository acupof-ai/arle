# FP8 operator-evidence convergence — guidellm, CUDA H20 Qwen3.6-27B-FP8, 2026-07-11

> After-run for commits `6cb2c0054`/`fb1e1896f`/`a7fd6ff37`/`4b80a4e3d`
> (content-addressed bundle id, request-boundary stats, probe-truth, multiproc
> fix). Host-path-only change — no kernel or scheduler bytes changed — so this is
> a serving no-regression + engagement-proof run, not a matched kernel Δ%.

## SLO-shape probed?  Y — canonical 4096-in / 256-out, 60s per rate, bounded concurrency

## Goal

Prove the three converged paths engage on real serving and do not regress the
FP8 dense lane: (1) `/v1/stats` materializes operator-dispatch + build identity
only at the request boundary; (2) a component probe cannot self-qualify; (3) the
`arle` bin builds and serves Qwen FP8 dense after the multiproc worker fix.

## Hypothesis

Moving operator/build-identity materialization off the per-tick path (drop the
per-tick `OperatorDispatchStats` clone + the runtime `git rev-parse`/sha
subprocess) is invisible at GPU-bound decode and removes host overhead. No
kernel change → no throughput regression.

## Environment

- H20 (sm_90, sm_count=78), CUDA 12.8, source build `--features cuda`.
- **Kernel set `full`** (not the pod default `dsv4_flash`) + TileLang AOT — the
  `dsv4_flash` set links `CUDA_ERROR_NOT_SUPPORTED` stubs for non-DSv4 TileLang
  symbols and crashes the Qwen forward. See Problems.
- Model `Qwen3.6-27B-FP8` (qwen3_5 dense executor, FP8 e4m3 dynamic), single GPU.
- Build id: source build → `kernel_bundle_id="unreported"` (expected; a verified
  bundle id only appears on the exported-pack path).

## Results

### Engagement — `/v1/stats` request-boundary materialization (PASSED)

Baseline pre-request: `implementation_hits: []`, `fallback_count: 0`.
After 3 FP8 dense completions (real deterministic generation):

```json
{
  "build_identity": {
    "product_binary_sha256": "sha256:6ef937b4eed16a0237085973411ba9b38955526258af26f64813d0238e8c4c0c",
    "kernel_bundle_id": "unreported"
  },
  "operator_dispatch": {
    "policy_hash": "sha256:b222f26a2cbaf2e54f8bf405add4679fcc11ca2b680a3f1d97c1a79378dfc279",
    "implementation_hits": [
      {"implementation_id": "cuda.qwen.fp8_pack_deepgemm", "hits": 1200},
      {"implementation_id": "cuda.qwen.fp8_gemv", "hits": 27600}
    ],
    "fallback_count": 27600
  }
}
```

Counts move only after requests (baseline `[]`/0 → populated), drained per read
via `std::mem::take` at the boundary. Independent launch evidence confirmed:
`fp8_pack_deepgemm`/`fp8_gemv` are the real FP8 dense projection entry points.

### GuideLLM serving (Err=0 every rate)

| conc | TTFT mean (ms) | TTFT p99 | TPOT mean (ms) | ITL p99 (ms) | req/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4894 | 9802 | 54.9 | 55.2 | 0.067 |
| 2 | 14315 | 18971 | 72.4 | 56.8 | 0.083 |
| 4 | 22459 | 26245 | 97.6 | 52.1 | 0.133 |
| 8 | 42173 | 47560 | 166.2 | 3.8 | 0.133 |
| 16 | 41376 | 59667 | 230.0 | 205.8 | 0.033 |

conc16 degrades as 4096-token prefills serialize on one GPU (TTFT p99 59.7 s)
— expected single-GPU scaling, not a fault.

## Problems

1. **Pod default `dsv4_flash` kernel set cannot serve Qwen.** Non-DSv4 TileLang
   symbols link `CUDA_ERROR_NOT_SUPPORTED` stubs → engine thread dies on the
   first Qwen forward. A Qwen serve/bench must build `ARLE_CUDA_KERNEL_SET=full`.
   Tracked: cklxx/arle#161.
2. **Canonical `sweep` profile crashes this model on one H20.** Its `throughput`
   strategy floods unbounded concurrency (observed 83 in-flight) → `TokenKVPool:
   out of pages` → lockstep loop closes fatally. Pre-existing engine gap (no
   KV-admission backpressure under exhaustion), not a regression from these
   commits. Bounded `--concurrencies` fits the 21460-page pool. Tracked:
   cklxx/arle#162.
3. A crashed serve's arle child lingers holding ~50 GB after the wrapper `kill`,
   starving the next serve's KV pool — reap by exact PID between runs.

## Rule

**A host-path stats/identity refactor is proven by engagement + no-regression,
not a kernel Δ%.** The gate is: counters move only across the request boundary
(materialization is not per-tick), the real implementation IDs appear, and
serving throughput is unchanged. A component probe's numeric PASS is parity
only — exact-cell qualification still needs an identity-bound actual-model E2E.
