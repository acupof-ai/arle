# Isolated microbench settles single-block sizing: both prologue kernels want 1024 threads (−48%/−51%) — and overturns a contention-poisoned e2e verdict

**Date:** 2026-06-11. **Tool:** `crates/infer-cuda/examples/dsv4_microbench.rs`
(new, durable kit) — CUDA-stream timing on an idle GPU, production shapes
(H=4096, hc_mult=4, sinkhorn=20), 2000 iters after warmup.

## Numbers (µs/instance, H20, idle GPU0)

| kernel | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| `mhc_params` (warp tail) | 26.64 | 16.22 | 11.05 | **8.51** |
| `hc_pre_rms_norm` (fused) | 16.81 | 9.70 | 6.24 | **4.80** |

Monotonic: a single block that owns a 32KB stream-row read alone wants the
maximum in-flight loads. Production launches set to 1024 (`d7be8c9b`).

## E2e matched A/B (same session, back-to-back, both arms co-tenant-clean)

| arm (.cu differs ONLY in the two launch constants) | B=1 p50 | mean |
|---|---|---|
| A: 256/256 (`edf2a309`) | 40.29 | 40.17 |
| **B: 1024/1024 (`d7be8c9b`)** | **44.04** | **43.81** |

**Δ = +9.3% (−2.1 ms/token)** — twice the component prediction (gap removal
and serve-clock effects on top of kernel time). Output correct, 8/8 both
arms. Campaign cumulative: 39.51 → 44.04 (**+11.5%**); vs the 38.99
pre-campaign default: **+13.0%**. 22.7 ms/token.

Cross-session e2e numbers meanwhile bounced 39.7–42.4 on the SAME binary —
session-to-session drift is ±6%, far above small-effect sizes. **E2e
verdicts for ≤10% effects require same-session matched A/B pairs; only
component microbenches and matched pairs count** (the Metal matched-A/B
rule now confirmed on the CUDA pod).

## The methodology lesson (why this entry exists)

The e2e A/B said params-1024 was a −4.5% regression (40.49) — and the
"confirmation" run collapsed to 20.2 tok/s, which exposed the real cause: a
**co-tenant process (16GB, 100% util) had appeared on GPU 7**, time-slicing
with rank 7 of the TP=8 serve. The e2e verdict was contention noise, fully
overturned by the isolated bench.

- **Check `nvidia-smi --query-compute-apps` BEFORE and AFTER every e2e
  measurement on shared pods.** A co-tenant on ANY rank's GPU poisons the
  whole pipeline number (lockstep makes one slow rank everyone's wall).
- **Single-kernel decisions get isolated microbenches, not e2e A/Bs** —
  faster (seconds vs 10-min serve cycles), contention-immune, and the
  signal isn't diluted 86:1 by the rest of the step (comm_bench precedent).
- **One commit = one variable.** `8f18e00a` bundled two block-size changes;
  its revert silently un-did the GOOD one too. The bundling error cost two
  serve cycles.

## State after this entry

Rung-1 ladder: 39.51 → 41.86 (mhc_params warp tail) → 42.41 (fused prologue,
last clean e2e) → **+~1.1 ms/token pending** (both kernels at 1024). Next:
hc_enter fusion (mhc_params INTO the fused prologue — same launch shape,
saves another boundary + the pre round-trip, ~0.3-0.4 ms), then
pack_quantize epilogue fusion, then Rung 2.
