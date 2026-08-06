# The OPD_SEQ_CHUNK knob is a null — the backward wall is total-work CPU, not per-chunk overhead

**Date:** 2026-08-06 · **Verdict:** KILL `OPD_SEQ_CHUNK 4096→8192` · **Pod-only**
(local git untouched, HEAD user's `740660c52`; pod const restored to 4096) ·
Ref: [[reference_opd_backward_is_72pct_host_idle_launch_bound]]

## Context

The 80K OPD step is 84% backward. An `nsys -t cuda,nvtx,osrt` host trace of one
backward (305.44s, ARM-A champion, cp=2 seq=81920 GPU 4,5) attributed the wall:

| Bucket | Time | % backward |
|---|---|---|
| **(4) CPU host compute (NOT in any CUDA API)** | **131.7 s** | **43.1%** |
| (3) HtoD re-upload (327.8 GB / 53,708 copies) | 38.2 s | 12.5% |
| (2) DtoH park/offload (101.5 GB / 384 copies) | 32.0 s | 10.5% |
| GPU compute (kernels) | 84.6 s | 27.7% |
| (1) launch-gap orchestration (CPU in CUDA API) | 18.6 s | 6.1% |

Thread-state check: the two kernel-launcher (critical-path) threads block only
~16 s each in the backward, 100% on `ioctl` (driver submit), **zero
`futex`/`pthread_cond_wait`**. The large `poll`(2446s)/`futex`(1970s) aggregates
are background pool/watchdog threads parked the whole step (sum ≫ wall). So the
131.7 s is genuine on-CPU Rust work on the critical path.

The backward runs a seq-chunked-recompute loop of 640 nested sub-backwards
(local_rows 40960 / `OPD_SEQ_CHUNK` 4096 = 10 chunks/layer × 64 layers), each
re-running the forward + a fresh `Tape::new()`+HashMap + its own post-order
backward (`tape.rs:1116-1173`). The hypothesis: the 131.7 s is per-chunk fixed
overhead, so doubling the chunk (640→320 iters) halves it. `OPD_SEQ_CHUNK` is a
compile-time `const` (`runtime_flags.rs:71`), so each arm was a source edit +
`arle` rebuild.

## Result

Matched A/B, both freshly built + per-second VRAM-sampled, SEQ=81920 cp=2 GPU 4,5:

| Metric | 4096 | 8192 | Δ |
|---|---|---|---|
| step wall | 381.9 s | 372.7 s | −9.1 s |
| backward wall | 315.6 s | 307.1 s | −8.5 s |
| peak VRAM/rank | 92.1 GB | 92.3 GB | **+0.2 GB (flat)** |
| loss | 4.537510 | 4.537510 | exact |
| grad_norm | 7.967 | 7.978 | in-envelope |

**KILL — below the noise floor.** The −8.5 s backward delta sits inside the 4096
run-to-run spread: the earlier un-sampled 4096 backward was 305.8 s, the
matched-sampled 4096 was 315.6 s → 9.8 s variance > the 8.5 s effect. 8192's
307.1 s falls inside [305.8, 315.6].

## Root cause of the null

Halving the sub-backward count barely moved the wall, so the 131.7 s is **not**
per-chunk fixed overhead (`Tape::new`/HashMap-per-chunk). It scales with **total
work** — forward-recompute + per-op tape/autograd bookkeeping churn — invariant
of chunk count. Corroborating: peak VRAM was flat (+1.6 GB, not the expected
+16 GB), so the peak is set by the resident-checkpoint set, not per-chunk
activation liveness; the chunk-size↔memory tradeoff the knob was supposed to
exercise did not materialize.

## Fix

`OPD_SEQ_CHUNK` stays 4096 (pod const restored). The chunk knob is not the lever.
The next-lever search is now scoped to the 131.7 s itself, which scales with total
op count (the ~61k per-op host dispatches), not chunk count. Two structural levers
survive, both orthogonal to chunk count: **(1) CUDA-graph capture** of the backward
— graph replay issues the whole DAG with one host call and re-runs none of the
per-op Rust orchestration between launches, amortizing the 131.7 s to ~0 on steps
2..N (blocked today by the non-static per-iter alloc/free/park; needs a
capture-safe pool); **(2) op-count reduction / fusion**. Async/pinned overlap of
the 70 GB copy-engine traffic (bucket 2+3) can hide but not shrink those bytes —
secondary. Decisive next step before building capture infra: a pod `perf record -g`
flamegraph to split the 131.7 s into launch-orchestration (graph-fixable) vs
genuine host-math (needs op reduction).

## Rule

Before picking a knob as the lever, name **what the dominant cost scales with**,
then pick the knob that moves that variable. The chunk knob moves iteration
count; the 131.7 s scales with total op count, so it was orthogonal by
construction — the A/B only confirmed what the scaling argument already implied.
And when an effect (−8.5 s) is the same size as the run-to-run spread (9.8 s),
it is not a result: sample the baseline's own variance before crediting a delta.
