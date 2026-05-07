# M_pf-gemm — cuBLASLt prefill GEMM auto-tune (algo benchmarking instead of heuristic top-1) — ⛔ Phase 0 KILLED

> **2026-05-07 EOD+2 update — Phase 0 KILLED.**
> [Bench evidence](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)
> shows `INFER_GEMM_AUTOTUNE=1` yields ~1% TTFT improvement at
> long-ctx 4k/c=4 — within bench noise (3% intrinsic). cuBLAS
> heuristic top-1 IS the best cuBLAS algo for ARLE's prefill
> shapes; the 1.65× gap to vLLM is at a level cuBLAS algo
> selection cannot reach. Substrate stays committed (env-gated,
> default off) for future Phase 0.5 experiments.
>
> **Phase 2 still on the table**: TileLang prefill GEMM port,
> gated on M_world1 Phase 0 baseline (SGLang + TRT-LLM at
> long-ctx 4k/c=4). If vLLM/SGLang prefill GEMM beats cuBLAS
> top-1 by >20%, the gap is in kernel implementation, not
> dispatch — TileLang port becomes the right move.

> Created 2026-05-07 EOD+2 from H_LP3 finding
> ([`cae08b7`](../experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)):
> long-ctx 4k/c=4 prefill TTFT 800 ms gap is **cutlass small-tile
> GEMM dispatch (56.7% of TTFT window)**, not per-chunk launch
> overhead. M_pf-gemm closes this gap by adding warmup-time algo
> benchmarking to `gemm_cublaslt_impl()`.

## Priority & ROI

**Priority**: **P1 license-or-kill** (after M_b.1 BF16 split-KV
lands, codex 41m+ in progress at time of writing). Phase 0 of
M_pf-gemm is a 5-10 LOC experiment that can fire by next /loop
tick.

**Why P1, not P0**:
- M_world1 Phase 0 (SGLang + TRT-LLM baseline measurement) is
  the absolute P0 — without it, we don't know who #2 actually
  is.
- M_pf-gemm has STRONG trace evidence (cutlass Kernel2 = 56.7%
  TTFT window) but the fix is technically novel (cuBLASLt at
  warmup) and needs Phase 0 validation.

**ROI basis (math from trace evidence)**:

Current state (long-ctx 4k/c=4 post-Phase-1A-v3-default-Split):
- ARLE TTFT mdn = 1976.4 ms; vLLM s8 same shape = 1177 ms
- Gap = 799 ms = vLLM 1.68× faster
- cutlass `Kernel2` (16×16×128 wmma bf16) = **1417 ms / 2500 ms
  TTFT window = 56.7%** of measured prefill time

Fix mechanism:
- `cublasLtMatmulAlgoGetHeuristic` already returns 8 candidate
  algos (`gemv.cu:362`, `requestedAlgoCount=8`)
- Current code uses only `heuristic_results[0]` (`gemv.cu:371`),
  discarding the other 7
- Heuristic top-1 is optimized for "average cost across many
  shapes"; for a specific (M, N, K), the fastest algo is
  frequently at index 1-3 (this is why vLLM, SGLang, TensorRT-
  LLM all auto-tune at warmup)

Expected gain (rough, validated by Phase 0):
- If best algo is 30% faster than top-1 at prefill shape → cutlass
  GEMM time 1417 → ~990 ms in TTFT window → TTFT 1976 → **~1550 ms**
- If best algo is 40% faster → TTFT ~1410 ms = parity with vLLM
  1177 ms within 20%
- Stretch: combined with M_world1 SGLang baseline measurement
  may identify additional non-GEMM opportunities

**Negative case**:
- cuBLAS heuristic top-1 may already be optimal at our specific
  shape (large M / aligned dims). Benchmark would show < 10%
  improvement. Phase 0 catches this in 1-2 hours of work.
- Auto-tuning at warmup adds 50-200 ms one-time per (M, N, K)
  shape × 8 algos × 5 iterations. For ARLE's ~6 unique GEMM
  shapes per layer × 36 layers × 6 algos = small overhead.
- Some algo IDs may fail at runtime (occasional cuBLAS bug);
  need to skip-and-retry without crashing.

**Kill criteria**:
- If Phase 0 shows < 10% TTFT improvement after warmup auto-tune
  → ABANDON; the H_LP3 finding suggests a different fix axis
  (TileLang prefill GEMM port — M_pf-gemm Phase 2).
- If warmup time exceeds 5 seconds (cumulative across all GEMM
  shapes) → ABANDON or move to lazy auto-tune (first-call benchmark).

## Phase 0 — License-or-kill experiment (2-4 hours)

**Single-file diff** in `crates/cuda-kernels/csrc/gemm/gemv.cu`,
function `gemm_cublaslt_impl()`:

Replace lines 370-371:
```cpp
if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0) {
  algo_it = state->algo_cache.emplace(key, heuristic_results[0].algo).first;
}
```

With:
```cpp
if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0) {
  // Benchmark all returned algos at this exact (M,N,K), pick fastest.
  // One-time cost amortized across all subsequent calls of this shape.
  cublasLtMatmulAlgo_t best_algo = heuristic_results[0].algo;
  float best_time_ms = std::numeric_limits<float>::infinity();
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  // Warmup once + measure 5 iters per algo
  for (int i = 0; i < returned_algo_count; ++i) {
    auto& candidate = heuristic_results[i].algo;
    // Warmup
    if (cublasLtMatmul(state->lt_handle, operation_desc, &h_alpha,
                        W, w_desc, X, x_desc, &h_beta,
                        Y, y_desc, Y, y_desc, &candidate,
                        state->cublaslt_workspace, kWorkspaceBytes, stream)
        != CUBLAS_STATUS_SUCCESS) continue;
    // Measure
    cudaEventRecord(e_start, stream);
    for (int it = 0; it < 5; ++it) {
      cublasLtMatmul(state->lt_handle, operation_desc, &h_alpha,
                     W, w_desc, X, x_desc, &h_beta,
                     Y, y_desc, Y, y_desc, &candidate,
                     state->cublaslt_workspace, kWorkspaceBytes, stream);
    }
    cudaEventRecord(e_stop, stream);
    cudaEventSynchronize(e_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, e_start, e_stop);
    if (ms < best_time_ms) { best_time_ms = ms; best_algo = candidate; }
  }
  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  algo_it = state->algo_cache.emplace(key, best_algo).first;
}
```

LOC delta: ~30 lines added in one function. No header changes,
no cross-file impact, no Rust binding changes.

**Validation flow**:
1. `cargo build --release --features cuda` (verify still compiles)
2. `cargo test --release --test e2e --features cuda` (correctness)
3. `cargo test --release --test greedy_consistency --features cuda`
   (numerical stability)
4. Bench longctx 4k/c=4 post-fix vs control:
   - `scripts/bench_guidellm.sh m_pf-gemm-autotune` (5 min)
   - Compare against baseline
     `bench-output/2026-05-07-longctx-4k-c4`
5. **License decision**:
   - ≥ 30% TTFT reduction → PROCEED to Phase 1 (multi-shape
     coverage, env flag for opt-in/out, telemetry)
   - 10-30% improvement → PROCEED but lower priority; document
     as opt-in flag
   - < 10% improvement → ABANDON; pivot to TileLang prefill GEMM
     (Phase 2 below)

## Phase 1 — Production wiring (after Phase 0 license)

If Phase 0 ≥ 30% TTFT improvement:
- Add env flag `INFER_GEMM_AUTOTUNE` (default on) to allow opt-out
  for deterministic-build flows.
- Add telemetry: log selected algo per (M,N,K) shape + measured
  speedup.
- Ensure auto-tune runs only at first-call per shape (lazy);
  graph-safe path (decode) keeps using `heuristic_results[0]`
  unchanged because tail latency matters more than peak there.
- Multi-bench: validate auto-tune holds at long-ctx 8k/c=4,
  high-conc 1k/256/c=64, multi-tenant shared-prefix.
- Wins entry with full per-shape Δ table.

LOC est: ~80-120.

## Phase 2 — TileLang prefill GEMM port (P2, conditional)

If Phase 0 fails (< 10% improvement) OR Phase 1 plateaus
< 30% TTFT improvement:

Author `crates/cuda-kernels/tools/tilelang/batch_prefill_bf16_gemm.py`:
- M-major tile specialization (256×128 or 256×256 for prefill
  shapes)
- Mirror existing prefill attention TileLang scripts
  (`batch_prefill_paged_hd128.py`)
- AOT-compile + dispatch via shape-based routing (M ≥ 1024 →
  TileLang; M < 1024 → cuBLAS)
- LOC est: 200-300 (kernel + AOT integration + ffi/Rust wiring)
- Risk: medium (kernel correctness vs cuBLAS, autotuning across
  shape space)

**Trigger to start Phase 2**: M_world1 Phase 0 baseline shows
SGLang/TRT-LLM use a TileLang-or-equivalent prefill GEMM that
explains their lead, AND Phase 1 ROI plateaus.

### 2026-05-07 EOD+8 evidence escalation — cuBLAS large-N blind spot

[M_pf-fuse Phase 0 KILL](../experience/wins/2026-05-07-m_pf-fuse-phase0-gateup-killed.md)
delivered a counter-intuitive datapoint:
- Predicted: fused 22k-output gate_up GEMM beats 2× separate 11k
  GEMMs by ~8% TTFT (call-count + tensor-core util math).
- Measured: fused 22k GEMM is **+1.5% slower** at long-ctx 4k/c=4.
- Implication: cuBLAS heuristic + algo space is **non-monotonic
  in N**. Selecting a single fat GEMM at N=22016 lands on a
  worse kernel/tile than two N=11008 calls.

This is independent evidence that **cuBLAS leaves real performance
on the table at large N** — exactly the regime a hand-rolled
TileLang kernel can target. Combined with H_LP3's finding (cutlass
`Kernel2` 16×16 wmma at 56.7% of TTFT window), Phase 2 now has TWO
trace-grounded reasons:
1. Per-launch tile is small (16×16); a large-tile TileLang variant
   (256×128) at large M could amortize launches.
2. cuBLAS algo selector is suboptimal at large N (M_pf-fuse data);
   a shape-tuned TileLang kernel can pick the right tile per
   (M, N, K) without the heuristic's averaging.

### Phase 2.5 candidate — Hybrid TileLang + cuBLAS dispatch (P2 alt)

Lateral idea inspired by the M_pf-fuse blind spot: instead of
porting **all** prefill GEMMs to TileLang, dispatch by shape:

```
if (N >= N_threshold && M >= M_threshold) {
    tilelang_prefill_gemm(...);  // hand-tuned 256×128
} else {
    cublasLt_gemm(...);           // existing path
}
```

- Smaller scope than full Phase 2 (~150 LOC vs 250-300)
- Targets exactly the regime M_pf-fuse exposed
- Falls back to cuBLAS for shapes where it IS optimal
- Risk: lower (existing decode/small-prefill paths untouched)

Phase 2.5 is gated on the SAME trigger as Phase 2 (M_world1 P0
baseline confirms kernel-impl gap). If SGLang lead at long-ctx 4k
turns out to be scheduler/attention rather than GEMM, BOTH 2 and
2.5 demote to P3.

## Acceptance criteria

- Long-ctx 4k/c=4: ARLE TTFT ≥ 30% lower than current 1976 ms
  AND within 20% of vLLM 1177 ms (i.e., ≤ 1412 ms).
- Long-ctx 8k/c=4: TTFT improvement ≥ 20% vs current.
- High-conc 1k/256/c=64: out tok/s no regression (decode-
  dominated workload, GEMM less critical).
- All e2e + greedy_consistency tests pass.
- Wins entry with per-shape data table cross-referenced to
  bench artifacts and sha256.

## Tasks

| # | Task | File | LOC | Owner | Trigger |
|---|---|---|---|---|---|
| Phase 0 | autotune algo selection | `gemv.cu:gemm_cublaslt_impl` | ~30 | Codex (after M_b.1 lands) | M_b.1 commits |
| Phase 0.1 | bench validation | `scripts/bench_guidellm.sh m_pf-gemm-autotune` | 0 | Claude | Phase 0 lands |
| Phase 0.2 | wins entry + license decision | `docs/experience/wins/...` | 0 | Claude | 0.1 done |
| Phase 1.1 | env flag + telemetry | `gemv.cu` + Rust binding | ~50 | Codex | License fires |
| Phase 1.2 | multi-shape bench | bench scripts | 0 | Claude | 1.1 done |
| Phase 2 (conditional) | TileLang prefill GEMM | NEW python + lockstep wire | ~250 | Codex | Phase 1 plateau |

## Cross-references

- H_LP3 finding: [`cae08b7`](../experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)
- M_nsys P1 validation: [`f3ff34f`](../experience/wins/2026-05-07-m_nsys-p1-validated-longctx-kernel-data-captured.md)
- Source survey: `crates/cuda-kernels/csrc/gemm/gemv.cu:280-404`
- M_world1 roadmap: [`docs/plans/M_world1-30-percent-lead-roadmap.md`](M_world1-30-percent-lead-roadmap.md)
- Codex H_LP1+H_LP2 license-kill (precedent): [`c219434`](../experience/wins/2026-05-07-longctx-4k-c4-prefill-gap-license-kill.md)

## Rules (per memory `feedback_docs_priority_roi_evidence.md`)

- **Trace-driven hypothesis ranking**: H_LP3 was promoted from
  intuition to evidence-backed before this plan was drafted
  (M_nsys P1 trace + SQLite query). All future plans must follow
  this pattern — no plan body without a P0 survey of trace +
  source.
- **Single-file Phase 0**: ~30 LOC in one C++ function lets us
  validate the hypothesis in 2-4 hours. Higher-cost Phase 1/2
  gated on Phase 0 evidence.
- **Auto-tune at warmup is industry standard**: vLLM, SGLang,
  TensorRT-LLM all do it. ARLE not having it is a real omission;
  this plan corrects it.
