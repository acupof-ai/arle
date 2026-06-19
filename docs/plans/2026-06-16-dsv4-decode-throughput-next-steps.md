# DSv4 8×H20 decode throughput — next steps + code-health audit (2026-06-16)

Status capture after the compressor-batch lever campaign, so the plan + the
DSv4 code-mess audit are recorded in one place. Verification lane = `.62`
(8×H20, build+serve, see [[reference_dsv4_pod_build_topology_61_62]]).

## Where we are (measured, clean = profiling OFF)

**Standard clean bench "姿势" (now fixed — use this every time):** serve with profiling
**OFF** (NO `ARLE_DSV4_DECODE_PHASE_TIME` / `ARLE_DSV4_LINEAR_PROFILE` — each adds a
per-step `cudaStreamSynchronize` that kills async overlap and **understates tok/s
by ~25–35%**), all non-MTP opt gates ON; fixed prompt + `max_tokens=128`,
c ∈ {1,2,4,8(,…)}, metric = aggregate output tok/s. `scripts/dsv4_c_sweep.py`
now has repeat/median/spread support; `scripts/bench_dsv4_trace_http.py` uses
text `/v1/completions` SSE for TTFT/ITL capture. guidellm still needs pod-side
install if we want the canonical bench wrapper there.

**c1–8 baseline (clean, profiling OFF) — re-measured as a same-session A/B
([[2026-06-16-dsv4-c1-8-baseline-clean-ab]]):**

| c | OFF (gate off) | ON (compressor-batch) |
|---|----------------|------------------------|
| 1 | 43.8 | 44.9 |
| 2 | 44.1 | 44.8 |
| 4 | 44.2 | 69.8 (+58%) |
| 8 | 74.0 | 77.6 (+5%) |

The OFF column **replicates** the prior clean session (43.0/45.0/45.0/74.8) → the
clean gate-OFF baseline is solid cross-session. The earlier "c1=32" was a
profiling artifact (`serve_bench_62.sh` had `DECODE_PHASE_TIME`+`LINEAR_PROFILE`);
the committed `2026-06-16-dsv4-c1-8-baseline-snapshot` table is now marked
SUPERSEDED. The compressor-batch lever's biggest marginal win is c=4 (+58%);
narrows to +5% at c=8 (other batched paths saturate). Single-sweep — the per-c
magnitude needs ≥3 repeats before enshrining (direction/sign solid, matches the
n=22 +38%).

**MTP arm — NOW MEASURED on `.61` (gcc-13). The `.62` "hang" was the clang-11 JIT
host compiler, confirmed by direct test** (`.61`/gcc-13 serves MTP in 136s, no
hang; `.62`/clang-11 hangs the MTP-head shapes). `.61` same-binary A/B (commit
`2f021c0`, profiling OFF):

| c | no-MTP | MTP | Δ |
|---|--------|-----|---|
| 1 | 43.4 | 47.9 | +10% |
| 2 | 43.6 | 48.1 | +10% |
| 4 | 67.2 | 48.1 | −28% |
| 8 | 73.5 | 79.1 | +8% |

MTP +10% @B=1; −28% @c4 (MTP flat ~48 c1→c4 then 79 @c8 = per-row-plateau, batched
only passes no-MTP at c8). On `2f021c0`, NOT my compressor-batch binary — not
cross-comparable to the table above. Prior-session envelope (different base/slots):
B=1 chain-fold 53.3 (+20%, [[2026-06-13-dsv4-mtp-d2-chain-fold-53]]); batched MTP
76.7 @c8 (+77% vs per-row, [[2026-06-15-dsv4-batched-mtp-prod-shape-flip]]).
Remaining: serve *my* `3e3e50e0`+compressor-batch binary on `.61` to get the
compressor-batch × MTP combined number. **`.61` (gcc-13, 8×H20) is the MTP lane's
serve host; `.62` (clang-11) cannot serve MTP.**

**Committed levers (now canonical since `68261b28`; old
`ARLE_DSV4_DECODE_COMPRESSOR_BATCH` gate removed):**
- `a4239598` compressor-GEMV batch (bf16 cublasLt m=N): n=22 perrow 162→92, step
  302→237, +28% (profiling-on relative; re-confirm clean).
- `3e3e50e0` full-flatten (batched per-slot compressor_update + inverse-rope +
  sw-window): n=22 step 237→219, **+8% over the GEMV lever** — MARGINAL; the step
  is still ∝n (residual is irreducible per-row compute). Reconsider (see audit).

## Open items / next steps (ranked by value)

1. **MTP enablement — B=1 is ~1.2×, batched-MTP-vs-per-row reaches ~1.7× at
   c≥8.** Two-part finding on this build:
   - **Head load fixed in the cleanup pass:** `--spec-type mtp` is now
     self-sufficient, the MTP draft head log reports the effective loader state,
     and the default draft depth is d2. `ARLE_DSV4_SPEC_DECODE` remains only a
     backwards-compatible env fallback, not a required companion to the flag.
   - **NOT a crash — the head-load deepgemm-JIT hangs/pathologically-slow
     (RESOLVED diagnosis, isolated on a CLEAN origin/main build).** Built clean
     origin/main on .62 from an scp'd 18.5MB source bundle (`/data01/arle-clean`,
     no franken). MTP head load: 8 workers stay ALIVE, no CUDA error/panic, but it
     never reaches "serving" in >14 min — the deepgemm JIT for the MTP head's
     expert shapes stalls at 16 cached kernels (cache stops growing). The earlier
     "crashes" were my short poll windows timing out on this slow/hung load.
     **NOT my campaign** (clean tree; it's the loader/deepgemm-JIT, not the decode
     path) and **NOT a throughput regression** (no-MTP clean = 43 tok/s c=1).
     **Prime suspect: the clang-11 deepgemm-JIT host compiler** I had to force on
     .62 (`-ccbin clang++-11`, because gcc-8.3 can't do the bridge's `-std=c++20`)
     — it likely chokes/hangs on the MTP head's specific shapes. So this is most
     likely a **.62-toolchain artifact**, not a real origin/main MTP regression.
     TO GET THE REAL ~53: build+serve MTP on a proper build host (gcc≥10, no
     clang-11 JIT workaround), where the MTP-head deepgemm JIT compiles normally.
     Transfer note: `scp` via jumpbox (18.5MB one-shot) >> base64 chunks.
2. **Compute/comm overlap for the BATCHED (n>1) lane.** As of `087df440`
   (2026-06-18), B=1 allreduce decode always overlaps shared-expert compute
   with the routed-MoE all-reduce; the old `ARLE_DSV4_COMM_OVERLAP` env is gone.
   This still does not cover `seq_len>1` verify/batched rows. Extend to n>1:
   hide the MoE all-reduce under cross-layer / cross-expert compute (overlap
   sources at any batch: token-chunk, intra-GEMM tile, cross-layer op,
   cross-expert). Three-stage handshake:
   ev_c (compute record) → comm wait_event(ev_c)+AllReduce+ev_m record →
   compute wait_event(ev_m) before the consumer. Raises the saturation level.
3. **True linear scaling** (throughput ∝ n) — needs the per-row COMPUTE to stop
   growing with n: DP-attention (distribute the batch across ranks; earlier
   flagged H20-compute-saturated in
   [[reference_dsv4_industry_baseline_and_h20_ceiling]] — re-measure) OR a
   compressor-cache/indexer-select redesign. Comm overlap alone does NOT give
   linear scaling (compute ∝n remains).

## Code-health audit (是否很乱 — yes, getting messy)

- **22k LOC** across dsv4.rs (5.7k) / attention.rs (10.6k) / moe.rs (5.5k) —
  inherent (most complex model), but the hot files are huge.
- **`ARLE_DSV4_*` env-gate sprawl** = real. First cleanup pass deleted the
  MTP rollback/attention/CSA/K/V/tail/probe dump paths and the strict
  tree-verify machinery; remaining buckets to prune (per the
  `f342c24f` classification compile→cfg / proven→locked / experiment→env):
  - **Debug/dump (consolidate under ONE gate or delete):** ATTN_DUMP, CSA_DUMP,
    KNEW_DUMP, TAIL_DUMP, MTP_ROLLBACK_DUMP(_LAYER), DSA_LOGITS_PROBE(_LIMIT/_SMS),
    LINEAR_PROFILE, STAGE_PROFILE, DECODE_PHASE_TIME, NVTX, FLASHMLA_PROBE.
  - **Proven default-on → lock to cfg/remove the gate:** DECODE_PROJ_DEEPGEMM,
    PREFILL_PROJ_DEEPGEMM, PREFILL_INDEXER_DEEPGEMM, FLASHMLA_DECODE(_BATCHED),
    DSA_INDEXER (all default-on per their wins).
  - **Keep as runtime knobs:** MOE_BACKEND, MOE_TRANSPORT, SPEC_DECODE,
    DECODE_GRAPH, GPU_ROUTER.
  Target: ~42 → ~10.
- **Decode loop (`forward_decode_batch_stream_impl`)** has **8 `for r in 0..n`
  passes**; the full-flatten added gated P1a/P1b/F1/F2 splits. Complexity high
  for +8%.
- **`3e3e50e0` full-flatten = top revert candidate**: +8% marginal, adds the
  loop-split complexity, and is **MTP-incompatibility-suspect** (validated
  without MTP). If item 1 shows the flatten breaks MTP, revert it and keep the
  cleaner `a4239598` GEMV lever (+28%, 2 files).

## Decision log
- "baseline" = current-best / all-opts-ON config (not the gate-OFF reference) —
  [[feedback_baseline_means_current_best_all_opts_on]].
- Every measured gain kept (even ~1%), but the flatten's +8% is under review
  against its complexity + MTP risk.
