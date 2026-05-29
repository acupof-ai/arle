# DSv4 — beat SGLang throughput by 30%+ (systematic optimization campaign)

> **Goal**: ARLE DSv4-Flash decode throughput **> SGLang × 1.30** on 8×H20
> TP=8, same model / ISL / OSL / concurrency. Continuous loop:
> measure → profile → optimize → A/B → repeat until met.

## Baselines

- **SGLang baseline** = image `eic-test-cn-shanghai.cr.volces.com/sglang/dsv4:v0`
  (27.2 GB, id `044ecc2e823fe`). The in-tree `/sgl-workspace/sglang` checkout
  has DSv4 source removed (HEAD = "Remove in-GitHub deployment for deepseek
  v4"; DSv4 ships as the prebuilt image, not source) → cannot serve DSv4
  from source. **Run the image** (kubectl pod/job from `sglang/dsv4:v0`) for
  the baseline.
- **ARLE current** (8×H20 TP=8, c=1, ISL≈17 / OSL=128): **15.4 tok/s** e2e,
  TTFT 274 ms, FlashMLA decode default ON (+14% vs legacy, byte-identical).
  Foundation validated 2026-05-29 (correct coherent output).

## Harness

`scripts/dsv4_beat_sglang_bench.sh <arle|sglang> <serve|bench|both>` — dep-free
OpenAI-compat concurrent-completion throughput bench, SLO shape ISL=1024
OSL=512, concurrency {1,8,32}. Results → `docs/trace-artifacts/beat-sglang/`.
Access: `~/bin/pod '<cmd>'` (tn tunnel — see [[project_h20_pod_access]]).
**GPU is shared (8×H20 = ~780 GB; one DSv4 instance ≈ 159 GB weights + KV) —
SGLang and ARLE must bench SEQUENTIALLY, not concurrently.**

## Loop (each iteration)

1. **Baseline** (once, then cache): deploy `sglang/dsv4:v0` → bench → record
   SGLang tok/s at each concurrency. Set TARGET = SGLang × 1.30.
2. **Measure ARLE**: build latest in-pod (`target-pod`) → serve → bench same
   shape. Record ARLE tok/s.
3. **Gap**: if ARLE ≥ TARGET at the SLO shape → DONE (write wins, stop loop).
4. **Profile** the dominant decode cost (nsys / per-NVTX wall-clock framing —
   CLAUDE.md §0: wall-clock is ground truth, not narrow window %). Rank
   bottlenecks.
5. **Optimize** the top bottleneck (delegate impl to worktree subagent;
   license-or-kill each hypothesis with a paired component A/B). Candidates
   from the operator map ([[../docs/projects/2026-05-29-dsv4-operator-map]]):
   MoE grouped-GEMM (GAP-A MMA), route kernel (GAP-B), FP8 KV cp.async
   (GAP-C), AllReduce/AllGather overlap, CUDA-graph decode capture,
   per-step launch fusion.
6. **A/B** same-binary env-flip, ≥2 shapes, confirm correctness (byte parity
   vs prior) + perf gain. Land if win; revert if wash/regression (errors entry).
7. **Record** iteration in this doc's log; reschedule.

## Iteration log

- **I0 (2026-05-29)**: campaign set up. SGLang baseline source = `dsv4:v0`
  image (in-tree source removed). ARLE foundation fixed (P→D KV handoff) →
  correct output + FlashMLA decode default ON. Harness landed. Next: deploy
  `sglang/dsv4:v0` baseline + ARLE full-SLO-shape bench → first gap number.

## Guardrails (CLAUDE.md §0)

- Wall-clock / per-request framing is ground truth for license-or-kill (not
  narrow nsys window %).
- One variable per A/B; same binary, same shell, two env flips, ≥2 shapes.
- Correctness gate before perf: every optimization must keep greedy output
  byte-identical (or within FP tolerance) vs the validated baseline.
- No legacy deletion (SM<90 fallback stays — see
  `wins/2026-05-29-dsv4-gpu-native-coherent-output-pd-handoff.md`).

- **I1 (2026-05-29)**: ARLE SLO bench (8×H20 TP=8, ISL=1024/OSL=512).
  c=1 = **5.65 tok/s** (88ms/decode-step). c=8/c=32 **timeout >300s** —
  unreasonable: step×512 ≈ 264s, no batching speedup. Decode-step profile
  (ARLE_DSV4_TRACE_LAYER, relative split):
  - **attn_core dominant** (attn_hybrid_kernel 0.186ms + csa_select 0.205ms
    + compressor/indexer/csa_project) — DSv4 sparse-attn machinery runs
    full CSA-select + compressor + indexer EVERY decode step.
  - **NCCL allreduce ≈21ms/step serial-blocking** (ffn_all_reduce 0.328ms +
    attn_all_reduce 0.103ms × 43 layers × 2) — NOT overlapped with compute.
  Top levers: (1) decode allreduce multi-stream overlap (SGLang DSv4 day-0
  approach; un-killed — distinct from the prefill AllGather-Q overlap killed
  2026-05-28); (2) c=8+ concurrency is broken (timeout) — continuous-batching
  decode at concurrency needs fixing BEFORE perf is comparable.
  SGLang baseline still pending user GPU-strategy choice. Next: fix c≥8
  decode concurrency (correctness/throughput) + decode allreduce overlap.

- **I2 (2026-05-29)**: DSv4 true batched decode — FFN half + NCCL allreduce
  now batch over N rows (c3e46932); attention core per-row (byte-identical).
  **Parity PASS** (8×H20 TP=8): c=4 batched output byte-identical to c=1
  per-row (`137 + 269 = 406...` all 4 rows). **Helps**: c=4 wall 6.1s vs
  ~10.8s if fully serial (≈1.8× from FFN/allreduce batching); not flat
  (attention still per-row). c=8 → `FlashMLA FP8 KV pool alloc OOM` (server
  graceful-aborts, not the I1 300s timeout) — the FP8 KV pool is allocated
  PER-SEQUENCE (per-state, state.rs), so 8 concurrent pools blow the
  mem-fraction-static budget.
  Batched-attention primitives landed on branch 7bfb9084 (batched indices
  kernel + KV-addressing design) but fwd-wiring gated: FlashMLA has no
  block_table (indices absolute into one kv base, splitkv_mla.cuh:468-545),
  so the subagent's per-step D2D staging-arena copy is the only b=N path
  WITHOUT a vendor patch — shape-dependent cost.
  **UNIFYING NEXT FIX (I3): shared persistent FP8 decode KV pool** (one
  allocation sized for num_slots, each slot owns a block range). Solves BOTH
  (a) c=8 OOM (one pool, not N) AND (b) batched FlashMLA attention without
  the per-step staging copy (indices offset per-row into the persistent
  shared pool). This is the qwen3 PagedKVPool model for DSv4 decode and the
  real "搞定 gpu 路径" fix.

- **I3a (2026-05-29)**: mem-fraction 0.6 does NOT fix c=8 — still
  `FlashMLA FP8 KV pool alloc OOM` (batch_width=7). The per-sequence FP8
  pools are allocated DYNAMICALLY at first-decode and are unbudgeted by
  mem-fraction-static → N concurrent pools OOM regardless of static %.
  c=4 still fine (parity PASS). Conclusion: the shared persistent pool is
  REQUIRED (not a perf nicety) — it's the only way c≥8 fits. Dispatching
  the shared-pool implementation. Step-1 mem-fraction shortcut KILLED.
