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
