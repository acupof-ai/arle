# DSv4 decode → SGLang-class (5–6 ms/token) perf roadmap

**Date:** 2026-06-04. **Target:** DSv4-Flash TP=8/EP=8 decode at SGLang-class
latency (~5–6 ms/token) via full CUDA-graph capture + on-device routing, on 8×H20.
**Status:** roadmap; gated behind the incremental-decode correctness fix (#23).

## 1. Why we're far from 5–6 ms today

DSv4 decode is currently (a) **broken** (incremental-decode `start_pos>0` diverges from
prefill — #23) and (b) architecturally **launch-bound + host-round-trip-bound + eager**:

| Gap | Evidence (our code) | Cost |
|---|---|---|
| Static expert weights rebuilt every layer/step | ~~`moe.rs` `build_grouped_cache` per call~~ **FIXED** `d5115de5` | was ~529 ms/token |
| MoE routing on the **host** every MoE layer every step | `moe.rs:181/512/708` `clone_dtoh(&logits.data)` → `infer_moe::route`; comment `moe.rs:20`: *"a device router is a perf follow-up"* | per-layer H2D/D2H round-trip + host compute, **un-capturable** |
| TP all-reduce stays **eager** under TP | `collective.rs:406` `supports_graph_capture()` → `false`; `executor.rs:408` "NCCL not graph-capturable" | ~250–400 `cuLaunchKernel`/token, no graph under TP |
| No full decode-graph for DSv4 | `executor.rs:563` "decode graph disabled (MLA host-routing per step)" | per-kernel launch overhead |

## 2. SGLang / DeepEP precedents (how 5–6 ms is reached)

- **NCCL itself need not be graph-capturable.** SGLang captures the full TP decode via
  either (a) a **custom graph-capturable all-reduce** (one-shot/two-shot single kernel,
  used for the small decode tensors) in its `GroupCoordinator`, or (b) **"breakable CUDA
  graph"** — capture the graphable segments between all-reduces, re-invoke the collective
  (with the same tensor refs) at the break points. So our `supports_graph_capture=false`
  is a self-imposed limit, not hardware.
  ([SGLang breakable cuda graph](https://sgl-project.github.io/advanced_features/breakable_cuda_graph.html),
  [SGLang cuda-graph](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph))
- **DeepEP low-latency decode mode** — pure-RDMA GPU dispatch/combine, **hook-based
  comm/compute overlap that occupies 0 SMs**, ~20 SMs to saturate, GPU-native group-limited
  gating (DeepSeek-V3 routing on-device), FP8 supported. This is the "GPU instructions
  reduce H2D/D2H": routing + token-shuffle run on-device, no host round-trip.
  ([DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md))

## 3. Roadmap (sequenced — correctness before perf)

0. **[DONE `d5115de5`]** Grouped-cache prebuild — static expert weights built once at load
   (~529 ms/token host rebuild eliminated). Plus the wq_b TP-shard correctness fix.
1. **[#23, Codex — PREREQ]** Fix DSv4 incremental decode (`start_pos>0`). Decode must match
   prefill before optimizing it. Hypothesis: SW ring-cache carry + RoPE position at
   `start_pos>0`. Nothing below is safe until decode is numerically correct.
2. **On-device MoE routing.** Compute top-k / group-limited gating on-GPU (the "device
   router" follow-up at `moe.rs:20`); drop the `clone_dtoh(&logits)`→`infer_moe::route`
   round-trip at `moe.rs:181/512/708`. Kills the per-layer H2D/D2H **and** makes the MoE
   step capturable. License: decode-step D2H count → 0 (nsys), no correctness change.
3. **DeepEP low-latency decode mode.** Route the EP all-to-all through DeepEP's
   decode-optimized kernels (GPU dispatch/combine, hook overlap, ~20 SM) instead of the
   host-routed `allreduce` path. Pairs with (2). License: decode tok/s A/B vs the
   allreduce path at the production shape.
4. **Graph-capturable all-reduce.** Add a custom one-shot/two-shot all-reduce kernel
   (capturable) for the decode-sized TP tensors, OR adopt breakable-graph; flip
   `supports_graph_capture`. License: captured == eager logits (greedy bit-exact).
5. **Full decode-graph capture.** Capture the whole DSv4 decode step (MLA + on-device-routed
   MoE + capturable all-reduce) into one graph; remove the `executor.rs:408` TP gate for the
   capturable path. License: ~250–400 launches/token → 1; decode tok/s A/B.

**Endpoint:** launch-overhead-free + host-round-trip-free + comm/compute-overlapped decode
→ SGLang-class 5–6 ms/token. Each step is its own commit + bench entry (per the bench
mandate); steps 2–5 are individually license-or-kill at the production decode shape (not a
smoke shape — distilled lesson: SLO verdict from the SLO workload).

## 4. Note

Steps 2–5 are a multi-commit architecture arc, not a single fix. The hard prerequisite is
#23 (decode correctness); the biggest single launch-reduction is step 5 (full graph), and
the biggest host-overhead kill is step 2 (on-device routing). Sources are upstream
precedent (hypothesis-grade per the upstream-scan skill); each lands only on a local
H20 A/B at the production shape.
