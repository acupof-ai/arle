# Doc index

Last refreshed: 2026-04-15 (post tiered-KV M2b + M0.3 + M3a + M3b + M3c **L4 remote acceptance**).

PARA layout: **Projects** (time-bound efforts) · **Plans** (in-flight design + execution) · **Research** (feasibility studies) · **Reviews** (standalone audits) · **Resources** (references) · **Areas** (long-running concerns) · **Archives** (inactive). Experience entries (`errors/`, `wins/`, `reviews/`) are listed at the bottom in reverse chronological order; the latest 3 of each are always-loaded per `CLAUDE.md`.

| Path | Status | TL;DR |
| --- | --- | --- |
| **Projects** | | |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | **Active — M2b + M0.3 + M3a + M3b + M3c accepted on L4 2026-04-15** | Hierarchical KV cache (T0 GPU → T1 host pinned → T2 NVMe → T3 NIXL). Scheduler selector flip, BF16 `page_size=16`, host-tier skeleton, staged-lookup/page-lifecycle contract, plannerless runtime wiring, and the legacy contiguous CPU-offload retirement all shipped and signed off on the L4 host; only real staged completion / promotion still pending |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active | Priority ledger for agent-grade serving — radix wiring, session routing, constrained decoding, speculative decoding. P-labels superseded by tiered-kv M-milestones |
| [projects/kv-quantization-long-context.md](projects/kv-quantization-long-context.md) | **Partially shipped** | TurboQuant Phases 1–3 (KV + weight + fused decode attention) shipped via [`turboquant-integration.md`](plans/turboquant-integration.md); FP8-native FlashInfer track deferred |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | MLX Metal: Qwen3/3.5 direct `mlx-sys` bridge, but roadmap now explicitly prioritizes scheduler-first serving over more single-request-only tuning |
| [projects/qwen35-batched-decode.md](projects/qwen35-batched-decode.md) | Done | Qwen3.5 batched decode: FlashInfer HD256 + scheduler integration. Historical record |
| [projects/xma-future-research.md](projects/xma-future-research.md) | Research radar | Observations on accelerated-model-architecture repos (training/experimental, not serving) |
| **Plans** | | |
| [plans/tiered-kv-cache-tasks.md](plans/tiered-kv-cache-tasks.md) | Active | Tiered KV Cache execution split: local Mac / remote GPU / parallel-GPU lanes, milestone-by-milestone |
| [plans/tiered-kv-cache-m2b-remote-acceptance.md](plans/tiered-kv-cache-m2b-remote-acceptance.md) | **Accepted 2026-04-15 on L4** | See `wins/2026-04-15-tiered-kv-m2b-remote.md` |
| [plans/tiered-kv-cache-m0.3-m3a-remote-acceptance.md](plans/tiered-kv-cache-m0.3-m3a-remote-acceptance.md) | **Accepted 2026-04-15 on L4** | See `wins/2026-04-15-tiered-kv-m0.3-m3a-remote.md` |
| [plans/tiered-kv-cache-m3b-remote-acceptance.md](plans/tiered-kv-cache-m3b-remote-acceptance.md) | **Accepted 2026-04-15 on L4** | See `wins/2026-04-15-tiered-kv-m3b-remote.md` |
| [plans/tiered-kv-cache-m3c-remote-acceptance.md](plans/tiered-kv-cache-m3c-remote-acceptance.md) | **Accepted 2026-04-15 on L4** | See `wins/2026-04-15-tiered-kv-m3c-remote.md` |
| [plans/tiered-kv-cache-remote-validation.md](plans/tiered-kv-cache-remote-validation.md) | Active (2026-04-13 batch) | Older remote validation checklist; for the M2b batch use the M2b remote acceptance doc above |
| [plans/2026-04-15-metal-backend-execution-checklist.md](plans/2026-04-15-metal-backend-execution-checklist.md) | Active | Prioritized execution checklist for turning Metal from serial beta into production-grade Apple Silicon serving |
| [plans/2026-04-15-metal-backend-acceptance-plan.md](plans/2026-04-15-metal-backend-acceptance-plan.md) | Active | Strict acceptance gates for Metal serving, API, DX, and the remaining live-scheduler blockers |
| [plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md) | Blueprint | Forward plan for the eventual `infer-cuda-kernels` crate extraction (already partially extracted). Trip wires: FA-3 H100, MLA, NCCL, FP8 GEMM, spec-decode GPU, second consumer |
| [plans/turboquant-integration.md](plans/turboquant-integration.md) | Complete (post-impl) | TurboQuant Phases 1–3 reference: KV TurboQuant → weight ITQ3_S → fused decode attention. Canonical post-implementation design |
| [plans/qwen35-sglang-parity.md](plans/qwen35-sglang-parity.md) | Active | Qwen3.5 SGLang parity — prefix cache fixed, batched prefill remaining |
| [plans/speculative-decoding-impl.md](plans/speculative-decoding-impl.md) | Architecture plan | DraftEngine + SpeculativeScheduler integration plan; CPU-only `speculative.rs` exists, GPU integration not started |
| [plans/q4k-native-gpu.md](plans/q4k-native-gpu.md) | **Shipped (post-impl)** | Native Q4_K GPU kernel (`q4k_gemv`) + packed GGUF loader fast path. Preserved as post-implementation design reference. Carnice-27B fits on L4-24GB. |
| [plans/gemma-gguf-support.md](plans/gemma-gguf-support.md) | Split — GGUF shipped, Gemma queued | GGUF loading (BF16/F16/Q8_0/Q4_K_M) **shipped** alongside the Q4_K path; Gemma 4 model still unwired (detection in `model_registry` only, no `model/gemma*.rs`) |
| [plans/kv-quant-remote-validation.md](plans/kv-quant-remote-validation.md) | Active | Validation checklist for the suffix-only KV migration patch (BF16/FP8/INT8/TQ3); no recorded results yet |
| [plans/backend-reorg-followups.md](plans/backend-reorg-followups.md) | Tracker | Round 2 backend reorg follow-ups: F1 done (`19a433d`), F2 parked (Option B), F3 done (`4b493c8`), Round 3 reverted by Route-A |
| **Research** | | |
| [research/speculative-decoding-feasibility.md](research/speculative-decoding-feasibility.md) | Feasibility | Draft model, EAGLE/EAGLE3, Medusa, MTP — phased integration plan, draft model first |
| [research/dflash-metal-feasibility.md](research/dflash-metal-feasibility.md) | Feasibility | DFlash (speculative) on Metal: feasible, medium priority. Now partially shipped via the experimental Metal DFlash path |
| [research/kv-quantization-metal.md](research/kv-quantization-metal.md) | Feasibility | KV quant on Metal deprioritized: MLX has no FP8 dtype, BF16-only; defer until C>4 with long contexts |
| **Reviews** | | |
| [reviews/2026-04-15-metal-ecosystem-route-correction.md](reviews/2026-04-15-metal-ecosystem-route-correction.md) | Active | External Apple Silicon serving reality check: `vllm-metal` / Docker / `mlx-lm` / Ollama calibration, local serial-server symptoms, and the corrected Metal execution order |
| [reviews/2026-04-14-cuda-kernel-six-principles-review.md](reviews/2026-04-14-cuda-kernel-six-principles-review.md) | Living reference | CUDA kernel 六要素审计 + Heat Map (P0/P1/P2 priorities) + first-wave optimizations. Source of truth for the next kernel-perf wave |
| [reviews/2026-04-06-10k-star-readiness.md](reviews/2026-04-06-10k-star-readiness.md) | Assessment (updated 2026-04-10) | 10K-star readiness review: technical foundation strong, governance/release discipline scoped as follow-up |
| [experience/reviews/2026-04-02-cuda-link-optimization-gaps.md](experience/reviews/2026-04-02-cuda-link-optimization-gaps.md) | Code review | CUDA path optimization gaps in Qwen3.5 batched decode |
| **Resources** | | |
| [resources/kv-cache-quantization.md](resources/kv-cache-quantization.md) | Reference | KV cache quantization research: methods, frameworks, eval metrics |
| [resources/metal-dflash.md](resources/metal-dflash.md) | User guide | Metal DFlash: supported models, CLI/env usage, benchmark workflow, limits |
| [resources/metal-dflash-params.md](resources/metal-dflash-params.md) | Reference | Metal DFlash parameter reference for `metal_request`, `metal_bench`, `metal_serve` |
| **Areas** | | |
| [areas/forward-pass-precision-vs-llamacpp.md](areas/forward-pass-precision-vs-llamacpp.md) | Long-running | Forward-pass numerical precision investigation vs llama.cpp reference |
| **Top-level utility / governance** | | |
| [architecture.md](architecture.md) | Reference | Current workspace/package topology and runtime split |
| [codebase-map.md](codebase-map.md) | Reference | Current code map: runtime, crates, docs, tests, entrypoints |
| [environment.md](environment.md) | Reference | Environment variable reference for CLI, build, tests, integration, setup |
| [stability-policy.md](stability-policy.md) | Governance | Stable / beta / experimental / internal surface classification |
| [support-matrix.md](support-matrix.md) | Governance | Current backend / platform / model / quantization support levels |
| [compatibility.md](compatibility.md) | Governance | Breaking-change, deprecation, migration policy |
| [perf-and-correctness-gates.md](perf-and-correctness-gates.md) | Governance | Minimum validation expectations for correctness and performance-sensitive changes |
| [release-checklist.md](release-checklist.md) | Governance | Repeatable release checklist: docs, validation, artifacts, compatibility |
| [from-zero-to-inference-engine.md](from-zero-to-inference-engine.md) | Tutorial | Walkthrough: from zero to inference engine |
| [gpu-benchmark-a100.md](gpu-benchmark-a100.md) | Reference | A100 GPU benchmark methodology and results |
| [gpu-test-prompt.md](gpu-test-prompt.md) | Active | Route-A verification checklist for the post-refactor workspace layout |
| [code-review.md](code-review.md) | Reference | Standing code-review guidelines |
| [../crates/README.md](../crates/README.md) | Reference | Workspace crate ownership and dependency boundaries |
| **Experience — errors (latest first)** | | |
| [experience/errors/2026-04-15-e2e-phase3-replay-drift.md](experience/errors/2026-04-15-e2e-phase3-replay-drift.md) | | e2e Phase 3 replay drift: `PrefixReuseAction::ReplayFinalToken` in the single-request engine is not numerically consistent with cold batch prefill — fall back to full recompute on exact match |
| [experience/errors/2026-04-14-p0-page16-blocker.md](experience/errors/2026-04-14-p0-page16-blocker.md) | | M0.3 page_size=16 blocker: BF16 prefill→pool migration kernel is NHD-only; FlashInfer reads HND |
| [experience/errors/2026-04-14-broken-rebase-baseline.md](experience/errors/2026-04-14-broken-rebase-baseline.md) | | Rebase / migration must verify upstream baseline before attributing errors |
| [experience/errors/2026-04-13-batched-decode-high-concurrency.md](experience/errors/2026-04-13-batched-decode-high-concurrency.md) | | Batched decode regression at high concurrency |
| [experience/errors/2026-04-10-remaining-gguf-bugs.md](experience/errors/2026-04-10-remaining-gguf-bugs.md) | | Remaining GGUF bugs after the load-path fix |
| [experience/errors/2026-04-10-qwen35-attn-output-gate-missing.md](experience/errors/2026-04-10-qwen35-attn-output-gate-missing.md) | | Qwen3.5 attention-output gate missing |
| [experience/errors/2026-04-10-gguf-load-path-forward-garbage.md](experience/errors/2026-04-10-gguf-load-path-forward-garbage.md) | | GGUF load path producing garbage forward outputs |
| [experience/errors/2026-04-09-w4-gptq-quality.md](experience/errors/2026-04-09-w4-gptq-quality.md) | | W4 GPTQ quality regression investigation |
| [experience/errors/2026-04-09-scheduler-greedy-divergence.md](experience/errors/2026-04-09-scheduler-greedy-divergence.md) | | Scheduler greedy divergence: Triton vs FlashInfer numerical diff |
| [experience/errors/2026-04-09-metal-optimization-pitfalls.md](experience/errors/2026-04-09-metal-optimization-pitfalls.md) | | Metal optimization pitfalls collection |
| [experience/errors/2026-04-09-carnice-27b-q4k-oom.md](experience/errors/2026-04-09-carnice-27b-q4k-oom.md) | | Carnice 27B Q4_K OOM root cause |
| [experience/errors/2026-04-02-rope-axis-bug.md](experience/errors/2026-04-02-rope-axis-bug.md) | | RoPE axis bug in Qwen3.5 |
| [experience/errors/2026-03-31-flashinfer-segfault-debug.md](experience/errors/2026-03-31-flashinfer-segfault-debug.md) | | 3 bugs causing FlashInfer batch decode crash |
| **Experience — wins (latest first)** | | |
| [experience/wins/2026-04-15-bench-metal-rerun.md](experience/wins/2026-04-15-bench-metal-rerun.md) | | Metal rerun snapshot on `f7b3b84`: direct path stayed flat and `metal_serve` quick sweep reproduced the post-`M0.2b` shape (`512/256 C=4` at `65.5 tok/s`, `TTFT p50 1742ms`) |
| [experience/wins/2026-04-15-tiered-kv-m2b-remote.md](experience/wins/2026-04-15-tiered-kv-m2b-remote.md) | | Tiered KV M2b L4 remote acceptance: static/build/test gates, agent-trace 14/14, 100x shared-prefix `bad=0`, found + fixed e2e Phase 3 replay drift in the single-request engine |
| [experience/wins/2026-04-15-tiered-kv-m0.3-m3a-remote.md](experience/wins/2026-04-15-tiered-kv-m0.3-m3a-remote.md) | | Tiered KV M0.3 + M3a L4 remote acceptance: BF16 `page_size=16` sweep flat on C≤4, long contexts slightly up, C≥8 recovered from 2026-04-13 zero-throughput regression |
| [experience/wins/2026-04-15-tiered-kv-m3b-remote.md](experience/wins/2026-04-15-tiered-kv-m3b-remote.md) | | Tiered KV M3b L4 remote acceptance: contract + runtime-wire tranche signed off, all `lookup_or_stage_*` and `PageLifecycleState` tests green on CUDA |
| [experience/wins/2026-04-15-tiered-kv-m3c-remote.md](experience/wins/2026-04-15-tiered-kv-m3c-remote.md) | | Tiered KV M3c L4 remote acceptance: legacy contiguous CPU KV offload retirement confirmed non-regressing on long-session agent trace, TTFT within noise of M2b baseline |
| [experience/wins/2026-04-15-metal-m0.2a-bench-validation.md](experience/wins/2026-04-15-metal-m0.2a-bench-validation.md) | | Metal M0.2a bench check: direct path stayed flat and quick HTTP sweep still shows the serial-serving shape |
| [experience/wins/2026-04-15-metal-m0.2a-request-state-local.md](experience/wins/2026-04-15-metal-m0.2a-request-state-local.md) | | Metal M0.2a local landing: resumable Qwen3/Qwen3.5 request state now owns prefill/decode/cleanup ahead of serving rewiring |
| [experience/wins/2026-04-15-tiered-kv-m3b-runtime-local.md](experience/wins/2026-04-15-tiered-kv-m3b-runtime-local.md) | | Tiered KV M3b local runtime wire: admission uses plannerless `lookup_or_stage`, eviction scores use live signals, and published blocks stamp session/keepalive metadata |
| [experience/wins/2026-04-15-tiered-kv-m3c-local.md](experience/wins/2026-04-15-tiered-kv-m3c-local.md) | | Tiered KV M3c local cleanup: legacy contiguous CPU KV offload removed, tests/docs/CLI aligned, compatibility shim kept as no-op |
| [experience/wins/2026-04-15-tiered-kv-m3b-local.md](experience/wins/2026-04-15-tiered-kv-m3b-local.md) | | Tiered KV M3b local contract tranche: `lookup_or_stage`, `StageTicket`, and pure page lifecycle landed without pretending CUDA runtime wiring exists |
| [experience/wins/2026-04-15-tiered-kv-m2b-local.md](experience/wins/2026-04-15-tiered-kv-m2b-local.md) | | Tiered KV M2b local landing: scheduler selector flip, safe same-slot resurrection, alloc retry, retain hard cap, tombstone GC |
| [experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md](experience/wins/2026-04-15-route-a-cuda-internal-hygiene.md) | | Route-A revert + CUDA internal hygiene: 4-shell-crate split reverted, ffi.rs split into 10 domain modules, prelude as proto-API |
| [experience/wins/2026-04-14-tiered-kv-m1-m2a.md](experience/wins/2026-04-14-tiered-kv-m1-m2a.md) | | Tiered KV M1a→M1b→M2a: directory.rs retired, RadixCache wired, TokenKVPool gains refcount + watermark eviction |
| [experience/wins/2026-04-14-bench-single-token-kernel-port.md](experience/wins/2026-04-14-bench-single-token-kernel-port.md) | | Qwen3-4B single-token decode Triton→CUDA C port bench (−8.8% Qwen3-4B; Qwen3.5 unchanged) |
| [experience/wins/2026-04-14-metal-dflash-qwen3.md](experience/wins/2026-04-14-metal-dflash-qwen3.md) | | Qwen3 Metal DFlash on M4 Pro: 5.9× decode throughput on 20/256 with native draft block size |
| [experience/wins/2026-04-14-kv-quant-audit.md](experience/wins/2026-04-14-kv-quant-audit.md) | | KV quantization audit: format coverage, kernel paths, regression gates |
| [experience/wins/2026-04-13-bench-baseline.md](experience/wins/2026-04-13-bench-baseline.md) | | 2026-04-13 benchmark baseline snapshot |
| [experience/wins/2026-04-13-bench-page1.md](experience/wins/2026-04-13-bench-page1.md) | | Page 1 (page_size=1) bench reference for the M0.3 lift |
| [experience/wins/2026-04-13-bench-agent-trace-baseline.md](experience/wins/2026-04-13-bench-agent-trace-baseline.md) | | Agent-trace bench baseline for repeated-prefix scoring |
| [experience/wins/2026-04-13-qwen35-metal-cpp-path-tuning.md](experience/wins/2026-04-13-qwen35-metal-cpp-path-tuning.md) | | Qwen3.5 Metal C++ path tuning: fused blocks + split() vs slice() |
| [experience/wins/2026-04-10-metal-final-optimization.md](experience/wins/2026-04-10-metal-final-optimization.md) | | Metal final optimization wave (round 1) |
| **2026-04-08 → 04-09 — Quantization wave** | Collapsed era | [marlin-prefill](experience/wins/2026-04-09-marlin-prefill.md) 5–25× TTFT · [weight-quantization](experience/wins/2026-04-08-weight-quantization.md) W8 +72% / W4 +39% · [kv-quant-fused-dequant](experience/wins/2026-04-08-kv-quant-fused-dequant.md) FP8 2× capacity · [int8-kv-batched-decode](experience/wins/2026-04-08-int8-kv-batched-decode.md) · [w4-int8kv-combo](experience/wins/2026-04-09-w4-int8kv-combo.md) · [native-w4-throughput](experience/wins/2026-04-09-native-w4-throughput.md) · [tq-weight-analysis](experience/wins/2026-04-09-tq-weight-analysis.md) · [ppl-weight-quant](experience/wins/2026-04-09-ppl-weight-quant.md) · [bench-quant-quality](experience/wins/2026-04-09-bench-quant-quality.md) · [bench-l4-qwen3-4b](experience/wins/2026-04-09-bench-l4-qwen3-4b.md) · [long-context-agent](experience/wins/2026-04-09-long-context-agent.md) · [metal-optimization-progress](experience/wins/2026-04-09-metal-optimization-progress.md) |
| **2026-04-01 → 04-02 — Qwen3.5 SGLang parity + batched-kernel wave** | Collapsed era | [sglang-parity-final](experience/wins/2026-04-02-sglang-parity-final.md) · [sglang-parity-achieved](experience/wins/2026-04-02-sglang-parity-achieved.md) C=1-16 ahead · [sglang-vs-infer-qwen35](experience/wins/2026-04-02-sglang-vs-infer-qwen35.md) · [128slots-high-concurrency](experience/wins/2026-04-02-128slots-high-concurrency.md) · [piecewise-cuda-graph](experience/wins/2026-04-02-piecewise-cuda-graph.md) · [modelforward-trait-redesign](experience/wins/2026-04-02-modelforward-trait-redesign.md) · [batched-kernels-results](experience/wins/2026-04-02-batched-kernels-results.md) · [qwen35-baseline-8slots](experience/wins/2026-04-02-qwen35-baseline-8slots.md) · [sglang-parity-steps1-4](experience/wins/2026-04-01-sglang-parity-steps1-4.md) Qwen3-8B C=4 exceeds SGLang · [throughput-vs-sglang](experience/wins/2026-04-01-throughput-vs-sglang.md) · [bench-raw-data](experience/wins/2026-04-01-bench-raw-data.md) · [long-seq-agent-bench](experience/wins/2026-04-01-long-seq-agent-bench.md) · [qwen35-scheduler-support](experience/wins/2026-04-01-qwen35-scheduler-support.md) · [mlx-metal-alignment-and-optimization](experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md) |
| **2026-03-31 — Foundation: batched decode origin** | Collapsed era | [batched-decode-throughput](experience/wins/2026-03-31-batched-decode-throughput.md) 128→811 tok/s (6.3×) · [throughput-profiling](experience/wins/2026-03-31-throughput-profiling.md) · [nsys-profiling-decode](experience/wins/2026-03-31-nsys-profiling-decode.md) |
| **Archives (inactive)** | | |
| [archives/cuda-crate-extraction.md](archives/cuda-crate-extraction.md) | Archived | Round-3 four-shell-crate extraction proposal reverted by Route-A on 2026-04-15; kept as the cautionary "overly broad" reference cited from the active kernel-crate blueprint |
| [archives/art-grade-architecture-for-long-agent-infer.md](archives/art-grade-architecture-for-long-agent-infer.md) | Archived | Earlier "art-grade architecture" workspace topology proposal — reverted by Route-A, but §六 governance and §七 acceptance criteria are still load-bearing for current architecture decisions |
