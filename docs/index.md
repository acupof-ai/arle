# Doc index

Last refreshed: 2026-04-15 (post tiered-kv M3b contract tranche + remote acceptance update).

PARA layout: **Projects** (time-bound efforts) · **Plans** (in-flight design + execution) · **Research** (feasibility studies) · **Reviews** (standalone audits) · **Resources** (references) · **Areas** (long-running concerns) · **Archives** (inactive). Experience entries (`errors/`, `wins/`, `reviews/`) are listed at the bottom in reverse chronological order; the latest 3 of each are always-loaded per `CLAUDE.md`.

| Path | Status | TL;DR |
| --- | --- | --- |
| **Projects** | | |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | **Active — M2b + M0.3 + M3a + M3b-contract local shipped** | Hierarchical KV cache (T0 GPU → T1 host pinned → T2 NVMe → T3 NIXL). Scheduler selector flip, BF16 `page_size=16`, host-tier skeleton, and the first `lookup_or_stage` / page-lifecycle contract all landed locally; remote CUDA acceptance and runtime wiring still pending |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active | Priority ledger for agent-grade serving — radix wiring, session routing, constrained decoding, speculative decoding. P-labels superseded by tiered-kv M-milestones |
| [projects/kv-quantization-long-context.md](projects/kv-quantization-long-context.md) | **Partially shipped** | TurboQuant Phases 1–3 (KV + weight + fused decode attention) shipped via [`turboquant-integration.md`](plans/turboquant-integration.md); FP8-native FlashInfer track deferred |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | MLX Metal: Qwen3/3.5 direct `mlx-sys` bridge, serial serving runtime, scheduler not yet on hot path |
| [projects/qwen35-batched-decode.md](projects/qwen35-batched-decode.md) | Done | Qwen3.5 batched decode: FlashInfer HD256 + scheduler integration. Historical record |
| [projects/xma-future-research.md](projects/xma-future-research.md) | Research radar | Observations on accelerated-model-architecture repos (training/experimental, not serving) |
| **Plans** | | |
| [plans/tiered-kv-cache-tasks.md](plans/tiered-kv-cache-tasks.md) | Active | Tiered KV Cache execution split: local Mac / remote GPU / parallel-GPU lanes, milestone-by-milestone |
| [plans/tiered-kv-cache-m2b-remote-acceptance.md](plans/tiered-kv-cache-m2b-remote-acceptance.md) | Active | Remote CUDA acceptance checklist for the 2026-04-15 M2b local batch |
| [plans/tiered-kv-cache-m0.3-m3a-remote-acceptance.md](plans/tiered-kv-cache-m0.3-m3a-remote-acceptance.md) | Active | Remote CUDA acceptance checklist for the 2026-04-15 M0.3 + M3a local batch |
| [plans/tiered-kv-cache-m3b-remote-acceptance.md](plans/tiered-kv-cache-m3b-remote-acceptance.md) | Active | Remote CUDA acceptance checklist for the 2026-04-15 M3b contract/state-machine local batch |
| [plans/tiered-kv-cache-remote-validation.md](plans/tiered-kv-cache-remote-validation.md) | Active (2026-04-13 batch) | Older remote validation checklist; for the M2b batch use the M2b remote acceptance doc above |
| [plans/2026-04-15-metal-backend-execution-checklist.md](plans/2026-04-15-metal-backend-execution-checklist.md) | Active | Prioritized execution checklist for turning Metal from serial beta into production-grade Apple Silicon serving |
| [plans/2026-04-15-metal-backend-acceptance-plan.md](plans/2026-04-15-metal-backend-acceptance-plan.md) | Active | Strict acceptance gates for Metal serving, API, DX, and the remaining live-scheduler blockers |
| [plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md) | Blueprint | Forward plan for the eventual `infer-cuda-kernels` crate extraction (already partially extracted). Trip wires: FA-3 H100, MLA, NCCL, FP8 GEMM, spec-decode GPU, second consumer |
| [plans/turboquant-integration.md](plans/turboquant-integration.md) | Complete (post-impl) | TurboQuant Phases 1–3 reference: KV TurboQuant → weight ITQ3_S → fused decode attention. Canonical post-implementation design |
| [plans/qwen35-sglang-parity.md](plans/qwen35-sglang-parity.md) | Active | Qwen3.5 SGLang parity — prefix cache fixed, batched prefill remaining |
| [plans/speculative-decoding-impl.md](plans/speculative-decoding-impl.md) | Architecture plan | DraftEngine + SpeculativeScheduler integration plan; CPU-only `speculative.rs` exists, GPU integration not started |
| [plans/q4k-native-gpu.md](plans/q4k-native-gpu.md) | Queued | Native Q4_K GPU kernel design for GGUF; not yet started |
| [plans/gemma-gguf-support.md](plans/gemma-gguf-support.md) | Queued | Gemma 4 model support + full GGUF loader; planning only |
| [plans/kv-quant-remote-validation.md](plans/kv-quant-remote-validation.md) | Active | Validation checklist for the suffix-only KV migration patch (BF16/FP8/INT8/TQ3); no recorded results yet |
| [plans/backend-reorg-followups.md](plans/backend-reorg-followups.md) | Tracker | Round 2 backend reorg follow-ups: F1 done (`19a433d`), F2 parked (Option B), F3 done (`4b493c8`), Round 3 reverted by Route-A |
| **Research** | | |
| [research/speculative-decoding-feasibility.md](research/speculative-decoding-feasibility.md) | Feasibility | Draft model, EAGLE/EAGLE3, Medusa, MTP — phased integration plan, draft model first |
| [research/dflash-metal-feasibility.md](research/dflash-metal-feasibility.md) | Feasibility | DFlash (speculative) on Metal: feasible, medium priority. Now partially shipped via the experimental Metal DFlash path |
| [research/kv-quantization-metal.md](research/kv-quantization-metal.md) | Feasibility | KV quant on Metal deprioritized: MLX has no FP8 dtype, BF16-only; defer until C>4 with long contexts |
| **Reviews** | | |
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
| [experience/wins/2026-04-15-tiered-kv-m3b-local.md](experience/wins/2026-04-15-tiered-kv-m3b-local.md) | | Tiered KV M3b local contract tranche: `lookup_or_stage`, `StageTicket`, and pure page lifecycle landed without pretending CUDA runtime wiring exists |
| [experience/wins/2026-04-15-tiered-kv-m0.3-m3a-local.md](experience/wins/2026-04-15-tiered-kv-m0.3-m3a-local.md) | | Tiered KV M0.3 + M3a local landing: BF16 `page_size=16`, page-aware pool, host-tier skeleton, tier-aware node metadata |
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
| [experience/wins/2026-04-09-marlin-prefill.md](experience/wins/2026-04-09-marlin-prefill.md) | | Marlin W4 prefill: 5–25× TTFT speedup for long prompts |
| [experience/wins/2026-04-09-w4-int8kv-combo.md](experience/wins/2026-04-09-w4-int8kv-combo.md) | | W4 weight + INT8 KV combined throughput |
| [experience/wins/2026-04-09-native-w4-throughput.md](experience/wins/2026-04-09-native-w4-throughput.md) | | Native W4 throughput results |
| [experience/wins/2026-04-09-tq-weight-analysis.md](experience/wins/2026-04-09-tq-weight-analysis.md) | | TurboQuant weight analysis |
| [experience/wins/2026-04-09-ppl-weight-quant.md](experience/wins/2026-04-09-ppl-weight-quant.md) | | PPL evaluation for weight quantization |
| [experience/wins/2026-04-09-bench-quant-quality.md](experience/wins/2026-04-09-bench-quant-quality.md) | | Quantization quality benchmark results |
| [experience/wins/2026-04-09-bench-l4-qwen3-4b.md](experience/wins/2026-04-09-bench-l4-qwen3-4b.md) | | Qwen3-4B benchmark on L4 GPU |
| [experience/wins/2026-04-09-long-context-agent.md](experience/wins/2026-04-09-long-context-agent.md) | | Long-context agent benchmark |
| [experience/wins/2026-04-09-metal-optimization-progress.md](experience/wins/2026-04-09-metal-optimization-progress.md) | | Metal optimization progress snapshot |
| [experience/wins/2026-04-08-weight-quantization.md](experience/wins/2026-04-08-weight-quantization.md) | | Weight quantization: W8 +72%, W4 +39%, W2 TurboQuant framework |
| [experience/wins/2026-04-08-kv-quant-fused-dequant.md](experience/wins/2026-04-08-kv-quant-fused-dequant.md) | | KV fused-dequant attention: FP8 2× capacity, INT8 parity |
| [experience/wins/2026-04-08-int8-kv-batched-decode.md](experience/wins/2026-04-08-int8-kv-batched-decode.md) | | INT8 KV batched decode initial benchmark |
| [experience/wins/2026-04-02-sglang-parity-final.md](experience/wins/2026-04-02-sglang-parity-final.md) | | Final SGLang parity numbers (Qwen3.5) |
| [experience/wins/2026-04-02-sglang-parity-achieved.md](experience/wins/2026-04-02-sglang-parity-achieved.md) | | Qwen3.5 SGLang parity achieved (C=1–C=16 ahead) |
| [experience/wins/2026-04-02-sglang-vs-infer-qwen35.md](experience/wins/2026-04-02-sglang-vs-infer-qwen35.md) | | Qwen3.5 head-to-head vs SGLang |
| [experience/wins/2026-04-02-128slots-high-concurrency.md](experience/wins/2026-04-02-128slots-high-concurrency.md) | | 128-slot high-concurrency fixes |
| [experience/wins/2026-04-02-piecewise-cuda-graph.md](experience/wins/2026-04-02-piecewise-cuda-graph.md) | | Piecewise CUDA Graph: per-group capture for Qwen3.5 |
| [experience/wins/2026-04-02-modelforward-trait-redesign.md](experience/wins/2026-04-02-modelforward-trait-redesign.md) | | ModelForward trait redesign for multi-model support |
| [experience/wins/2026-04-02-batched-kernels-results.md](experience/wins/2026-04-02-batched-kernels-results.md) | | Batched kernel optimization results |
| [experience/wins/2026-04-02-qwen35-baseline-8slots.md](experience/wins/2026-04-02-qwen35-baseline-8slots.md) | | Qwen3.5 baseline with 8 slots |
| [experience/wins/2026-04-01-sglang-parity-steps1-4.md](experience/wins/2026-04-01-sglang-parity-steps1-4.md) | | Qwen3-8B scheduler fixes: C=4 exceeds SGLang |
| [experience/wins/2026-04-01-throughput-vs-sglang.md](experience/wins/2026-04-01-throughput-vs-sglang.md) | | Qwen3-8B throughput head-to-head vs SGLang |
| [experience/wins/2026-04-01-bench-raw-data.md](experience/wins/2026-04-01-bench-raw-data.md) | | Raw benchmark data for Qwen3-8B |
| [experience/wins/2026-04-01-long-seq-agent-bench.md](experience/wins/2026-04-01-long-seq-agent-bench.md) | | Long-sequence agent benchmark |
| [experience/wins/2026-04-01-qwen35-scheduler-support.md](experience/wins/2026-04-01-qwen35-scheduler-support.md) | | Qwen3.5 scheduler + FlashInfer HD256 |
| [experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md](experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md) | | MLX Metal benchmark alignment + 4-bit optimization |
| [experience/wins/2026-03-31-batched-decode-throughput.md](experience/wins/2026-03-31-batched-decode-throughput.md) | | 128 → 811 tok/s (6.3×) via batched decode optimizations |
| [experience/wins/2026-03-31-throughput-profiling.md](experience/wins/2026-03-31-throughput-profiling.md) | | Profiling-driven throughput optimization |
| [experience/wins/2026-03-31-nsys-profiling-decode.md](experience/wins/2026-03-31-nsys-profiling-decode.md) | | nsys profiling methodology |
| **Archives (inactive)** | | |
| [archives/quantization-architecture-design-spec.md](archives/quantization-architecture-design-spec.md) | Archived 2026-04-15 | LinearWeight enum design spec — never implemented; TurboQuant became the actual production path |
| [archives/sglang-parity-qwen3-8b.md](archives/sglang-parity-qwen3-8b.md) | Archived 2026-04-15 | Qwen3-8B SGLang parity — Steps 1–4 done, C=4 exceeds. (Qwen3.5 parity is still active under `plans/qwen35-sglang-parity.md`) |
| [archives/backend-metal-split.md](archives/backend-metal-split.md) | Archived 2026-04-15 | F1 metal split executed — commits `64c0baa` / `32875b2` / `19a433d` / `f59238c` |
| [archives/2026-04-13-cuda-kv-long-prefix-followup.md](archives/2026-04-13-cuda-kv-long-prefix-followup.md) | Archived 2026-04-15 | KVCache offload panic in legacy CPU offload path; whole code path slated for deletion in tiered-kv M3c |
| [archives/cuda-crate-extraction.md](archives/cuda-crate-extraction.md) | Archived | 4-shell-crate extraction proposal reverted by Route-A on 2026-04-15 |
| [archives/art-grade-architecture-for-long-agent-infer.md](archives/art-grade-architecture-for-long-agent-infer.md) | Archived | Earlier "art-grade architecture" workspace topology proposal — reverted by Route-A; PR discipline still applies |
| [archives/mlx-optimization.md](archives/mlx-optimization.md) | Archived | Early MLX optimization plan — superseded by current Metal roadmap |
| [archives/mlx-metal-phase1-validation.md](archives/mlx-metal-phase1-validation.md) | Archived | MLX Metal Phase 1 validation record |
| [archives/mlx-performance-analysis.md](archives/mlx-performance-analysis.md) | Archived | MLX performance analysis snapshot |
