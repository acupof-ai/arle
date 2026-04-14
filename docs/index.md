| Path | TL;DR |
| --- | --- |
| **Projects** | |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | **Active** — Priority ledger for turning agent-infer into the strongest engine for agent sequences: radix cache wiring, session routing, constrained decoding, speculative decoding |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | **Active** — Hierarchical KV cache (T0 GPU → T2 host pinned → T3 NVMe → T4 NIXL) with coordinator-driven auto offload; folds in A1/B1/B3 |
| [projects/qwen35-batched-decode.md](projects/qwen35-batched-decode.md) | **Done** — Qwen3.5 batched decode: FlashInfer HD256 + scheduler integration |
| [projects/quantization-architecture.md](projects/quantization-architecture.md) | **Complete** — INT4 quantization architecture: LinearWeight dispatch, W4A16 GEMV, Marlin prefill |
| [projects/kv-quantization-long-context.md](projects/kv-quantization-long-context.md) | **Active** — KV 量化存储 + Agent 长上下文：FP8 native + INT8/INT4 fused-dequant + TurboQuant 3-bit |
| [projects/xma-future-research.md](projects/xma-future-research.md) | Research — XMA / accelerated model architectures future implications |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | **Active** — MLX Metal: Qwen3/3.5, direct `mlx-sys` bridge, serial serving |
| **Plans** | |
| [plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md) | **Blueprint** — Forward plan for the eventual `infer-cuda-kernels` crate extraction. Documents trip wires (FA-3 H100, MLA, NCCL, FP8 GEMM, spec-decode GPU, second consumer) and the one-day mechanical migration that executes when any trigger fires |
| [plans/tiered-kv-cache-tasks.md](plans/tiered-kv-cache-tasks.md) | **Active** — Tiered KV Cache execution split: local Mac / remote GPU / parallel-GPU lanes per phase |
| [plans/tiered-kv-cache-remote-validation.md](plans/tiered-kv-cache-remote-validation.md) | **Active** — Consolidated cargo / python command checklist for remote CUDA validation of the 2026-04-13 local batch |
| [gpu-test-prompt.md](gpu-test-prompt.md) | **Active** — Route-A verification checklist for the post-refactor workspace layout |
| [plans/sglang-parity.md](plans/sglang-parity.md) | **Done** — Qwen3-8B sglang parity (C=4 exceeded) |
| [plans/qwen35-sglang-parity.md](plans/qwen35-sglang-parity.md) | **Near-complete** — prefix cache fixed, batched prefill remaining |
| [plans/turboquant-integration.md](plans/turboquant-integration.md) | **Complete** — TurboQuant Phases 1-3: KV cache + weight + fused decode attention |
| **Resources** | |
| [resources/kv-cache-quantization.md](resources/kv-cache-quantization.md) | KV cache quantization research: methods, frameworks, eval metrics |
| [architecture.md](architecture.md) | Current workspace/package topology and runtime split |
| [codebase-map.md](codebase-map.md) | Current code map: runtime, crates, docs, tests, and entrypoints |
| [stability-policy.md](stability-policy.md) | Governance draft: stable/beta/experimental/internal surface classification |
| [support-matrix.md](support-matrix.md) | Governance draft: current backend / platform / model / quantization support levels |
| [compatibility.md](compatibility.md) | Governance draft: breaking-change, deprecation, and migration policy |
| [perf-and-correctness-gates.md](perf-and-correctness-gates.md) | Governance draft: minimum validation expectations for correctness and performance-sensitive changes |
| [release-checklist.md](release-checklist.md) | Governance draft: repeatable release checklist for docs, validation, artifacts, and compatibility review |
| [environment.md](environment.md) | Environment variable reference for CLI, build, tests, integration, and setup workflows |
| [../crates/README.md](../crates/README.md) | Workspace crate ownership and dependency boundaries |
| **Reviews** | |
| [reviews/2026-04-06-10k-star-readiness.md](reviews/2026-04-06-10k-star-readiness.md) | 10K star readiness review |
| [experience/reviews/2026-04-02-cuda-link-optimization-gaps.md](experience/reviews/2026-04-02-cuda-link-optimization-gaps.md) | CUDA path review: optimization gaps in Qwen3.5 batched decode |
| **Experience — Errors** | |
| [experience/errors/2026-04-09-scheduler-greedy-divergence.md](experience/errors/2026-04-09-scheduler-greedy-divergence.md) | Scheduler greedy divergence: Triton vs FlashInfer numerical diff |
| [experience/errors/2026-04-09-w4-gptq-quality.md](experience/errors/2026-04-09-w4-gptq-quality.md) | W4 GPTQ quality regression investigation |
| [experience/errors/2026-04-02-rope-axis-bug.md](experience/errors/2026-04-02-rope-axis-bug.md) | RoPE axis bug in Qwen3.5 |
| [experience/errors/2026-03-31-flashinfer-segfault-debug.md](experience/errors/2026-03-31-flashinfer-segfault-debug.md) | 3 bugs causing FlashInfer batch decode crash |
| **Experience — Wins** | |
| [experience/wins/2026-04-09-marlin-prefill.md](experience/wins/2026-04-09-marlin-prefill.md) | Marlin W4 prefill: 5-25x TTFT speedup for long prompts |
| [experience/wins/2026-04-09-native-w4-throughput.md](experience/wins/2026-04-09-native-w4-throughput.md) | Native W4 throughput results |
| [experience/wins/2026-04-09-ppl-weight-quant.md](experience/wins/2026-04-09-ppl-weight-quant.md) | PPL evaluation for weight quantization |
| [experience/wins/2026-04-09-bench-quant-quality.md](experience/wins/2026-04-09-bench-quant-quality.md) | Quantization quality benchmark results |
| [experience/wins/2026-04-09-w4-int8kv-combo.md](experience/wins/2026-04-09-w4-int8kv-combo.md) | W4 weight + INT8 KV combo performance |
| [experience/wins/2026-04-08-kv-quant-fused-dequant.md](experience/wins/2026-04-08-kv-quant-fused-dequant.md) | KV fused-dequant attention: FP8 2x capacity, INT8 parity |
| [experience/wins/2026-04-08-weight-quantization.md](experience/wins/2026-04-08-weight-quantization.md) | Weight quantization: W8 +72%, W4 +39%, W2 TurboQuant framework |
| [experience/wins/2026-04-08-int8-kv-batched-decode.md](experience/wins/2026-04-08-int8-kv-batched-decode.md) | INT8 KV batched decode initial benchmark |
| [experience/wins/2026-04-02-sglang-parity-achieved.md](experience/wins/2026-04-02-sglang-parity-achieved.md) | Qwen3.5 SGLang parity achieved (C=1–C=16 ahead) |
| [experience/wins/2026-04-02-sglang-parity-final.md](experience/wins/2026-04-02-sglang-parity-final.md) | Final SGLang parity numbers |
| [experience/wins/2026-04-02-sglang-vs-infer-qwen35.md](experience/wins/2026-04-02-sglang-vs-infer-qwen35.md) | Qwen3.5 head-to-head vs SGLang |
| [experience/wins/2026-04-02-128slots-high-concurrency.md](experience/wins/2026-04-02-128slots-high-concurrency.md) | 128 slots high concurrency fixes |
| [experience/wins/2026-04-02-piecewise-cuda-graph.md](experience/wins/2026-04-02-piecewise-cuda-graph.md) | Piecewise CUDA Graph: per-group capture for Qwen3.5 |
| [experience/wins/2026-04-02-modelforward-trait-redesign.md](experience/wins/2026-04-02-modelforward-trait-redesign.md) | ModelForward trait redesign for multi-model support |
| [experience/wins/2026-04-02-batched-kernels-results.md](experience/wins/2026-04-02-batched-kernels-results.md) | Batched kernel optimization results |
| [experience/wins/2026-04-02-qwen35-baseline-8slots.md](experience/wins/2026-04-02-qwen35-baseline-8slots.md) | Qwen3.5 baseline with 8 slots |
| [experience/wins/2026-04-01-sglang-parity-steps1-4.md](experience/wins/2026-04-01-sglang-parity-steps1-4.md) | Qwen3-8B scheduler fixes: C=4 exceeds sglang |
| [experience/wins/2026-04-01-throughput-vs-sglang.md](experience/wins/2026-04-01-throughput-vs-sglang.md) | Qwen3-8B vs sglang head-to-head |
| [experience/wins/2026-04-01-bench-raw-data.md](experience/wins/2026-04-01-bench-raw-data.md) | Raw benchmark data for Qwen3-8B |
| [experience/wins/2026-04-01-long-seq-agent-bench.md](experience/wins/2026-04-01-long-seq-agent-bench.md) | Long-sequence agent benchmark |
| [experience/wins/2026-04-01-qwen35-scheduler-support.md](experience/wins/2026-04-01-qwen35-scheduler-support.md) | Qwen3.5 scheduler + FlashInfer HD256 |
| [experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md](experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md) | MLX Metal benchmark alignment + 4-bit optimization |
| [experience/wins/2026-03-31-batched-decode-throughput.md](experience/wins/2026-03-31-batched-decode-throughput.md) | 128 → 811 tok/s (6.3x) via batched decode optimizations |
| [experience/wins/2026-03-31-nsys-profiling-decode.md](experience/wins/2026-03-31-nsys-profiling-decode.md) | nsys profiling methodology |
| [experience/wins/2026-03-31-throughput-profiling.md](experience/wins/2026-03-31-throughput-profiling.md) | Profiling-driven throughput optimization |
