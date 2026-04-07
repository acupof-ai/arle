| Path | TL;DR |
| --- | --- |
| **Projects** | |
| [projects/qwen35-batched-decode.md](projects/qwen35-batched-decode.md) | **Done** — Qwen3.5 batched decode: FlashInfer HD256 + scheduler integration |
| [projects/quantization-architecture.md](projects/quantization-architecture.md) | **Design Review** — INT4 quantization architecture: LinearWeight enum, GEMV kernel, risk analysis |
| **Plans** | |
| [plans/sglang-parity.md](plans/sglang-parity.md) | **Done** — Qwen3-8B sglang parity plan (C=4 exceeded) |
| [plans/qwen35-sglang-parity.md](plans/qwen35-sglang-parity.md) | **In Progress** — Qwen3.5 sglang parity: ITL gap from per-request recurrent ops |
| **Resources** | |
| [resources/kv-cache-quantization.md](resources/kv-cache-quantization.md) | KV cache quantization research: methods, frameworks, eval metrics, implementation plan |
| **Experience — Errors** | |
| [experience/errors/2026-03-31-flashinfer-segfault-debug.md](experience/errors/2026-03-31-flashinfer-segfault-debug.md) | 3 bugs causing FlashInfer batch decode crash: hardcoded MAX_SEQ, GPU plan_info, double alloc |
| **Experience — Reviews** | |
| [experience/reviews/2026-04-02-cuda-link-optimization-gaps.md](experience/reviews/2026-04-02-cuda-link-optimization-gaps.md) | CUDA path review: implemented-but-not-landed optimizations, mainly Qwen3.5 batched decode / recurrent / graph gaps |
| **Experience — Wins** | |
| [experience/wins/2026-03-31-batched-decode-throughput.md](experience/wins/2026-03-31-batched-decode-throughput.md) | 128 → 811 tok/s (6.3x) via token pool + FlashInfer + buffer reuse + plan-once + CUDA Graph + argmax/scatter |
| [experience/wins/2026-03-31-nsys-profiling-decode.md](experience/wins/2026-03-31-nsys-profiling-decode.md) | nsys profiling methodology for decode kernel analysis |
| [experience/wins/2026-03-31-throughput-profiling.md](experience/wins/2026-03-31-throughput-profiling.md) | Profiling-driven throughput optimization for Qwen3 |
| [experience/wins/2026-04-01-sglang-parity-steps1-4.md](experience/wins/2026-04-01-sglang-parity-steps1-4.md) | Qwen3-8B scheduler fixes: C=4 exceeds sglang (260 vs 256 tok/s) |
| [experience/wins/2026-04-01-throughput-vs-sglang.md](experience/wins/2026-04-01-throughput-vs-sglang.md) | Qwen3-8B vs sglang head-to-head: C=1 -8%, C=4 +2%, TTFT 2.5x faster |
| [experience/wins/2026-04-01-bench-raw-data.md](experience/wins/2026-04-01-bench-raw-data.md) | Raw benchmark data for Qwen3-8B optimization runs |
| [experience/wins/2026-04-01-long-seq-agent-bench.md](experience/wins/2026-04-01-long-seq-agent-bench.md) | Long-sequence agent benchmark results |
| [experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md](experience/wins/2026-04-01-mlx-metal-alignment-and-optimization.md) | MLX Metal benchmark alignment log: prompt_tps / generation_tps / e2e_tps / TTFT, then 4-bit hot-path optimization |
| [experience/wins/2026-04-01-qwen35-scheduler-support.md](experience/wins/2026-04-01-qwen35-scheduler-support.md) | Qwen3.5 scheduler + FlashInfer HD256 batched decode; C=1 100tok/s, C=4 290tok/s vs sglang 107/349 |
