# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog should record more than feature additions. It should also record:

- breaking changes
- deprecated surfaces
- support-matrix changes
- migration notes when user action is required

Related governance docs:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)
- [docs/compatibility.md](docs/compatibility.md)

## [Unreleased]

### 2026-04-15 — Workspace consolidation + CUDA layer hygiene

Coordinated refactor round finishing Route-A and pre-staging the internal seams for a future CUDA kernel crate extraction. No user-facing behavior changes — all work is structural.

#### Workspace consolidation (Route-A)
- Folded four shell crates back into `infer`. The workspace is now a flat `infer` crate again, with submodules as the only seam between backends.
- Collapsed the `agent_engine` duplicate façade into `server_engine`. There is now one engine trait (`InferenceEngine`) and one loaded enum (`LoadedInferenceEngine`) covering CUDA, Metal, and CPU backends through a single dispatch path.
- Renamed types for unambiguous semantics: `ServerEngine` → `InferenceEngine`, `CompleteRequest` → `CompletionRequest`, `Usage` → `TokenUsage`, etc. Imports across the HTTP server, agent CLI, and scheduler updated in lock-step.
- `LoadedInferenceEngine` is now the single entry point for both the HTTP server and the agent CLI — no more backend-specific façades above the engine trait.

#### Chat naming disambiguation
- Renamed the OpenAI wire-format chat types to `OpenAi*` (`OpenAiChatMessage`, `OpenAiToolCall`, `OpenAiFunctionCall`, …) so they no longer collide with the internal protocol names re-exported from `infer_chat::protocol`. The HTTP layer now consistently uses `OpenAi*` on the wire and converts to the internal types before handing work to the engine.

#### CUDA layer hygiene
- Split `backend/cuda/ffi.rs` (1500 lines) into ten domain submodules (attention, gemm, kv, quant, graph, stream, etc.) with a clean re-export surface — no behavioral change, just navigability.
- Introduced `backend::cuda::prelude` as the proto-API contract for downstream modules. Only genuinely universal CUDA handles land in the prelude; per-discipline writeup, `TokenKVPool` explicitly stays out (see `prelude.rs` doc comment for the rule).
- Deleted four dead Triton kernels that had no live callers and removed the vestigial `replaced_cuda_files` bookkeeping directory left over from an earlier migration.
- `build.rs` Triton `cargo:rerun-if-changed` list is now auto-derived from a directory walk instead of a hand-maintained constant, so new Triton kernels don't silently skip rebuilds.

#### Future direction
- Locked in [`docs/plans/cuda-kernel-crate-extraction.md`](plans/cuda-kernel-crate-extraction.md) as the forward blueprint. The current `infer` layout is option A (monolithic); option B (standalone `infer-cuda-kernels` crate) is pre-staged through the `backend::cuda::prelude` seam and executes as a one-day mechanical refactor when any of the six documented trip wires fires (FA-3 for H100, MLA / DeepSeek-V3, NCCL tensor parallel, FP8 GEMM, speculative decode GPU path, or a second in-tree consumer of the kernel layer).

### Governance
- Added a formal stability policy
- Added a support matrix document
- Added a compatibility and deprecation policy
- Added performance and correctness gate guidance
- Documented the CPU backend as a development-oriented smoke and validation path

### Added
- Radix-tree prefix cache for cross-request KV reuse
- Paged KV block manager with copy-on-write sharing
- Token-level KV pool (page_size=1, FlashInfer-compatible)
- GPU-CPU KV offload for contexts beyond VRAM capacity
- CUDA Graph batched decode (per batch size 1-32)
- Continuous batching scheduler with decode-priority and chunked prefill
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`, SSE streaming)
- Prometheus metrics (`/metrics`) and stats endpoint (`/v1/stats`)
- Qwen3 (0.5B-72B) and Qwen3.5-4B model support
- Built-in agent runtime with tool calling (shell, python)
- Benchmark suite (throughput, agent, multi-request)
- macOS Metal backend (experimental)
- Development-oriented CPU backend path for non-GPU request validation

### Performance
- TTFT 4.6x faster than SGLang v0.5.9 (8.6ms vs 39.3ms at C=1)
- Throughput parity: 0.99x at C=1, 0.92x at C=8
- 100% KV cache hit rate on multi-turn agent benchmarks
