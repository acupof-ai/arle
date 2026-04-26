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

## [Unreleased]

### 2026-04-26 — Open-source usability and `arle` front door cleanup

#### CLI / DX
- Added `arle serve`, a unified front door that launches the matching serving
  binary (`infer`, `metal_serve`, or `cpu_serve`) from the release artifact or
  PATH.
- Added `--no-tools` for the local agent runtime so one-shot and REPL prompts
  can explicitly disable built-in shell/python tool execution.
- Extended `arle --doctor --json` to schema version 3 with tool/sandbox
  diagnostics, including the detected sandbox backend.

#### Packaging
- Renamed release tarballs to `arle-<version>-<platform>.tar.gz`.
- macOS release artifacts now include both `arle` and `metal_serve`; Linux
  artifacts include `arle`, `infer`, and `bench_serving`.
- The Docker image now uses `ghcr.io/cklxx/arle` and enters through `arle`
  instead of exposing only `infer`.

#### Docs and examples
- Added copyable examples under `examples/` for curl, stdlib Python,
  Docker Compose, Apple Silicon local serving, and the tiny train fixture.
- Updated README, Chinese README, support matrix, release checklist, and
  security guidance for the unified front door and tool safety controls.

### 2026-04-25 — Truth-surface cleanup

Documentation-only refactor that collapses `docs/` to a single source of
truth per [`docs/plans/2026-04-20-project-constitution-and-refactor-plan.md`](docs/plans/2026-04-20-project-constitution-and-refactor-plan.md)
§2. No code or behavior change.

Net effect: the documentation tree shrinks by ~330 markdown files. After
this commit series, `docs/index.md` lists every document that counts as a
source of truth; anything not on that index is not.

Retired surfaces:

- `docs/archives/` and `docs/areas/` removed; the surviving "Workspace
  governance rules" (PR discipline + crate-admission criteria) inlined
  into `docs/architecture.md`.
- `docs/plans/` collapsed from 58 entries to 10 — the 8 active plans
  listed in `docs/index.md` plus the canonical bench-parameter
  (`guidellm-integration.md`) and kernel-crate-extraction blueprints.
  Six tiered-kv `*-remote-acceptance.md` checklists folded into the
  `docs/projects/tiered-kv-cache.md` milestone ledger as one-line
  "completed YYYY-MM-DD; see wins/<file>" pointers.
- `docs/projects/` collapsed from 8 to 5; the dropped three
  (`kv-quantization-long-context`, `qwen35-batched-decode`,
  `xma-future-research`) are either superseded by `docs/resources/
  kv-cache-quantization.md` or off-roadmap.
- `docs/research/` collapsed from 6 to 1 (only the
  `mni-ml-framework-notes.md` reference referenced from the agent-RL
  project survives).
- `docs/reviews/` collapsed from 4 to 2 (cuda-kernel-six-principles +
  metal-ecosystem-route-correction; both still cited from active docs).
- `infer/docs/` parallel tree retired; `profiling-guide.md` consolidated
  into `docs/resources/`.
- 45 `pending-remote` / `pending-local-rerun` stub wins/ entries that
  never converted to real measurements deleted.
- 44 pre-2026-04-15 micro-cleanup wins/ + early errors/ retired (history
  preserved in git log + this CHANGELOG).
- 150 superseded bench wins/ entries retired (intra-step iterations of
  CUDA c1–c16 closure, Qwen3.5 paged-prefill landing, Qwen3.6 MoE
  DFlash bring-up, and per-step scheduler / KV-tier redesigns) — kept
  the milestone summaries and the latest-per-topic entry only.

Survival criteria for future cleanups: `docs/index.md` lists every
source-of-truth file. Adding a second index, a parallel doc tree, or a
sibling status matrix is a regression and must be rejected at PR time.

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

#### Kernel crate extraction (option B landed same day)
- Followed the hygiene round by executing the option B extraction locked in [`docs/plans/cuda-kernel-crate-extraction.md`](docs/plans/cuda-kernel-crate-extraction.md) — the `backend::cuda::prelude` seam was the staging point, and commits `a4e12f5` → `0ab2cd1` → `081cf32` landed the one-day mechanical refactor. `backend/cuda/` now contains only `bootstrap.rs`; all kernel sources, FFI, paged KV, FlashInfer wrappers, graph pool, tensor primitives, KV quant, and TurboQuant live under [`crates/cuda-kernels/`](crates/cuda-kernels/) with a one-way `infer → cuda-kernels` dependency. CUDA kernel C++ sources moved from `infer/csrc/cuda/` to `crates/cuda-kernels/csrc/{attention,gemm,kv,misc,quant}/`.
- The `mlx-sys` bridge was promoted from `infer/mlx-sys/` to [`crates/mlx-sys/`](crates/mlx-sys/) as part of the same Route-A flattening so both native layers (CUDA, Metal) sit peer-level under `crates/`.

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
