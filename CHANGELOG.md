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

### Metal

- Qwen3.5-0.8B MLX 4bit single-request step-driver reaches 305.5 tok/s mean
  / 304.7 p50 on M4 Pro 20c for `1024/256`. The matched GGUF Q4_K_M
  exact default remains 202.1 tok/s direct for correctness, while the
  opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s
  step-driver, so current status surfaces no longer present the historical
  211.7 tok/s GGUF-only profile as the Metal SOTA headline. Evidence:
  [`docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`](docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md).
  Native-q4 GGUF evidence:
  [`docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md`](docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md).

## [0.1.4] — 2026-04-28

Correctness fix release for Metal chat generation. Multimodal Qwen3.5 /
Qwen3.6 MoE configs (HF `Qwen3_5MoeForConditionalGeneration` family)
declare `eos_token_id` as an array at the root of `config.json` while
nesting a single base-LM EOS inside `text_config`. Our Metal config
loader took only the first element of the array and let `text_config`'s
scalar shadow the root, then the C++ generate path passed only that one
id into the stop check — so on `mlx-community/Qwen3.6-35B-A3B-4bit` the
model walked past `<|im_end|>` after the first reply and hallucinated
fresh user/assistant turns, with `skip_special_tokens=true` decoding
leaving "user"/"assistant" plain-text role names visible.

### Metal

- `infer/src/backend/metal/config.rs` — replaced ad-hoc `get_eos` +
  `load_stop_token_ids` with `resolve_stop_token_ids`, a generic
  HuggingFace-precedence resolver: `generation_config.json` (HF
  inference-time authority) → `config.json` root → `text_config`,
  scalar-or-array normalized to `Vec<u32>`, dedup preserves first-seen
  order, fall back to 151645. `MetalModelConfig.eos_token_id` is now
  the first id from the resolved array; `stop_token_ids` is the
  authoritative list.
- `infer/src/backend/metal/{qwen35.rs, generate.rs}` — C++ generate
  paths now extend `stop_ids` with the full `config.stop_token_ids`,
  not just `config.eos_token_id`. Sort + dedup after merge.
- `infer/src/backend/metal/request_state.rs` — scheduler/HTTP path
  also folds `config.stop_token_ids` into `ResumableRequestState`,
  guarded by `params.ignore_eos` so benchmarks that explicitly want
  to generate past EOS still can.
- Three new HF-precedence unit tests in `metal::config::tests`.

Verified on Apple M4 Max with `mlx-community/Qwen3.6-35B-A3B-4bit`:
multi-turn REPL produces clean, on-topic, EOS-terminated replies; tool
execution works on the third turn. Background entry under
`docs/experience/wins/2026-04-28-fix-metal-eos-token-id-array.md`.

### CUDA · breaking — SM env-var policy

- `INFER_CUDA_SM` and `CUDA_SM` are removed. The new build-time env
  var is `TORCH_CUDA_ARCH_LIST` (alias `CMAKE_CUDA_ARCHITECTURES`),
  matching the PyTorch / vLLM convention. Migration: replace
  `INFER_CUDA_SM=89` with `TORCH_CUDA_ARCH_LIST=8.9`.
- T1 (sm_80/86/89/90) is the default fat-binary set when no env is
  set. T2 (sm_100/120) is opt-in. T3 (sm < 80) is rejected at build
  time with a hint. Unknown SMs hard-fail rather than silently
  falling back.
- Plan + tier policy live in `docs/plans/sm-coverage.md`; four
  per-SM bench stubs are tracked under `docs/experience/wins/` as
  `pending-remote` and will be filled in alongside the multi-cubin
  AOT dispatch (Phase B/C).

## [0.1.3] — 2026-04-27

Packaging fix release: macOS bottles and release tarballs were missing
`mlx.metallib`, so every `metal_serve` model load on a fresh install
hit "Failed to load the default metallib". v0.1.3 ships the metallib
alongside the binaries on every distribution channel. Also folds in
the scheduler `cuda/runtime.rs` + `cuda/core.rs` splits and the Phase 2
KV swap revert. Pre-built artifacts on the
[GitHub Release page](https://github.com/cklxx/arle/releases/tag/v0.1.3)
and on GHCR (`ghcr.io/cklxx/arle:0.1.3`, `:0.1`, `:latest`).

### Tooling / packaging

- `scripts/package_macos_metal_artifact.sh` bundles `mlx.metallib`
  into the macOS tarball (auto-discovers from the cargo build dir;
  override via `ARLE_MLX_METALLIB`). MLX's `load_default_library`
  searches binary-colocated paths, so without this the brew bottle
  and curl-installed binaries fell straight through to the
  compile-time `METAL_PATH` (a cmake build dir absent in
  distribution).
- `scripts/install.sh` installs `mlx.metallib` next to the binaries
  on macOS-arm64.
- `crates/mlx-sys/build.rs` stages `mlx.metallib` into
  `target/<profile>/` after the cmake build, so locally-built
  `metal_serve` works without colocating the metallib by hand.
- `cklxx/homebrew-tap` formula picks up `mlx.metallib` via
  `bin.install "mlx.metallib" if File.exist?("mlx.metallib")` — no
  effect on installs of older tarballs that don't ship the file.

### Scheduler

- `infer/src/scheduler/cuda/runtime.rs` split into
  `runtime/{admission,fetch,helpers,scheduler_loop,swap,tests}.rs` to
  contain the per-iteration scheduler loop. Behavior unchanged.
- `infer/src/scheduler/cuda/core.rs` extracted into
  `helpers/state_types/emit_worker/construction/warmup` siblings —
  same flat-module-no-`mod.rs` shape used elsewhere.
- Phase 2 KV swap path deleted entirely (vLLM V1 / SGLang precedent
  — tier demote/promote already covers the workload that motivated
  swap; the dual path was not paying its complexity cost).
- Host pool for the now-removed swap path right-sized while the
  revert was in flight; entry recorded under `errors/`.

### Metal

- Qwen3.5 prefill: Rust fallback path now materializes its hidden
  state correctly when the C++ prefill bails out (previously
  returned a stale handle).
- Qwen3.6 MoE wired through the C++ prefill alongside Qwen3.5 so the
  large model uses the same fused step path.
- `mlx-sys` FFI guard shared across the `infer` and `autograd`
  crates so MLX process-global state has one Rust synchronization
  boundary regardless of which side enters MLX first.

### HTTP

- `infer/src/http_server.rs` split into
  `http_server/{types,handlers,router,tests}.rs`. Behavior unchanged.

### Refactors

- `infer/src/backend/metal/request_state.rs` extracted into helper
  modules + standalone tests.

## [0.1.2] — 2026-04-27

Engine consolidation, Metal GGUF perf, train binary migration, and a
v2 landing site. Pre-built artifacts on the
[GitHub Release page](https://github.com/cklxx/arle/releases/tag/v0.1.2)
and on GHCR (`ghcr.io/cklxx/arle:0.1.2`, `:0.1`, `:latest`).

### Runtime

- `ModelInferenceEngine` and per-model aliases removed; everything now
  routes through `LoadedInferenceEngine`. The scheduler grew a unified
  `Cuda(RequestHandle)` variant so CUDA drives the same path as Metal.
- E2E tests migrated to the `LoadedInferenceEngine` scheduler path,
  matching the runtime that production serves use.
- Unused `PreemptionMode` enum and config field dropped from the
  scheduler — the unification project chose tier demote/promote over
  the separate preemption type.
- `kv-tier` gained a `WholeKv` variant on `ReadmissionPlan` plus a
  `PlanKind` discriminator; the Plan path is documented as
  reserved-for-distributed scaffolding (no runtime change).
- Metal Qwen3.5-0.8B GGUF Q4_K_M decode now crosses 200 tok/s on M4 Pro
  (211.7 tok/s for 512 prompt / 1024 decode) after Q5_K/Q8_0 affine
  repack and Q6/group16 qmv tile tuning. Evidence:
  [`docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md`](docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md).
- Qwen3.6-35B-A3B Metal rechecked locally with a short load/execute
  diagnostic. This is not DFlash acceptance evidence; future DFlash
  optimization claims for Qwen3.6 should use long-context /
  ultra-long-sequence workloads only. Evidence:
  [`docs/experience/wins/2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md`](docs/experience/wins/2026-04-27-bench-metal-qwen36-a3b-dflash-quick-check.md).

### HTTP

- `session_id` now threads through `CompletionRequest`; the hardcoded
  `None` in the OpenAI-compatible handlers is gone.

### Metal

- Qwen3.5 GGUF decode path tuned (perf); Q4/Q6 weights are now repacked
  into MLX affine layout at load time so decode hits the fast kernel.
- `qwen35` GGUF projections kept packed end-to-end (avoids a redundant
  unpack/repack round-trip).
- Metal GGUF decode status doc refreshed under
  `infer/src/backend/metal/AGENTS.md`.

### Train

- `crates/train/src/bin/*` moved to `crates/train/src/commands/*` —
  `arle train …` and `arle data …` are the only entry points; the
  `[[bin]]` declarations in `crates/train/Cargo.toml` are gone.

### Docs & web

- Landing site v2: 3-pane docs shell + Quickstart page; homepage
  split-layout hero with dark footer and framed sections; logo
  redrawn as a two-story humanist 'a'; wordmark swapped for topology
  mark with v2 design tokens.
- Install cards on the landing site now lead with `brew install
  cklxx/tap/arle` and the `curl | sh` installer (Docker / source
  demoted to fallbacks).
- README, roadmap, support matrix, and maintainer docs now share the
  same Metal GGUF support and benchmark wording.
- Intra-doc links cleaned up for `rustdoc -D warnings`: links from
  `kv-tier` and elsewhere now resolve to public items only.
- `CONTRIBUTING.md` Getting Started leads with an end-user pointer to
  the README install section so contributors don't assume source
  build is the canonical path.

### Tooling

- Homebrew formula in `cklxx/homebrew-tap` simplified to macOS-arm64
  only. The `mislav/bump-homebrew-formula-action` only bumps one URL,
  so a multi-platform formula went stale on every release; Linux
  brew users for a CUDA-driven tool are vanishingly rare (Docker /
  install.sh / source cover that path).

## [0.1.1] — 2026-04-27

Install ergonomics + a batch of TileLang, KV-tier, and Metal/Qwen3.5
follow-ups. Pre-built artifacts on the
[GitHub Release page](https://github.com/cklxx/arle/releases/tag/v0.1.1)
and on GHCR (`ghcr.io/cklxx/arle:0.1.1`, `:0.1`, `:latest`).

### Install

- **Homebrew tap**: `brew install cklxx/tap/arle`
  ([cklxx/homebrew-tap](https://github.com/cklxx/homebrew-tap)). The
  `bump-homebrew` job in `release.yml` keeps the formula in lockstep
  with each `v*` tag.
- **One-line installer**: `curl -fsSL
  https://github.com/cklxx/arle/releases/latest/download/install.sh
  | sh`. Detects platform, SHA256-verifies the tarball, installs to
  `~/.local/bin` (override via `INSTALL_DIR`).
- New `docs/install.md` documents the full matrix, env-var overrides,
  and uninstall steps; `docs/release-checklist.md` §4a covers the
  `HOMEBREW_TAP_TOKEN` secret needed for tap automation.

### Runtime

- TileLang prefill HD128 path behind the `tilelang-attn` feature
  (Experimental); per-Qwen3 head-config specialization. Prefill HD256
  + decode HD256 + tc-decode AOT kernels added under the same flag.
  L4 floor verified; H100 verification runbook landed.
- macOS Metal link fix: `compiler-rt` now linked so mlx-sys resolves
  `__isPlatformVersionAtLeast`; release tarballs and Metal CI build
  cleanly on hosted runners.
- Metal Qwen3.5 GGUF Q4 path: aligned `gdr` and prefill matmul; gated
  GGUF DFlash; preserved greedy C++ generate. Long bench coverage
  recorded under `docs/experience/wins/`.
- `release.yml` now installs `flashinfer-python` (in lockstep with
  Dockerfile) so Linux CUDA tarballs build without the
  `flashinfer/pos_enc.cuh` not-found regression.
- KV-tier P3 follow-ups: typed `FailureClass` in events with explicit
  emit policy; `OrchestratorEvent` unified into `CoordinatorEvent`;
  `emit_observability` switched to `try_send` while `Store*` paths
  retain required delivery.
- `cargo build` no longer requires an explicit backend feature on
  macOS — `default = ["cuda"]` was dropped from the workspace,
  `agent-infer`, and `crates/cli` so a flag-less build no longer pulls
  cudarc on platforms without nvcc.
- Standalone `pretrain` / `train_sft` / `train_grpo` /
  `train_multi_turn` / `eval_lm` / `download_dataset` /
  `convert_dataset` binaries removed from `crates/train`;
  `arle train …` and `arle data …` are the single front door
  (sources still in `crates/train/src/bin/*.rs` as in-process
  dispatch modules under `autobins = false`).

### Docs & web

- `docs/support-matrix.md` §4b documents the multi-turn KV reuse /
  tiered-KV stability tiers (T0 GPU Supported, T1 host-pinned Beta,
  T2 NVMe Beta, T3 cluster-shared Experimental — NIXL stub-only).
- `docs/experience/wins/README.md` explains the `*pending-remote*.md`
  filename convention so external readers do not mistake stubs for
  published claims.
- `docs/troubleshooting.md` and `docs/comparison.md` added.
- Astro/Vite landing site at `web/` (deploys via `pages.yml` to
  <https://cklxx.github.io/arle/>); homepage IA refactor — runnable
  hero, dated bench, dropped chrome; switched to minimal-white
  aesthetic.

### CI

- `cuda-ci.yml` gated behind `workflow_dispatch` until the
  self-hosted CUDA runner returns.

## [0.1.0] — 2026-04-26

First tagged release. Pre-built artifacts live on the
[GitHub Release page](https://github.com/cklxx/arle/releases/tag/v0.1.0)
and on GHCR (`ghcr.io/cklxx/arle:0.1.0`, `:0.1`, `:latest`).

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
