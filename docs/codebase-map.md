# ARLE Codebase Map

> **新人请先读 [`onboarding.md`](onboarding.md)**（30 分钟当前真相 + 验证清单）。
> 本文是 workspace 拓扑 canonical source；战略变化见下方 pointer。

This document is the canonical workspace-topology truth: where files live,
what each crate owns, and where to start reading. For ownership boundaries
and crate-admission governance see [architecture.md](architecture.md);
support status by surface lives in [support-matrix.md](support-matrix.md).
Qwen3.6 now serves on CUDA (FP8 MoE + batched paged decode — no longer
Metal-only); the in-flight model additions are GLM-5.2 (wired on the DSv4 CUDA
path, verification pending-remote) and the Metal VLMs (DeepSeek-OCR).

## 1. Workspace at a glance

The repository has three practical layers:

- workspace root package (`arle`): thin binary wrapper in `src/main.rs`
 that calls `cli::run()` to produce the `arle` binary (the only binary the
 workspace builds; gated on the default-on `cli` feature).
- `crates/infer-*`: the device-neutral runtime crate graph (IR → seam → core →
 backends → server → front-door). One `Engine` drives both backends; the
 single public front door is `infer-api`.
- `crates/`: the reusable control-plane/helper crates around the runtime
 (agent, chat, cli, tools), the GPU/bridge crates (cuda-kernels, mlx-sys), the
 spec/topology/routing leaves (qwen3-spec, qwen35-spec, deepseek-spec,
 infer-topo, infer-moe, infer-util), and the training stack (autograd, train).
- `docs/`: architecture and implementation notes (single
 source of truth; the historical `infer/docs/` parallel tree was retired
 during the 2026-04-25 truth-surface cleanup).

Current workspace members (ownership and boundaries are listed in
[architecture.md §Package Boundaries](architecture.md#package-boundaries)):

- workspace root package `arle` (produces the `arle` binary)
- **runtime crate graph:** `crates/infer-plan`, `crates/infer-seam`,
 `crates/infer-core`, `crates/infer-cuda`, `crates/infer-metal`,
 `crates/infer-hip`, `crates/infer-vulkan`,
 `crates/infer-server`, `crates/infer-api`, `crates/infer-topo`,
 `crates/infer-moe`, `crates/infer-util`
- **GPU / bridge:** `crates/cuda-kernels`, `crates/mlx-sys`,
 `crates/deepep-sys`, `crates/hip-sys`, `crates/hip-kernels`,
 `crates/vulkan-sys`, `crates/vulkan-kernels`
- **control plane / helpers:** `crates/agent`, `crates/chat`, `crates/cli`,
 `crates/tools`
- **specs:** `crates/qwen3-spec`, `crates/qwen35-spec`, `crates/deepseek-spec`,
 `crates/deepseek-ocr-spec`
- **training:** `crates/autograd`, `crates/train`, `crates/spec-train`
- **substrate:** `crates/infer-gguf`, `crates/kv-native-sys`,
 `crates/xgrammar-sys`

## 2. Main execution paths

### Agent CLI path

```text
src/main.rs
 -> cli::run()
 -> infer_util::hf_hub::resolve_model_source()
 + infer_api::LoadedInferenceEngine::load()
 -> agent::AgentSession (uses `dyn infer_api::InferenceEngine`)
 -> tools builtin tools + chat protocol
 -> LoadedInferenceEngine dispatches to CUDA / Metal / CPU backend
 (each request: tokenize -> ServeHandle::submit -> collect -> detokenize)
```

Key files:

- `src/main.rs`: `arle` binary entrypoint from the root `arle` package
- `crates/cli/src/lib.rs`: CLI startup, backend selection,
 `infer_api::{InferenceEngine, LoadedInferenceEngine}` load + REPL drive
- `crates/cli/src/repl.rs`: REPL loop, slash commands, terminal UX
- `crates/infer-api/src/lib.rs`: the single public front door — re-exports the
 `InferenceEngine` trait, `LoadedInferenceEngine` enum, and the
 `CompletionRequest`/`CompletionOutput`/`TokenUsage`/`CompletionStreamDelta`
 request/output/stream/telemetry types
- `crates/infer-api/src/loaded.rs`: `EngineLoadConfig` + `LoadedInferenceEngine`
 feature-gated backend dispatch (metal/cuda/cpu)
- `crates/infer-api/src/serve_engine.rs`: `ServeInferenceEngine` —
 tokenize → `ServeHandle::submit` → collect → detokenize per request
- `crates/infer-util/src/hf_hub.rs`: local model discovery + `resolve_model_source`
- `crates/agent/src/lib.rs`: session state, prompt assembly, turn loop
- `crates/tools/src/lib.rs`: builtin tools and shared tool hooks
- `crates/chat/src/lib.rs`: `OpenAiChatMessage` / `OpenAiToolDefinition` wire format + re-exports of the internal `ChatMessage` / `ToolCall` / `ToolDefinition` protocol types from `crate::protocol`

### Serving path (`arle serve` / `infer-server`)

```text
src/main.rs -> cli::run()
 -> cli/src/serve.rs (front-door arg parse + backend resolution)
 -> infer_api front door (feature-selected backend)
 infer_server::ServeHandle::spawn (engine thread)
 + infer_server::coordinator_local_router / coordinator_router (single HTTP facade for all backends)
 -> infer_core::Engine<E, K> (scheduler + radix prefix + overlap)
 -> infer_cuda::CudaExecutor (CUDA: paged KV, TileLang + native CUDA, TP/EP, DSv4-Flash)
 -> infer_metal::MetalExecutor (Metal: MLX packed varlen decode)
 -> crates/cuda-kernels kernels / TileLang / CUDA graph path
```

Key files:

- `crates/cli/src/serve.rs`: `arle serve` front-door — backend resolution +
 invocation (arg parse, port/bind, backend label)
- `crates/infer-server/src/lib.rs`: `ServeHandle::spawn` / `submit` / `collect`
 engine thread; `coordinator_local_router` (single-process: wraps ServeHandle in
 an in-process relay)
- `crates/infer-server/src/execution.rs`: per-request execution loop driving the
 engine
- `crates/infer-server/src/coordinator.rs` + `crates/infer-server/src/schema.rs`:
 the one OpenAI v1 HTTP facade (axum) for all backends — chat/completions, models,
 `/v1/stats`, `/metrics`; schema and request/response types
- `crates/infer-server/src/multiproc_relay.rs`: relay protocol (`RelayCoordinator`,
 `LocalChannel*` in-process variant, `WireStats`)
- `crates/infer-server/src/tokenizer.rs`: tokenizer wiring for the serve path
- `crates/infer-core/src/lib.rs`: device-neutral `Engine<E, K>` + `SchedulerConfig`
- `crates/infer-cuda/src/executor.rs`: CUDA `BackendExecutor` impl
- `crates/infer-metal/src/executor.rs`: Metal `BackendExecutor` impl
 (shipped, parity-verified)

> `infer-server` depends on `infer-core` + `infer-plan` + `infer-seam`; it does
> **not** depend on any backend crate. CUDA/Metal/HIP/Vulkan are wired one layer
> up at `infer-api`. The `cpu` smoke path still depends on `infer-metal` for the
> feature-free placeholder executor, but its KV pool is the shared
> `infer-seam::HostPagedKvPool`.

### Current OPD train path (post-OPD-pivot, 2026-05-24)

```text
crates/cli/src/train_cli.rs::run_opd()
 -> run_opd_from_dirs() or run_opd_smoke()
 -> train::qwen35_loader::load_qwen35_from_hf_dir()
 -> train::opd::opd_step()
 -> autograd Tape + AdamW + Qwen3.5 teacher/student weights
```

Scratch pretrain, SFT, GRPO, and multi-turn RL surfaces were retired
in commit `bd94c09` (see OPD-only pivot).
Their dispatch sources, supporting modules, and tests have been deleted
from `crates/train`; the empty legacy command namespace is also gone.
The autograd + Trainer + checkpoint codec + tokenizer + LoRA
remain as OPD substrate. The OPD-teacher
raw-logits + per-step student-LoRA re-merge surface is exposed at
`infer-api` (`RawLogits`, `StudentLora*`) under `--features cuda`.

Key files (surviving the pivot):

- `crates/cli/src/train_cli.rs`: `arle train env`, `estimate-memory`, and `opd` front door
- `crates/train/src/trainer.rs`: `Trainer<O, C, S>` skeleton — kept; OPD will provide its own `step_fn`
- `crates/train/src/{checkpoint,cli_args,grad_accum,grad_clip,loss,lora,tokenizer,causal_lm,qwen35,qwen35_checkpoint,model_family}.rs`: substrate kept for OPD

## 3. Runtime crate map (the `infer-*` graph)

The runtime is the device-neutral crate graph. Each crate owns one concern;
the dependency direction is strictly downward (IR → seam → core → backends →
server → front door), with `infer-core` carrying **no** backend dependency.

### 3.1 `infer-plan` — backend-neutral IR

- `crates/infer-plan/src/lib.rs`: the `ForwardPlan` / `ForwardMode` /
 `SamplingParams` / `SlotToken` / `StepOutput` data contract — every layer
 speaks this; no backend types.
- `crates/infer-plan/src/sample.rs`: pure host `sample_token`
 (temp/top-k/top-p/min-p, deterministic by `(seed, position)`).

### 3.2 `infer-seam` — host-only trait seam

- `crates/infer-seam/src/lib.rs`: `BackendExecutor` — the proven seam trait:
 `submit`/`poll`/`warmup` core plus opt-in capability default-methods
 (model stop ids, `max_rows_per_step`/`max_live_requests`, prefix-reuse
 hooks, page-tier and whole-slot tier demote/promote, OPD weight
 offload/reload). Also `ResourceGovernor` (+ `Permissive`/`Cooperative`
 impls) — **driven**: `Engine` holds a `Box<dyn ResourceGovernor>`
 (`infer-core/src/lib.rs`) and `infer-api/src/loaded.rs` wires the
 cooperative governor. The earlier
 `Communicator`/`Sampler`/`GraphRunner`/`ModelArch` hypothesis traits were
 deleted.
- `crates/infer-seam/src/kv.rs` + `kv_query.rs` + `allocator.rs` +
 `prefix_store.rs`: the three-way `KvPool = KvQuery + KvAllocator + KvPrefixStore`
 split (alloc/grow/truncate + prefix retain/release lookup) — the host-only KV
 contract both backend KV pools implement.
- `crates/infer-seam/src/kv_batch.rs`: `KvBatchDescriptor`/`KvBatchRow` — the
 host-only batch-addressable description backends lower from
 (the Phase 1 unified-batched plan's seam piece).
- `crates/infer-seam/src/{kv_dtype,resource,host_paged_kv_pool}.rs`:
 `KvCacheDtype`, `SlotBudget`/`HostTierBudget`/`split_host_tiers`, and the
 shared production host page allocator.

This is the old `infer/src/backend.rs` backend-trait surface, recast as a
host-only seam with zero device coupling.

### 3.3 `infer-core` — device-neutral Engine + scheduler

> Old home: `infer/src/scheduler/**`, `infer/src/prefix_cache.rs`,
> `infer/src/block_manager.rs`, the chunked-prefill/overlap logic, and the
> shared scheduler types/events.

- `crates/infer-core/src/lib.rs`: `Engine<E, K>` generic over the seam traits +
 `SchedulerConfig` — the continuous-batching coordinator + in-file tests.
- `crates/infer-core/src/planner.rs`: the hot scheduling axis (admission /
 chunking / batch assembly).
- `crates/infer-core/src/prefix.rs` + `crates/infer-core/src/radix.rs`: the
 RadixCache prefix-cache (the device-neutral replacement for legacy
 `infer/src/prefix_cache.rs`).

`infer-core` depends only on `infer-plan` + `infer-seam` — no backend crate.

### 3.4 `infer-cuda` — CUDA executor

> Old home: `infer/src/model/{qwen3,qwen35,deepseek}.rs`, `infer/src/ops/**`,
> `infer/src/scheduler/cuda/**`, `infer/src/speculative/cuda.rs`,
> `infer/src/tp.rs`, plus the CUDA kernels in `crates/cuda-kernels`.

- `crates/infer-cuda/src/executor.rs`: the `BackendExecutor` impl (CPU-testable
 placeholder without `cuda`, real cuda-kernels path with it).
- `crates/infer-cuda/src/ops.rs` + `crates/infer-cuda/src/attention.rs`: the two
 perf hotspots over `cuda-kernels` (`attention.rs` is the DSv4 MLA / FlashMLA /
 DSA path; Qwen3.5/3.6 attention lives in `qwen35_attention.rs`).
- `crates/infer-cuda/src/loader.rs`: safetensors weight loading.
- `crates/infer-cuda/src/qwen35.rs`: Qwen3.5/3.6 **hybrid** model
 (gated-delta linear attention + periodic full attention) with shape- and
 architecture-dependent MoE dispatch: grouped FP8 DeepGEMM on Hopper or CUTLASS
 on SM120 for eligible prefill shapes, with hand/BF16 fallbacks elsewhere.
- `crates/infer-cuda/src/dsv4.rs` + `crates/infer-cuda/src/hc.rs`: DSv4-Flash
 FP8 model (loader + structs + MLA KV arena) and DSv4 hyper-connections
 (`hc_mult > 1` wide-residual wrap), cuda-gated.
- `crates/infer-cuda/src/deepep.rs`: native DeepEP transport for DSv4 MoE,
 gated on `cuda` + `deepep`.
- `crates/infer-cuda/src/moe.rs` + `crates/infer-cuda/src/moe_config.rs`: CUDA
 MoE forward + config (uses `infer-moe` for the routing reference).
- `crates/infer-cuda/src/tp.rs` + `crates/infer-cuda/src/shard_slice.rs`:
 TP/EP shard-aware load + slicing (uses `infer-topo` for the placement math).
- `crates/infer-cuda/src/graph.rs`: CUDA graph capture/reuse primitives
 (`CudaGraphState`), driven by the Qwen3.5/3.6 and DSv4 executors.
- `crates/cuda-kernels/src/{paged_kv,tensor,kv_quant}.rs`
 + `crates/cuda-kernels/csrc/{attention,comm,elementwise,gemm,kv,moe,norm,recurrent,sampling}/`:
 the kernel layer `infer-cuda` calls into. `deepep_sidecar/` is a separate C++
 sidecar; the legacy Rust `ffi::misc` module still exists, but no `csrc/misc/`
 directory exists.

`infer-cuda` depends on `infer-plan` + `infer-seam` + `cuda-kernels` +
`infer-topo` + `infer-moe` + `qwen35-spec` (and optionally `qwen3-spec` /
`deepseek-spec` by feature); never `infer-core`.

### 3.5 `infer-metal` — Metal MLX executor

> Old home: `infer/src/backend/metal/**`.

- `crates/infer-metal/src/executor.rs`: the `BackendExecutor` impl (shipped,
 parity-verified) — the cleanest example of a thin seam impl.
- `crates/infer-metal/src/kv_pool.rs`: `MetalKvPool` compatibility alias to
 `infer-seam::HostPagedKvPool`; device-side MLX KV stays in the executor.
- `crates/infer-metal/src/qwen35.rs`: the stable C++ MLX bridge for the
 Qwen3.5/3.6 forward (split deferred until FFI churn justifies it).
- `crates/infer-metal/src/{mlx,loader,weights,config,model_source,wired_limit}.rs`:
 MLX glue, safetensors load, weight tables, config, model-source resolution,
 and the auto-wired-limit pin.

`infer-metal` depends on `infer-plan` + `infer-seam` (+ `mlx-sys` under the
`metal` feature); never `infer-core`.

### 3.6 `infer-server` — OpenAI v1 HTTP frontend

> Old home: `infer/src/http_server.rs` + `infer/src/http_server/openai_v1.rs`.

- `crates/infer-server/src/lib.rs` + `execution.rs`: the `ServeHandle` engine
 loop; `coordinator_local_router` entry for single-process backends (wraps
 ServeHandle in `RelayCoordinator::new_local()` + `serve_handle_relay_driver`).
- `crates/infer-server/src/coordinator.rs` + `schema.rs`: the **single** OpenAI v1
 HTTP facade used by all backends (single-process and multi-process alike) —
 `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/stats`, `/metrics`.
- `crates/infer-server/src/multiproc_relay.rs`: relay protocol shared by both paths
 (`RelayCoordinator`; TCP channels for multi-process, `LocalChannel*` in-process
 for single-process).
- `crates/infer-server/src/multimodal.rs`: image extract / preprocess helpers for
 vision backends (DeepSeek-OCR) routed via
 `LocalMultimodalTx` in-process channel to `run_on_executor`.
- `crates/infer-server/src/tokenizer.rs`: tokenizer wiring.

Metal is wired via the `metal` feature; non-Metal builds fall back to an
`EchoExecutor`. CUDA is wired one layer up at `infer-api`.

### 3.7 `infer-api` — the single public front door

> Old home: the public surface of `infer/src/server_engine.rs`
> (`InferenceEngine` trait + `LoadedInferenceEngine` dispatch + request/output
> types + the OPD-teacher raw-logits surface).

- `crates/infer-api/src/lib.rs`: re-exports the public contract — the
 `InferenceEngine` trait, `LoadedInferenceEngine` enum, request/output/stream/
 telemetry types, and (under `cuda`) `RawLogits` + `StudentLora*`.
- `crates/infer-api/src/loaded.rs`: `EngineLoadConfig` + `LoadedInferenceEngine`
 feature-gated backend dispatch (metal/cuda/cpu).
- `crates/infer-api/src/serve_engine.rs`: `ServeInferenceEngine` —
 tokenize → submit → collect → detokenize.
- `crates/infer-api/src/types.rs`: the public request/output/sampling/telemetry
 types + `RawLogits` (cuda-only).

`infer-api` depends on `infer-core` + `infer-server` + `infer-plan` +
`infer-seam`, and pulls `infer-cuda` / `infer-metal` (+ `cuda-kernels` +
`cudarc`) by feature. Backends plug in here. `cli` depends on `infer-api` (the
front door) + `infer-util`.

### 3.8 `infer-topo` / `infer-moe` / `infer-util` — pure leaves

- `crates/infer-topo/src/{lib,sharding,topology,error}.rs`: pure,
 CPU-verifiable TP/EP topology + sharding math (TP rank placement,
 `head_shard` / `column_shard` / `row_shard`, SGLang-style multi-axis rank
 groups). Ported from legacy `infer/src/tensor_parallel.rs` with all GPU/NCCL
 coupling dropped.
- `crates/infer-moe/src/{lib,route,config,error,tests}.rs`: pure,
 CPU-verifiable MoE routing/gating math — the reference the GPU kernel is
 verified against (`route`, `RoutingDecision`, `MoeConfig`; DSv4 vs Qwen3.6
 routing rules; `group_limited_mask` reference for grouped routing).
- `crates/infer-util/src/{lib,hf_hub,logging}.rs`: backend-agnostic
 HuggingFace model-id/path resolution + download (`hf_hub`, relocated from
 `infer/src/hf_hub.rs`) and stderr logger init (`logging`, from
 `infer/src/logging.rs`). A leaf crate so host-only commands (cli
 `doctor`/`download`) avoid dragging in a backend-gated engine crate.

### 3.9 `infer-hip` / `infer-vulkan` — AIPC backends (experimental)

The AIPC lane (#71/#76/#77) landed ahead of the Phase 3 ordering — ratification
pending.

- `crates/infer-hip/src/{executor,kv_pool,model,loader}.rs`:
 `HipDsv4Executor` + `HipKvPool` seam impls; DSv4-Flash GGUF 2-bit
 shim-portable lane (FlashMLA/official-DSA/DeepGEMM datacenter paths
 excluded by design). GGUF reader + CPU dequant + deepseek4 GGUF→config
 mapping live in the neutral `infer-gguf` leaf (extracted 2026-06-12,
 roadmap tranche R1 done).
- `crates/infer-vulkan/src/{executor,kv_pool,model_*.rs}`: seam-correct
 skeleton — `VulkanExecutor` + `VulkanKvPool` implement the seam; forward
 order pinned for Qwen3/3.5/3.6, DSv4; device execution pending
 the shader ABI. Re-exports `infer-gguf`'s GGUF host modules
 (`deepseek4`/`dequant`/`gguf`).

Both compile and test on any host (device features off ⇒ stub layers), so
they ride the normal CPU CI lanes.

## 4. Surrounding crate map

These crates sit around the runtime graph:

- `crates/agent`: agent session state, tool recovery, turn loop
- `crates/chat`: shared protocol parsing/formatting and OpenAI chat types
- `crates/cli`: CLI entry, arg parsing, REPL UX, `arle serve` front door, train front door
- `crates/tools`: builtin tools, sandbox/tool execution, shared tool hooks
- `crates/cuda-kernels`: CUDA kernel layer (extracted from the legacy `infer` crate in commit `a4e12f5`, 2026-04-15). Owns `csrc/{attention,comm,elementwise,gemm,kv,moe,norm,recurrent,sampling}/`, separate C++ `deepep_sidecar/`, `tools/tilelang/`, Rust FFI, `paged_kv`, `tensor`, `kv_quant`; legacy Rust `ffi::misc` remains, but no `csrc/misc/` exists
- `crates/mlx-sys`: MLX C++ bridge for the Metal backend, including vendored MLX qmv kernels used by Qwen3.5 GGUF affine/tiled quant decode
- `crates/deepep-sys`: DeepEP all-to-all transport bindings used by `infer-cuda`'s DSv4 MoE path
- `crates/hip-sys`: thin hand-declared HIP runtime FFI (no bindgen; every entry point stubs to `HIP_NOT_COMPILED` off-box)
- `crates/hip-kernels`: HIP kernel build + FFI layer for the AIPC DSv4 2-bit lane (llama.cpp-adapted IQ2_XXS/Q2_K mmvq corpus; hipcc-gated, layout helpers always compiled)
- `crates/vulkan-sys`: ash-backed Vulkan loader wrapper (stub off the `vulkan` feature, mirroring `hip-sys`)
- `crates/vulkan-kernels`: glslc-compiled shader corpus adapted from llama.cpp `vulkan-shaders` (typecheck-only without `glslc`)
- `crates/infer-gguf`: GGUF v2/v3 memmap reader + llama.cpp-port CPU dequantizers + per-arch GGUF→spec-config mappers (`deepseek4`); consumers: `infer-hip`, `infer-vulkan`
- `crates/kv-native-sys`: local persistence substrate for KV tier disk transport — `KvMmapStore` (file-backed sparse mmap page-slot store: memcpy writes, zero-copy `&[u8]` reads, slot allocator + free list). Unused: WAL, shm, mmap descriptors (kept for future shared-memory tier). Sharded block ops (`write_block_cache_sharded` / `read_block_into_sharded` / `remove_block_sharded`) — consumers: `infer-cuda` KV-tier hooks (`executor/dsv4/slot_tier.rs`; the store itself is `kv-native-sys::KvTierStore`) and `infer-metal`'s SSD tier (`kv_ssd.rs`).
- `crates/xgrammar-sys`: Rust wrapper over upstream mlc-ai/xgrammar matcher (grammar-constrained decode) — consumer: `infer-server/src/grammar.rs` (OpenAI `response_format` → `GrammarHook`). The `real` feature builds the C++ engine (requires `XGRAMMAR_SOURCE_DIR`); without it the crate exports stubs that reject at runtime.
- `crates/qwen3-spec`: Qwen3 config + tensor-parallel `Shard` enum (TP layout authority)
- `crates/qwen35-spec`: shared train↔infer Qwen3.5 config + canonical tensor-name contract + `Shard` annotations consumed by the sharded loader path
- `crates/deepseek-spec`: DeepSeek-V4-only spec — owns `DeepSeekV4Config`, V4 tensor-name builders, shard annotations, attention operator summaries (`DeepSeekV4AttentionLayerPlan` — consumed by `infer-hip` today; making it the single DSv4 forward-order authority is roadmap tranche R3), and MoE route helpers (`deepseek-spec/src/v4.rs`). CUDA V4 hybrid attention + MoE + MTP kernels live in `infer-cuda` (`dsv4.rs` / `hc.rs` / `moe.rs` / `deepep.rs`). DS4 is the **#1 next-model priority**
- `crates/autograd`: from-scratch autograd + optimizer + lr-schedule + AdamW codec (OPD substrate)
- `crates/train`: train-side control plane + OPD stack (post-2026-05-18 pivot)

Current dependency direction (runtime graph):

```text
workspace root package (arle / arle bin)
 -> cli
 -> infer-api (the single front door)
 -> infer-util
 -> agent
 -> chat
 -> tools
 -> train

infer-api
 -> infer-core
 -> infer-server
 -> infer-plan, infer-seam
 -> infer-cuda (feature = "cuda")
 -> infer-metal (feature = "metal"; also pulled feature-free for cpu executor)

infer-server
 -> infer-core, infer-plan, infer-seam (no backend crates)

infer-core
 -> infer-plan, infer-seam (no backend dependency — device-neutral)

infer-cuda
 -> infer-plan, infer-seam, cuda-kernels, infer-topo, infer-moe,
 qwen35-spec (+ qwen3-spec / deepseek-spec by feature) (never infer-core)

infer-metal
 -> infer-plan, infer-seam (+ mlx-sys under "metal") (never infer-core)

infer-gguf
 -> deepseek-spec (neutral GGUF host substrate leaf; spec crates never depend back)

infer-hip
 -> infer-plan, infer-seam, deepseek-spec, infer-gguf
 (+ hip-sys, hip-kernels under "hip") (never infer-core)

infer-vulkan
 -> infer-plan, infer-seam, deepseek-spec, qwen3-spec, qwen35-spec,
 infer-gguf
 (+ vulkan-sys, vulkan-kernels under "vulkan") (never infer-core)
```

`infer-api` pulls `infer-hip` / `infer-vulkan` behind the optional `hip` /
`vulkan` features, mirroring `cuda`/`metal`.

## 5. Tests and validation map

### Where tests live

Each runtime crate carries its hot-path unit tests in-file (`#[cfg(test)]`).
Integration / adapter tests:

- `tests/cli_smoke.rs`, `tests/cli_agent_live.rs`, `tests/cli_test_support.rs`:
 root-package (`arle`) CLI smoke + live agent paths
- `crates/autograd/tests/`, `crates/train/tests/`: training-stack tests
- `infer-moe` carries its reference-routing tests in `crates/infer-moe/src/tests.rs`

CI (`.github/workflows/ci.yml`) builds and tests `infer-api`, `cli`, the
`arle` smoke path, and the support crates per backend (metal/cpu).

### Bench and helper entrypoints

- `scripts/bench_throughput.py`: canonical streaming throughput / latency runner
- `scripts/bench_agent_trace.py`: agent-style trace replay
- additional `scripts/bench_*.{sh,py}` for per-workload A/B and trace replay

## 6. Where to start reading

- Data contract first: `crates/infer-plan/src/lib.rs` (`ForwardPlan` /
 `ForwardMode`) — every layer speaks this.
- The seam: `crates/infer-seam/src/lib.rs` (`BackendExecutor`) +
 `crates/infer-seam/src/kv.rs` (the `KvPool` split).
- The scheduler: `crates/infer-core/src/lib.rs` (`Engine<E, K>`), then
 `crates/infer-core/src/planner.rs`.
- A real backend end-to-end: `crates/infer-metal/src/executor.rs` (shipped,
 parity-verified) — the cleanest thin seam impl; for CUDA see
 `crates/infer-cuda/src/executor.rs` + `executor/qwen35.rs`.
- The front door: `crates/infer-api/src/lib.rs` (`InferenceEngine` /
 `LoadedInferenceEngine`) → `crates/infer-api/src/serve_engine.rs`.
- Serving: `crates/infer-server/src/lib.rs` (`ServeHandle::spawn` / `submit` /
 `collect`) + `coordinator.rs` (HTTP facade).
- Agent CLI path: `src/main.rs` → `crates/cli/src/lib.rs` →
 `crates/infer-api/src/lib.rs` → `crates/agent/src/lib.rs`.
- Model discovery: `crates/infer-util/src/hf_hub.rs` (`resolve_model_source`)
 → `crates/infer-api/src/loaded.rs` (`LoadedInferenceEngine::load`).
