# agent-infer Codebase Map

Updated 2026-04-15. This document describes the repository as it exists after
the Route-A refactor folded the partial runtime split back into `infer`.

## 1. Workspace at a glance

The repository has four practical layers:

- `agent-infer` (root package): thin binary wrapper in `src/main.rs` that calls
  `infer_cli::run()`.
- `infer/`: the runtime-heavy crate. It owns the HTTP server, scheduler,
  backends, model/runtime modules, and the unified
  `server_engine::InferenceEngine` contract used by the HTTP server and agent
  CLI alike.
- `crates/`: reusable control-plane/helper crates around the runtime.
- `docs/` and `infer/docs/`: architecture, plans, research, and implementation
  notes.

Current workspace members:

- `agent-infer`
- `infer`
- `crates/mlx-sys`
- `crates/infer-agent`
- `crates/infer-chat`
- `crates/infer-cli`
- `crates/infer-tools`

## 2. Main execution paths

### Agent CLI path

```text
src/main.rs
  -> infer_cli::run()
  -> infer::hf_hub::resolve_model_source() + infer::server_engine::LoadedInferenceEngine::load()
  -> infer_agent::AgentSession (uses `dyn InferenceEngine`)
  -> infer_tools builtin tools + infer_chat protocol
  -> LoadedInferenceEngine dispatches to CUDA / Metal / CPU backend
```

Key files:

- `src/main.rs`: root binary entrypoint
- `crates/infer-cli/src/lib.rs`: CLI startup and backend selection
- `crates/infer-cli/src/repl.rs`: REPL loop, slash commands, terminal UX
- `infer/src/server_engine.rs`: unified `InferenceEngine` trait, `CompletionRequest`/`CompletionOutput`/`TokenUsage`/`CompletionStreamDelta` types, and `LoadedInferenceEngine` backend dispatch enum
- `infer/src/hf_hub.rs`: local model discovery + `resolve_model_source`
- `crates/infer-agent/src/lib.rs`: session state, prompt assembly, turn loop
- `crates/infer-tools/src/lib.rs`: builtin tools and shared tool hooks
- `crates/infer-chat/src/lib.rs`: `OpenAiChatMessage` / `OpenAiToolDefinition` wire format + re-exports of the internal `ChatMessage` / `ToolCall` / `ToolDefinition` protocol types from `crate::protocol`

### CUDA serving path

```text
infer/src/main.rs
  -> backend/cuda/bootstrap.rs
  -> http_server.rs
  -> server_engine.rs
  -> scheduler/cuda/*
  -> model.rs + model/*
  -> ops.rs + ops/*
  -> crates/infer-cuda-kernels kernels / FlashInfer / CUDA graph path
```

Key files:

- `infer/src/main.rs`: CUDA server binary
- `infer/src/backend/cuda/bootstrap.rs`: model loading, runtime config, scheduler bring-up
- `infer/src/http_server.rs` and `infer/src/http_server/openai_v1.rs`: HTTP API
- `infer/src/server_engine.rs`: synchronous/streaming generation façade
- `infer/src/scheduler/cuda/`: production CUDA scheduler implementation

### Serial backend runtime path

```text
cpu_serve / metal_serve
  -> backend/runtime.rs
  -> CpuBackend or MetalBackend
  -> request streaming through StopChunkProcessor
```

Key files:

- `infer/src/backend/runtime.rs`: serial runtime handle for non-CUDA backends
- `infer/src/backend/cpu.rs`: development CPU backend
- `infer/src/backend/metal.rs`: Apple Silicon backend via `mlx-sys`
- `infer/src/bin/cpu_serve.rs`
- `infer/src/bin/metal_serve.rs`

## 3. `infer/` crate map

### Runtime entry, serving, and wiring

- `infer/src/server_engine.rs`: unified `InferenceEngine` trait, `CompletionRequest`/`CompletionOutput`/`TokenUsage`/`CompletionStreamDelta` types, CUDA generation loop, and the `LoadedInferenceEngine` enum that dispatches to Qwen3/Qwen35/GLM4 (CUDA), `BackendInferenceEngine<MetalBackend>` (Metal), or `BackendInferenceEngine<CpuBackend>` (CPU)
- `infer/src/backend/cuda/bootstrap.rs`: builds CUDA engines and schedulers
- `infer/src/backend/runtime.rs`: serial backend runtime for CPU/Metal
- `infer/src/http_server.rs`: axum wiring for serving
- `infer/src/request_handle.rs`: generic request submission interface
- `infer/src/logging.rs`: default logging init
- `infer/src/metrics.rs`: metrics export surface
- `infer/src/hf_hub.rs`: local model discovery / HuggingFace integration
- `infer/src/model_registry.rs`: model architecture detection

### Scheduling and lifecycle control

- `infer/src/scheduler/batch.rs`: pure CPU accounting scheduler with lifecycle events
- `infer/src/scheduler/types.rs`: request types, handles, config, queue admission
- `infer/src/scheduler/policy.rs`: admission/chunking/eviction policy traits and defaults
- `infer/src/scheduler/cuda/`: production CUDA scheduler
- `infer/src/backend/metal/scheduler.rs`: Metal scheduling/accounting layer

### Shared runtime contracts that Route A folded back in

- `infer/src/types.rs`: request/session identifiers and shared scheduler enums
- `infer/src/events.rs`: engine event schema and sink trait
- `infer/src/scheduler/policy.rs`: admission/chunking/eviction policy traits
- `infer/src/server_engine.rs`: unified `InferenceEngine` trait — the old
  `agent_engine.rs` duplicate facade was deleted and its responsibilities
  collapsed into `server_engine.rs` so HTTP and agent CLI share one contract

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-observability`, `infer-policy`, and `infer-engine` crates back into
these in-tree modules because the split never achieved real independence.
A follow-up pass in the same day also deleted `infer/src/agent_engine.rs`
after confirming every `Agent*` type exactly duplicated a corresponding
`Completion*` / `InferenceEngine` / `LoadedInferenceEngine` type in
`server_engine.rs`.

### Memory, KV, caching, and batching support

- `infer/src/block_manager.rs`: KV block accounting for the batch scheduler
- `infer/src/backend/cuda/paged_kv.rs`: token-level KV pool for CUDA paged attention
- `infer/src/prefix_cache.rs`: radix-tree prefix cache for CUDA/runtime reuse
- `infer/src/memory_planner.rs`: memory planning helpers
- `infer/src/backend/cuda/graph_pool.rs`: CUDA graph capture/reuse support
- `infer/src/backend/cuda/flashinfer.rs`: paged-KV metadata staging for FlashInfer
- `infer/src/backend/metal/kv_pool.rs`
- `infer/src/backend/metal/prefix_cache.rs`
- `infer/src/backend/metal/gdr.rs`

### Models, kernels, and numerics

- `infer/src/model.rs`: `ModelForward`, `GenerationState`, decode-context abstractions
- `infer/src/model/qwen3.rs`
- `infer/src/model/qwen35.rs`
- `infer/src/model/glm4.rs`
- supporting files under `infer/src/model/`
- `infer/src/ops.rs` and `infer/src/ops/*`
- `infer/src/backend/cuda/tensor.rs`: CUDA tensor/device abstractions
- `infer/src/weight_loader.rs`: weight loading
- `infer/src/gguf.rs`: GGUF parsing
- `infer/src/quant.rs`: quantization metadata + dispatch
- `infer/src/speculative.rs`: speculative decoding experiments
- `infer/src/tensor_parallel.rs`: tensor-parallel scaffolding
- `infer/src/tokenizer.rs`: tokenizer wrapper

### Backends and binaries

- `infer/src/backend.rs`: backend traits and shared generate result types
- `infer/src/backend/cpu.rs`
- `infer/src/backend/metal.rs`
- runtime/benchmark binaries in `infer/src/bin/`

## 4. Extracted crate map

These crates remain independent after Route A:

- `crates/infer-agent`: agent session state, tool recovery, turn loop
- `crates/infer-chat`: shared protocol parsing/formatting and OpenAI chat types
- `crates/infer-cli`: CLI entry, arg parsing, REPL UX
- `crates/infer-tools`: builtin tools, sandbox/tool execution, shared tool hooks
- `crates/mlx-sys`: MLX C++ bridge for the Metal backend

Current dependency direction:

```text
agent-infer
  -> infer-cli
     -> infer
     -> infer-agent
     -> infer-chat
     -> infer-tools

infer-agent
  -> infer
  -> infer-chat

infer
  -> infer-chat
  -> mlx-sys (feature = "metal")
```

## 5. Tests and validation map

### Rust integration tests in `infer/tests/`

- scheduler/runtime: `e2e.rs`, `e2e_qwen35.rs`, `greedy_consistency.rs`
- GGUF/quantization/kernel regressions: `q4k_kernel_correctness.rs`, `ground_truth_q4k.rs`, `smoke_*`
- golden/test-data tooling: `regen_test_data.rs`, `gen_test_data_35.rs`

### Bench and helper entrypoints

- `scripts/bench_throughput_sweep.py`: standard throughput sweep
- `scripts/bench_agent_trace.py`: agent-style trace replay
- `infer/src/bin/metal_bench.rs`: Metal micro/macro benchmark entrypoint

## 6. Where to start reading

- Backend loading / model discovery: start at `infer/src/hf_hub.rs` for
  `resolve_model_source`, then `infer/src/server_engine.rs` for
  `LoadedInferenceEngine::load` and the `InferenceEngine` trait, then
  `infer/src/backend/cuda/bootstrap.rs` for the CUDA bring-up
- CUDA serving path: `infer/src/main.rs` → `infer/src/http_server.rs` →
  `infer/src/scheduler/cuda/`
- Agent CLI path: `src/main.rs` → `crates/infer-cli/src/lib.rs` →
  `infer/src/server_engine.rs` → `crates/infer-agent/src/lib.rs`
