# agent-infer Codebase Map

Updated 2026-04-13. This document describes the repository as it exists today.
It is a navigation map, not a future architecture proposal.

## 1. Workspace at a glance

The repository currently has four layers:

- `agent-infer` (root package): thin binary wrapper in `src/main.rs` that calls `infer_cli::run()`.
- `infer/`: the runtime and backend-heavy crate. This is still where most inference complexity lives.
- `crates/`: extracted control-plane and shared-boundary crates around the runtime.
- `docs/` and `infer/docs/`: governance, project writeups, performance notes, and runtime-focused implementation notes.

The workspace members are:

- `agent-infer`
- `infer`
- `infer/mlx-sys`
- `crates/infer-agent`
- `crates/infer-chat`
- `crates/infer-cli`
- `crates/infer-core`
- `crates/infer-engine`
- `crates/infer-observability`
- `crates/infer-policy`
- `crates/infer-tools`

## 2. Main execution paths

### Agent CLI path

The interactive agent path is:

```text
src/main.rs
  -> infer_cli::run()
  -> infer_engine::resolve_model_source() + LoadedAgentEngine::load()
  -> infer_agent::AgentSession
  -> infer_tools builtin tools + infer_chat protocol
  -> runtime backend selected by infer-engine
```

Key files:

- `src/main.rs`: root binary entrypoint
- `crates/infer-cli/src/lib.rs`: CLI startup and backend selection
- `crates/infer-cli/src/repl.rs`: REPL loop, slash commands, terminal UX
- `crates/infer-engine/src/lib.rs`: runtime-facing façade for agent use
- `crates/infer-agent/src/lib.rs`: session state, prompt assembly, turn loop
- `crates/infer-tools/src/lib.rs`: builtin tool definitions, execution, tool policy hooks
- `crates/infer-chat/src/lib.rs`: protocol parsing/formatting for chat + tool calls

### CUDA serving path

The CUDA/OpenAI-compatible serving path is:

```text
infer/src/main.rs
  -> bootstrap.rs
  -> http_server.rs
  -> server_engine.rs
  -> scheduler/cuda/*
  -> model.rs + model/*
  -> ops.rs + ops/*
  -> infer/csrc kernels / FlashInfer / CUDA graph path
```

Key files:

- `infer/src/main.rs`: CUDA server binary
- `infer/src/bootstrap.rs`: model loading, runtime config, scheduler bring-up
- `infer/src/http_server.rs` and `infer/src/http_server/openai_v1.rs`: HTTP API
- `infer/src/server_engine.rs`: synchronous/streaming server-engine façade and generation loop
- `infer/src/scheduler/cuda/`: production CUDA scheduler implementation

### Serial backend runtime path

The non-CUDA runtime path uses a serial request loop:

```text
cpu_serve / metal_serve
  -> backend_runtime.rs
  -> CpuBackend or MetalBackend
  -> request streaming through StopChunkProcessor
```

Key files:

- `infer/src/backend_runtime.rs`: serial runtime handle for backends without the CUDA scheduler
- `infer/src/cpu_backend.rs`: development-oriented CPU backend
- `infer/src/metal_backend.rs`: Apple Silicon backend via `mlx-sys`
- `infer/src/bin/cpu_serve.rs`
- `infer/src/bin/metal_serve.rs`

## 3. `infer/` crate map

### 3.1 Runtime entry, serving, and wiring

These files sit near the top of the runtime stack:

- `infer/src/bootstrap.rs`: builds CUDA engines, loads tokenizer/model, and applies runtime config
- `infer/src/server_engine.rs`: shared request contract (`CompleteRequest`, `CompleteOutput`, streaming deltas) and CUDA generation loop
- `infer/src/backend_runtime.rs`: serial backend runtime for CPU/Metal request handling
- `infer/src/http_server.rs`: axum wiring for serving
- `infer/src/request_handle.rs`: generic request submission interface
- `infer/src/logging.rs`: default logging init
- `infer/src/metrics.rs`: metrics export surface
- `infer/src/hf_hub.rs`: local model discovery / HuggingFace integration
- `infer/src/model_registry.rs`: model architecture detection

### 3.2 Scheduling and lifecycle control

Scheduling is split into three pieces today:

- `infer/src/scheduler/batch.rs`: pure CPU accounting scheduler with block-level KV accounting, policy hooks, and shared lifecycle events
- `infer/src/scheduler/types.rs`: request types, handle, config, queue admission
- `infer/src/scheduler/cuda/`: real CUDA serving scheduler

The CUDA scheduler is internally split by behavior:

- `core.rs`: scheduler struct and initialization
- `runtime.rs`: run loop, slot assignment, cleanup
- `execution.rs`: per-step orchestration
- `prefill.rs`: prefix reuse and chunked prefill
- `decode.rs`: batched decode and preemption behavior
- `request.rs`: active request state and token streaming helpers

Metal has a separate planning/accounting scheduler in:

- `infer/src/metal_scheduler.rs`

This Metal scheduler is currently useful as a policy/accounting layer and test target; the actual Metal serving path still goes through `backend_runtime.rs`.

### 3.3 Memory, KV, caching, and batching support

The runtime’s memory/caching pieces live here:

- `infer/src/block_manager.rs`: GPU/CPU KV block accounting for the batch scheduler
- `infer/src/paged_kv.rs`: token-level KV pool for CUDA paged attention
- `infer/src/prefix_cache.rs`: radix-tree prefix cache for CUDA/runtime reuse
- `infer/src/memory_planner.rs`: memory planning helpers
- `infer/src/cuda_graph_pool.rs`: CUDA graph capture/reuse support
- `infer/src/flashinfer_metadata.rs`: paged-KV metadata staging for FlashInfer

Metal-specific cache/state helpers are separate:

- `infer/src/metal_kv_pool.rs`
- `infer/src/metal_prefix_cache.rs`
- `infer/src/metal_gdr.rs`

### 3.4 Models, kernels, and numerics

Model integration starts at:

- `infer/src/model.rs`: `ModelForward`, `GenerationState`, decode-context abstractions

Per-model code lives in:

- `infer/src/model/qwen3.rs`
- `infer/src/model/qwen35.rs`
- `infer/src/model/glm4.rs`
- plus supporting files under `infer/src/model/`

The operator layer lives in:

- `infer/src/ops.rs`
- `infer/src/ops/attention.rs`
- `infer/src/ops/embedding.rs`
- `infer/src/ops/linear.rs`
- `infer/src/ops/norm.rs`
- `infer/src/ops/recurrent.rs`
- `infer/src/ops/sampling.rs`
- `infer/src/ops/kv_ops.rs`
- `infer/src/ops/kv_quant.rs`
- `infer/src/ops/kv_turboquant.rs`

Lower-level runtime support:

- `infer/src/tensor.rs`: CUDA tensor/device abstractions
- `infer/src/weight_loader.rs`: weight loading
- `infer/src/gguf.rs`: GGUF parsing
- `infer/src/quant.rs`: quantization metadata + dispatch
- `infer/src/speculative.rs`: speculative decoding experiments
- `infer/src/tensor_parallel.rs`: tensor-parallel scaffolding
- `infer/src/tokenizer.rs`: tokenizer wrapper

### 3.5 Backends and binaries

Backend interfaces:

- `infer/src/backend.rs`: backend traits and shared generate result types
- `infer/src/cpu_backend.rs`
- `infer/src/metal_backend.rs`

Runtime/benchmark binaries:

- `infer/src/bin/bench_serving.rs`
- `infer/src/bin/cpu_serve.rs`
- `infer/src/bin/metal_serve.rs`
- `infer/src/bin/metal_bench.rs`
- `infer/src/bin/metal_request.rs`
- `infer/src/bin/triton_silu_smoke.rs`

### 3.6 Tracing and diagnostics

- `infer/src/trace_reporter.rs`: file-based fastrace reporter
- `infer/src/error.rs`: runtime error helpers
- `infer/src/ffi.rs`: CUDA-only FFI bindings

## 4. Extracted crate map

### Root binary

- `agent-infer`: just `src/main.rs`, no REPL or backend logic

### Control-plane crates

- `crates/infer-cli`: CLI entry, arg parsing, REPL UX
- `crates/infer-agent`: agent session state, turn loop, tool-call recovery, persistence
- `crates/infer-tools`: builtin tools, sandbox/tool execution, shared tool policy hooks
- `crates/infer-chat`: protocol structures and parsing/formatting for tool-aware chat

### Runtime-facing control boundary

- `crates/infer-engine`: model source resolution, logging init, runtime façade used by agent mode

### Shared small crates

- `crates/infer-core`: small shared domain types such as inference modes and request events
- `crates/infer-policy`: admission/chunking policy traits and defaults
- `crates/infer-observability`: event schema + sink trait

Current dependency direction is:

```text
agent-infer
  -> infer-cli
     -> infer-engine
     -> infer-agent
     -> infer-tools
     -> infer-chat

infer-engine
  -> infer

infer-agent / infer-tools / infer-engine
  -> infer-chat / infer-core / infer-policy / infer-observability (as needed)
```

## 5. Tests and validation map

### Rust integration tests in `infer/tests/`

These cover runtime correctness, parity, and regression behavior:

- scheduler/runtime: `e2e.rs`, `e2e_qwen35.rs`, `greedy_consistency.rs`
- GGUF/quantization/kernel regressions: `q4k_kernel_correctness.rs`, `ground_truth_q4k.rs`, `smoke_*`
- golden/test-data tooling: `regen_test_data.rs`, `gen_test_data_35.rs`

### Python tests in `tests/`

These are lightweight black-box or algorithm-level checks:

- `test_openai_api.py`
- `test_kv_cache.py`
- `test_radix_tree.py`
- `test_sampling.py`
- `test_scheduler.py`
- `test_speculative.py`

### Agent CLI live tests

- `tests/cli_agent_live.rs`

These validate CLI agent behavior when a local model can be auto-detected. They are intentionally ignored unless the environment has the required model.

## 6. Documentation map

The repository has two documentation tiers:

### `docs/`

Repository-wide and governance-facing material:

- `docs/architecture.md`: workspace/package topology
- `docs/stability-policy.md`, `docs/support-matrix.md`, `docs/compatibility.md`: surface/governance policy
- `docs/perf-and-correctness-gates.md`, `docs/release-checklist.md`: merge/release expectations
- `docs/plans/`: planned feature work
- `docs/projects/`: project writeups and architecture notes
- `docs/reviews/` and `docs/experience/`: reviews, wins, error writeups

### `infer/docs/`

Runtime- and model-focused implementation notes:

- `infer/docs/projects/`: runtime architecture, model-forward, optimization, accuracy investigations
- `infer/docs/resources/`: profiling, onboarding, parity/optimization playbooks

## 7. Where to start for common tasks

- Agent/CLI behavior: start at `crates/infer-cli/src/lib.rs`, `crates/infer-cli/src/repl.rs`, `crates/infer-agent/src/lib.rs`
- Tool behavior: start at `crates/infer-tools/src/lib.rs`
- Backend loading / model discovery: start at `crates/infer-engine/src/lib.rs`, then `infer/src/hf_hub.rs` and `infer/src/bootstrap.rs`
- CUDA serving scheduler: start at `infer/src/scheduler/mod.rs`, then `infer/src/scheduler/cuda/`
- Metal serving path: start at `infer/src/metal_backend.rs`, `infer/src/backend_runtime.rs`, and `infer/src/metal_scheduler.rs`
- GGUF / quantization / kernels: start at `infer/src/gguf.rs`, `infer/src/quant.rs`, `infer/src/ops/`, and `infer/csrc/`
- HTTP API behavior: start at `infer/src/http_server.rs` and `infer/src/server_engine.rs`

## 8. Current reality check

Two things are true at once:

- The control-plane split is real: the root package is thin, and CLI/agent/tool/protocol code no longer lives in `infer`.
- The runtime is still mostly concentrated in `infer/`: scheduler internals, CUDA kernels, model integration, backend runtime, and serving logic remain there.

That is the current codebase shape. Future extraction work should be judged against this map, not against a hypothetical end-state tree.
