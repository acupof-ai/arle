# agent-infer Codebase Map

Updated 2026-04-21 (post `cuda-kernels` extraction, tiered KV
M3a/M3b/M3c local, the current Phase 6 `crates/autograd` + `crates/train`
landing, and the train-side control-plane / observability refresh).
Supplemented 2026-04-21 with the canonical `pretrain` entrypoint and
the latest Phase 6 runtime truth.
(Earlier 2026-04-20 supplement introduced Phase 6 `crates/autograd` + `crates/train`
(from-scratch autograd + LoRA/GRPO trainer; see [`docs/plans/rust-agent-rl-single-node.md`](plans/rust-agent-rl-single-node.md))
and the canonical guidellm bench SSOT alignment.)
Current train control-plane truth: `crates/train` owns the active
training server, surfaced by `pretrain --serve`, `train_sft --serve`,
`train_grpo --serve`, and `train_multi_turn --serve`; the current
surface is `/v1/train/status|events|stop|save`. The
`infer/src/http_server/train.rs` `/v1/train/*` surface is the target
unified entrypoint, not the current implementation.
Current train-side model reality is a generic Qwen-family control plane
with Qwen3.5 as the optimized default: `train_sft` and `train_grpo`
dispatch across Qwen3 / Qwen3.5 families, `train_multi_turn` already
runs on the dense/full-attn Qwen3.5 path and now exposes a stepwise-GRPO
vs sequence-level-GSPO objective switch, `eval_lm` reads the same
checkpoint dirs for tokenized or chat JSONL evaluation, the canonical
scratch-pretrain entrypoint is `pretrain`, checkpoints are written as
HF-style directories, the handwritten Transformer/TinyLM runtime
compatibility path has been deleted, and the hybrid linear-attn Qwen3.5
LoRA/eval path is now landed while scratch pretrain + RL acceptance
remain dense/full-attn only.
This document describes the repository as it exists after the Route-A
refactor folded the partial runtime split back into `infer`, and after
the CUDA kernel layer was extracted into `crates/cuda-kernels/`
(commit `a4e12f5`).

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
- `crates/cuda-kernels`
- `crates/mlx-sys`
- `crates/agent`
- `crates/chat`
- `crates/cli`
- `crates/tools`
- `crates/qwen3-spec` (shared Qwen3 config + canonical tensor-name contract)
- `crates/qwen35-spec` (shared Qwen3.5 config + canonical tensor-name contract)
- `crates/autograd` (Phase 6 — from-scratch autograd with `Backend` trait; the current local Metal train path already uses the device-resident / lazy-eval tranche for the active training-critical ops, while CUDA remains the primary full-acceptance target)
- `crates/train` (Phase 6 — generic Qwen-family pretrain/SFT/GRPO trainer, train-side server exposed by the active train binaries' `--serve` flag; current optimized path is Qwen3.5-family with dense/full-attn scratch pretrain + RL, plus hybrid linear-attn support on the LoRA/eval path; HF-style checkpoint dirs and shared async observability, bounded backpressure + `dropped_metrics` status reporting, MLflow export, OTLP log export, and optional W&B sidecar export; depends on `autograd`)
- `crates/kv-native-sys` (Zig-backed local persistence substrate for `infer/src/kv_tier/transport/disk.rs`; now owns file + block object ABI plus mmap/WAL/shm descriptor primitives used by local validation and by coordinator spill/rehydrate persistence paths)

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
- `crates/cli/src/lib.rs`: CLI startup and backend selection
- `crates/cli/src/repl.rs`: REPL loop, slash commands, terminal UX
- `infer/src/server_engine.rs`: unified `InferenceEngine` trait, `CompletionRequest`/`CompletionOutput`/`TokenUsage`/`CompletionStreamDelta` types, and `LoadedInferenceEngine` backend dispatch enum
- `infer/src/hf_hub.rs`: local model discovery + `resolve_model_source`
- `crates/agent/src/lib.rs`: session state, prompt assembly, turn loop
- `crates/tools/src/lib.rs`: builtin tools and shared tool hooks
- `crates/chat/src/lib.rs`: `OpenAiChatMessage` / `OpenAiToolDefinition` wire format + re-exports of the internal `ChatMessage` / `ToolCall` / `ToolDefinition` protocol types from `crate::protocol`

### CUDA serving path

```text
infer/src/main.rs
  -> backend/cuda/bootstrap.rs
  -> http_server.rs
  -> server_engine.rs
  -> scheduler/cuda/*
  -> model.rs + model/*
  -> ops.rs + ops/*
  -> crates/cuda-kernels kernels / FlashInfer / CUDA graph path
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

### Current train control-plane path

```text
crates/train/src/bin/{pretrain,train_sft,train_grpo,train_multi_turn}.rs
  -> train::server::bind_and_serve_on_thread()
  -> std TcpListener control plane on /v1/train/{status,events,stop,save}
  -> train::control::TrainingController + ControllerSink
  -> SharedSink background worker
  -> local JSONL/stdout + optional MLflow / OTLP / W&B export
  -> autograd + train runtime loop
```

This is the **current implementation truth** for train-side control.
Docs that describe infer-side `/v1/train/*` routes are target architecture,
not the current repository surface.

Key files:

- `crates/train/src/bin/eval_lm.rs`: standalone eval entrypoint for checkpoint dirs written by current train code
- `crates/train/src/bin/pretrain.rs`: canonical `pretrain` entrypoint; `--serve` starts the train-side control plane for scratch pretraining
- `crates/train/src/bin/train_sft.rs`: supervised fine-tune entrypoint; `--serve` starts the same control plane
- `crates/train/src/bin/train_grpo.rs`: single-turn GRPO entrypoint; `--serve` starts the same control plane
- `crates/train/src/bin/train_multi_turn.rs`: current multi-turn entrypoint on the Qwen3.5-family dense/full-attn path; `--serve` starts the same control plane
- `crates/train/src/server.rs`: minimal HTTP control plane for `/v1/train/status|events|stop|save`
- `crates/train/src/control.rs`: shared controller / status state plus recent event ring buffer used by the server thread and trainer loop
- `crates/train/src/metrics.rs`: shared async observability sink, lifecycle/artifact events, bounded-queue backpressure accounting, and MLflow / OTLP / W&B export adapters

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
- `crates/cuda-kernels/src/paged_kv.rs`: token-level KV pool for CUDA paged attention (page-aware, BF16 `page_size=16`)
- `infer/src/prefix_cache.rs`: radix-tree prefix cache for CUDA/runtime reuse; tier-aware `RadixNode` metadata (`hit_count`, `tier_location`, `session_id`, `fingerprint`, `soft_pin_until`, `byte_len`) + `lookup_or_stage` contract
- `infer/src/kv_tier.rs` + `infer/src/kv_tier/{lookup,coordinator,host_pool,transport,tier,id}.rs`: tiered KV cache module (T0 GPU → T1 host pinned → T2 NVMe → T3 NIXL); M3a host-tier skeleton + M3b `lookup_or_stage` contract + page-lifecycle state machine landed locally 2026-04-15
- `infer/src/memory_planner.rs`: memory planning helpers
- `crates/cuda-kernels/src/graph_pool.rs`: CUDA graph capture/reuse support
- `crates/cuda-kernels/src/flashinfer.rs`: paged-KV metadata staging for FlashInfer
- `infer/src/backend/metal/kv_pool.rs`
- `infer/src/backend/metal/prefix_cache.rs`
- `infer/src/backend/metal/gdr.rs`
- `infer/src/backend/metal/request_state.rs`: resumable Metal request state layer for Qwen3 / Qwen3.5 (prefill in chunks, one-step decode, deterministic cleanup); M0.2a landed locally 2026-04-15

### Models, kernels, and numerics

- `infer/src/model.rs`: `ModelForward`, `GenerationState`, decode-context abstractions
- `infer/src/model/qwen3.rs`
- `infer/src/model/qwen35.rs`
- `infer/src/model/glm4.rs`
- supporting files under `infer/src/model/`
- `infer/src/ops.rs` and `infer/src/ops/*`
- `crates/cuda-kernels/src/tensor.rs`: CUDA tensor/device abstractions (`DeviceContext`, `DeviceVec`, `DeviceMatrix`, `HiddenStates`, `RawDevicePtr`)
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

- `crates/agent`: agent session state, tool recovery, turn loop
- `crates/chat`: shared protocol parsing/formatting and OpenAI chat types
- `crates/cli`: CLI entry, arg parsing, REPL UX
- `crates/tools`: builtin tools, sandbox/tool execution, shared tool hooks
- `crates/cuda-kernels`: CUDA kernel layer extracted from `infer` in commit `a4e12f5` (2026-04-15). Owns `csrc/{attention,gemm,kv,quant,misc}/`, `tools/triton/`, Rust FFI, `paged_kv`, `flashinfer`, `graph_pool`, `tensor`, `kv_quant`, `kv_turboquant`
- `crates/mlx-sys`: MLX C++ bridge for the Metal backend
- `crates/kv-native-sys`: Zig-native persistence layer used by `infer/src/kv_tier/transport/disk.rs` for local file and content-addressed block object operations; also exports substrate APIs for WAL append/replay, mmap descriptors, and shared-memory descriptors

Current dependency direction:

```text
agent-infer
  -> cli
     -> infer
     -> agent
     -> chat
     -> tools

agent
  -> infer
  -> chat

infer
  -> chat
  -> cuda-kernels  (one-way; never the reverse)
  -> mlx-sys (feature = "metal")
```

## 5. Tests and validation map

### Rust integration tests in `infer/tests/`

- scheduler/runtime: `e2e.rs`, `e2e_qwen35.rs`, `greedy_consistency.rs`
- GGUF/quantization/kernel regressions: `q4k_kernel_correctness.rs`, `ground_truth_q4k.rs`, `smoke_*`
- golden/test-data tooling: `regen_test_data.rs`, `gen_test_data_35.rs`

### Bench and helper entrypoints

- `scripts/bench_guidellm.sh`: canonical throughput / latency sweep wrapper
- `scripts/bench_throughput.py`: legacy helper for narrower synthetic/sharegpt runs; not canonical throughput / latency truth
- `scripts/bench_agent_trace.py`: agent-style trace replay
- `infer/src/bin/metal_bench.rs`: Metal micro/macro benchmark entrypoint

## 6. Where to start reading

- Backend loading / model discovery: start at `infer/src/hf_hub.rs` for
  `resolve_model_source`, then `infer/src/server_engine.rs` for
  `LoadedInferenceEngine::load` and the `InferenceEngine` trait, then
  `infer/src/backend/cuda/bootstrap.rs` for the CUDA bring-up
- CUDA serving path: `infer/src/main.rs` → `infer/src/http_server.rs` →
  `infer/src/scheduler/cuda/`
- Agent CLI path: `src/main.rs` → `crates/cli/src/lib.rs` →
  `infer/src/server_engine.rs` → `crates/agent/src/lib.rs`
