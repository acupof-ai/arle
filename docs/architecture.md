# agent-infer Architecture

Updated 2026-04-15 after the Route-A refactor.

## Workspace Layout

The repository is a small control-plane workspace around one runtime-heavy
crate:

- `agent-infer`: thin binary wrapper that delegates to `infer-cli`
- `infer`: inference runtime, HTTP server, scheduler, backend runtime, model
  loading, CUDA/Metal/CPU backends, and the unified `server_engine::InferenceEngine`
  contract that both the HTTP server and the agent CLI call through
- `infer-chat`: shared chat/tool-call protocol and OpenAI chat surface types
- `infer-tools`: builtin tool definitions plus sandboxed execution helpers
- `infer-agent`: agent session state, prompt assembly, tool-call recovery, and
  turn loop logic
- `infer-cli`: REPL and CLI wiring for the `agent-infer` binary
- `mlx-sys`: MLX C++ bridge used by the Metal backend

The 2026-04-15 Route-A refactor folded `infer-core`, `infer-observability`,
`infer-policy`, and `infer-engine` back into `infer` because the split never
achieved real independence.

## Runtime Paths

### Agent CLI

```text
agent-infer (root binary)
  -> infer-cli::run()
  -> infer::hf_hub::resolve_model_source() / infer::server_engine::LoadedInferenceEngine::load()
  -> infer-agent::AgentSession (uses `dyn InferenceEngine`)
  -> infer-tools builtin tools
  -> infer-chat prompt/tool-call protocol
  -> infer::server_engine::LoadedInferenceEngine
     - CUDA: Qwen3InferenceEngine / Qwen35InferenceEngine / GLM4InferenceEngine
     - Metal: BackendInferenceEngine<MetalBackend>
     - CPU:   BackendInferenceEngine<CpuBackend>
```

### OpenAI-Compatible Serving

```text
infer binary / metal_serve / cpu_serve
  -> axum handlers
  -> request parsing + sampling params
  -> scheduler or backend::runtime
  -> model forward path
  -> backend-specific kernels/runtime
```

## Package Boundaries

| Crate | Owns | Does not own |
| --- | --- | --- |
| `agent-infer` | Binary entrypoint only | REPL logic, backend loading |
| `infer-cli` | CLI args, REPL commands, terminal UX | Session state, runtime internals |
| `infer-agent` | Conversation state, tool recovery, request/response contract for agent turns | Concrete backend/runtime implementations |
| `infer-tools` | Tool schemas and execution wrappers | Prompt formatting, model inference |
| `infer-chat` | Shared protocol formatting/parsing | Runtime scheduling and backend logic |
| `infer` | Scheduler, HTTP server, backend runtime, model/kernel integration, `server_engine::InferenceEngine` contract | Terminal UX and agent-session orchestration |

## Backend Split

- `cuda`: full scheduler path with chunked prefill, decode-priority batching,
  paged KV, FlashInfer/Triton/CUDA kernels
- `metal`: serial backend path for Apple Silicon via `mlx-sys`
- `cpu`: development-oriented serial backend for smoke tests, CLI wiring, and
  end-to-end validation on non-GPU machines

## Current Notes

- The old agent-facing adapter (`infer::agent_engine::AgentEngine` / `LoadedAgentEngine`)
  was deleted entirely. Its responsibilities are now satisfied by
  `infer::server_engine::InferenceEngine` and `LoadedInferenceEngine`, which
  serve both the HTTP server and the agent CLI through a single unified
  contract. `resolve_model_source` moved into `infer::hf_hub`.
- Shared runtime contracts for request/session ids, scheduler policies, and
  event sinks now live inside `infer` as `types.rs`, `scheduler/policy.rs`, and
  `events.rs`.
- Architecture details for CUDA kernels, paged KV, and scheduler internals
  still live in the `infer` crate; this document tracks package ownership, not
  every implementation detail.
