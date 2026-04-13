# agent-infer Architecture

Updated 2026-04-13 after the workspace split.

## Workspace Layout

The repository is now split into focused crates instead of keeping agent-facing
logic in the root package:

- `agent-infer`: thin binary wrapper that delegates to `infer-cli`.
- `infer`: inference runtime, HTTP server, scheduler, backend runtime, model
  loading, CUDA/Metal/CPU backends.
- `infer-chat`: shared chat/tool-call protocol and OpenAI chat surface types.
- `infer-tools`: builtin tool definitions plus sandboxed execution helpers.
- `infer-agent`: agent session state, prompt assembly, tool-call recovery, and
  turn loop logic.
- `infer-engine`: backend-facing adapter that loads a backend and exposes the
  stable `infer-agent` engine interface.
- `infer-cli`: REPL and CLI wiring for the `agent-infer` binary.
- `infer-core`, `infer-policy`, `infer-observability`: shared domain, policy,
  and event-sink crates extracted from runtime internals.

## Runtime Paths

### Agent CLI

```text
agent-infer (root binary)
  -> infer-cli::run()
  -> infer-engine::resolve_model_source() / load()
  -> infer-agent::AgentSession
  -> infer-tools builtin tools
  -> infer-chat prompt/tool-call protocol
  -> infer backend
     - CUDA: LoadedServerEngine
     - Metal: MetalBackend adapter
     - CPU: CpuBackend adapter
```

### OpenAI-Compatible Serving

```text
infer binary / metal_serve / cpu_serve
  -> axum handlers
  -> request parsing + sampling params
  -> scheduler or backend_runtime
  -> model forward path
  -> backend-specific kernels/runtime
```

## Package Boundaries

| Crate | Owns | Does not own |
| --- | --- | --- |
| `agent-infer` | Binary entrypoint only | REPL logic, backend loading |
| `infer-cli` | CLI args, REPL commands, terminal UX | Model discovery internals, backend-specific request types |
| `infer-engine` | Model discovery, logging init, backend loading, adapter conversions | Session state, tool execution UX |
| `infer-agent` | Conversation state, tool recovery, request/response contract for agent turns | Concrete backend/runtime implementations |
| `infer-tools` | Tool schemas and execution wrappers | Prompt formatting, model inference |
| `infer-chat` | Shared protocol formatting/parsing | Runtime scheduling and backend logic |
| `infer` | Scheduler, HTTP server, backend runtime, model/kernel integration | CLI session logic |

## Backend Split

- `cuda`: full scheduler path with chunked prefill, decode-priority batching,
  paged KV, FlashInfer/Triton/CUDA kernels.
- `metal`: serial backend path for Apple Silicon via `mlx-sys`.
- `cpu`: development-oriented serial backend for smoke tests, CLI wiring, and
  end-to-end validation on non-GPU machines.

## Current Notes

- The old root modules `src/agent.rs`, `src/chat.rs`, `src/tools.rs`, and
  `src/engine.rs` have been removed.
- `infer/src/chat.rs` and `infer/src/chat_protocol.rs` were replaced by the
  shared `infer-chat` crate.
- Architecture details for CUDA kernels, paged KV, and scheduler internals
  still live in the `infer` crate; this document only tracks the current
  workspace/package topology.
