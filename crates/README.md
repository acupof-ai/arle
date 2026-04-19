# Workspace crates

This folder hosts the reusable crates around `infer`:

- `agent`: reusable agent session/control-plane logic
- `chat`: shared ChatML/tool-call protocol and OpenAI chat surface types
- `cli`: reusable REPL/CLI flow for the `agent-infer` binary
- `tools`: reusable builtin tool definitions and sandboxed execution
- `mlx-sys`: MLX C++ bridge used by the Metal backend
- `cuda-kernels`: extracted CUDA kernel layer (CUDA C/Triton sources,
  FFI, `DeviceContext` / `DeviceVec` / `HiddenStates`, `PagedKVPool` /
  `FlashInferDecodeMetadata`, `graph_pool`). Added 2026-04-15 by
  `a4e12f5 refactor(cuda): extract cuda-kernels api`. The dependency
  edge is one-way: `infer → cuda-kernels`, never the reverse. See
  [`cuda-kernels/AGENTS.md`](cuda-kernels/AGENTS.md) and
  [`../docs/plans/cuda-kernel-crate-extraction.md`](../docs/plans/cuda-kernel-crate-extraction.md)
  for the proto-API / prelude discipline.

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-engine`, `infer-observability`, and `infer-policy` crates back into
`infer` as in-tree modules (`types`, `events`, and `scheduler::policy`). The
old `agent_engine` adapter was also collapsed: its trait and types merged into
`infer::server_engine` (`InferenceEngine`, `LoadedInferenceEngine`,
`CompletionRequest`, `CompletionOutput`, `TokenUsage`, …), so the HTTP server
and the agent CLI now share a single engine contract.
