# Workspace crates

This folder hosts the reusable crates around `infer`:

- `infer-agent`: reusable agent session/control-plane logic
- `infer-chat`: shared ChatML/tool-call protocol and OpenAI chat surface types
- `infer-cli`: reusable REPL/CLI flow for the `agent-infer` binary
- `infer-tools`: reusable builtin tool definitions and sandboxed execution
- `mlx-sys`: MLX C++ bridge used by the Metal backend

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-engine`, `infer-observability`, and `infer-policy` crates back into
`infer` as in-tree modules (`types`, `events`, and `scheduler::policy`). The
old `agent_engine` adapter was also collapsed: its trait and types merged into
`infer::server_engine` (`InferenceEngine`, `LoadedInferenceEngine`,
`CompletionRequest`, `CompletionOutput`, `TokenUsage`, …), so the HTTP server
and the agent CLI now share a single engine contract.
