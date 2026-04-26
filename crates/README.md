# Workspace crates

This folder hosts the reusable crates around `infer`. The canonical workspace
map lives in [`../docs/codebase-map.md`](../docs/codebase-map.md); this README
is a quick orientation.

Runtime-facing control plane:

- `agent`: agent session state, prompt assembly, tool-call recovery, turn loop
- `chat`: shared chat / tool-call protocol and OpenAI chat surface types
- `cli`: REPL and slash-command flow for the `arle` binary
- `tools`: builtin tool definitions and sandboxed execution helpers

Backend bridges and kernel layer:

- `cuda-kernels`: extracted CUDA kernel layer (CUDA C / Triton sources, Rust
  FFI, `DeviceContext` / `DeviceVec` / `HiddenStates`, `PagedKVPool` /
  `FlashInferDecodeMetadata`, `graph_pool`). Extracted 2026-04-15 by commit
  `a4e12f5`; the dependency edge is one-way: `infer → cuda-kernels`, never
  the reverse. See [`cuda-kernels/AGENTS.md`](cuda-kernels/AGENTS.md) and
  [`../docs/plans/cuda-kernel-crate-extraction.md`](../docs/plans/cuda-kernel-crate-extraction.md)
  for the proto-API / prelude discipline.
- `mlx-sys`: MLX C++ bridge used by the Metal backend
- `kv-native-sys`: local persistence substrate (Zig) for the `infer/src/kv_tier/`
  disk and shared-memory transport paths

Shared model contract:

- `qwen3-spec`: canonical Qwen3 config + tensor-name contract shared between
  train and infer
- `qwen35-spec`: canonical Qwen3.5 config + tensor-name contract

Train-side runtime extension (per
[`../docs/projects/agent-rl-self-evolving.md`](../docs/projects/agent-rl-self-evolving.md)):

- `autograd`: from-scratch Rust autograd — `TensorStore` + `Tape` + `Backend`
  trait with the device-resident / lazy-eval Metal path
- `train`: generic Qwen-family pretrain / SFT / GRPO / multi-turn trainer,
  train-side `/v1/train/{status,events,stop,save}` control plane, shared
  async observability sinks (JSONL + MLflow + OTLP + W&B sidecar)

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-engine`, `infer-observability`, and `infer-policy` crates back into
`infer` as in-tree modules (`types`, `events`, and `scheduler::policy`). The
old `agent_engine` adapter was also collapsed: its trait and types merged into
`infer::server_engine` (`InferenceEngine`, `LoadedInferenceEngine`,
`CompletionRequest`, `CompletionOutput`, `TokenUsage`, …), so the HTTP server
and the agent CLI now share a single engine contract.
