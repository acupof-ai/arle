# Workspace crates

This folder hosts the reusable crates around the inference runtime. The canonical workspace
map lives in [`../docs/codebase-map.md`](../docs/codebase-map.md); new
contributors should start at [`../docs/onboarding.md`](../docs/onboarding.md).

Runtime-facing control plane:

- `agent`: agent session state, prompt assembly, tool-call recovery, turn loop
- `chat`: shared chat / tool-call protocol and OpenAI chat surface types
- `cli`: REPL and slash-command flow for the `arle` binary
- `tools`: builtin tool definitions and sandboxed execution helpers

Backend bridges and kernel layer:

- `cuda-kernels`: extracted CUDA kernel layer (CUDA C / TileLang sources, Rust
  FFI, `DeviceContext` / `DeviceVec` / `HiddenStates`, `PagedKVPool` /
  `TileLangDecodeMetadata`, `graph_pool`). The dependency edge is one-way: `infer-cuda → cuda-kernels`, never
  the reverse. See [`cuda-kernels/AGENTS.md`](cuda-kernels/AGENTS.md)
  for the proto-API / prelude discipline.
- `mlx-sys`: MLX C++ bridge and vendored MLX Metal qmv kernels used by the
  Metal backend
- `kv-native-sys`: pure-Rust persistence substrate — `KvTierStore`, the
  backend-neutral two-level KV-tier store shared by the CUDA and Metal
  executors

Shared model contract:

- `qwen3-spec`: canonical Qwen3 config + tensor-name contract
- `qwen35-spec`: canonical Qwen3.5 config + tensor-name contract
- `deepseek-spec`: DeepSeek V4 config + tensor-name contract (DS0 scaffold)

OPD training lives in the separate
[arle-opd](https://github.com/acupof-ai/arle-opd) repository and consumes these
crates.

`infer-api` (`LoadedInferenceEngine`) is the single programmatic engine entry
point.
