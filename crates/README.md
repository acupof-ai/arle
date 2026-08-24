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
  `TileLangDecodeMetadata`, `graph_pool`). Extracted 2026-04-15 by commit
  `a4e12f5`; the dependency edge is one-way: `infer-cuda → cuda-kernels`, never
  the reverse. See [`cuda-kernels/AGENTS.md`](cuda-kernels/AGENTS.md)
  for the proto-API / prelude discipline.
- `mlx-sys`: MLX C++ bridge and vendored MLX Metal qmv kernels used by the
  Metal backend
- `kv-native-sys`: pure-Rust persistence substrate — `KvTierStore`, the
  backend-neutral two-level KV-tier store shared by the CUDA and Metal
  executors

Shared model contract:

- `qwen3-spec`: canonical Qwen3 config + tensor-name contract shared between
  train and infer
- `qwen35-spec`: canonical Qwen3.5 config + tensor-name contract
- `deepseek-spec`: DeepSeek V4 config + tensor-name contract (DS0 scaffold)

Train-side runtime extension (OPD-only since 2026-05-18 pivot):

- `autograd`: from-scratch Rust autograd — `TensorStore` + `Tape` + `Backend`
  trait with CPU + CUDA + Metal paths
- `train`: OPD substrate — `opd_step`, LoRA, checkpoint codec, tokenizer,
  train-side `/v1/train/{status,events,stop,save}` control plane, shared async
  observability sinks (JSONL + MLflow + OTLP + W&B sidecar). **Retired surfaces:**
  scratch pretrain, SFT, GRPO, multi-turn RL (commit `bd94c09`).

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-engine`, `infer-observability`, and `infer-policy` crates back into
`infer` as in-tree modules; the monolith was itself deleted in the 2026-06-04
rewrite. `infer-api` (`LoadedInferenceEngine`) is now the single programmatic
engine entry point.
