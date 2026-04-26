# ARLE Architecture

This document is the canonical source for ownership boundaries, dependency
direction, and crate-admission governance. For "what files exist and where
to start reading", see [codebase-map.md](codebase-map.md). For the
extraction story behind `crates/cuda-kernels`, see
[plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md).

Project framing (also in [index.md](index.md) §Current Positioning):
`infer` owns serving/runtime truth, `arle` is the local front door built on
top of it, and the train stack extends the same runtime/model authority
rather than defining a second equal architecture.

## Package Boundaries

| Crate | Owns | Does not own |
| --- | --- | --- |
| workspace root package | `arle` binary entrypoint only | REPL logic, backend loading |
| `cli` | CLI args, REPL commands, terminal UX | Session state, runtime internals |
| `agent` | Conversation state, tool recovery, request/response contract for agent turns | Concrete backend/runtime implementations |
| `tools` | Tool schemas and execution wrappers | Prompt formatting, model inference |
| `chat` | Shared protocol formatting/parsing, OpenAI chat surface types | Runtime scheduling and backend logic |
| `infer` | Scheduler, HTTP server, backend runtime, model/kernel integration, `server_engine::InferenceEngine` contract | Terminal UX and agent-session orchestration |
| `cuda-kernels` | CUDA kernel layer (`csrc/`, Triton AOT, Rust FFI, paged-KV / FlashInfer / graph-pool / tensor / kv_quant / kv_turboquant) | Model code, scheduler logic, tokenizer |
| `mlx-sys` | MLX C++ bridge for the Metal backend | Anything that is not the Metal bridge |
| `kv-native-sys` | Local persistence substrate (file/block ABI, mmap, WAL, shm descriptors) for the KV-tier disk/shared transport path | Tier policy, scheduler, GPU code |
| `qwen3-spec` / `qwen35-spec` | Shared train↔infer Qwen config + canonical tensor names | Implementation code |
| `autograd` | From-scratch autograd: `TensorStore` + `Tape` + `Backend` trait | Trainer loop, control plane |
| `train` | Generic Qwen-family pretrain/SFT/GRPO stack, train-side `/v1/train/*` control plane, shared async observability | GPU kernels, scheduler |

## Dependency Direction

```text
workspace root package
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

Reverse dependencies from `runtime-*` (or any `infer`-internal layer) into
`http`/`cli` are rejected on sight.

## Backend Split

- `cuda`: full scheduler path with chunked prefill, decode-priority batching,
  paged KV, FlashInfer/Triton/CUDA kernels.
- `metal`: serial backend path for Apple Silicon via `mlx-sys`.
- `cpu`: development-oriented serial backend for smoke tests, CLI wiring, and
  end-to-end validation on non-GPU machines.

## Route-A Note (Historical)

The 2026-04-15 Route-A refactor folded the experimental `infer-core`,
`infer-observability`, `infer-policy`, and `infer-engine` crates back into
`infer` because the split never achieved real independence. A follow-up the
same day deleted `infer/src/agent_engine.rs` after confirming every `Agent*`
type duplicated a corresponding `Completion*` / `InferenceEngine` type in
`server_engine.rs`.

The old agent-facing adapter (`AgentEngine` / `LoadedAgentEngine`) is gone;
`server_engine::InferenceEngine` and `LoadedInferenceEngine` now serve both
the HTTP server and the agent CLI through one contract. `resolve_model_source`
moved into `infer::hf_hub`. Shared runtime contracts (request/session ids,
scheduler policies, event sinks) live inside `infer` as `types.rs`,
`scheduler/policy.rs`, and `events.rs`.

## Crate-Split Governance

These rules govern when a new crate may be cut, and when one must not.

1. New module → prefer placing it in an existing crate; cut a new crate only
   when the existing one cannot contain it without leaking concerns.
2. Cross-crate calls go through public traits; never import private
   implementation modules across the boundary.
3. Every new crate must name **at least two direct consumers** in its PR
   description. If you cannot, the split is premature.
4. Every PR states its "affected layer" and "does this break a dependency
   direction" up front; reverse dependencies from `runtime-*` into
   `http/cli` are rejected on sight.
5. Branches must arrive as single-topic commits; if a reviewer must hold
   kernel + scheduler + workspace semantics in their head at once, the
   split has already failed.

### Active anti-goals

The kernel-crate extraction (`a4e12f5`, 2026-04-15) was deliberately narrow.
The items below remain anti-goals **unless** a concrete second consumer
forces them.

- **No `infer-ops` crate.** Ops are tightly coupled to model data layouts.
- **No `infer-scheduler-core` crate.** The CUDA scheduler reaches into
  `PagedKVPool`, `FlashInferDecodeMetadata`, and model-specific types in
  `bootstrap`.
- **No `infer-runtime-api` trait crate.** Already covered by
  `infer::server_engine::InferenceEngine`.
- **No `*-sys` / Rust-types split for the kernel crate.** One crate holds
  both layers; splitting them creates a `*-sys` boundary with one consumer.
- **No CPU backend extraction.** `infer/src/backend/cpu.rs` is a 309-line
  smoke-test backend that generates synthetic responses; extracting it
  would create a one-consumer crate with zero independence benefit.

The original kernel-crate trip wires (T1 NCCL, T2 FA-3, T3 MLA/FP8 GEMM,
T4 spec decoding, T5 second external consumer) are arguments for the
**next** extraction boundary — whichever one, if any, eventually peels
scheduler or model layers out. They are not arguments about the kernel
crate itself.
