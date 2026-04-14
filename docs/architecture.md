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

## Future Evolution

The current monolithic `infer` shape is **option A** in a two-step path that
ends at **option B (kernel-only crate extraction)**. The decision and the
trip wires were locked in on 2026-04-15 after a strategic discussion between
Claude and Codex, with explicit framing on long-term maintainability and
upgradeability rather than short-term build speed.

### Why we stopped at option A for now

Today's `infer/src/backend/cuda/bootstrap.rs` reaches into `crate::model::*`,
`crate::scheduler::*`, `crate::tokenizer::Tokenizer`, and
`crate::model_registry::*` to load and dispatch model-specific CUDA engines.
Extracting `backend/cuda/` now would force `Tokenizer`, `KVCacheDtype`,
`ModelType`, and the per-model weight structs to become `pub` cross-crate,
and `bootstrap.rs::load_*_components` would straddle the new crate boundary —
the exact failure mode of the four-shell Route-A split that was reverted in
the same week. So we keep it monolithic and instead **pre-stage the future
boundary** through three internal moves that landed as commits `26c8f39` and
`efcc991`:

1. `infer/src/backend/cuda/ffi.rs` is split into 10 domain submodules
   (`ffi/{attention,gemm,kv,norm,quant,sampling,embedding,elementwise,
   recurrent,misc}.rs`) with re-exports from the parent.
2. `infer/src/backend/cuda/prelude.rs` is the **proto-API contract** —
   seven cross-cutting types that 25+ files import through. The prelude is
   intentionally narrow and policed by a "≥3 consumers + stable + would not
   force any `infer` type to become cross-crate `pub`" rule documented in
   the file itself.
3. `infer/build.rs` derives Triton `cargo:rerun-if-changed` from a
   directory walk, so the build script can never drift from the actual
   kernel set.

### When option B kicks in

Any **one** of these triggers (all on `ROADMAP.md::Missing`) executes the
extraction blueprint at `docs/plans/cuda-kernel-crate-extraction.md`:

| Trigger | Why it forces extraction |
|---|---|
| **T1** Parallel kernel build configs producing incompatible `.a` artifacts | A single `infer/build.rs` cannot ship two different `libkernels_cuda.a` flavors. |
| **T2** NCCL tensor parallel communication | Adds `libnccl` linkage + multi-GPU coordination kernels on top of an already-crowded build feature matrix. |
| **T3** FlashAttention-3 (H100 / sm_90) prefill | Two parallel attention kernel implementations selected by SM target → triggers T1. |
| **T4** MLA attention (DeepSeek-V2/V3/R1) + FP8 GEMM (sm_90) | Doubles the size of `csrc/cuda/`; new attention algorithm + new GEMM family. |
| **T5** Speculative decoding GPU integration | Adds a draft kernel surface called from inside the scheduler hot loop. |
| **T6** A second consumer of the kernel layer | The "two direct consumers" admission criterion is met. |

The blueprint is one focused day of mechanical work on the CUDA host because
the seams (ffi domain split, prelude proto-API, Triton auto-derive) are
already in place. **Trip wire response is "execute the plan", not "decide
again".**

### What option B looks like

After extraction:

```
crates/infer-cuda-kernels/    ← NEW: csrc/cuda/, tools/triton/, ffi/, tensor,
                                paged_kv, flashinfer, graph_pool, prelude
infer/                        ← thin runtime + HTTP shell; bootstrap.rs stays
                                here because it pulls model + scheduler
```

`infer/src/backend/cuda.rs` becomes a thin `pub use infer_cuda_kernels::*;`
re-export shim so 60+ existing `crate::backend::cuda::…` call sites do not
need to change. The dependency edge is **`infer → infer-cuda-kernels`,
never the reverse.** No `infer-ops`, no `infer-scheduler-core`, no
`infer-runtime-api` — those are the anti-goals documented in the blueprint
because they would re-create the bootstrap straddle problem.

### Anti-goals

- **No `infer-ops` crate.** Ops are tightly coupled to model layouts.
- **No `infer-scheduler-core` crate.** The CUDA scheduler reaches into
  `PagedKVPool`, `FlashInferDecodeMetadata`, and model-specific types in
  bootstrap.
- **No CPU backend extraction.** `infer/src/backend/cpu.rs` is a 309-line
  smoke-test backend that generates synthetic responses; extracting it
  would create a one-consumer crate with zero independence benefit.
- **No `*-sys` / Rust-types split for the kernel crate.** One crate holds
  both layers; splitting them creates a `*-sys` boundary with one consumer.

### Cross-references

- `docs/plans/cuda-kernel-crate-extraction.md` — full execution blueprint.
- `infer/src/backend/cuda/prelude.rs` — the proto-API contract.
- `docs/archives/art-grade-architecture-for-long-agent-infer.md` — the
  ambitious 8-crate split that Route-A reverted; §六 governance and §七
  acceptance criteria still inform the trip wire bar.
- `docs/archives/cuda-crate-extraction.md` — the original (overly broad)
  Round-3 extraction plan that bundled `backend + ops + model + scheduler`,
  superseded by the kernel-only blueprint above.
