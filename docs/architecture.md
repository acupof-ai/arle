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
- `infer-cuda-kernels`: CUDA kernel layer — `csrc/` + Triton AOT + Rust FFI
  (`paged_kv`, `flashinfer`, `graph_pool`, `tensor`, `kv_quant`,
  `kv_turboquant`). Extracted from `infer` in commit `a4e12f5` (2026-04-15).
- `mlx-sys`: MLX C++ bridge used by the Metal backend

Phase 6 training stack (orthogonal to the inference runtime; see
[projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md)):
- `autograd`: from-scratch Rust autograd — `TensorStore` + `Tape` + `Backend` trait with CPU/Metal/CUDA matmul
- `train`: LoRA + GRPO trainer (`train_multi_turn` binary), depends on `autograd`

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

## Kernel-Crate Extraction (post-2026-04-15)

The workspace originally had two decision points: whether to split the CUDA
kernel layer out of `infer` (option B), and, if so, what the minimal safe
surface looked like. The discussion happened between Claude and Codex on
2026-04-15 and the trip wires were locked in before execution so the work
could land as mechanical refactor, not re-debate.

The extraction **landed in commit `a4e12f5 refactor(cuda): extract
infer-cuda-kernels api`**, followed by `f81e2d5 style(workspace): run
cargo fmt after kernel-crate extraction`. From that point on, the
"option A vs option B" framing is historical; the current tree is
option B.

### Current layout (what the extraction moved)

```
crates/infer-cuda-kernels/
├── Cargo.toml
├── build.rs                 ← nvcc + Triton AOT, lifted from infer/build.rs
├── csrc/
│   ├── attention/           ← flashinfer_*, fused_attention, prefill_attention, decode_prep_paged, …
│   ├── gemm/                ← gemv, quantized_gemv, marlin_*, turboquant_weight_gemv
│   ├── kv/                  ← paged_kv_append, kv_cache_to_paged, kv_quant, scatter_kv
│   ├── quant/               ← turboquant, turboquant_fast, dtype_convert
│   ├── misc/                ← norm, sampling, pos_enc, conv1d, gdr, fused_mlp, …
│   └── common.cuh
├── tools/triton/            ← flash_attention_prefill_hd256, gated_delta_rule_chunkwise, silu_mul, basic, …
└── src/
    ├── lib.rs
    ├── ffi.rs + ffi/{attention,gemm,kv,norm,quant,sampling,embedding,elementwise,recurrent,misc}.rs
    ├── tensor.rs            ← DeviceContext / DeviceVec / DeviceMatrix / HiddenStates / RawDevicePtr
    ├── paged_kv.rs          ← PagedKVPool / TokenKVPool
    ├── flashinfer.rs        ← FlashInferDecodeMetadata / workspace
    ├── graph_pool.rs
    ├── kv_quant.rs / kv_turboquant.rs / kv_types.rs / turboquant_state.rs
    └── prelude.rs           ← public API surface (the proto-API graduated)
```

```
infer/
└── src/backend/
    ├── cuda.rs              ← thin `pub use infer_cuda_kernels::*;` re-export
    │                           shim; ~60 `crate::backend::cuda::…` call sites
    │                           keep resolving unchanged.
    └── cuda/
        └── bootstrap.rs     ← STAYS in infer; reaches into crate::model::*,
                                crate::scheduler::*, crate::tokenizer::Tokenizer,
                                crate::model_registry::*.
```

The dependency edge is **`infer → infer-cuda-kernels`, never the reverse**.
`bootstrap.rs` is the only file where kernel concerns and model/scheduler
concerns meet; keeping it in `infer` is what let the kernel crate ship
without forcing `Tokenizer`, `KVCacheDtype`, `ModelType`, or per-model
weight structs to become cross-crate `pub`.

### Why the earlier internal hygiene work still matters

Three pre-extraction moves landed in commits `26c8f39` and `efcc991` and
are what made the single-day extraction actually mechanical:

1. `infer/src/backend/cuda/ffi.rs` was split into 10 domain submodules
   (`ffi/{attention,gemm,kv,norm,quant,sampling,embedding,elementwise,
   recurrent,misc}.rs`). They carried over intact to
   `crates/infer-cuda-kernels/src/ffi/`.
2. `prelude.rs` was the **proto-API contract** — seven cross-cutting types
   that 25+ files imported through, gated by a "≥3 consumers + stable +
   would not force any `infer` type to become cross-crate `pub`" rule.
   At extraction time those `pub(crate)` items became real cross-crate
   `pub` with no new symbols added.
3. Triton `cargo:rerun-if-changed` is derived from a `read_dir(tools/triton)`
   walk so the build script can never drift from the actual kernel set.

### Further extraction (still future)

The kernel-crate extraction was deliberately narrow. The items below remain
anti-goals **unless** a concrete second consumer forces them — the bar
documented in `docs/archives/art-grade-architecture-for-long-agent-infer.md`
§六 / §七 applies:

- **No `infer-ops` crate.** Ops are tightly coupled to model data layouts.
- **No `infer-scheduler-core` crate.** The CUDA scheduler reaches into
  `PagedKVPool`, `FlashInferDecodeMetadata`, and model-specific types in
  `bootstrap`.
- **No `infer-runtime-api` trait crate.** Already covered by
  `infer::server_engine::InferenceEngine`.
- **No `*-sys` / Rust-types split for the kernel crate.** One crate holds
  both layers; splitting them creates a `*-sys` boundary with one consumer.

The original trip wires (T1 NCCL, T2 FA-3, T3 MLA/FP8 GEMM, T4 spec
decoding, T5 second external consumer) are now arguments for the **next**
extraction boundary — whichever one, if any, eventually peels scheduler
or model layers out. They are no longer arguments about the kernel crate.

### Additional anti-goal (CPU backend)

- **No CPU backend extraction.** `infer/src/backend/cpu.rs` is a 309-line
  smoke-test backend that generates synthetic responses; extracting it
  would create a one-consumer crate with zero independence benefit.

### Cross-references

- `docs/plans/cuda-kernel-crate-extraction.md` — full execution blueprint.
- `crates/infer-cuda-kernels/src/prelude.rs` — the proto-API contract (graduated from `infer/src/backend/cuda/prelude.rs` at extraction time).
- `docs/archives/art-grade-architecture-for-long-agent-infer.md` — the
  ambitious 8-crate split that Route-A reverted; §六 governance and §七
  acceptance criteria still inform the trip wire bar.
- `docs/archives/cuda-crate-extraction.md` — the original (overly broad)
  Round-3 extraction plan that bundled `backend + ops + model + scheduler`,
  superseded by the kernel-only blueprint above.
