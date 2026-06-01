# ARLE Architecture

This document is the canonical source for ownership boundaries, dependency
direction, and crate-admission governance. New contributors: start at
[onboarding.md](onboarding.md) (30 min). For "what files exist and where
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
| `cuda-kernels` | CUDA kernel layer (`csrc/`, TileLang AOT, Rust FFI, paged-KV / TileLang metadata / graph-pool / tensor / kv_quant / kv_turboquant) | Model code, scheduler logic, tokenizer |
| `mlx-sys` | MLX C++ bridge for the Metal backend | Anything that is not the Metal bridge |
| `kv-native-sys` | Local persistence substrate (file/block ABI, mmap, WAL, shm descriptors) for the KV-tier disk/shared transport path | Tier policy, scheduler, GPU code |
| `qwen3-spec` / `qwen35-spec` | Shared train↔infer Qwen config + canonical tensor names + `Shard` annotations | Implementation code |
| `deepseek-spec` | DS0 readiness scaffold (2026-05-01): DeepSeek V3/V4 config, tensor-name contracts, MLA/MoE/MTP `Shard` annotations | Runtime model code, MLA/MoE kernels (gated on F0–F4 multi-GPU collectives in forward) |
| `autograd` | From-scratch autograd: `TensorStore` + `Tape` + `Backend` trait | Trainer loop, control plane |
| `train` | On-Policy Distillation substrate (teacher in `infer`, student LoRA), train-side `/v1/train/*` control plane, shared async observability. Pretrain / SFT / GRPO / multi-turn retired 2026-05-18 — see `docs/projects/2026-05-18-opd-only-pivot.md`. | GPU kernels, scheduler |

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
  paged KV, TileLang AOT, and native CUDA C kernels.
- `metal`: serial backend path for Apple Silicon via `mlx-sys`.
- `cpu`: development-oriented serial backend for smoke tests, CLI wiring, and
  end-to-end validation on non-GPU machines.

## Backend Parity Matrix

Cross-backend differences are intentional — hardware and maturity differ.
Status labels mirror [`support-matrix.md`](support-matrix.md); this table is
the architecture-level view only. Do not treat a ✅ in one column as implying
parity in another.

| Capability | CUDA | Metal | CPU |
| --- | --- | --- | --- |
| Production serving target | Supported | Beta | No (smoke only) |
| Continuous batching scheduler | Yes (`scheduler/cuda/`) | Partial (serial runtime + Metal scheduler; varlen batched decode path exists but not full CUDA parity) | No |
| Paged / batched KV | Yes (`cuda-kernels` `PagedKVPool`, page_size=16) | Yes (`BatchKVCache` pattern via `mlx-sys`) | No |
| Chunked prefill + decode-priority | Yes | Partial | No |
| Quantized KV cache (`--kv-cache-dtype`) | Yes (INT8/FP8/TQ*) | No (native model dtype) | No |
| Radix prefix cache + tiered KV (T0–T3) | Yes (T0 prod; T1–T2 Beta; T3 stub) | Beta (prefix reuse via snapshots) | No |
| Speculative decode | Not shipped (plumbing only) | Beta (DFlash for Qwen3.5) | No |
| Multi-GPU TP/PP/EP | Scaffold (F0–F4; forward not wired) | No | No |
| OPD teacher surface | Yes | No | No |
| OpenAI HTTP (`/v1/chat/completions`, SSE) | Yes | Yes | Yes (synthetic) |

Evidence pointers:

- Backend tiers: support-matrix §1
- Model reach: support-matrix §3
- Quant / KV: support-matrix §4, §4b
- Spec decode: support-matrix §4a
- Multi-GPU scaffold: §Multi-GPU below + `infer/src/distributed/`

## Change Impact Map

Minimum verification after touching a layer. Runtime hot-path changes also
require a dated entry under `docs/experience/wins/` or `errors/` per
[`AGENTS.md`](../AGENTS.md) §Benchmarks.

| Layer touched | Minimum verify | Notes |
| --- | --- | --- |
| `crates/cuda-kernels/csrc/` or `crates/cuda-kernels/src/` | `cargo test --release -p infer --features cuda`; kernel regressions in `infer/tests/smoke_*` | Bench: `scripts/bench_guidellm.sh` for perf claims |
| `infer/src/scheduler/cuda/` | `cargo test --release --test e2e` | Scheduler invariants: [`infer/src/scheduler/AGENTS.md`](../infer/src/scheduler/AGENTS.md) |
| `infer/src/model/qwen35/` (CUDA) | `cargo test --release --test e2e_qwen35` | Golden baselines in `infer/test_data/` |
| KV quant / paged KV gating | `cargo test --release -p infer --features cuda --test kv_precision_parity -- --nocapture --test-threads=1` | See AGENTS.md §Build & run |
| `infer/src/backend/metal/` or `crates/mlx-sys/` | `cargo test --release --no-default-features --features metal --test e2e_qwen35` | Canonical Metal model: Qwen3.6 MoE (AGENTS.md) |
| `infer/src/kv_tier/` or `infer/src/prefix_cache.rs` | `cargo test --release -p infer --features cuda` + tier-specific tests | MR stability: [`infer/src/kv_tier/AGENTS.md`](../infer/src/kv_tier/AGENTS.md) |
| `crates/agent/`, `crates/cli/`, `crates/chat/` | `cargo test --release -p agent -p cli -p chat` | No GPU required |
| `crates/train/` OPD path | `cargo test --release -p train` | End-to-end OPD needs CUDA GPU |
| Docs-only | — | State `docs-only` in commit body; no bench gate |

## Multi-GPU Parallel Axes (single-node F0–F4 scaffold)

The single-node multi-GPU foundation lives under `infer/src/distributed/`
and is currently a scaffold: type surfaces, group metadata, and an NCCL
group-coordinator smoke are proven, but real production collectives are
**not yet wired into model forward**. Mainline default behavior is one
rank, one model load, one scheduler — unchanged.

Axes scaffolded today (see
[`docs/projects/2026-05-01-multi-gpu-f0-readiness.md`](projects/2026-05-01-multi-gpu-f0-readiness.md)
and [`docs/plans/2026-04-28-single-node-multi-gpu.md`](plans/2026-04-28-single-node-multi-gpu.md)):

- **TP (tensor parallel):** F1 `parallel_state.rs` + `TpLoadContext` shard
  helpers; F2 Qwen3.5 forward sharding through
  `LayerCommunicator` (`post_attn_all_reduce`, `post_mlp_all_reduce`,
  DP-attention gather hook). TP=1 is no-op; TP>1 production model load
  fails fast until F2 collectives complete.
- **PP (pipeline parallel):** F0.7 `ForwardBatch.pp_proxy:
  Option<IntermediateTensors>` + F3 `pipeline_state.rs` scaffold.
- **EP (expert parallel):** F4 `expert_state.rs` scaffold; no CUDA MoE
  forward consumer yet.
- **NCCL backend:** `--features cuda,nccl` gate; 2-thread `all_reduce(sum)`
  smoke passes via `infer --nccl-smoke` and
  `infer/tests/distributed_nccl_smoke.rs`.

These axes are the dependency floor for both the longctx Phase 3
(disaggregated prefill/decode) lever and the DeepSeek V4 readiness path
(`crates/deepseek-spec` DS0 scaffold + DS3 MLA + DS4 CUDA MoE + DS5 NCCL
collectives in forward). They must complete real collectives in forward
before either downstream consumer can claim multi-rank serving.

DeepSeek V4 is the #1 next-model priority and Qwen 3.6 is #2; the canonical
ranking and rationale live in
[`ROADMAP.md` §Next-Model Priority Order](../ROADMAP.md#next-model-priority-order),
with current support status in
[`docs/support-matrix.md` §3](support-matrix.md#3-model-family-matrix).

## Speculative Decode Framework

The Phase 2 spec-decode plumbing landed but does not yet produce a
throughput lift:

- `infer/src/speculative.rs`: `SpecConfig`, `DraftMode`, persistent
  per-request external draft state, K-token proposals, greedy verifier
  accounting, bonus-token commit, and live spec counters.
- `infer/src/speculative/cuda.rs`: CUDA integration entry points.
- `infer/src/scheduler/cuda/spec_path.rs`: per-step `SpecPath` dispatch
  threading through the CUDA execution loop.

The first end-to-end real-spec bench regressed -62.8% vs the Phase 1
SGLang-row close because the correctness-first verifier still runs the
target paged decode once per verifier position. Phase 2 throughput
claims are paused until a packed K+1 verifier or MagicDec sparse-KV
self-spec lands. See
[`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`](projects/2026-04-30-longctx-32k-128k-leadership.md)
§13 and the regression entry in
[`docs/experience/errors/2026-05-01-phase2-real-spec-regression.md`](experience/errors/2026-05-01-phase2-real-spec-regression.md).

For Qwen3.5 / Medusa specifically, the current gate is recurrent-state
rollback: paged KV can be truncated, but hybrid linear-attention recurrent
state needs a model-owned accepted-length commit/rollback before spec-on
results are valid. See
[`docs/plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md).

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
  `PagedKVPool`, `TileLangDecodeMetadata`, and model-specific types in
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
