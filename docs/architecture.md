# ARLE Architecture

This document is the canonical source for ownership boundaries, dependency
direction, and crate-admission governance. New contributors: start at
[onboarding.md](onboarding.md) (30 min). For "what files exist and where
to start reading", see [codebase-map.md](codebase-map.md).

Project framing (also in [index.md](index.md) §Current Positioning): the
`infer-*` crate graph owns serving/runtime truth, `arle` is the local front
door built on top of it, and the train stack extends the same runtime/model
authority rather than defining a second equal architecture.

> **The device-neutral rewrite is the product (PR #53, merged to `main`).**
> The legacy welded `infer/` monolith has been **deleted**. The runtime is
> now the `crates/infer-*` crate graph: a **device-neutral engine core → a
> narrow host-only seam → thin per-device executors** — one scheduler serves
> all backends; a backend is a seam impl, not a scheduler fork. Source of
> truth for the rewrite's executed state:

## Package Boundaries

| Crate | Owns | Does not own |
| --- | --- | --- |
| workspace root package (`arle`) | `arle` binary entrypoint only | REPL logic, backend loading |
| `cli` | CLI args, REPL commands, terminal UX | Session state, runtime internals |
| `agent` | Conversation state, tool recovery, request/response contract for agent turns | Concrete backend/runtime implementations |
| `tools` | Tool schemas and execution wrappers | Prompt formatting, model inference |
| `chat` | Shared protocol formatting/parsing, OpenAI chat surface types | Runtime scheduling and backend logic |
| `infer-plan` | Backend-neutral data IR: `ForwardPlan`, `ForwardMode`, `SamplingParams`, `StepOutput`, the pure host `sample_token`. No behavior, no device. | Any device or backend type |
| `infer-seam` | Host-only trait seam: `BackendExecutor` (submit/poll core + opt-in capability default-methods: stop ids, row/request caps, prefix reuse, KV page-tier and slot-tier hooks, OPD weight offload) + `KvPool` (`KvQuery`/`KvAllocator`/`KvPrefixStore`) + `KvBatchDescriptor` + `ResourceGovernor` + the backend-neutral `HostPagedKvPool`. No device types. | Concrete kernels, scheduler, model code |
| `infer-core` | The one device-neutral `Engine<E,K>`: admission, continuous batching, RadixCache, chunked prefill, overlap, slot lifecycle, sampling/streaming/telemetry. No backend dependency. | Device kernels, HTTP, CLI |
| `infer-cuda` | CUDA `BackendExecutor` + KV pool: paged KV, TileLang AOT + native-CUDA kernels, TP/EP, DeepGEMM, DeepEP, DSv4-Flash, GLM-5.2 (`glm_moe_dsa`, on the DSv4 path; forward landed, verification pending-remote), Qwen3.5/3.6 hybrid+MoE (FP8 MoE via DeepGEMM, batched paged decode), Qwen3.8 NVFP4 (Marlin W4A16 `marlin_fp4_gemm`, repacked at load; prefill m≥512 dequantizes to FP8 for DeepGEMM), opt-in DSv4 whole-step decode CUDA graph (`ARLE_DSV4_DECODE_GRAPH=1`, below eager today), over `cuda-kernels` | Scheduler logic, HTTP, terminal UX |
| `infer-metal` | Metal MLX `BackendExecutor` over `mlx-sys`: target-only Qwen execution is single-row; a loaded DFlash/NextN runtime enables configurable multi-row prefill, mixed, and decode plans. Supports Qwen3.5/3.6 hybrid+MoE, LFM2.5-8B-A1B hybrid (gated short-conv + full attention, behind the `CompiledMetalModel` trait), DeepSeek-OCR; `MetalKvPool` is a compatibility alias to `infer-seam::HostPagedKvPool` | Scheduler logic, HTTP, terminal UX |
| `infer-hip` | HIP/ROCm `BackendExecutor` + KV pool (experimental AIPC lane, #76/#77): DSv4-Flash GGUF 2-bit shim-portable forward over `hip-sys`/`hip-kernels`; consumes the `infer-gguf` host substrate | Scheduler logic, HTTP, datacenter CUDA paths (FlashMLA/DeepGEMM/DeepEP) |
| `infer-vulkan` | Vulkan `BackendExecutor` + KV pool (experimental AIPC skeleton): host forward-order pins for Qwen3/3.5/3.6, DSv4 over `vulkan-sys`/`vulkan-kernels`; device execution pending the shader ABI; consumes the `infer-gguf` host substrate | Scheduler logic, HTTP |
| `infer-topo` | TP/EP sharding: `head_shard`, column/row shard | Kernels, scheduler, HTTP |
| `infer-moe` | Backend-neutral MoE routing: `route`, `RoutingDecision`, `MoeConfig` | Backend kernels, scheduler |
| `infer-server` | OpenAI v1 HTTP frontend (`coordinator.rs` — single facade for all backends) + tokenizer; `ServeHandle<E,K>` engine thread; relay protocol for both single-process (`LocalChannel*`) and multi-process (TCP) | Terminal UX, agent-session orchestration |
| `infer-api` | The single front-door lib: `LoadedInferenceEngine`, `EngineLoadConfig`, `RawLogits`, OPD-teacher surface. Backends plug in behind it. | Terminal UX, REPL logic |
| `infer-util` | Backend-agnostic `hf_hub` + logging leaf crate | Anything backend- or model-specific |
| `cuda-kernels` | CUDA kernel layer (`csrc/`, TileLang AOT, Rust FFI, paged-KV / TileLang metadata / graph-pool / tensor / kv_quant, Marlin W4A16 fp4 GEMM, unified quantized paged attention `paged_attention_quantized_fa3`) | Model code, scheduler logic, tokenizer |
| `mlx-sys` | MLX C++ bridge for the Metal backend (Qwen3.5/3.6, LFM2.5 compiled models) | Anything that is not the Metal bridge |
| `deepep-sys` | DeepEP/NVSHMEM FFI (`internode_ll` dispatch/combine) for EP collectives | Routing policy, scheduler |
| `xgrammar-sys` | Grammar-constrained decode FFI (xgrammar) | Sampling policy, scheduler |
| `hip-sys` / `hip-kernels` | Thin hand-declared HIP runtime FFI (stubs off-box) / HIP kernel build + FFI layer (llama.cpp-adapted IQ2_XXS/Q2_K mmvq corpus, hipcc-gated) | Model code, scheduler logic |
| `vulkan-sys` / `vulkan-kernels` | ash-backed Vulkan loader wrapper / glslc-compiled shader corpus adapted from llama.cpp `vulkan-shaders` | Model code, scheduler logic |
| `infer-gguf` | GGUF container reading (v2/v3 memmap reader), CPU dequant (llama.cpp ports), per-arch GGUF→spec-config mappers (`deepseek4`) | Model forward code, scheduler |
| `kv-native-sys` | `KvMmapStore` (sparse mmap page-slot store): memcpy writes, zero-copy reads. WAL/shm/mm/descriptors unused — kept for future shared-memory tier. | Tier policy, scheduler, GPU code |
| `qwen3-spec` / `qwen35-spec` | Shared train↔infer Qwen config + canonical tensor names + `Shard` annotations | Implementation code |
| `deepseek-spec` | DS0 readiness scaffold (2026-05-01): DeepSeek V3/V4 config, tensor-name contracts, MLA/MoE/MTP `Shard` annotations, `DeepSeekV4AttentionLayerPlan` operator summaries | Runtime model code beyond the spec |
| `autograd` | From-scratch autograd: `TensorStore` + `Tape` + `Backend` trait | Trainer loop, control plane |
| `train` | Runtime-led post-training substrate (teacher via `infer-api`, student LoRA, rollout→score→LoRA-backward). Pretrain / SFT / GRPO / multi-turn retired 2026-05-18 — see OPD-only product boundary. Two families share it: **OPD** (`opd`/`self-opd`, teacher/EMA + KL) and **RFT** (`agent-opd`/`rubric-opd`, reward-selected + masked CE, no teacher/KL). "OPD-only" is the positioning, not that every subcommand is distillation. | GPU kernels, scheduler |

## Dependency Direction

The spine is **IR → host-only seam → device-neutral engine → executors →
serving**. The IR (`infer-plan`) has no dependencies; the seam
(`infer-seam`) names only host types; the engine (`infer-core`) depends on
plan + seam but never on a backend. Executors (`infer-cuda` / `infer-metal`)
implement the seam against plan + seam only — they do **not** depend on
`infer-core`. The serving layer (`infer-server` / `infer-api`) wires a chosen
executor into `Engine<E,K>`.

```text
infer-plan (no deps — the IR)
 ▲
infer-seam -> infer-plan
 ▲
infer-core -> infer-plan, infer-seam (the one Engine<E,K>; no backend dep)

infer-cuda -> infer-plan, infer-seam, infer-topo, infer-moe, cuda-kernels, [deepep-sys, deepseek-spec, qwen3-spec], qwen35-spec
infer-metal -> infer-plan, infer-seam, [mlx-sys]
infer-gguf -> deepseek-spec (neutral GGUF host substrate leaf; spec crates never depend back)
infer-hip -> infer-plan, infer-seam, deepseek-spec, infer-gguf, [hip-sys, hip-kernels]
infer-vulkan -> infer-plan, infer-seam, deepseek-spec, qwen3-spec, qwen35-spec,
 infer-gguf, [vulkan-sys, vulkan-kernels]

infer-server -> infer-core, infer-seam, infer-plan
infer-api -> infer-core, infer-seam, infer-plan, infer-server, [infer-metal, infer-cuda, infer-hip, infer-vulkan, cuda-kernels]

workspace root package (arle)
 -> cli
 -> infer-api
 -> agent (-> infer-api, chat, tools)
 -> chat
 -> tools
 -> autograd, train, infer-util, deepseek-spec, qwen3-spec, qwen35-spec
```

The backend-agnostic-scheduler win: one `Engine<E,K>` in `infer-core` drives
both CUDA and Metal. Adding a third backend means implementing the **two
host-only seam traits** (`BackendExecutor` + `KvPool`) — no scheduler fork,
no `infer-core` change.

Reverse dependencies from any runtime layer (`infer-core` / `infer-cuda` /
`infer-metal`) into `infer-server` / `infer-api` / `cli` are rejected on
sight.

## Engine Core, Seam, and Executors

The runtime answers one structural problem: device, parallelism, and kernel
concerns must not be welded into the scheduler, because then adding a backend
means a second scheduler. The shape is a **device-neutral engine core → a
narrow host-only seam → thin per-device executors** — one scheduler serves all
backends; a backend is a seam impl, not a scheduler fork.

```text
infer-plan data IR (ForwardPlan / ForwardMode / SamplingParams / StepOutput)
 ▲
infer-seam host-only trait seam (no device types):
 │ BackendExecutor (submit/poll core + opt-in capability
 │ default-methods), KvPool = KvQuery+KvAllocator+KvPrefixStore,
 │ KvBatchDescriptor, ResourceGovernor
 ▲
infer-core device-neutral Engine<E,K> + scheduler + radix prefix + overlap (no backend dep)

infer-metal infer-cuda thin executors, one seam impl each
 (real MLX Qwen; (CUDA paged KV, (implement plan + seam only;
 target-only single-row, TileLang + native zero scheduler)
 DFlash multi-row) kernels, TP/EP,
 DeepGEMM, DeepEP,
 DSv4-Flash)
 ▲ ▲
infer-server OpenAI v1 HTTP frontend: coordinator.rs (single HTTP facade for all backends)
 + ServeHandle<E,K> engine thread + relay (LocalChannel / TCP)
 ▲
infer-api single front-door lib (LoadedInferenceEngine, EngineLoadConfig,
 RawLogits, OPD teacher); backends plug in behind it
```

**Seam growth pattern (deliberate).** Cross-cutting engine features (KV
page/slot tiering, prefix-reuse limits, OPD weight offload, row/request
caps) land as **opt-in default-methods on `BackendExecutor`**: the engine
drives the feature generically, backends opt in by overriding, and
non-participating backends are untouched (`0`/`false`/no-op defaults keep
the baseline byte-for-byte). The cost is a wide trait (~15 methods today)
whose capability×backend×model coverage must stay documented (the parity
matrix above). Regrouping into capability traits is trigger-gated; do not
pre-split speculatively.

### Engine-core crate boundaries

| Crate | Owns |
| --- | --- |
| `infer-plan` | The data contract: `ForwardPlan`, `ForwardMode{Prefill,Decode,Mixed,Idle,Verify,Draft}`, `SamplingParams`, `StepOutput`, the pure host `sample_token`. No behavior, no device — the sole engine↔executor bridge. |
| `infer-seam` | Host-only trait seam: `BackendExecutor` (submit/poll) + the `KvPool` split (`KvQuery`/`KvAllocator`/`KvPrefixStore`) + `HostPagedKvPool`, the shared production host page allocator. No device types. |
| `infer-core` | The one device-neutral scheduler: admission, continuous batching, RadixCache, chunked prefill, overlap, slot lifecycle, sampling/streaming/telemetry, `Engine<E,K>`. No backend dependency. |
| `infer-metal` | Metal MLX Qwen3.5/3.6 hybrid+MoE forward plus LFM2.5-8B-A1B hybrid (gated short-conv + full attention, `CompiledMetalModel` trait), DeepSeek-OCR as a thin `BackendExecutor`. Target-only Qwen execution is single-row; loaded DFlash/NextN enables configurable multi-row prefill, mixed, and decode plans. `MetalKvPool` names the shared host allocator for compatibility. |
| `infer-cuda` | CUDA executor as a thin seam impl over `cuda-kernels`: paged KV, TileLang AOT + native-CUDA kernels, TP/EP, DeepGEMM, DeepEP, DSv4-Flash, GLM-5.2 (DSv4 path, verification pending-remote), Qwen3.5/3.6 hybrid+MoE (FP8 MoE via DeepGEMM), Qwen3.8 NVFP4 (Marlin W4A16, repacked at load), opt-in DSv4 decode CUDA graph (`ARLE_DSV4_DECODE_GRAPH=1`). |
| `infer-topo` | TP/EP sharding helpers: `head_shard`, column/row shard. |
| `infer-moe` | Backend-neutral MoE routing: `route`, `RoutingDecision`, `MoeConfig`. |
| `infer-server` | OpenAI v1 HTTP frontend (`coordinator.rs` — single axum router for all backends, both single-process and multi-process); `ServeHandle<E,K>` engine thread; relay protocol (`RelayCoordinator`, `LocalChannel*`, `WireStats`). |
| `infer-api` | The single front-door lib: `LoadedInferenceEngine`, `EngineLoadConfig`, `RawLogits`, OPD-teacher surface. Backends plug in behind it via Cargo features (`cuda`/`metal`). |

### How the parallelism axes map onto the stack

| Axis | Where it lands | Seam fit |
| --- | --- | --- |
| TP (tensor) | `all_reduce` inside the executor's model forward, below the seam | clean — scheduler stays rank-neutral |
| EP (expert/MoE) | `all_to_all` dispatch/combine (DeepEP) inside the executor, below the seam | clean — hidden in the executor |
| DP (data) | N `Engine` instances + router above, in `infer-server` | clean — the `Engine` is the DP unit |
| PP (pipeline) | microbatch ring in `infer-core` + stage-aware executor | the one known gap — single-inflight assumption must be revisited |

CUDA TP=8 / EP=8 (DeepGEMM FP8 MoE + DeepEP) is live in `infer-cuda` for
DeepSeek-V4-Flash; PP is not yet wired into a forward path.

## Backend Split

- `cuda`: full scheduler path with chunked prefill, decode-priority batching,
 paged KV, TileLang AOT, and native CUDA C kernels.
- `metal`: Apple Silicon via `mlx-sys`; target-only Qwen execution is
 single-row, while loaded DFlash/NextN enables backend-specific multi-row plans.
- `cpu`: development-oriented serial backend for smoke tests, CLI wiring, and
 end-to-end validation on non-GPU machines.
- `hip`: experimental AIPC lane (AMD ROCm) — DSv4 GGUF 2-bit shim-portable
 forward; on-box validation pending-remote.
- `vulkan`: experimental AIPC skeleton (cross-vendor) — seam impls + host
 order pins; device execution pending the shader ABI.

## Backend Parity Matrix

Cross-backend differences are intentional — hardware and maturity differ.
Status labels mirror [`support-matrix.md`](support-matrix.md); this table is
the architecture-level view only. Do not treat a ✅ in one column as implying
parity in another.

| Capability | CUDA | Metal | CPU | HIP | Vulkan |
| --- | --- | --- | --- | --- | --- |
| Production serving target | Supported | Beta | No (smoke only) | No (experimental) | No (skeleton) |
| Continuous batching scheduler | Yes (one `Engine<E,K>` in `infer-core`) | Same `Engine<E,K>`; target-only Qwen is single-row, loaded DFlash/NextN enables configurable multi-row/mixed plans | No | Seam impl (single-stream MVP) | Seam impl (skeleton) |
| Paged / batched KV | Yes (`cuda-kernels` `PagedKVPool`, page_size=16) | Yes (`BatchKVCache` pattern via `mlx-sys`) | No | Host KV pool (DSv4 slot shape) | Host KV pool (bookkeeping) |
| Chunked prefill + decode-priority | Yes | Partial | No | No | No |
| Quantized KV cache (`--kv-cache-dtype`) | Yes (INT8/FP8/TQ4), Qwen3.5/3.6 only — DSv4 MLA KV is already FP8-packed and rejects the flag | Yes (INT8 default via MLX affine groups; BF16 fallback) | No | No | No |
| Radix prefix cache + tiered KV (T0–T3) | Yes (T0 prod; T1–T2 Beta; T3 stub) | Beta (prefix reuse via snapshots; T2 local-SSD write-through) | No | No | No |
| Speculative decode | MTP for DSv4/Qwen3.6 (depth-2 ~1.03× net-win on H20); else plumbing only | Beta (DFlash for Qwen3.5; NextN/MTP shipped for Qwen3.6, +44% tok/s) | No | No | No |
| Multi-GPU TP/PP/EP | TP=8 / EP=8 live (DSv4: DeepGEMM + DeepEP); PP not wired | No | No | No | No |
| OPD teacher surface | Yes | No | No | No | No |
| OpenAI HTTP (`/v1/chat/completions`, SSE) | Yes | Yes | Yes (synthetic) | Wired (validation pending-remote) | Wired (device pending) |

The HIP/Vulkan columns describe the experimental AIPC lane (#76/#77): seam
impls compile and test on any host (device layers stub off-feature); HIP
device validation is pending-remote, Vulkan device execution pends the
shader ABI. The lane started ahead of strategy v2 Phase 3 ordering —
ratification pending.

Evidence pointers:

- Backend tiers: support-matrix §1
- Model reach: support-matrix §3
- Quant / KV: support-matrix §4, §4b
- Spec decode: support-matrix §4a
- Multi-GPU: §Multi-GPU below + `crates/infer-cuda/src/{tp,deepep,moe,dsv4}.rs`

## Change Impact Map

Minimum verification after touching a layer. Runtime hot-path changes also
require a dated entry under `docs/experience/wins/` or `errors/` per
[`AGENTS.md`](../AGENTS.md) §Benchmarks.

| Layer touched | Minimum verify | Notes |
| --- | --- | --- |
| `crates/cuda-kernels/csrc/` or `crates/cuda-kernels/src/` | `cargo test --release -p cuda-kernels --features cuda` + the affected `infer-cuda` path | Bench: `scripts/bench_throughput.py` for perf claims; kernel heat map in [`crates/cuda-kernels/AGENTS.md`](../crates/cuda-kernels/AGENTS.md) |
| `crates/infer-core/` (scheduler / RadixCache / chunked prefill) | `cargo test --release -p infer-core` | One `Engine<E,K>` drives both backends — a scheduler change touches all of them |
| `crates/infer-cuda/` (CUDA executor, model, TP/EP, DSv4) | `cargo test --release -p infer-cuda --features cuda` (GPU) | Golden parity validated on the multi-GPU pod, not locally on a Mac |
| KV quant / paged KV gating (`crates/infer-cuda/`) | `cargo test --release -p infer-cuda --features cuda` (GPU) | See AGENTS.md §Build & run |
| `crates/infer-metal/` or `crates/mlx-sys/` | `cargo test --release -p infer-metal --no-default-features --features metal,no-cuda` | Canonical Metal model: Qwen3.6 MoE (AGENTS.md); MLX bridge in [`crates/mlx-sys/AGENTS.md`](../crates/mlx-sys/AGENTS.md) |
| `crates/infer-seam/` or `crates/infer-plan/` (the host-only contract) | `cargo test --release -p infer-core -p infer-api` | A seam-signature change ripples through every executor |
| `crates/agent/`, `crates/cli/`, `crates/chat/` | `cargo test --release -p agent -p cli -p chat` | No GPU required |
| `crates/train/` OPD path | `cargo test --release -p train --features no-cuda --lib` | End-to-end OPD needs CUDA GPU |
| Docs-only | — | State `docs-only` in commit body; no bench gate |

## Multi-GPU Parallel Axes

Multi-GPU lives entirely below the seam, inside `infer-cuda` — the
`infer-core` scheduler stays rank-neutral. The default build is one rank, one
model load, one `Engine` — unchanged; collectives only engage when a
multi-GPU config is selected.

- **TP (tensor parallel):** `crates/infer-cuda/src/tp.rs` — `TpRuntime` /
 `TpConfig` / `resolve_tp_config_from_env`, `all_reduce_sum` post-attn /
 post-MLP. TP=1 is no-op; TP>1 collectives are live (DSv4 prefill verified at
 TP=8). Sharding helpers come from `infer-topo` (`head_shard`, column/row).
- **EP (expert parallel):** `crates/infer-cuda/src/{moe,deepep}.rs` —
 DeepEP `all_to_all` dispatch/combine, gated by the `deepep` feature; routing
 is the backend-neutral `infer-moe`. Live at EP=8 for DSv4.
- **PP (pipeline parallel):** not yet wired into a forward path — the one
 known gap (microbatch ring in `infer-core` + a stage-aware executor).
- **NCCL backend:** `--features cuda,nccl` gate forwards
 `infer-cuda/nccl → cuda-kernels/nccl`; `deepep` implies `nccl`.

DeepSeek-V4-Flash is the binding multi-GPU consumer (TP=8 / EP=8, FP8
DeepGEMM MoE + DeepEP). The DSv4 contract scaffold lives in
`crates/deepseek-spec`. GLM-5.2 (`glm_moe_dsa`, DSv4-V3.2-DSA family, 256
experts) is the in-flight DSv4-family addition riding the same CUDA path via
an adapter — forward tranches landed, verification pending-remote (not
production-verified).

DSv4 decode is under active kernel optimization on 8×H20 (adopt-best-first):
gated, license-or-kill on a same-load resident A/B at the B=1 SLO shape, with
KV-precision parity as the
precondition for any default flip. Landed gated: FlashMLA fused sparse decode,
FP8 fused `wqkv_a`, contiguous active-row MoE layout.
Prefill at production shapes is in repair (a MoE padded-layout i32 work-size
overflow at >~1560 tokens).

Qwen3.6 hybrid+MoE now serves on CUDA (FP8 MoE via DeepGEMM, batched paged
decode — no longer Metal-only); the next DSv4-family in-flight addition is
GLM-5.2 (verification pending-remote). Current model support status lives in
[`docs/support-matrix.md` §3](support-matrix.md#3-model-family-matrix).

## Speculative Decode Framework

Speculative control flow is executor-owned rather than driven by
`ForwardMode::{Verify,Draft}`. CUDA implements DSv4 MTP/DSpark and Qwen
MTP/DSpark; Metal implements DFlash/NextN. Model support, batching, enablement,
and defaults are backend-specific: DSv4 MTP is explicit opt-in; Qwen
multi-request DSpark/MTP loops per row; DSv4 has a separate cross-slot batched
path; compatible Metal DFlash/NextN uses its advertised row cap. See
[support-matrix.md §4a](support-matrix.md#4a-speculative-decoding-matrix).

The historical caveats still bound any port:

- The first end-to-end real-spec bench regressed -62.8% because the
 correctness-first verifier ran the target paged decode once per verifier
 position; a packed K+1 verifier (or MagicDec sparse-KV self-spec) is the
 prerequisite for a throughput lift.
- For Qwen3.5 / Medusa the gate is recurrent-state rollback: paged KV can be
 truncated, but hybrid linear-attention recurrent state needs a model-owned
 accepted-length commit/rollback.

## Route-A Note (Historical)

The 2026-04-15 Route-A refactor folded an earlier experimental `infer-core`,
`infer-observability`, `infer-policy`, and `infer-engine` set back into the
(now-deleted) monolithic `infer` crate because that split never achieved real
independence. A follow-up the same day deleted the legacy `agent_engine.rs`
after confirming every `Agent*` type duplicated a corresponding `Completion*` /
`InferenceEngine` type.

> Note: the 2026-04-15 `infer-core` named here is unrelated to today's
> `crates/infer-core` (the device-neutral scheduler in
> [§Engine Core, Seam, and Executors](#engine-core-seam-and-executors)). The
> 2026-04-15 crate was dissolved into the old monolith; the current
> `infer-core` is the fresh, independent device-neutral scheduler from the
> PR #53 rewrite. Same name, different crate.

The lesson carried into the rewrite: one contract serves both the HTTP server
and the agent CLI. Today that single front door is `infer-api`'s
`LoadedInferenceEngine` / `EngineLoadConfig`, with model resolution in
`infer-util`'s `hf_hub` and the OpenAI HTTP surface in `infer-server`.

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

- **No `infer-ops` crate.** Ops are tightly coupled to model data layouts and
 live inside each executor (`infer-cuda` / `infer-metal`).
- **Scheduler extraction already done.** The PR #53 rewrite extracted the
 scheduler into `infer-core` cleanly by pushing all device coupling
 (`PagedKVPool`, TileLang metadata, model-specific bootstrap) below the
 host-only seam into the executors. Do not re-couple the scheduler to a
 backend — that is the regression this split exists to prevent.
- **No `infer-runtime-api` trait crate beyond what exists.** The runtime
 contract is already the `infer-seam` traits (`BackendExecutor` + `KvPool`)
 plus the `infer-api` front door; a further trait crate would be redundant.
- **No `*-sys` / Rust-types split for the kernel crate.** One crate holds
 both layers; splitting them creates a `*-sys` boundary with one consumer.
- **No separate CPU backend crate.** The smoke executor still reuses the
 feature-free placeholder `MetalExecutor`, but the host paged KV allocator is
 now the shared `infer-seam::HostPagedKvPool`, not a Metal-owned pool.

The original kernel-crate trip wires (T1 NCCL, T2 FA-3, T3 MLA/FP8 GEMM,
T4 spec decoding, T5 second external consumer) are arguments for the
**next** extraction boundary — whichever one, if any, eventually peels
scheduler or model layers out. They are not arguments about the kernel
crate itself.
