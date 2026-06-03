# Backend seam redesign — one device-neutral scheduler, backends as plug-ins

**Status:** design (hypothesis-grade; SGLang-grounded, ARLE-audited). Not yet
licensed for L1+ — see §Open hypotheses.
**Driver:** ckl — "the current architecture does not support scaling horizontally
to HIP and other backends; the architectural design has problems. Optimize
bottom-up, taking SGLang as reference."
**Anchor scan:** [`wins/2026-06-03-sglang-backend-seam-scan.md`](../experience/wins/2026-06-03-sglang-backend-seam-scan.md)

## Problem

Adding a backend today means **writing another scheduler**. `backend/AGENTS.md`
§Extension pattern says so explicitly ("if multi-request, write a scheduler
analogous to `scheduler/cuda/`"). Evidence:

- `scheduler/cuda/` = 13.7k lines, generic over a CUDA-shaped `ModelForward` and
  holding `paged_kv_pool: PagedKVPool` (a CUDA concrete type) directly.
- `backend/metal/scheduler.rs` = a **separate** 1.1k-line `MetalScheduler` that
  reimplements continuous batching, producing `MetalLogicalServePlan`.
- A third backend (AMD HIP/ROCm) would be a third scheduler. O(backends).

Root cause: the backend contract sits at the **wrong altitude**. The seam is
`ModelForward` — a ~50-method god-trait fusing model forward + KV-pool migrate +
`DeviceContext`/`DeviceVec` + CUDA-graph (`cuda_graph_decode_support`) + NCCL
(`ep_nccl() -> NcclGroup`) + CUDA-stream async overlap (`launch_prefill_batch` /
`complete_prefill_batch`, `sample_batch_greedy_launch` / `readback`) + sampling +
spec. A new backend must implement all of it, CUDA-shaped concepts included.

## What SGLang does (grounded: `sgl-project/sglang@3e681d7`)

Device-neutral scheduling **above**, device-specific execution **below**, split
by three narrow contracts:

| Contract | Role | Shape |
|---|---|---|
| `ForwardBatch` + `ForwardMode` | logical plan (data) | `forward_mode` (EXTEND/DECODE/MIXED/IDLE/TARGET_VERIFY/DRAFT_EXTEND/…) + `input_ids`/`seq_lens`/`out_cache_loc`/`positions`/`spec_info` |
| `AttentionBackend` (ABC) | compute seam | `forward_decode/forward_extend/forward_mixed(q,k,v,layer,forward_batch)` + `init_forward_metadata(fb)` + graph hooks — **~8 methods** |
| `KVCache`/`TokenToKVPool` (ABC) | KV memory seam | `alloc(need_size)`/`free(idx)`/`get_kv_buffer(layer)`/`set_kv_buffer(...)`/`get_kv_size_bytes()` |

**24** `AttentionBackend` implementations span NVIDIA (flashinfer/flashmla/trtllm),
**AMD ROCm/HIP (aiter, deepseek_v4_backend_hip_radix, wave)**, **Intel (xpu,
intel_amx)**, portable (triton, torch_native), NPU — all behind **one** scheduler.
HIP is "cuda-alike": same `ModelRunner`, `_is_hip = is_hip()`, `_use_aiter =
SGLANG_USE_AITER and _is_hip` → swap attention backend + kernels, scheduler
unchanged. Sampling, TP/EP collectives, and graph capture are **separate orthogonal
layers**, not folded into the compute seam.

## ARLE gap

| Concern | SGLang | ARLE |
|---|---|---|
| Scheduler | 1 device-neutral impl | 2 impls (`scheduler/cuda` 13.7k + `MetalScheduler` 1.1k); guidance = write a 3rd |
| Compute seam | narrow `AttentionBackend` (~8) | `ModelForward` 50-method god-trait |
| KV memory seam | abstract `KVCache` | `PagedKVPool` (CUDA) held directly by scheduler |
| Logical plan | `ForwardBatch` canonical everywhere | `LogicalServePlan` exists; CUDA shadow-converts from `StepPlan`, Metal reimplements |
| Orthogonal layers | sampling/comm/graph separate | folded into `ModelForward` |

## CUDA-touchpoint audit (the L1 work estimate)

Per-file scan of `scheduler/cuda/` for CUDA-specific symbols
(`paged_kv_pool|PagedKVPool`, `DeviceContext|DeviceVec|backend::cuda|cudarc`,
`decode_bufs|prefill_ctx|*Context|launch/complete_prefill|sample_batch_greedy_*`,
`nccl|synchronize_token|DistributedRequest`, `cuda_graph|warmup_cuda|force_eager`,
`kv_tier|DiskStore|host_pinned|coordinator|tier_policy`, `tilelang`, `nvtx`):

**506 / 13,638 lines (~3.7%) touch any CUDA-specific symbol.** The scheduling
logic is already backend-agnostic in expression; the coupling is thin, concentrated,
and mostly in the *types* (`ModelForward`, `PagedKVPool` field) — not a rewrite.

| coupling | ~lines | concentration | maps to |
|---|--:|---|---|
| tiered-KV substrate | 248 | core.rs 81, admission 34, scheduler_loop 25, fetch 21 | memory layer (mostly device-*neutral*) |
| KV pool | 120 | core.rs 44, spec 20, warmup 13, prefill 10, decode 8 | `KvPool` trait (L0.2) |
| exec ctx / async overlap | 71 | warmup 20, decode 18, spec 16, prefill 9 | `BackendExecutor` (L0.3) |
| nvtx | 24 | nvtx_scopes 7 + scattered | feature-gated profiling |
| dist (NCCL) | 16 | decode 6, admission 4, request 4 | `Communicator` (LayerCommunicator exists) |
| tilelang | 12 | prefill 10 | executor-internal |
| cuda graph | 9 | warmup 7 | executor-internal |

Already 0% CUDA (move as-is): `emit_worker` (251), `core/helpers` (126). Near-0%:
`execution.rs` planner (0.5%, only nvtx), `decode.rs` (3%), `prefill.rs` (3%),
`admission.rs` (4%). The single heaviest coupling is the **tiered-KV coordinator
(248)** — a memory-tier substrate (host-pinned + disk + coordinator) that is
conceptually device-neutral, currently CUDA-coupled.

## Target architecture

```
┌──────────────────────────────────────────────┐
│  Scheduler  (device-neutral, 1 impl)           │  ← scheduler/cuda 13.7k, de-CUDA'd
│  continuous batch / admission / radix / retract │
│  produces ForwardPlan (= LogicalServePlan)      │
└────────────────┬───────────────────────────────┘
                 │ ForwardPlan  (data contract)
┌────────────────▼───────────────────────────────┐
│  BackendExecutor  (compute seam, narrow)         │ execute_prefill/decode/mixed(plan)
│  KvPool           (KV memory seam)               │ alloc/free/page/seq_len/migrate
│  + orthogonal: Sampler / Communicator(LayerComm) │
└──┬───────────────┬────────────────┬─────────────┘
   │CudaExecutor   │MetalExecutor    │HipExecutor    ← kernels + pool only, 0 scheduler
   │+CudaKvPool    │+MetalKvPool     │+HipKvPool
```

## Bottom-up migration (incremental, each step compiles + benches)

Discipline: extract the seam from the **CUDA↔Metal commonality** (two real
backends already produce `LogicalServePlan` — the proven shared contract). HIP only
validates. Do not design HIP-first in a vacuum
([[feedback_no_speculative_interface_shaping]], [[feedback_first_principles_best_practice]]).

- **L0.1** Normalize `ForwardPlan`: `LogicalServePlan` becomes the single plan; CUDA
  planner emits it directly (kill `StepPlan→Logical→candidates` round-trip,
  execution.rs:880-882); delete `unified_scheduler` feature.
- **L0.2** Extract `KvPool` trait from `PagedKVPool`'s alloc/free/page/seq_len/migrate
  surface; CUDA impls it. Smallest, clearest, **first**. Mac-typecheckable.
- **L0.3** Split `ModelForward` god-trait: forward/launch/readback/mixed →
  `BackendExecutor`; sampling → `Sampler`; NCCL → `Communicator` (LayerCommunicator
  exists); graph → executor-internal.
- **L1** De-CUDA the scheduler: `scheduler/cuda/` → `scheduler/`, generic over
  `<B: BackendExecutor, K: KvPool>`; thread the ~506 touchpoints through traits;
  decide tiered-KV substrate stays device-neutral operating on `KvPool`.
- **L2** Collapse the two schedulers: delete `MetalScheduler`; Metal becomes
  `MetalExecutor` on the shared scheduler. **Now ARLE has one scheduler.**
- **L3** HIP: new `HipExecutor + HipKvPool`, zero scheduler code. Abstraction holds →
  done; leaks → back to L0.

Hot-path steps (L1/L2) verify on H20 pod: `bench_guidellm` + `greedy_consistency`
(pending-remote per CLAUDE.md). Non-hot steps: `cargo test --workspace` +
`cargo check -p infer --no-default-features --features cuda,no-cuda`.

## Open hypotheses (license-or-kill with local evidence before L1+)

1. **Tiered-KV (248 touchpoints) placement** — keep the host/disk/coordinator
   substrate in the device-neutral layer operating on `KvPool`, or push into the
   executor? Needs a read of which tier ops are genuinely GPU-specific.
2. **Async launch/readback overlap behind a narrow seam** — ARLE currently fuses
   the CUDA-stream overlap (`pending_decode`/`pending_prefill` across loop turns)
   into the scheduler + `ModelForward`. SGLang keeps it narrow (forward_stream/TBO).
   Verify the overlap survives a `BackendExecutor` seam without latency loss.
3. **CUDA vs Metal `LogicalServePlan` field diff** — determines the `ForwardPlan`
   normalization union. Read both producers.

## Kept / killed from the SGLang scan

- **Kept:** 3-seam split (ForwardBatch / AttentionBackend / KVCache); HIP =
  cuda-alike + swap attention backend; orthogonal sampling/comm/graph layers.
- **Killed for ARLE:** runtime `device`-string dispatch (ARLE = Rust traits +
  compile-time generics / cargo features, not a Python `ModelRunner` with a device
  string); torch device abstraction.
