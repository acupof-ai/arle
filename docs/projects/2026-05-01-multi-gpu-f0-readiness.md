# Multi-GPU F0 readiness assessment

Date: 2026-05-01
Status: Readiness assessment only; no implementation.
Scope: single-node multi-GPU F0. F9 multi-host is explicitly out of scope for
this tranche, even where the existing plan describes a future shared path.

## Source Plan

Primary plan:
`docs/plans/2026-04-28-single-node-multi-gpu.md`

The plan's F0 foundation deliverables are:

- `parallel_state.rs`, `group_coordinator.rs`, and `init_method.rs`, including
  all 10 SGLang-style accessors: world plus TP, PP, EP, attn-TP/DP/CP, and
  MoE-TP/EP/DP groups.
- Full NCCL FFI and a `CollectiveBackend` trait with NCCL as the first backend.
- `TpModelWorker` skeleton with the full rank-axis signature.
- `ForwardBatch.pp_proxy: Option<IntermediateTensors>`.
- `LayerCommunicator` skeleton.
- TCP rendezvous and device selection plumbing, including `INFER_CUDA_DEVICE`.
- Acceptance: unchanged single-GPU path within +/-2%, greedy token identity, a
  CUDA+NCCL build, and a 2-thread `all_reduce(sum)` smoke.

## Current In-Tree State

ARLE already has several useful pieces, but they are not wired into the runtime
path:

- CPU-side TP and multi-axis rank math exists in
  `infer/src/tensor_parallel.rs:34`, `infer/src/tensor_parallel.rs:129`,
  `infer/src/tensor_parallel.rs:166`, `infer/src/tensor_parallel.rs:289`, and
  `infer/src/tensor_parallel.rs:381`.
- `NcclComm` in `infer/src/tensor_parallel.rs:237` is still a local stub:
  `new()` returns a wrapper under `feature=cuda`, while
  `all_reduce_sum_f16()` and `all_gather_f16()` are `todo!()` at
  `infer/src/tensor_parallel.rs:259` and `infer/src/tensor_parallel.rs:268`.
- TCP rendezvous exists and is unit-tested in
  `infer/src/distributed/init_method.rs:41` and
  `infer/src/distributed/init_method.rs:48`; the public distributed module
  states that parallel-state and group coordinators are later work at
  `infer/src/distributed.rs:3`.
- NCCL FFI and `CollectiveBackend` exist in
  `crates/cuda-kernels/src/ffi/nccl.rs:70` and
  `crates/cuda-kernels/src/collective.rs:35`; `NcclBackend::init_rank()` exists
  at `crates/cuda-kernels/src/collective.rs:154`. The Cargo `nccl` feature is
  declared in `infer/Cargo.toml:88` and `crates/cuda-kernels/Cargo.toml:20`,
  but the latter explicitly says linking is deferred.
- Device selection exists for one device via `INFER_CUDA_DEVICE` and
  `DeviceContext::on_device()` at `crates/cuda-kernels/src/tensor.rs:29` and
  `crates/cuda-kernels/src/tensor.rs:61`; the server still constructs the
  default context through `DeviceContext::new()` in `infer/src/main.rs:247`.
- Qwen3/Qwen3.5 model specs already annotate sharding in
  `crates/qwen3-spec/src/lib.rs:70` and `crates/qwen35-spec/src/lib.rs:174`,
  but runtime weight loading does not consume those annotations.
- The CUDA bootstrap still loads one complete model and spawns one scheduler
  thread in `infer/src/backend/cuda/bootstrap.rs:288` and
  `infer/src/backend/cuda/bootstrap.rs:432`.
- Qwen3 and Qwen3.5 safetensor loaders load full tensors with
  `load_tensor_2d()` / `load_linear()` at `infer/src/model/qwen3/weights.rs:119`
  and `infer/src/model/qwen35/weights.rs:124`, without `TpConfig` or
  `Shard`-aware narrowing.
- `LocalCudaTransport` is not an intra-node multi-GPU transport. It only accepts
  GPU <-> host-pinned transfer shapes at
  `infer/src/kv_tier/transport/local_cuda.rs:52`, and `poll()` returns a
  structural-stub error at `infer/src/kv_tier/transport/local_cuda.rs:105`.

## Gap Matrix

| Gap | Plan described | Current code state | Required commit sequence | Est. size |
|---|---|---|---|---:|
| F0.1 distributed parallel state | `parallel_state.rs` with init/dispose and accessors for world, TP, PP, EP, attn-TP/DP/CP, MoE-TP/EP/DP groups. | Only `init_method` is exported; `infer/src/distributed.rs:3` says group coordinators and parallel state land later. Rank math exists separately in `infer/src/tensor_parallel.rs:289`. | 1. Add `distributed/parallel_state.rs` wrapping `MultiAxisConfig`/`RankCoord`. 2. Add typed group descriptors and thread-local accessors. 3. Add CPU tests for all 10 accessor layouts. | 250-400 LoC |
| F0.2 group coordinator | SGLang-style `GroupCoordinator` holding a collective backend and exposing all collectives/P2P methods. | `CollectiveBackend` exists in `crates/cuda-kernels/src/collective.rs:35`, but there is no `GroupCoordinator` in `infer/src/distributed/`. | 1. Add `distributed/group_coordinator.rs`. 2. Bridge raw pointer collectives without leaking CUDA types into cross-backend modules. 3. Wire accessors from `parallel_state`. | 250-350 LoC |
| F0.3 NCCL link and smoke | Full NCCL FFI linked under `--features cuda,nccl`; 2-thread `all_reduce(sum)` smoke passes. | FFI declarations exist at `crates/cuda-kernels/src/ffi/nccl.rs:70`; `crates/cuda-kernels/Cargo.toml:20` says linking is deferred. No smoke test exercises `NcclBackend`. | 1. Add build/link discovery for system NCCL. 2. Use rendezvous to broadcast `ncclUniqueId`. 3. Add CUDA-gated 2-thread all-reduce smoke. | 180-280 LoC |
| F0.4 reconcile `NcclComm` stub | Plan wants `infer/src/tensor_parallel.rs` to re-export real group coordinator, not retain a fake communicator. | `NcclComm` methods are `todo!()` in `infer/src/tensor_parallel.rs:259` and `infer/src/tensor_parallel.rs:268`, while the real trait lives in `cuda-kernels`. | 1. Delete or deprecate `NcclComm` stub. 2. Re-export `GroupCoordinator`/backend types through the planned surface. 3. Update tests to use `CollectiveBackend` path. | 80-140 LoC |
| F0.5 runtime bootstrap shape | Thread-per-rank worker spawn from a single binary, with per-rank device ordinal. | Server path creates one early context in `infer/src/main.rs:247`, loads one model at `infer/src/backend/cuda/bootstrap.rs:288`, and spawns one scheduler thread at `infer/src/backend/cuda/bootstrap.rs:444`. | 1. Add a no-op `MultiGpuRuntimeConfig` defaulting to one rank. 2. Parse `INFER_CUDA_DEVICES`/`INFER_TP_SIZE` for F0 but reject unsupported values until NCCL smoke passes. 3. Introduce rank-thread spawn wrapper without changing the single-rank path. | 250-450 LoC |
| F0.6 `TpModelWorker` skeleton | Full SGLang-shaped worker struct with TP/PP/EP/attn/MoE rank fields and `is_draft_worker`. | No `tp_worker.rs` exists under `infer/src/scheduler/cuda/`; scheduler owns one generic `Scheduler<M>` path. | 1. Add inert `TpModelWorker` data struct and constructor. 2. Feed it from `parallel_state` in tests. 3. Do not move request scheduling yet. | 180-300 LoC |
| F0.7 `ForwardBatch.pp_proxy` slot | `ForwardBatch` with `pp_proxy: Option<IntermediateTensors>` present from F0. | No `infer/src/scheduler/forward_batch.rs`; no `ForwardBatch`, `IntermediateTensors`, or PP proxy references in scheduler/model search. | 1. Add type-only `ForwardBatch` and `IntermediateTensors`. 2. Keep existing forward signatures unchanged until F2, or add an adapter that is unused by default. | 120-220 LoC |
| F0.8 `LayerCommunicator` skeleton | Model-level communicator methods for post-attn/post-MLP AR, DP-attn, CP, and future fused paths. | No `LayerCommunicator`/parallel-linear runtime module exists. Only comments in specs mention SGLang layer classes. | 1. Add `infer/src/model/layer_communicator.rs` with no-op single-rank behavior. 2. Keep Qwen forward call sites untouched until F1. 3. Add tests that single-rank no-op preserves buffers. | 180-280 LoC |
| F0.9 sharded weight loading | F1 consumes Qwen spec `Shard` annotations to narrow safetensors per rank; F0 should pick the first commit boundary. | `Shard` metadata exists in `crates/qwen3-spec/src/lib.rs:70` and `crates/qwen35-spec/src/lib.rs:174`, but loaders call full `load_tensor_2d()` in `infer/src/model/qwen3/weights.rs:119` and `infer/src/model/qwen35/weights.rs:124`. | 1. First landing commit should add a `TpLoadContext` and shard-aware loader helpers behind `tp_size=1` behavior. 2. Next commit narrows Qwen3 weights for TP=2. 3. Then wire all-reduce/logits gather. | 350-700 LoC |
| F0.10 local CUDA KV transport | Plan assumes per-device/runtime buffers can be reasoned about before sharded KV lands. | `LocalCudaTransport` is a tiering stub, not peer or rank transport: `poll()` returns a structural-stub error at `infer/src/kv_tier/transport/local_cuda.rs:105`. | Keep out of F0 critical path. Add a readiness note/test that multi-GPU TP does not depend on this transport. Peer/NVLink KV transfer belongs to later KV-tier/disaggregation work, not F0 TP. | 20-60 LoC |
| F0.11 docs/env completeness | F0 plan lists `INFER_TP_SIZE`, `INFER_PP_SIZE`, `INFER_EP_SIZE`, `INFER_ATTN_*`, `INFER_CUDA_DEVICES`, `INFER_NCCL_PORT`. | `docs/environment.md:160` documents only `INFER_CUDA_DEVICE`; lines `168-171` describe rank threads as future F1+. | 1. Document F0 accepted envs and rejected combinations. 2. Add startup logging showing parsed single-rank/multi-rank config. | 80-140 LoC |

## Stub Versus Wired Verdict

`infer/src/tensor_parallel.rs` is useful CPU-side scaffolding, not runtime TP:
the shard math and group-layout math are real, but there is no call from model
loading, forward, scheduler admission/execution, or CUDA bootstrap. Qwen spec
shard annotations are likewise real metadata but unwired.

`infer/src/distributed.rs` is partial F0: TCP rendezvous is implemented and
tested, but `parallel_state` and `group_coordinator` are absent.

`crates/cuda-kernels/src/collective.rs` is the closest to a real F0 primitive:
the trait and NCCL method bodies exist, but without Cargo/system NCCL linking,
rendezvous-driven unique-id construction, and a GPU smoke, it is not proven
operational.

`infer/src/kv_tier/transport/local_cuda.rs` has not run an intra-node multi-GPU
bench and should not be treated as evidence for single-node multi-GPU readiness.
It is a local GPU <-> host-pinned transport skeleton, not TP communication.

## Phase 2 Spec Decode Interaction

Phase 2 spec decode and single-node F0 can proceed in parallel only at the
planning/type-surface level. They should not both modify scheduler decode
execution in the same commit window:

- P2.3 just landed `spec_enabled` dispatch through `SpecPath` in
  `infer/src/scheduler/cuda/execution.rs`.
- F0 primarily touches bootstrap, distributed state, collectives, and loader
  surfaces. Those can be developed without changing decode semantics.
- F1 and F6 will collide with Phase 2 more directly: F1 introduces
  per-rank logits gather/token broadcast, and F6 introduces draft workers.

Recommendation: run F0 through group coordinator + NCCL smoke + inert
multi-rank config in parallel with P2.4 design, but serialize any changes to
`execution.rs`, `decode.rs`, sampling, or verifier acceptance feedback. For
mainline stability, F0 must preserve the default single-rank behavior exactly.

## First Landing Commit

The first F0 landing commit should be:

`feat(cuda): add nccl group coordinator smoke behind nccl feature`

Rationale:

- It converts the highest-risk foundation from declared API to proven runtime:
  system NCCL link, rendezvous unique-id exchange, `NcclBackend::init_rank()`,
  and one 2-thread all-reduce.
- It does not require touching Qwen forward, scheduler decode, or weight
  loading, so it can run beside Phase 2 without semantic conflict.
- It retires the biggest readiness ambiguity before adding `TpModelWorker` or
  shard-aware loaders.

Only after that smoke passes should F0 add rank-thread bootstrap and then
`TpLoadContext`. Starting with weight loading would create a second model-load
path before collectives are proven; starting with multi-process/bootstrap would
spawn ranks that still cannot communicate.

## F0 Commit Sequence

1. `feat(cuda): add nccl group coordinator smoke behind nccl feature`
   - Files: `crates/cuda-kernels/build.rs`, `crates/cuda-kernels/src/collective.rs`,
     `infer/src/distributed/{group_coordinator,parallel_state}.rs`, one
     CUDA-gated smoke test.
   - Exit: `cargo test -p infer --release --features cuda,nccl <smoke>` passes.

2. `feat(cuda): parse single-node rank topology without changing default path`
   - Files: `infer/src/backend/cuda/bootstrap.rs`, `infer/src/main.rs`,
     `docs/environment.md`.
   - Exit: default `tp_size=1` path logs equivalent config; invalid multi-rank
     configs fail early with actionable errors until the worker path lands.

3. `feat(scheduler): add inert tp worker and forward batch skeletons`
   - Files: `infer/src/scheduler/cuda/tp_worker.rs`,
     `infer/src/scheduler/forward_batch.rs`,
     `infer/src/model/layer_communicator.rs`.
   - Exit: type surfaces compile; no scheduler execution changes.

4. `feat(qwen3): add tp load context in single-rank mode`
   - Files: Qwen3/Qwen3.5 loaders plus common weight helpers.
   - Exit: `tp_size=1` loads byte-for-byte equivalent tensor shapes; no
     collectives called.

5. `feat(cuda): spawn rank threads for tp=2 smoke`
   - Files: CUDA bootstrap and distributed accessors.
   - Exit: two ranks initialize, rendezvous, all-reduce, then shutdown cleanly;
     no HTTP multi-rank serving claim yet.

## Readiness Verdict

F0 is not ready to implement TP serving immediately. It is ready for a narrow
foundation tranche whose first measurable output is a NCCL group-coordinator
smoke. The existing code is valuable prior art, but the runtime path remains
single-GPU: one model load, one scheduler, one default device context, no
sharded weights, no collectives in forward, no multi-rank admission/execution.

Do not start F1 TP dense until the F0 smoke proves:

- `--features cuda,nccl` links on the target machine.
- two rank threads exchange an NCCL unique id via ARLE rendezvous.
- `CollectiveBackend::all_reduce(sum)` produces the expected result.
- the default single-GPU longctx row is unchanged within the F0 +/-2% gate.
