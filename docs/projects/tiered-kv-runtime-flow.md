# Tiered KV Runtime Flow

## Purpose

This file is the runtime-facing companion to
[tiered-kv-cache.md](./tiered-kv-cache.md) and
[tiered-kv-hicache-readmission.md](../plans/tiered-kv-hicache-readmission.md).
It answers one narrower question:

- which module owns each part of the local / cluster-shared KV path
- which scheduler branch fires first
- what the fallback path is when a staged hit cannot progress

The descriptions below are intended to match the current implementation in:

- `infer/src/scheduler/cuda/runtime.rs`
- `infer/src/scheduler/cuda/core.rs`
- `infer/src/prefix_cache.rs`
- `infer/src/kv_tier/{backend,coordinator,policy,readmission,transport/*}.rs`

## Architecture Graph

```text
                                         request submit
                                              |
                                              v
                    +------------------------------------------------------+
                    | CUDA scheduler runtime                               |
                    | runtime.rs / execution.rs / core.rs                  |
                    | - assign_slots()                                     |
                    | - plan_step()                                        |
                    | - step()                                             |
                    | - cleanup()                                          |
                    +-----------------------------+------------------------+
                                                  |
                                prefix lookup / stage classify / publish
                                                  |
                                                  v
                    +------------------------------------------------------+
                    | RadixCache / CacheIndex                               |
                    | prefix_cache.rs                                       |
                    | - lookup_or_stage()                                   |
                    | - publish metadata / tier location / block refs       |
                    +-------------------+-------------------+---------------+
                                        |                   |
                           ready on T0  |                   | staged below T0
                                        |                   |
                                        v                   v
                    +---------------------------+   +---------------------------+
                    | paged_kv / scheduler T0   |   | ReadmissionPlan           |
                    | direct attach + tail COW  |   | readmission.rs            |
                    +-------------+-------------+   +-------------+-------------+
                                  |                                 |
                                  |                           submit_fetch()
                                  |                                 |
                                  v                                 v
                    +------------------------------------------------------+
                    | Coordinator / Orchestrator                           |
                    | coordinator.rs                                       |
                    | - PlanQueue                                          |
                    | - FetchQueue                                         |
                    | - StoreQueue                                         |
                    +----------------------+-------------------------------+
                                           |
                                fetch/store via CacheIO
                                           |
                 +-------------------------+--------------------------+
                 |                         |                          |
                 v                         v                          v
         +---------------+       +------------------+      +----------------------+
         | T1 host pinned|       | T2 local disk    |      | T3 cluster-shared    |
         | HostPinnedPool|       | DiskStore        |      | ClusterSharedBackend |
         | host_pool.rs  |       | transport/disk.rs|      | backend.rs           |
         +-------+-------+       +------------------+      +----------+-----------+
                 |                                                       |
                 +-------------------- promote / restore -----------------+
                                              |
                                              v
                                  +---------------------------+
                                  | T0 GPU pages              |
                                  | paged_kv.rs               |
                                  | attach_pages / copy_from  |
                                  +---------------------------+

                    +------------------------------------------------------+
                    | Emit worker                                           |
                    | core.rs::spawn_emit_worker()                          |
                    | - UTF-8 decode / delta emission                       |
                    | - stop-sequence scan                                  |
                    | - gate result channel back to scheduler               |
                    +------------------------------------------------------+
```

## Scheduler Decision Order

The canonical order lives in `runtime.rs::build_prefix_admission_plan()` and
`runtime.rs::assign_slots()`.

## Scheduler Iteration Timeline

The live CUDA scheduler is iteration-level, but it no longer performs decode
launch and decode readback in the same logical turn.

Current order inside `runtime.rs::run()` is:

1. drain newly arrived requests
2. drain coordinator fetch/store completions
3. if every active request is parked in `Phase::WaitingFetch`, block on a
   coordinator event with a short timeout instead of busy-spinning
4. `assign_slots()`
5. `step()`
6. `cleanup()`
7. `spill_host_blocks_if_pressured()`

Current order inside `execution.rs::step()` is:

1. read back the previous iteration's pending decode
2. wait for any outstanding stop-sensitive emit gate results
3. plan the next batch shape (`Idle` / `DecodeOnly` / `Mixed` / `PrefillOnly`)
4. launch the next decode batch or prefill chunk
5. dispatch newly materialized decode tokens to the emit worker

That keeps `pending_decode` alive across loop turns so CPU-side intake and
admission can overlap the previous GPU decode launch, while text decode /
streaming stays off the scheduler hot path.

### 1. Lookup first

`RadixCache::lookup_or_stage()` classifies each matched block as:

- `ReadyOnGpu`
- `StagingFromHost`
- `StagingFromDisk`
- `Miss`

It also returns `recompute_advised` when the current bandwidth model says
recompute is cheaper than staged fetch.

### 2. Direct GPU attach beats every slower-tier path

Trigger:

- model uses paged prefill
- at least one matched block is `ReadyOnGpu`
- all matched blocks are runnable on T0
- `recompute_advised == false`
- no staged plan is needed

Path:

- scheduler stores `attached_prefix_blocks`
- request enters `Phase::Prefilling`
- `step_new()` and normal prefill resume immediately

Implementation:

- `runtime.rs::build_prefix_admission_plan()`
- `runtime.rs::assign_slots()`
- `core.rs::attach_gpu_prefix_blocks()`

### 3. Staged readmission is second

Trigger:

- model uses paged prefill
- at least one matched block lives in T1/T2/T3
- `recompute_advised == false`

Path:

- scheduler builds `ReadmissionPlan`
- request enters `Phase::WaitingFetch`
- runtime submits a `FetchTicket`
- coordinator loads bytes into T1 host memory
- `FetchCompleted` promotes bytes into T0
- request re-enters `Phase::Prefilling`
- if all active work is parked in `Phase::WaitingFetch`, `run()` sleeps on the
  coordinator event channel instead of spinning a hot scheduler loop

Implementation:

- `core.rs::build_staged_prefix_plan()`
- `readmission.rs::ReadmissionPlan`
- `runtime.rs::assign_slots()`
- `runtime.rs::drain_coordinator_events()`
- `runtime.rs::promote_fetched_prefix()`

### 4. Same-slot contiguous reuse is third

Trigger:

- no direct GPU attach
- no staged plan
- radix hit is already on T0
- a free slot still materializes that prefix in contiguous state

Path:

- scheduler reuses the free slot
- request enters `Phase::Prefilling`

Implementation:

- `runtime.rs::best_reusable_slot_for_radix_hit()`
- `runtime.rs::assign_slots()`

### 5. Cold prefill is the only fallback

Cold prefill fires when any of these are true:

- `lookup_or_stage()` returns no reusable blocks
- `recompute_advised == true`
- staged plan is invalid
- fetch queue is backpressured
- fetch submit returns `None`
- fetch later fails
- promotion back into T0 fails

There is intentionally no second parked admission path.

Implementation:

- `runtime.rs::fallback_to_cold_prefill()`
- `runtime.rs::fallback_to_cold_prefill_without_release()`

## Read Path Outcomes

### Hit path: T0

1. `lookup_or_stage()` returns `ReadyOnGpu`
2. scheduler direct-attaches or same-slot reuses
3. request continues without coordinator I/O

### Hit path: T1/T2/T3

1. `lookup_or_stage()` returns `StagingFromHost` / `StagingFromDisk`
2. scheduler builds a `ReadmissionPlan`
3. coordinator fetches into T1
4. runtime promotes into T0
5. request resumes normal prefill

### Failure path

Any failure after `Phase::WaitingFetch` resolves to:

1. release any held radix references
2. clear staged state
3. restart from cold prefill

This keeps one runnable path instead of parallel recovery branches.

## Write / Demotion Order

The write-side trigger currently starts in
`core.rs::spill_host_blocks_if_pressured()`, but it submits a single
`StoreRequest` path. There is no separate runtime `Spill*` queue surface
anymore.

### 1. Only T1 pressure starts store work

Trigger:

- `host_pool_usage_fraction() > t1_host_pinned_high_water`
- coordinator store queue is not already backpressured

### 2. Candidate selection

Source:

- `prefix_cache.select_blocks_with_policy(SessionBiasedLru, ..., Tier::HostPinned)`

Hard gates:

- block has fingerprint
- block has a valid host region
- no store for that block is already in flight
- if another block already submitted the same `fingerprint + target`, scheduler
  joins that existing `StoreTicket` instead of enqueueing a duplicate write

### 3. Store target selection

`TieredKvPolicy::choose_store_target()` decides:

- `Disk` when no cluster-shared backend exists
- `Remote` only when cluster-shared backend exists and policy allows it

Current remote policy:

- `WriteThroughSelective`
- requires `hit_count >= remote_store_min_hits`
- requires store queue not soft-saturated

### 4. Store completion

`runtime.rs::drain_coordinator_events()` handles:

- `StoreQueued` -> `mark_block_storing`
- `StoreCompleted` -> `mark_block_stored` + release T1 region
- `StoreFailed` -> `mark_block_store_failed`

## Queue Ownership

### Scheduler owns

- request lifecycle
- slot assignment
- fallback decisions
- T0 attachment / promotion

### Coordinator owns

- queue accounting
- cancellation bookkeeping
- fetch/store execution
- translation between `StoreTarget` and concrete backend calls

### Backends own

- byte persistence / retrieval
- backend-local existence checks
- descriptor encoding / decoding

## Cluster-shared Backend Notes

Current repo-local shape:

- scheduler config carries `cluster_shared_backend`
- scheduler builds a `ClusterSharedBackend`
- coordinator uses one backend surface for remote `exists/store/fetch`
- shared-fs is the only fully functional cluster-shared backend
- NIXL is still a compile-checked stub behind `rdma-nixl`

This is intentional: the control-plane path is already unified even though the
real RDMA data plane is still gated on external runtime dependencies.
