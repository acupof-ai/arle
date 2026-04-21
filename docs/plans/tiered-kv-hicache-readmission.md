# Tiered KV Cache — HiCache-aligned readmission / queue / backend plan

**Status**: active — Phases A/B/C/D landed locally on the CUDA lane, and Phase E now has a minimal shared-filesystem cluster-shared backend wired through the same coordinator fetch/store path; remote CUDA validation still pending  
**Scope**: implementation SSOT for the current local readmission tranche plus the remaining design for remote/shared backends  
**Purpose**: keep the local path (`RadixCache + paged_kv + Zig T1 arena + readmission + T1→T2 spill`) on one clean architecture while the remaining queue/backpressure/cluster-L3 work lands without reviving parallel side paths.

This doc records the target shape for:

- live `L2/L3 → L1` readmission
- a strict control-plane / data-plane split
- one canonical chunk abstraction for transfer and persistence
- asynchronous prefetch / write-back pipelines
- a cluster-shared L3 backend contract

Current implementation truth stays in [`../projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md) and [`../../infer/src/kv_tier.rs`](../../infer/src/kv_tier.rs). As of 2026-04-21, the local CUDA lane already ships:

- `KVBlock / KVSpan / KVHandle` control-plane objects
- `KVPayload*` and `KVBackend*` data-plane surfaces
- local `lookup_or_stage -> ReadmissionPlan -> FetchTicket -> WaitingFetch -> promote_fetched_prefix`
- explicit `plan/fetch/store` queue vocabulary in the coordinator
- local `DiskStore` node-local backend + remote `NixlTransport` stub

## 1. Why this tranche exists

The current local path is honest but incomplete:

- `RadixCache` already owns prefix metadata and tier metadata
- `paged_kv` already owns T0 page attachment and tail-page COW
- `HostPinnedPool` is already Zig-backed for T1 storage
- `Coordinator + DiskStore` already own T1→T2 persistence and local staged fetch preparation

The missing half is the rest of the HiCache loop:

- a canonical async pipeline for prefetch and write-back
- a backend boundary that can grow from local disk to cluster-shared L3
- cancellation / retry policy beyond the current local fetch/store path

Without that split, every attempt to add readmission risks coupling scheduler admission directly to disk or RDMA I/O.

## 2. Tier model and physical mapping

We use two overlapping views and keep them explicit:

### 2.1 Physical hierarchy

| Level | Medium | Ownership |
|---|---|---|
| `L0` | GPU SRAM / registers / shared memory | kernel-private, not orchestrated by `kv_tier` |
| `L1` | GPU HBM paged KV pool | `paged_kv` + scheduler |
| `L2` | CPU pinned DRAM | `HostPinnedPool` + CacheIO |
| `L3` | NVMe / remote shared storage | CacheIO backend |

### 2.2 Software tiers

| Tier | Physical level | Notes |
|---|---|---|
| `T0` | `L1` GPU HBM | block/page-attached decode working set |
| `T1` | `L2` host pinned DRAM | promotion / demotion buffer, instance-private |
| `T2-local` | `L3` local NVMe | content-addressed local durability layer |
| `T2-remote` | `L3` remote shared backend | cluster-shared KV warehouse |

`L0` is intentionally outside the orchestrator. It is a kernel/runtime concern, not a cache-controller concern. The tiered-KV runtime starts at `L1`.

## 3. Canonical data-flow rules

### 3.1 Read path

Read flow is always:

`Index lookup -> highest available tier hit -> asynchronous promotion upward until runnable at T0`

Concrete rules:

1. `L1/T0` hit:
   - return immediately
   - attach pages directly when the prefix is already materialized in the paged pool
2. `L2/T1` hit:
   - enqueue a prefetch job
   - promote into T0
   - allow layer-overlap later, but the contract is "bytes become runnable at T0"
3. `L3/T2-local or T2-remote` hit:
   - enqueue fetch into T1 first
   - then promote from T1 to T0
   - the scheduler sees staged readiness, not raw disk/RDMA I/O
4. full miss:
   - cold prefill

The scheduler never issues storage or RDMA calls directly. It only reacts to staged readiness and backpressure signals.

### 3.2 Write path

Write flow is always:

`decode/prefill materializes new KV in T0 -> policy decides retention -> asynchronous demotion / write-through / write-back`

Concrete rules:

1. New tokens are always produced into `T0/L1`
2. Partial decode tails remain `T0`-only until block/page sealing rules permit publish
3. Published sealed spans may:
   - stay hot in `T0`
   - demote to `T1`
   - write-through or write-back into `T2`
4. Slower-tier persistence is asynchronous and cancellable

## 4. Minimal maintainable split: four packages

The next implementation tranche should be built around four strict modules.

### 4.1 CacheIndex

Pure metadata. No byte transport.

Responsibilities:

- `HiRadixTree` / radix wrapper
- prefix lookup
- version / epoch checks
- TTL / hit count / last-access tracking
- placement metadata per span
- snapshot / restore of metadata only

Canonical shape:

`key -> {tier, location, span/page_ids, version, ttl, size, ref_cnt, last_access_ts, checksum}`

Required operations:

- `lookup(prefix)`
- `insert(span_meta)`
- `update_location(span_key, new_location)`
- `mark_pending(span_key, transfer_kind)`
- `complete_transfer(span_key, result)`
- `evict(plan)`
- `snapshot()`

The index must never parse or move raw KV bytes.

### 4.2 CacheIO

Pure data movement. No scheduling policy.

Responsibilities:

- fetch and store chunk payloads
- serialization / validation / checksums
- local disk, local host pool, remote backend adaptation
- optional compression in slower tiers
- dedupe of concurrent fetch/store for the same chunk

Required operations:

- `fetch(chunk_ref, target_tier)`
- `store(chunk_ref, source_tier)`
- `exists(chunk_key)`
- `abort(op)`
- `poll(op)`

This is the only layer allowed to touch raw payload bytes.

### 4.3 CachePolicy

Pure decision logic. No byte movement.

Responsibilities:

- admission to slower tiers
- eviction ordering
- prefetch window / concurrency / timeout / cancellation rules
- cost-aware scoring

Inputs:

- hit counts
- reuse probability hints
- prefill cost
- bytes to move
- queue depth / backpressure
- current tier occupancy

Outputs:

- `should_admit`
- `prefetch_priority`
- `write_policy`
- `eviction_plan`

### 4.4 CacheOrchestrator

Queue + state machine layer that runs the lifecycle end to end.

Responsibilities:

- connect scheduler requests to index lookups and IO jobs
- drive `miss -> prefetch -> ready -> decode -> store`
- own cancel / timeout / idempotency / rollback behavior
- aggregate per-queue metrics

The scheduler should ask the orchestrator for readiness, not orchestrate byte motion itself.

## 5. Canonical object model: `KVBlock` / `KVSpan` / `KVHandle`

The next tranche should stop overloading `BlockId` to mean both pool identity and transport identity.

### 5.1 `KVBlock`

The smallest transfer/storage unit.

Required fields:

- `block_id`: stable in-process block identity
- `layer_range`: which transformer layers the block covers
- `token_range`: which logical tokens it covers
- `dtype/shape`: `K/V`, heads, `head_dim`
- `bytes`

`KVBlock` is the unit used by allocators and by raw payload fetch/store.

### 5.2 `KVSpan`

A continuous prefix segment backed by one or more blocks.

Required fields:

- `span_id`
- `prefix_hash`
- `blocks: [KVBlockRef...]`
- `epoch/version`

`KVSpan` is closer to the radix-tree edge model than a bare block list.  
Index operations should reason in spans; allocators should reason in blocks.

### 5.3 `KVHandle`

The control-plane reference passed around between scheduler/orchestrator/index.

Required fields:

- `span_id`
- `layer`
- `location_ptr`
- `epoch`
- `size`

Rule:

- the control plane passes `KVHandle`
- the data plane resolves `KVHandle` into actual bytes/pages/descriptors
- no control-plane API should ship large payload objects

## 6. Canonical lookup and write interfaces

These interfaces are pseudocode-level SSOT for the coming refactor.

### 6.1 Read path

Unified entry point:

```python
class KVCache:
    def lookup(self, req: RequestCtx, prefix_tokens) -> LookupResult
```

`LookupResult` should carry:

- `hit_layer`: `L0 | L1 | L2 | L3 | MISS`
- `handle`: one or more `KVHandle`
- `prefetch_future`: optional future when the hit is below runnable GPU level

Implementation split:

```python
CacheIndex.lookup(prefix_hash, epoch) -> IndexEntry | None
CachePolicy.on_hit(entry) -> PromotePlan | None
CacheOrchestrator.enqueue_fetch(plan) -> future
```

Promotion must be explicit:

```python
class CachePolicy:
    def promote_plan(self, entry, target_layer) -> FetchPlan
```

### 6.2 Write path

Write path is split into synchronous hot writes and asynchronous background work.

#### Synchronous hot write

```python
class KVCache:
    def append_kv(self, req, new_token, kv_tensor) -> None
```

Hot-path guarantees:

- write KV into `L1/T0`
- update the request's current block pointer
- do **not** touch `L2/L3` IO

#### Asynchronous background work

```python
class CacheOrchestrator:
    def schedule_store(self, span, src_layer, dst_layer)
    def schedule_demotion(self, span)
```

This is where write-through / write-through-selective / write-back live.

## 7. Canonical chunk abstraction

The system already has:

- `BlockId`: ephemeral pool slot identity
- `BlockFingerprint`: durable semantic identity
- `Vec<u32>` page ids inside `paged_kv`

That is not enough for live multi-tier IO. We need one transport/persistence unit that is independent of the current slot.

### 7.1 Proposed transport/persistence types

#### `KvChunkKey`

Stable semantic identity.

Fields:

- `fingerprint`
- `model_epoch`
- `kv_format_tag`
- `block_tokens`

This is what local disk and remote backends key on.

#### `KvChunkRef`

Index-visible metadata for one chunk.

Fields:

- `key: KvChunkKey`
- `tier`
- `location`
- `token_span`
- `page_ids` when resident in T0
- `byte_len`
- `checksum`
- `version`

#### `KvChunkPayload`

Data-plane payload view.

Fields:

- raw bytes or typed descriptor
- source tier
- target tier
- optional host-region / shm / mmap descriptors

### 7.2 Why chunk, not slot

`BlockId` is allocator-local and dies with the current pool layout.  
`KvChunkKey` survives:

- restart
- re-pool
- cross-process restore
- cross-node sharing

The scheduler can still map runnable chunks back onto fresh pages in T0, but persistence and readmission must stop pretending that the live slot is the identity.

## 8. Queue model: three explicit queues

Each queue gets its own worker pool and metrics.

Every queued item should be one `SpanTask` keyed by:

- `span_id`
- `epoch`
- destination layer

That key is what queue-level dedupe operates on.

### 8.1 `PrefetchPlanQueue`

Lightweight planning only.

Responsibilities:

- query `CacheIndex`
- compute missing spans
- decide whether to recompute or fetch
- assign priorities
- dedupe requests that map to the same chunk

Inputs:

- request prefix
- current queue depth / backpressure
- tier occupancy
- policy knobs

Outputs:

- fetch plan
- immediate hit
- recompute decision

### 8.2 `FetchQueue`

Heavy IO queue.

Responsibilities:

- `T1 -> T0`
- `T2-local -> T1`
- `T2-remote -> T1`
- batched/merged transfers where possible
- inflight dedupe by `KvChunkKey`

Required behavior:

- futures/promises for waiters on the same chunk
- explicit cancel propagation
- bounded concurrency and backpressure

### 8.3 `StoreQueue`

Heavy IO queue for slower-tier persistence.

Responsibilities:

- `T0 -> T1`
- `T1 -> T2-local`
- `T1 -> T2-remote`
- async write-through / write-through-selective / write-back

Required behavior:

- best-effort operation for non-critical persistence
- dedupe on repeated writes of the same hot chunk
- policy-controlled dropping / downgrading under pressure

### 8.4 Required queue semantics

All three queues must support:

- **dedupe**: one `(span_id, epoch, dst_layer)` task in-flight at a time
- **abort/cancel**: cancelled requests must cancel queued work; in-flight work must short-circuit as soon as the backend permits
- **backpressure**: queue depth and backend bandwidth pressure must feed back into admission/prefetch policy

Without these three, the queue split only moves the bottleneck around.

## 9. Request and store state machines

### 9.1 Request chunk state

Every request-visible chunk should move through explicit states:

`NEW -> PLANNED -> FETCHING -> READY -> CONSUMED`

Interpretation:

- `NEW`: request admitted, no decision yet
- `PLANNED`: index lookup complete, fetch/recompute decision made
- `FETCHING`: an inflight IO op exists
- `READY`: runnable in T0
- `CONSUMED`: decode/prefill used the chunk

### 9.2 Store state

Persistence is independent:

`STORE_PENDING -> STORING -> STORED | FAILED`

Important rule:

- read readiness and write-back readiness must not be represented by one state variable

### 9.3 Observable timestamps

Every state transition should emit:

- queue wait time for `Q1/Q2/Q3`
- fetch latency bucketed by backend
- bytes moved
- pages moved
- abort / retry counts
- backpressure events

Without explicit state transitions, readmission debugging becomes guesswork.

## 10. Index metadata and two-phase commit

Every `IndexEntry` must include:

- `span_id`
- `prefix_hash`
- `layer`
- `location_ptr/page_ids`
- `ref_cnt`
- `last_access_ts`
- `size_bytes`
- `epoch`
- `state: READY | PENDING | EVICTING`

Suggested interface:

```python
class CacheIndex:
    def lookup(self, prefix_hash, epoch) -> Optional[IndexEntry]
    def pin(self, entry) -> None
    def unpin(self, entry) -> None
    def insert_pending(self, span_meta) -> IndexEntry
    def commit_ready(self, entry, location_ptr) -> None
```

Required invariant:

- write path first publishes `PENDING`
- data movement/store completes
- only then does the index flip to `READY`

That is the minimal defense against half-written KV becoming readable.

## 11. Policy model

### 11.1 Admission

Admit to slower tiers only when policy says the chunk is worth preserving.

Candidate inputs:

- access frequency / hit count
- estimated recompute cost
- tenant/session affinity
- object size
- target tier pressure

### 11.2 Eviction

Baseline stays tier-aware LRU, but the policy interface must allow cost-aware ranking.

Required inputs:

- last access
- hit count
- byte size
- prefill cost
- whether chunk is hot in multiple sessions
- queue pressure

Suggested interface:

```python
class CachePolicy:
    def maybe_evict(self, layer_state) -> List[EvictCandidate]
    def score(self, entry) -> float
```

Watermarks must stay per-layer and configurable:

- `high_watermark` triggers eviction
- `low_watermark` stops eviction

Eviction invariants:

- `ref_cnt > 0` cannot evict
- `state != READY` cannot evict

### 11.3 Prefetch

Prefetch must be configurable per workload:

- `best_effort`
- `wait_complete`
- `timeout`

These modes align with the public SGLang HiCache control-plane description and should be first-class policy values, not hidden heuristics.

### 11.4 Write-back

Required write modes:

- `write_through`
- `write_through_selective`
- `write_back`

`write_through_selective` needs hit-count or hotness thresholds in the index metadata.

## 12. Memory layout and allocators

The image-level storage guidance should map into code as explicit allocators, not implicit side effects.

### 12.1 Layout rules

- block/page-oriented transfer unit
- `K` and `V` remain physically separate
- GPU layout stays kernel-friendly
- host/storage layout stays transport-friendly

The data-plane layer is allowed to choose different physical layouts per tier as long as `KVHandle -> payload` remains stable.

### 12.2 Allocator boundary

Each tier gets an explicit allocator:

```python
class LayerAllocator:
    def alloc(self, n_blocks) -> Location
    def free(self, location) -> None
```

Concrete mapping:

- GPU: HBM allocator with page/block alignment and dtype-aware layout
- CPU: pinned host allocator, DMA-friendly
- NVMe: page-file or object-key allocator, sequential-write oriented

`CacheOrchestrator` and `CacheIO` consume allocators; `CacheIndex` does not.

## 13. Continuous batching and prefix sharing

Prefix merging belongs at the index layer, not at generation-time state mutation.

Required behavior:

1. scheduler groups requests by `prefix_hash`
2. requests sharing one prefix issue one logical lookup/prefetch
3. those requests share one `KVHandle`
4. ref-counting owns lifetime

Suggested batch interface:

```python
def batch_lookup(prefix_hashes: List[Hash]) -> Dict[Hash, LookupResult]
```

This is how continuous batching and multi-layer prefix reuse stay unified instead of becoming separate heuristics.

## 14. Index engineering pitfalls to bake in early

### 14.1 Epoch/versioning

The same textual prefix can become invalid under:

- new weights
- changed KV format
- changed tokenizer
- changed tenant/session scope

The index must carry epoch/version and fail closed on mismatch.

### 14.2 Memory management

Radix nodes should use slab/arena allocation instead of unconstrained heap churn.  
This is especially important once reads become highly concurrent.

### 14.3 Concurrency

Target assumption:

- many readers
- relatively few writers

Preferred shapes:

- shard by prefix hash, or
- RCU / epoch-style reclamation once the mutation rate justifies it

What we should avoid:

- ad hoc mutex layering across scheduler + coordinator + backend handles

### 14.4 Consistency

A chunk must not become readable in the index before data is durable enough for the chosen tier.

Allowed patterns:

- write data first, then publish ready index entry
- or publish `PENDING` and only flip to ready on completion

Disallowed pattern:

- make the index return a supposedly fetchable chunk whose payload is not committed yet

## 15. Metrics: minimum first-class surface

At minimum, the next tranche must expose:

- layered hit rate: `L0/L1/L2/L3`
- queue wait time and inflight for `Q1/Q2/Q3`
- IO throughput and p99 latency bucketed by backend
- promotion/demotion success rate and cancellation rate
- eviction count, bytes evicted, and "blocked by refcnt" count

If these are missing, policy tuning and regression triage will be blind.

## 16. Backend contract for cluster-shared L3

The cluster-shared layer should remain minimal.

Required backend contract:

- `get(key)`
- `exist(key)`
- `set(key, value)`

Optional capabilities:

- `delete(key)`
- `batch_get(keys)`
- `batch_set(items)`
- backend-native zero-copy descriptors

The orchestrator remains responsible for:

- retry
- dedupe
- queueing
- synchronization across request waiters

This follows the public HiCache description: backend integration should stay small while the cache controller owns orchestration complexity.

## 17. Implementation order

### Phase A — control/data-plane split

**Status (2026-04-21 local)**: landed

- carve `CacheIndex`, `CacheIO`, `CachePolicy`, `CacheOrchestrator` boundaries out of current `prefix_cache`, `kv_tier`, and scheduler code
- keep the current local spill/readmission path working

### Phase B — chunk abstraction

**Status (2026-04-21 local)**: landed

- introduce `KvChunkKey` / `KvChunkRef`
- move slower-tier reads/writes off raw `BlockId`

### Phase C — live local readmission

**Status (2026-04-21 local)**: landed on CUDA local (`host/disk -> host -> T0`)

- `T1 -> T0`
- `T2-local -> T1 -> T0`
- explicit `PrefetchPlanQueue` / `FetchQueue`

### Phase D — asynchronous write-back

**Status**: landed locally (`StoreQueue` vocabulary + local async spill/store path + queue cancellation + live local ServerMetrics queue/backpressure surface)

- `StoreQueue`
- `write_through_selective`
- queue metrics and cancellation

### Phase E — cluster-shared backend

**Status**: minimal shared-filesystem backend landed locally; RDMA / NIXL / Mooncake remain planned

- remote backend trait
- shared-filesystem backend implementation
- same coordinator fetch/store path for local disk and shared-fs handles

## 18. Acceptance for the tranche

This design is only considered implemented when all of the following are true:

1. The scheduler never calls disk/RDMA/backend code directly
2. Live T1/T2 readmission is driven by explicit queues and request states
3. Slower-tier persistence uses the canonical chunk abstraction, not raw slot identity
4. Local NVMe and remote backends share the same `CacheIO` contract
5. Queue metrics, aborts, retries, and backpressure are visible
6. The old spill-only/document-only staging story is deleted instead of kept in parallel

## 19. References

Used to align terminology and workflow:

- SGLang HiCache blog: <https://lmsys.org/blog/2025-09-10-sglang-hicache/>
- Mooncake x SGLang HiCache design: <https://kvcache-ai.github.io/Mooncake/design/hicache-design.html>
- SGLang HiCache docs: <https://docs.sglang.ai/advanced_features/hicache.html>

## 20. Image-to-plan mapping

This appendix maps the architecture image's concepts onto this plan's sections so the visual model and the design SSOT stay aligned.

| Image concept | Plan section |
|---|---|
| `L0/L1/L2/L3` layering (`SRAM / GPU HBM / CPU DRAM / NVMe`) | §2 |
| read path: hit high tier, otherwise promote upward | §3.1, §6.1 |
| write path: hot write first, then async demotion / write-back | §3.2, §6.2, §8.3 |
| global index / metadata (`location`, `layer`, `ref_cnt`, `last_access_ts`, `block_size`, `checksum`) | §4.1, §5, §10 |
| tiered LRU + cost-aware eviction | §11 |
| block-oriented memory layout, `K/V` separated, transport-friendly slower tiers | §12 |
| continuous batching + prefix sharing | §13 |
| end-to-end queueing / scheduling / IO observability | §8, §9, §15 |
