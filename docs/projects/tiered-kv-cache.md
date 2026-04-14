# Tiered KV Cache — hierarchical KV with auto offload and forward-compat RDMA

**Status**: Active — opened 2026-04-13, **revised 2026-04-15** after an
internal survey + 7-system industry comparison exposed three corrections to
the original design. See §13 for the corrections summary and §6 for the
revised execution path (Milestones M0–M5 replace the old P0–P5 phase plan).

**Goal**: Every agent session reuses prior work across requests, survives
memory pressure without OOM, survives process restarts without paying the
cold-prefill tax, and — when we are ready to build multi-node serving —
extends to cross-node KV migration as a new transport impl, not as a rewrite.

This doc is the **implementation spec** for work items A1, B1, and B3 in
[`agent-first-architecture.md`](agent-first-architecture.md). Those three
items share one code topology and must be built against one data-structure
contract; splitting them into independent designs is how we end up with three
incompatible caches. This doc owns that contract.

This doc operates under the Phase-1 PR discipline originally proposed in
[`../archives/art-grade-architecture-for-long-agent-infer.md`](../archives/art-grade-architecture-for-long-agent-infer.md)
(now archived; the crate topology was reverted by Route-A but the PR
discipline still applies): one main topic per PR, structure-before-behavior,
no mixed kernel+scheduler+workspace diffs in the same review.

---

## 1 · Naming

We use the generic industry term, not a branded one. SGLang's "HiCache",
LMCache, and Mooncake Store are products; reusing any of those names would
confuse readers about which project they are looking at.

- **Code**: module `infer/src/kv_tier/` (flat layout: `kv_tier.rs` +
  `kv_tier/`). Canonical types after the 2026-04-15 merge:
  - `RadixCache` + `RadixNode` (was `prefix_cache.rs`) — tree + tier-aware
    metadata. **Also** absorbs what was `TieredKvCache` / `TierDirectory` /
    `BlockDescriptor` in the original design.
  - `Tier`, `TierLocation`, `KVTransport`, `EvictionPolicy` — unchanged
    from the original shape.
  - `BlockId` = `u32` canonical (see §5.1). Content hash is a separate type
    `BlockFingerprint([u8; 16])`, only constructed when persistence or
    cross-node reuse is actually needed.
- **User-facing**: "Tiered KV Cache" or "hierarchical KV cache" — matches the
  vLLM/SGLang vocabulary without claiming implementation parity.
- **Not** `kv_fabric` — collides with `libfabric`/OpenFabrics, which is a
  concrete backend we may call through NIXL later.

---

## 2 · Non-goals

- **New attention kernels.** page_size is a pool/bookkeeping change, not a
  kernel rewrite; everything under `infer/csrc/cuda/` that touches paged KV
  already parameterizes page_size.
- **Metal hierarchy.** MLX unified memory makes T0↔T1 a self-memcpy; Metal
  only joins at M4 for T2 (disk), and only because the wired-memory
  kernel panic in mlx-lm #883 forces us to bound the KV pool somehow.
- **Multi-node scheduling.** M5 keeps the `KVTransport` trait RDMA-ready, but
  cross-node prefill/decode disagg is a separate project that lives on top
  of this one.
- **New storage backends beyond local disk and NIXL.** Mooncake Store, 3FS,
  S3, Valkey are all legitimate post-M5 work; not in scope here.
- **Non-prefix reuse (LMCache CacheBlend).** Only LMCache attempts arbitrary
  substring reuse via attention blending. 6 of 7 other production systems
  do prefix-only reuse. We follow the majority.
- **Replacing the CPU backend.** `infer/src/backend/cpu.rs` remains a 309-line
  synthetic-response smoke-test backend.

---

## 3 · Current state (2026-04-15)

| Area | State | File:line |
|---|---|---|
| CUDA paged pool | `page_size = 1`, token-granular LIFO slot allocator | `infer/src/backend/cuda/paged_kv.rs:7,760` |
| CUDA contiguous legacy KV | Has `k_host/v_host` CPU shadow buffers with `OFFLOAD_BLOCK_SIZE=64`, **default off** | `infer/src/model/kv_cache.rs:130-168` |
| `infer/src/prefix_cache.rs` | ~552 lines, leaf-LRU, refcount pinning, **3 known correctness bugs (§8 items 10-12)**, **no scheduler consumer** | `infer/src/prefix_cache.rs` |
| CUDA scheduler prefix logic | Still `cached_prompts: Vec<Vec<u32>>` single-slot linear compare — **P1(b) never shipped** | `infer/src/scheduler/cuda/runtime.rs:75-151`, `request.rs:437-475` |
| `infer/src/kv_tier/` | 719 lines (id.rs, tier.rs, directory.rs, transport.rs, transport/disk.rs, transport/nixl.rs). `DiskStore` fully implemented + 8 unit tests + round-trips cleanly. **`NixlTransport` stub compiles with `--features rdma-nixl`.** `EvictionPolicy` trait + 4 default impls in `scheduler/policy.rs`. **Every single one: zero production callers.** | `infer/src/kv_tier/**` |
| `BlockId` collision | Three incompatible types exist simultaneously: `kv_tier::BlockId(u64)`, `prefix_cache::BlockId(u32)`, `block_manager::BlockId(u32)`. No `use` or module path disambiguates them. | `infer/src/kv_tier/id.rs:12`, `infer/src/prefix_cache.rs:29`, `infer/src/block_manager.rs:19` |
| `TierDirectory` | `RwLock<HashMap<BlockId, BlockDescriptor>>`, 322 lines, zero production callers. Eviction policy code exists in `scheduler/policy.rs:179-189` but is never invoked. | `infer/src/kv_tier/directory.rs` |
| A2 session_id plumbing | `IncomingRequest::session_id` populated from HTTP; scheduler does not consume it yet | `infer/src/scheduler/types.rs:113-128`, `infer/src/http_server/openai_v1.rs:21-54,222-248` |
| Metal KV pool | `SlotLedger` refcount-only, MLX unified memory, no tier concept | `infer/src/backend/metal/kv_pool.rs` |
| Metal prefix cache | Wraps RadixCache, not wired into scheduler | `infer/src/backend/metal/prefix_cache.rs` |
| Storage deps | `nixl-sys = "1.0"` optional, behind `rdma-nixl` (stub-api) and `rdma-nixl-real` features | `infer/Cargo.toml:39` |
| `infer::scheduler::policy` | `SchedulerSignals { prefix_hit_tokens, session_affinity_slot, turn_depth }` + `AdmissionPolicy` + `ChunkingPolicy` + `PrefixAwareAdmission` + 4 `EvictionPolicy` impls | `infer/src/scheduler/policy.rs` |

Seven facts shape everything below:

1. **The production data path is single-tier today.** No CPU offload, no
   disk tier, no coordinator. The `KVCache::k_host` path exists but is not
   on in serving mode.
2. **`RadixCache` is built but orphaned.** It has three correctness bugs
   flagged in §8 that must be fixed before any scheduler wires go live.
3. **`page_size = 1` is a choice, not a FlashInfer constraint.** FlashInfer
   covers `{1, 5, 8, 16}` in its own tests. vLLM default 16, SGLang default
   64, Mooncake 512. **`page_size=1` is below every production system's lower
   bound** and makes tier-transfer DMA bandwidth impossible to saturate —
   small copies are bottlenecked by DMA engine launch overhead, not throughput.
4. **Three `BlockId` types collide.** The original §5.1 design assumed
   a single unified `BlockId(u64)` with blake3-reduced-to-u64 as content
   hash. The actual code has three different types (u64, u32, u32) that
   never converged. §5.1 is rewritten to resolve this.
5. **`TierDirectory` and `RadixCache` have zero API contact.** The original
   §4 target architecture diagram shows them as two layers with a
   `RadixCache → TierDirectory::resolve` call. That call never existed
   in code. 7 of 7 surveyed production systems (vLLM, SGLang, LMCache,
   Mooncake, Dynamo KVBM, TRT-LLM, TGI) merge radix + tier into a single
   data structure. The project's "two modules zero contact" shape is the
   failure mode; §5.2 is rewritten to merge them.
6. **P1(a) shipped; P1(b) never did.** The 2026-04-13 plan split P1 into a
   structural PR (types + tests) and a behavior PR (scheduler swap). The
   structural PR landed; the behavior PR never did. Every downstream phase
   (P2/P3/P4) was designed against a foundation that was never put in place.
7. **`NixlTransport` trait shape is the right bet.** The Transport trait
   was already revised in 2026-04-13 from `type Completion: Future` to
   `type Op: Send` + explicit `poll()` + `abort()` because NIXL has no native
   Future. Industry research 2026-04-15 confirms: NIXL has a Mooncake plugin
   and Mooncake has a NIXL plugin, and the trait shape matches both. This
   decision survives the 2026-04-15 revision unchanged.

---

## 4 · Target architecture (revised 2026-04-15)

Seven surveyed production systems merge the radix/hash index with the tier
location into a single data structure. The original diagram showed two
layers (`RadixCache → TierDirectory`) with a resolve hop between them; that
shape is not industry-proven and the project never implemented the hop in
code. The revised shape collapses them:

```text
              ┌────────────────────────────────────────────────┐
              │                 RadixCache                     │
              │  (private `Node` struct carries tokens,        │
              │   children, refcount, block_id,                │
              │   tier_location: Cell<TierLocation>,           │
              │   last_access, session_id, soft_pin_until,     │
              │   byte_len, optional fingerprint)              │
              │                                                │
              │   lookup(tokens) → (hit_len, Vec<BlockId>)     │
              │   ref_inc/ref_dec on slot assign / finish      │
              │   evict → free queue (dual residency, §4.3)    │
              └─┬────────┬───────────┬───────────┬─────────────┘
                │        │           │           │
         ┌──────▼───┐ ┌──▼─────┐ ┌───▼────┐ ┌────▼────┐
         │    T0    │ │   T1   │ │   T2   │ │   T3    │
         │  GPU HBM │ │  Host  │ │  NVMe  │ │ Remote  │
         │          │ │ pinned │ │  SSD   │ │ (NIXL)  │
         └────┬─────┘ └────┬───┘ └───┬────┘ └────┬────┘
              │            │         │           │
              │ cudaMemcpy │ io_uring│  NIXL     │
              │   Async    │   disk  │ put/get   │
              │            │         │           │
              └────────────┴─────────┴───────────┘
                      kv_tier::Coordinator
              (OS thread, dedicated CUDA copy stream,
               crossbeam channel — NOT tokio; see §4.4)
```

**Key change from 2026-04-13 shape:** `TierDirectory` / `BlockDescriptor`
no longer exist as separate types. Their fields (`tier`, `location`,
`last_access`, `session_id`, `pin_until`) move onto `RadixNode`. One lookup
returns both the block id and the tier location — no second hop.

### 4.1 Tier semantics (tier numbering updated to industry convention)

| Tier | Medium | Latency class | Who reads/writes |
|---|---|---|---|
| **T0** | GPU HBM | ~0 (kernel direct) | Attention kernels |
| **T1** | Host pinned DRAM | ~10 µs via PCIe copy engine | Coordinator only (never direct kernel access) |
| **T2** | NVMe SSD | 10–100 µs via io_uring / `O_DIRECT` | Coordinator only |
| **T3** | Remote node | 1–50 µs over RDMA via NIXL | Coordinator only; M5+ |

**Tier number change from original**: original doc used T0/T2/T3/T4 with T1
intentionally cut. The revised numbering is T0/T1/T2/T3, matching vLLM,
SGLang HiCache L1/L2/L3 (where L3 is the shared remote tier), Mooncake, and
NVIDIA KVBM. The reason for the rename is alignment with industry
documentation so that cross-system comparison is apples-to-apples; no
semantic change.

T1 on Apple Silicon (MLX / `MTLStorageModeShared`) is a compile-time no-op
because CPU and GPU share one physical DRAM region; "offloading to host" is
a self-memcpy that buys nothing. The Metal backend skips T1 and only
joins at T0+T2 in M4 (see §10).

### 4.2 Invariants

1. **`RadixCache` nodes carry `BlockId` and `TierLocation` together.** One
   data structure, one lookup, one atomic tier transition. The project's
   original "radix tree + separate directory" topology is explicitly
   superseded; no function takes a `BlockId` and needs a second query to
   know which tier it is in.
2. **Only the coordinator moves blocks between tiers.** The scheduler
   emits intents (`Demote`, `Promote`, `Pin`, `Unpin`); the coordinator
   owns the copy stream, the in-flight IO queue, and tier transitions.
   No tier transitions on the scheduler main loop.
3. **`RadixCache` is the single source of truth for `tier`.** A block's
   tier changes atomically at `Cell<TierLocation>` write; pool, transport,
   and eviction code never maintain their own tier bookkeeping.
4. **`BlockId` is a pool-slot identifier, not a content hash.** It lives
   only as long as the block is resident somewhere. For persistence
   (`BlockId` must survive a restart or a node migration), use the
   separate `BlockFingerprint([u8; 16])` content hash — see §5.1.
5. **MR registration stability.** T1 pinned regions are allocated once at
   pool init and never reallocated; this is a precondition for NIXL MR
   registration in M5.
6. **Dual residency (§4.3) is mandatory, not optional.** 5 of 7 production
   systems have it. The 2 that don't (LMCache, DeepSpeed) are not really
   tiered prefix caches in the same sense. A block whose refcount drops
   to zero stays reachable through the radix tree until it is physically
   overwritten.
7. **Refcount is the lease.** In-flight requests hold a refcount on every
   block they touch. Eviction may not remove a block with refcount > 0.
   Refcount increments at slot assignment, decrements at request finish.

### 4.3 Dual residency (the vLLM / SGLang / TRT-LLM pattern)

Five production systems (vLLM native, SGLang HiCache, TRT-LLM, Mooncake,
Dynamo KVBM) implement this, three with explicit documentation, two
implied. The shape:

1. When a radix node's refcount drops to 0, the block is **not** removed
   from the radix tree. It is moved from the "active" set to a "free queue"
   while its `TierLocation::Gpu { slot }` is preserved.
2. `TokenKVPool::alloc` prefers blocks from the free queue over fresh
   allocation. A popped block keeps its `block_id` and its location.
3. When a subsequent `RadixCache::lookup` reaches the block, the radix
   node "resurrects" it: refcount goes back to 1, it rejoins the active
   set.
4. Only when the free queue is empty (physical pressure) does the pool
   actually repurpose a block's physical memory. At that instant, the
   radix tree forgets the block.

**Why it matters**: without dual residency, a second request with the same
system prompt pays the full prefill cost because the block it would have
reused was physically alive but no longer findable. SGLang reports this
pattern alone takes prefix hit rate from ~0 to ~80% on Novita's workload
and cuts TTFT 56%.

**Where it lives**: entirely inside `infer/src/prefix_cache.rs::RadixCache`
and `infer/src/backend/cuda/paged_kv.rs::TokenKVPool`. No new module.

### 4.4 Coordinator threading model (revised 2026-04-15)

Original 2026-04-13 text said "tokio task, dedicated CUDA copy stream".
Task doc §3.3 course-correction argued for "OS thread + crossbeam, not
tokio". This revision commits to the course correction:

- **OS thread, not tokio task.** The coordinator does not need
  work-stealing or cancellation — it is a long-running single-consumer.
  `std::thread::spawn` + a `crossbeam_channel::bounded` intent queue.
- **Dedicated `CudaStream`.** Separate from the scheduler's compute stream
  so copy and compute overlap naturally. Event-based synchronization
  between streams.
- **Metal has no CUDA stream.** The Metal backend coordinator (M4 T2
  only) uses MLX async submit + wait; the abstraction is
  backend-specific, not shared across CUDA/Metal. A future cross-backend
  coordinator trait is a post-M5 concern, not in scope now.

---

## 5 · Data structures (the contract, revised 2026-04-15)

### 5.1 Block identity (unification)

The 2026-04-13 design assumed a single `BlockId(u64)` deterministically
derived from a blake3 content hash. Reality: the project shipped three
different `BlockId` types (`kv_tier::BlockId(u64)`, `prefix_cache::BlockId(u32)`,
`block_manager::BlockId(u32)`), and the content-hash derive function is
still a `todo!()` stub. Attempting to wire `RadixCache` to `TierDirectory`
directly surfaces the collision.

**Resolution**: split the two concepts that the original doc conflated into
two different types.

```rust
// infer/src/types.rs (new canonical location)

/// Opaque identifier for a KV block currently resident in some tier.
/// Scope: lives only as long as the block is in memory or on disk; not
/// stable across restarts, not stable across nodes. Used by the radix
/// tree, the pool, the directory-merged RadixNode, the transport trait.
///
/// u32 because the worst-case block count (page_size=16, T0=80GB on H100,
/// bf16 KV on DeepSeek-V3's 64 layers × 8 KV heads × 128 head_dim) is
/// ~2M blocks — well under 2^32. vLLM and SGLang both use u32.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockId(pub u32);

/// Content-addressable fingerprint for a KV block's semantic identity.
/// Stable across processes and across nodes. Two nodes that independently
/// prefill the same prefix produce the same fingerprint; that is the
/// foundation for cross-node remote-tier reuse (M5+) and for session
/// save/load (M4). Computed from:
///   1. model fingerprint (arch + weight digest + numeric profile)
///   2. layer index
///   3. kv format (bf16 / fp8e4m3 / int8 / turboquant-2/3/4)
///   4. parent fingerprint (chains the radix path)
///   5. token ids of THIS block, in order
///
/// blake3 truncated to 128 bits — enough for single-installation dedup
/// safety (birthday bound ~2^64 blocks before collision).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockFingerprint(pub [u8; 16]);
```

**Migration**:
1. Add `types::BlockId(u32)` and `types::BlockFingerprint([u8; 16])` as the
   canonical types.
2. `prefix_cache::BlockId(u32)` becomes a re-export of `types::BlockId`.
3. `block_manager::BlockId(u32)` is **deleted** — block_manager refers to
   `types::BlockId` directly.
4. `kv_tier::BlockId(u64)` is **deleted** — the file `infer/src/kv_tier/id.rs`
   is removed in the same PR.
5. `RadixNode` stores `block_id: BlockId` and optionally
   `fingerprint: Option<BlockFingerprint>` (populated only when a block
   gets persisted to disk or migrated to remote).

This is M0.1 — see §6.

### 5.2 `Node` with tier metadata (merges the old `BlockDescriptor`)

The 2026-04-13 design had a separate `TierDirectory` holding
`BlockDescriptor { id, tier, location, byte_len, ref_count, last_access,
session_id, pin_until }`. The 2026-04-15 revision moves every field onto
the existing **private** `Node` struct inside `RadixCache` (at
`infer/src/prefix_cache.rs:54-66`). The struct stays private; no `pub`
leakage; consumers still go through `RadixCache` methods.

Note: the previous draft of this section called the struct `RadixNode`.
It is actually named `Node` and is `pub(crate) struct Node` at the time
of writing. The rename to `RadixNode` is not required for M1; keeping
`Node` matches the existing code.

```rust
// infer/src/prefix_cache.rs — after M1

pub enum TierLocation {
    Gpu { slot: u32 },
    HostPinned { offset: u64 },         // offset within the pinned region
    Disk { file_id: u32, offset: u64 },
    Remote { node: NodeId, desc: OpaqueRemoteDesc },
}

pub(crate) struct Node {
    // existing fields (pre-M1 shape from prefix_cache.rs:54-66)
    tokens: Vec<u32>,
    block_id: BlockId,                              // pool slot id, u32
    children: HashMap<u32, Box<Node>>,
    ref_count: u32,
    last_access: u64,                               // monotonic tick

    // added in M1 (absorb BlockDescriptor's semantic fields)
    tier_location: Cell<TierLocation>,              // atomic tier transition
    session_id: Option<SessionId>,
    soft_pin_until: Option<Instant>,
    byte_len: u32,                                  // includes quantization scales
    fingerprint: Option<BlockFingerprint>,          // only when persisted
}

impl RadixCache {
    pub fn lookup(&self, tokens: &[u32]) -> LookupResult {
        // Returns (hit_len, Vec<BlockId>, tier_locations).
        // Single walk of the tree; no second hop to an external directory.
    }

    pub fn ref_inc(&self, block_ids: &[BlockId]) { /* walk ancestor chain */ }
    pub fn ref_dec(&self, block_ids: &[BlockId]) { /* symmetric */ }

    pub fn evict_into_free_queue(&self, block_id: BlockId) {
        // Dual residency (§4.3): block stays in the radix tree but is
        // removed from the active set.
    }

    pub fn promote(&self, block_id: BlockId, to: TierLocation) {
        // Called by coordinator when a block moves between tiers.
    }
}
```

**What the `kv_tier/directory.rs` file was supposed to do that is now
covered by `RadixCache`:**

| Old `TierDirectory` API | Replacement in `RadixCache` |
|---|---|
| `resolve(id) -> BlockDescriptor` | `lookup(tokens)` returns both `block_id` and `TierLocation` at once; no second hop |
| `insert(desc)` | `RadixNode` is allocated by the tree on insertion |
| `promote(id, to, loc)` | `RadixCache::promote` — atomic `Cell<TierLocation>` swap |
| `demote(id, to, loc)` | Same, `promote`/`demote` are the same API |
| `touch(id, now)` | `RadixNode::last_access.store(now)` |
| `pin(id) / unpin(id)` | `RadixNode::soft_pin_until` write |

**The `infer/src/kv_tier/directory.rs` file (322 lines) is deleted in M1.**
Nothing in code calls it today, so removal is a pure subtraction.

### 5.3 `KVTransport` trait (matches existing code)

The original 2026-04-13 sketch had `type Completion: Future`. The actual
code in `infer/src/kv_tier/transport.rs:94-144` uses `type Op: Send` with
explicit `poll()` + `abort()` because NIXL does not expose a Future type.
The revised §5.3 matches the shipped code:

```rust
// infer/src/kv_tier/transport.rs — this is what's already in the tree

pub enum MemKind { HostPinned, CudaDevice, CudaManaged, MetalUnified }

pub struct TransferOp {
    pub src: TierLocation,
    pub dst: TierLocation,
    pub len: u32,
}

pub trait KVTransport: Send + Sync {
    type Region;   // registered memory region handle (MR)
    type Op: Send; // transfer handle — NOT a Future

    // Registration — UCX, NIXL, Mooncake all require pre-registered MRs.
    fn register(&self, ptr: *mut u8, len: usize, kind: MemKind) -> Result<Self::Region>;
    fn deregister(&self, region: Self::Region) -> Result<()>;
    fn invalidate_region(&self, region: &Self::Region) -> Result<()>;

    fn put_batch(&self, ops: &[TransferOp]) -> Self::Op;
    fn get_batch(&self, ops: &[TransferOp]) -> Self::Op;

    fn poll(&self, op: &mut Self::Op) -> PollOutcome;   // NotReady | Ready | Err
    fn abort(&self, op: Self::Op);
}

/// Remote descriptor content is opaque; NIXL serializes agent metadata,
/// Mooncake serializes segment handles, raw verbs serialize (rkey, addr, qpn).
pub struct OpaqueRemoteDesc(pub smallvec::SmallVec<[u8; 32]>);
```

**Implementations, in order of planned delivery:**

- **`LocalCudaTransport`** (M3). `cudaMemcpyAsync` on a dedicated copy
  stream. Covers T0↔T1. Registration is a no-op beyond storing the base ptr.
  First version uses vanilla DMA; the SGLang GPU-assisted I/O kernel (3×
  speedup for small blocks) is a follow-up optimization, not a blocker.
- **`DiskStore`** (M4). Already implemented at `kv_tier/transport/disk.rs`
  (328 lines, 8 unit tests passing). Needs to (a) switch from raw-bytes
  dump to postcard header + blake3-hash filename (task doc §4.2 spec), and
  (b) be connected to the coordinator.
- **`NixlTransport`** (M5, `#[cfg(feature = "rdma-nixl-real")]`). Links the
  real `libnixl`; the M5 shape is the stub that compiles under
  `rdma-nixl` (stub-api). Only executes when the cross-node / prefill-
  decode disaggregation trigger fires.

### 5.4 `EvictionPolicy` (already shipped, needs to be wired)

`infer/src/scheduler/policy.rs:179-189` already defines the trait and four
implementations:

```rust
pub enum SessionState { Active, Keepalive, Cold }

pub struct EvictionCandidate {
    pub last_access: u64,
    pub ref_count: u32,
    pub block_count: u32,
    pub session_state: SessionState,
    pub session_id: Option<SessionId>,
}

pub trait EvictionPolicy: Send + Sync {
    fn score(&self, c: &EvictionCandidate, sig: &SchedulerSignals) -> i64;
}

pub struct LruEviction;           // shipped
pub struct ReuseBiasedLru { /*..*/ }  // shipped
pub struct HitCountLru { /*..*/ }     // shipped
pub struct SessionBiasedLru {     // shipped, matches KVFlow default
    pub active_weight: i64,
    pub keepalive_weight: i64,
    pub keepalive_ticks: u64,
}
```

**Status**: trait + 4 implementations shipped, **zero call sites in the
scheduler runtime**. M3 wires it in when the coordinator is introduced.
Industry reference: TRT-LLM's priority-bucket LRU gives +20% hit rate over
pure LRU; we can add a `PriorityLru` variant as a post-M3 experiment if the
benchmark shows the delta is real for agent workloads.

---

## 6 · Execution path (revised 2026-04-15, Milestones M0–M5)

Replaces the 2026-04-13 phase plan P0–P5. The M0–M5 shape rearranges the
same work to put **behavior changes first** — the original P1(b) "wire
RadixCache into scheduler" was the single point of failure, and every
downstream P2/P3/P4 was designed against it without it shipping.

Each milestone below is a **PR** (or a short stacked series). Exit criteria
are observable: test passes, benchmark delta, or the explicit code-path
appearance of a named type. Nothing is "done when it feels done".

### M0 — Pre-work (3 independent PRs, no ordering constraint between them)

#### M0.1 · `BlockId` unification

**What**: Add `infer/src/types.rs` (or extend the existing types module)
with `BlockId(u32)` canonical and `BlockFingerprint([u8; 16])` separate.
Delete `infer/src/kv_tier/id.rs`. Remove `block_manager::BlockId`. Update
`prefix_cache::BlockId` to be a re-export of `types::BlockId`.

**Why first**: resolves the 3-way collision that blocks M1. Pure type
rename + `use` path update, no algorithmic change.

**Files**:
- New: `infer/src/types.rs` (if not already created)
- Delete: `infer/src/kv_tier/id.rs`
- Modify: `infer/src/block_manager.rs:19`, `infer/src/prefix_cache.rs:29`
- Modify: every consumer of any of the three old types (grep `BlockId`)

**Exit**: `grep -rn 'pub struct BlockId' infer/src/` returns exactly one
match (in `types.rs`). `cargo check` passes under `cuda,no-cuda`,
`cpu,no-cuda`, `metal`.

#### M0.2 · Fix the three `prefix_cache.rs` correctness bugs

**What**: Port the SGLang RadixAttention fix patterns for the three bugs
listed in §8 (items 10–12). Each bug gets one unit test.

**Why first**: the scheduler wire-up in M1 will exercise code paths that
trip these bugs. Fixing them under unit tests is cheap; finding them
during M1 benchmark regressions is expensive.

**Files**:
- `infer/src/prefix_cache.rs::_split_node` — new node inherits child's
  `ref_count` (SGLang `new_node.lock_ref = child.lock_ref`)
- `infer/src/prefix_cache.rs::lookup` — ref_inc walks ancestor chain
  from leaf to root
- `infer/src/prefix_cache.rs::evict` — iterative, re-add orphan parent
  to eviction heap after leaf removal

**Exit**: three new unit tests, each reproducing the bug pre-fix and
passing post-fix.

#### M0.3 · `page_size = 1 → 16` (per-format dispatch)

**What**: Raise `TokenKVPool::page_size` default from 1 to 16. Rewrite the
pool as a two-level allocator (allocate a new page when
`seq_len % page_size == 0`, else append to the tail page). INT8, FP8, and
TurboQuant paths **remain at `page_size=1`** because their kernels are
written to assume per-token page granularity — this is per-format
dispatch, not a global bump.

**Sequencing clarification (Codex review 2026-04-15)**: M0.3 is **not a
prereq for M1**. M1's exit gate is TTFT / throughput parity versus the
existing `cached_prompts` path on T0 only — no tier transfer is involved,
so `page_size=1` does not break M1's benchmark gate. M0.3 is a **prereq
for M3**, where T0↔T1 transfer begins and small-block DMA launch overhead
starts to matter. The original draft of this section said "M0.3 must land
before M1" — that was an overstatement. M0.3 and M1 can be sequenced in
either order; the only hard rule is M0.3 lands before M3a.

**Why do it early anyway**: industry data shows `page_size=1` cripples
tier-transfer bandwidth — at small block sizes DMA engine launch overhead
dominates DMA throughput. vLLM floor is 16, SGLang 64, Mooncake 512. M3
and beyond all presume `page_size ≥ 16` for BF16 paths. Doing it in M0
instead of sandwiched into M3 keeps M3a focused on the transport code
without carrying a page-allocator rewrite in the same PR.

**Why it is not before the in-flight crate extraction**: the `.cu` files
this milestone touches (`kv_cache_to_paged.cu`, `kv_quant.cu`,
`scatter_kv.cu`) are being moved from `infer/csrc/cuda/` to
`crates/infer-cuda-kernels/csrc/` in a parallel extraction work stream.
M0.3 **blocks on** that extraction landing because the .cu file
locations change mid-PR otherwise.

**Files**:
- `infer/src/backend/cuda/paged_kv.rs:7,76-87,482-511,549-562,614,760`
  (will be at the new extraction path after the extraction lands)
- `infer/src/backend/cuda/flashinfer.rs` (incremental metadata update)
- `infer/src/model/qwen3/batch_decode.rs:384`,
  `qwen35/batch_decode.rs:724`, `glm4/batch_decode.rs:338` — drop the
  `let page_size = 1;` locals
- `infer/src/scheduler/cuda/decode.rs:193` — literal `1` → `pool.page_size`
- Kernels (post-extraction path): `kv_cache_to_paged.cu:64-103`,
  `kv_quant.cu:184,193,207,211`, `scatter_kv.cu` — per-format dispatch
  `match format { BF16 | FP16 => PAGE_SIZE_16, INT8 | FP8 | TurboQuant => 1 }`

**Exit**:
1. `cargo test --release --test e2e` and `--test e2e_qwen35` pass unchanged.
2. `greedy_consistency` passes unchanged.
3. `scripts/bench_throughput_sweep.py --label page16` recorded vs
   `--label page1` baseline in `docs/experience/wins/`.
4. FlashInfer split-KV scheduler does not lose parallelism on short-
   context single-request benches (watch the tail in the sweep).

**Risk**: FlashInfer's split-KV fans out by `max_num_pages_per_batch`; at
very short contexts and batch=1 a larger page size can reduce fan-out.
The bench gate catches it. Mitigation if real: keep `page_size=1` on a
short-context fast path and `page_size=16` otherwise — but do not build
this pre-emptively.

### M1 — Wire `RadixCache` into scheduler, delete `TierDirectory`

This is the original P1(b) behavior PR, expanded to also delete the
now-dead `kv_tier/directory.rs` shell. Previous draft said "must be one
atomic PR or the midway state is uncompilable"; Codex review 2026-04-15
showed that is too strong — a compilable 2-PR split exists. M1 is
**nominally one PR but may be split** if reviewer cost exceeds the
atomic-PR benefit.

**Scope**:
- Extend `RadixCache`'s internal `Node` struct (private, in
  `infer/src/prefix_cache.rs:54-66`; not `RadixNode` — the type name was
  a mistake in the previous draft of this section) with `tier_location:
  Cell<TierLocation>`, `session_id: Option<SessionId>`, `last_access:
  AtomicU64`, `soft_pin_until: Option<Instant>`, `byte_len: u32`, and
  optional `fingerprint: Option<BlockFingerprint>` fields.
- Delete `infer/src/kv_tier/directory.rs` (322 lines) and
  `infer/src/kv_tier.rs`'s `TierDirectory` re-export.
- Delete `scheduler/cuda/core.rs::cached_prompts: Vec<Vec<u32>>`, replace
  with `radix_cache: Arc<RadixCache>`.
- Replace the linear-compare prefix logic currently living in
  `scheduler/cuda/runtime.rs:11-30,111-117,167-184` and
  `server_engine.rs:475-547` with `radix_cache.lookup`.
- Add `ref_inc(&block_ids)` at slot assignment and `ref_dec(&block_ids)`
  at request completion in `scheduler/cuda/runtime.rs`.
- Only use `TierLocation::Gpu { slot }` in M1; other variants compile but
  are not populated yet. `LocalCudaTransport` is not wired.

**Split option (recommended if reviewer cost is high)**:
- **M1a · structural** — add the new fields on `Node`, keep `cached_prompts`
  and `TierDirectory` untouched. The new fields compile because the
  struct is private and no consumer looks at them. Full test suite
  passes unchanged. Zero behavior change. `cargo check` clean.
- **M1b · behavior** — swap `scheduler/cuda/*` to use the radix lookup,
  delete `cached_prompts`, delete `kv_tier/directory.rs`. Full M1
  benchmark gate applies to this PR.

This split is safe because there is no intermediate state where
`TierDirectory` is deleted but the radix tree has no tier fields — the
structural PR adds the fields first, the behavior PR removes the old
path second, and at every intermediate commit both paths exist
simultaneously. The single-PR form is still fine; choose based on
reviewer bandwidth.

**Exit**:
1. `grep -rn cached_prompts infer/src/` returns empty
2. `grep -rn RadixCache infer/src/scheduler/` returns non-empty
3. `cargo test --release --test e2e` and `e2e_qwen35`: golden outputs
   unchanged
4. `scripts/bench_throughput_sweep.py`: **TTFT and throughput ≤ 1%
   regression** on 1-batch, 4-batch, and 8-batch sweeps (T0 only — no
   tier transfer in M1, so `page_size=1` is acceptable even if M0.3 has
   not landed)
5. Prefix hit rate on an agent-style 2-session alternating trace
   (replayed by `scripts/bench_agent_trace.py`) is **≥ the old
   `cached_prompts` path**

### M2 — Dual residency (T0 only, no new tiers)

**Why it is its own milestone**: the hit-rate improvement that SGLang
reports (40%→80% cross-session) comes from dual residency, not tiering.
It is the single biggest prefix-hit lever and it is orthogonal to adding
T1/T2/T3 — it only touches `RadixCache::evict` and `TokenKVPool::alloc`.

**What**:
- `RadixCache::evict_into_free_queue`: when refcount reaches 0, move the
  node from active set to free queue, **preserve** `TierLocation::Gpu { slot }`
- `TokenKVPool::alloc_block`: prefer blocks from the free queue before
  allocating new
- `RadixCache::lookup`: when a hit touches a free-queue block, resurrect
  it (atomic move from free queue to active set, refcount back to 1)
- Only when the free queue is empty does the pool physically repurpose a
  block's memory — at that instant, the radix tree forgets the block

**Exit**:
1. Two-session alternating trace: second visit to the same system prompt
   achieves **≥ 95% prefix hit rate** (previous: ~0%)
2. `scripts/bench_agent_trace.py` TTFT drops measurably on the
   agent-workload benchmark
3. `cargo test --release --test e2e_qwen35` unchanged

### M3 — T1 host pinned tier + coordinator (stacked PR series)

**Sub-PRs (in order)**:

- **M3a** · `HostPinnedPool` + `cudaMemcpyAsync` one-way T0→T1 transfer.
  Coordinator is called manually (no watermark trigger yet). No behavior
  change in the scheduler — it just gets a new API to say "move this
  block to T1".
- **M3b** · Coordinator OS thread + crossbeam intent channel + dedicated
  CUDA copy stream + watermark triggers (T0_high = 0.85, T0_low = 0.70,
  matching vLLM / SGLang defaults). `EvictionPolicy::score` is now
  actually called.
- **M3c** · T1→T0 promotion on radix lookup miss in T0 (the promote
  path). Delete the dormant `infer/src/model/kv_cache.rs:130-168` CPU
  offload code after confirming zero production callers.

**Exit**:
1. A long-agent-session benchmark (32k+ cumulative tokens, num_slots=4)
   that OOMs on pre-M3 runs to completion on M3c
2. `scripts/bench_throughput_sweep.py --label tier-T1` recorded vs a
   pre-M3 baseline. Steady-state decode throughput regression ≤ 3%
3. `cargo test --release --test e2e_qwen35` unchanged

### M4 — T2 disk tier + session save/load + first Metal contact

**What**: Add the real coordinator path for T1→T2 spill under watermark.
Change `DiskStore` wire format from raw-bytes dump to postcard header +
blake3-hash filename (task doc §4.2 spec). Add
`POST /v1/sessions/{id}/save` and `POST /v1/sessions/{id}/load` routes
for session persistence.

Bind `mlx.metal.set_wired_limit` and `get_active_memory` in `mlx-sys` so
the Metal backend has the telemetry it needs to enforce a bounded KV
pool (the mlx-lm #883 panic mitigation).

**Why not CacheGen compression**: LMCache's CacheGen (quantization +
entropy coding of KV chunks) is the only system with it. Other production
systems skip it. The disk tier works without it; compression is a
post-M4 optimization if disk footprint becomes the bottleneck.

**Files**:
- `infer/src/kv_tier/transport/disk.rs` — wire format change
- `infer/src/http_server.rs:422-427` — new routes
- `infer/src/http_server/sessions.rs` — new, save/load handlers
- `crates/mlx-sys/src/lib.rs` — bindings for wired memory
- `infer/src/backend/metal/kv_pool.rs` — bounded `max_total_tokens` at init
- `infer/src/backend/metal/prefix_cache.rs` — T2 hook via `TieredKvCache`
  façade

**Exit**:
1. Restart smoke test: save a 30k-token system prompt session, kill the
   process, restart, reload, measure TTFT. Must land within 20% of the
   pre-restart warm-state baseline.
2. Metal backend with bounded `max_total_tokens` runs a long-context
   test without a `prepare count underflow` kernel panic.

### M5 — Real NIXL RDMA path (deferred)

**Not scheduled.** The M5 shape remains the `NixlTransport` stub that
compiles behind `rdma-nixl`. The jump from stub to real (link `libnixl`,
run against UCX, transfer KV across an InfiniBand fabric) happens only
when one of these triggers fires:

- **Prefill/decode disaggregation** — separate worker pools for prefill
  and decode need to migrate KV between them
- **Cluster-wide session roaming** — a session moves between nodes and
  its KV must follow
- **Second consumer of the kernel layer** — an external project wants to
  reuse `infer-cuda-kernels` + the transport, and needs a functional
  remote tier to do it

See [`../plans/cuda-kernel-crate-extraction.md`](../plans/cuda-kernel-crate-extraction.md)
§2 for the trip-wire discipline this follows. In the absence of any of
these triggers, M5 real-RDMA is post-project work.

---

## 7 · PR splitting discipline

| Milestone | PRs | Structure / behavior |
|---|---|---|
| M0.1 | 1 | Type rename + `use` path update (structural) |
| M0.2 | 1 | Three prefix_cache bug fixes (correctness, structural) |
| M0.3 | 1 | page_size lift with per-format dispatch (structural, benchmark gate) |
| M1 | **1 atomic** | RadixCache wire + TierDirectory merge + scheduler swap (combined because any split has an uncompilable midway state) |
| M2 | 1 | Dual residency (behavior, benchmark gate) |
| M3 | 3 stacked | M3a transport, M3b coordinator+watermarks, M3c promote path |
| M4 | 2 stacked | (a) disk + MLX bindings (structural); (b) HTTP routes + session save/load (behavior) |
| M5 | deferred | triggered, not scheduled |

No PR in this doc mixes kernel and scheduler changes. M0.3's kernel files
are already page-size-aware internally; only the dispatch locals change.
M1–M4 have zero kernel diffs.

---

## 8 · Pitfalls we already know about

Three of these (items 10–12) are M0.2's exact scope — the doc records them
here so that M0.2's test suite references them by number.

1. **MLX wired memory panic** (mlx-lm #883). MLX wires all allocations by
   default; an unbounded KV pool hits `prepare count underflow` in
   `IOGPUMemory` before the OS gets a chance to page. M4 must bind
   `set_wired_limit` and `get_active_memory` before enabling T2 on Metal.
2. **MR registration invalidation.** UCX, NIXL, and Mooncake all require
   pre-registered memory regions. If the T1 pinned pool ever reallocates
   or compacts, registered MRs become dangling. Allocate the pool once at
   engine init, never grow. If we need to grow, register the new region
   before freeing the old one.
3. **FlashInfer split-KV parallelism at short contexts.** M0.3's risk.
   The benchmark gate catches it; do not pre-emptively build a
   `page_size=1` fast path.
4. **`nvidia-peermem` vs old `nv_peer_mem`.** GDR needs the former; old
   docs and third-party crates still reference the latter. M5 must probe
   for `nvidia-peermem` and fall back to bounce buffer.
5. **NIXL stack requirements.** CUDA 12+, UCX 1.19/1.20, NIXL native lib
   at link time. `nixl-sys` `stub-api` feature is how we keep default CI
   green. Gate behind `rdma-nixl` feature for M5 compile, `rdma-nixl-real`
   for real link.
6. **Mooncake metadata service.** If we add a Mooncake transport later,
   Mooncake Store needs etcd (or its own master). That is a deployment-
   story decision, not a transport-trait one. Keep the trait oblivious.
7. **`BlockFingerprint` collision.** `BlockFingerprint` uses blake3
   truncated to 128 bits. At 2^64 blocks the birthday bound is non-zero;
   when two fingerprints are equal, verify the `parent_fingerprint` chain
   to catch the pathological case. u32 `BlockId` has no such issue —
   it is not a hash, just a pool slot id.
8. **Scheduler single-threadedness.** Today the scheduler owns all KV
   under one thread and needs no locks. The coordinator is a second owner.
   M3b's directory-owning structure (now `RadixCache`, not `TierDirectory`)
   must be audited for cancel-safety at every `await` point.
9. **`backend/metal/gdr.rs` is not GPUDirect RDMA.** The filename is
   misleading; it is the Qwen3.5 Gated Delta Rule linear-attention
   decoder. Do not reuse that module for transport work.

### 2026-04-15 bugs flagged by the internal survey

10. **`_split_node` does not inherit child's `ref_count`.** When the radix
    tree splits a node on insertion, the new parent-like node is created
    with `ref_count = 0` even if its child had `ref_count > 0`. SGLang's
    reference implementation sets `new_node.lock_ref = child.lock_ref`.
    Without it, a split while a request is mid-decode can free a block
    that is still in use. **Fix in M0.2.**

11. **`lookup` does not walk the ancestor chain.** Current code bumps
    only the matched leaf's `ref_count`; ancestors are unprotected.
    Scheduler that holds an intermediate radix node while decoding a
    continuation can see that node evicted from under it. **Fix in M0.2.**

12. **`evict()` does not iterate orphan parents.** When a leaf is
    evicted, its parent may become childless and eligible for eviction
    too, but the current code only considers the original leaf set.
    SGLang re-pushes orphaned parents onto the eviction heap in the
    same pass. **Fix in M0.2.**

---

## 9 · Relationship to other docs

- [`agent-first-architecture.md`](agent-first-architecture.md) — owns A1,
  B1, B3. This doc **supersedes** those three items' implementation shape.
  When M1 lands, A1 moves to the Done section; when M3 lands, B3 moves;
  when M4 lands, B1 moves. `agent-first-architecture.md` gets an update
  pointer to this doc in the same PR series.
- [`kv-quantization-long-context.md`](kv-quantization-long-context.md) —
  KV quantization formats (FP8, INT8, TurboQuant) live inside T0 blocks.
  `byte_len` on `RadixNode` must account for scale bytes. M0.3 must not
  regress the quantized fast paths; the M0.3 per-format dispatch keeps
  `page_size=1` for the quantized families exactly for this reason.
  M3 coordinator must preserve format across tier transitions.
- [`mlx-backend-roadmap.md`](mlx-backend-roadmap.md) — Metal side. M4 is
  the first point of contact; the MLX roadmap should link back here once
  M4 enters execution.
- [`cuda-kernel-crate-extraction.md`](../plans/cuda-kernel-crate-extraction.md) —
  M0.3 **blocks on** the extraction's `.cu` file moves landing. After the
  extraction lands, M0.3's kernel file paths change from
  `infer/csrc/cuda/kv/*.cu` to `crates/infer-cuda-kernels/csrc/kv/*.cu`.
- [`../archives/art-grade-architecture-for-long-agent-infer.md`](../archives/art-grade-architecture-for-long-agent-infer.md) —
  archived workspace crate topology proposal. PR discipline (§六) and
  crate admission criteria (§七) still apply; the §一 topology was
  reverted by Route-A. If M3 or later promotes `kv_tier` to a separate
  crate, the promotion still has to pass the §六 "two direct consumers"
  gate.

---

## 10 · Backend coverage summary (revised 2026-04-15)

| Backend | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| CUDA | `page_size=16` + 3 bug fixes + BlockId unify | RadixCache wired, T0-only | Dual residency | T0+T1 + coordinator | +T2 disk + session HTTP | NIXL stub only (real deferred) |
| Metal | n/a (unaffected) | RadixCache wired via `backend/metal/prefix_cache` | **no-op** (unified memory) | no-op | T0 bounded + T2 disk | n/a (no RDMA) |
| CPU backend | untouched (309-line smoke test) | untouched | untouched | untouched | untouched | untouched |

The Metal column's M3 entry is a no-op intentionally; see §4.1 (unified
memory argument).

---

## 11 · Industry comparison (added 2026-04-15)

Seven surveyed systems. Matrix with the five dimensions that gated the
2026-04-15 design decisions. Full notes in the stop-hook review transcript.

| System | Tier model | Addressing | Eviction | Dual residency | Radix-tier coupling |
|---|---|---|---|---|---|
| **vLLM native + llm-d** | T0 + T1 | block hash | LRU | ✅ (T1 ⊇ T0) | same hash table is tier-aware |
| **SGLang HiCache** | T0 + T1 + T3(shared L3) | HiRadixTree node | write-through/selective/back + LRU | ✅ (node records multiple tiers) | **HiRadixTree IS the tier system** |
| **LMCache + CacheGen** | T1/T2/T3 | content hash on chunks | backend-dependent | partial | orthogonal (supports non-prefix) |
| **Mooncake Store** | cluster pool (T0 split/T1/T2/T3) | Merkle prefix hash, 512-token blocks | LRU (empirical best) | ✅ (multi-replica) | hash-keyed |
| **Dynamo KVBM + NIXL** | T0/T1/T2/T3 | delegated to engine | pluggable | implied | delegated |
| **TRT-LLM native** | T0 + T1 | radix tree, partial match | priority-bucket LRU (0–100, +20% hit rate) | ✅ (`secondary_offload_min_priority`) | radix tree |
| **DeepSpeed OffloadedCache** | T0 + T1 (layer-rotating) | none | FIFO | ❌ | n/a |

Contested design questions (where the 7 systems genuinely disagree):

1. **Unified vs separate tier + prefix data structure** — SGLang and
   TRT-LLM merge them; vLLM keeps them in one hash table but conceptually
   separate; LMCache is orthogonal. **Our choice: merge (SGLang-style).
   §5.2.**
2. **Dual residency on eviction** — vLLM / SGLang / TRT-LLM / Mooncake all
   yes; LMCache and DeepSpeed no. **Our choice: yes, mandatory. §4.3.**
3. **L3/remote metadata locality** — vLLM (via llm-d) mirrors NATS events
   locally; SGLang queries through to L3 at lookup time; Mooncake uses
   etcd. **Our choice (M5+): query-through for simplicity. §4.2 invariant 3.**
4. **Block granularity** — vLLM 16, SGLang 64, Mooncake 512. **Our choice:
   `page_size=16` for BF16/FP16, 1 for quantized formats. §6 M0.3.**
5. **Eviction policy** — LRU is the baseline everyone has; TRT-LLM's
   priority-bucket LRU gives +20%. **Our choice: ship `LruEviction` or
   `SessionBiasedLru` in M3; consider `PriorityLru` as a post-M3
   experiment. §5.4.**
6. **Transport abstraction** — NIXL or Mooncake Transfer Engine or none
   (vLLM native `cudaMemcpyAsync`, TRT-LLM native `cudaMemcpyAsync`,
   SGLang custom kernels). **Our choice: trait + `LocalCudaTransport` +
   `DiskStore` + `NixlTransport`. M3 ships the CUDA impl with vanilla
   `cudaMemcpyAsync`; SGLang's 3×-faster custom I/O kernel is a post-M3
   optimization. §5.3.**
7. **Scope** — vLLM / TRT-LLM / DeepSpeed single-node; Mooncake / KVBM
   cluster. **Our choice: single-node through M4; cluster is M5+.**
8. **Non-prefix reuse** — only LMCache does it (CacheBlend). **Our choice:
   skip, prefix-only. §2 non-goals.**
9. **Refcount semantics** — SGLang and TRT-LLM use radix-leaf refcount;
   vLLM blocks scheduling on pending loads. **Our choice: refcount at
   slot assignment, decrement at request finish. §4.2 invariant 7.**
10. **Backpressure** — vLLM stalls, SGLang prefetches with early
    termination, Mooncake rejects at ingress, Dynamo autoscales. **Our
    choice: stall at scheduler (vLLM-style), add ingress rejection only
    if M3 exposes a need.**

Key industry numbers that grounded the path:

- **vLLM**: TTFT 2×–22× improvement with CPU cache hits, v0.12.0 4× TTFT
  reduction + 5× throughput
- **SGLang HiCache**: up to 6× throughput, 80% TTFT reduction; Ant Group
  DeepSeek-R1-671B: 84% TTFT reduction; Novita: 56% TTFT reduction,
  throughput 2×, prefix hit rate 40% → 80%
- **TRT-LLM**: 5× faster TTFT with early reuse; priority-bucket eviction
  +20% hit rate
- **Mooncake**: Kimi handles 75% more requests; K2 on 128 H200: 224k
  tokens/s prefill, 288k tokens/s decode

These numbers set the M1 benchmark gate: ≤ 1% TTFT regression vs the
`cached_prompts` path. We are not trying to match SGLang's 84% TTFT
reduction in M1 — that's M2+M3 territory. M1 is "survive the switch
with no regression".

### 11.1 Industry optimisations explicitly considered and deferred

Listing so future contributors know these were not accidentally
omitted. None of them is in the M0–M5 critical path; each is a
post-M4 candidate if its trigger condition materialises.

- **LMCache CacheGen (quantization + entropy coding of KV chunks on disk)** —
  considered for M4 (§6 M4). Deferred: disk footprint is not currently
  a bottleneck, and CacheGen adds algorithmic complexity on both write
  and read paths. Revisit when the M4 disk tier is full and users want
  persistence over more sessions than the disk pool holds.
- **SGLang GPU-assisted I/O kernels (3× faster than `cudaMemcpyAsync` for
  small blocks)** — considered for M3 (§6 M3). Deferred: the first version
  of `LocalCudaTransport` uses vanilla `cudaMemcpyAsync`. Only revisit
  if M3 bench shows DMA launch overhead dominates tier-transfer
  throughput at `page_size=16`, in which case ~100 lines of custom CUDA
  can close the gap.
- **TRT-LLM priority-bucket LRU (0–100 bucket eviction, +20% hit rate
  over pure LRU)** — considered as M3's eviction policy default (§5.4).
  Deferred: M3 ships with `SessionBiasedLru` (the KVFlow-matched default).
  Add `PriorityLru` as a post-M3 experiment if the agent-workload benchmark
  shows `SessionBiasedLru` under-performs by ≥ 10%.
- **Mooncake 512-token blocks** — considered for the `page_size` default
  (§6 M0.3). Rejected: 512 is too coarse for this project's agent
  workload (each tool call is typically 20–100 tokens), which would
  under-utilise blocks. vLLM's 16 and SGLang's 64 are the defensible
  defaults; we ship at 16.
- **cuFile / GPU-Direct Storage for T2 disk** — considered for M4 disk
  tier. Deferred: `cuFile` adds a driver dependency and a second code
  path alongside `io_uring`. M4's first version uses `io_uring`
  (Linux) + `tokio::fs` fallback. Revisit when M4's disk-tier benchmarks
  show userspace I/O is the bottleneck.
- **Mooncake Transfer Engine as the primary transport** — considered
  as a potential alternative to writing our own `KVTransport` trait.
  Rejected: the trait already exists (§5.3) and is NIXL-compatible;
  NIXL itself has a Mooncake plugin, so we can call into Mooncake
  through NIXL at M5 if we want to. No need to take a direct Mooncake
  dependency.
- **LMCache CacheBlend non-prefix reuse** — see §2 non-goals. Only one
  of seven surveyed systems does it (LMCache, via cross-attention
  blending). The algorithmic complexity and the research-stage status
  keep it out of this project.

---

## 12 · Sources

**Original 2026-04-13 references**:
- [SGLang HiCache blog (LMSYS 2025-09-10)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [SGLang `radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [SGLang `hiradix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py)
- [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
- [vLLM `kv_cache_manager`](https://docs.vllm.ai/en/v0.19.0/api/vllm/v1/core/kv_cache_manager/)
- [LMCache tech report](https://lmcache.ai/tech_report.pdf)
- [llm-d KV Cache Manager](https://llm-d.ai/docs/architecture/Components/kv-cache-manager)
- [Mooncake FAST '25 paper](https://www.usenix.org/system/files/fast25-qin.pdf)
- [Mooncake × SGLang HiCache design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [KVFlow paper (arXiv 2507.07400)](https://arxiv.org/abs/2507.07400)
- [NVIDIA NIXL blog](https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/)
- [NIXL GitHub (ai-dynamo/nixl)](https://github.com/ai-dynamo/nixl)
- [NIXL Backend Guide](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)
- [nixl-sys crate](https://crates.io/crates/nixl-sys)
- [vLLM NixlConnector docs](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [FlashInfer `page.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh)
- [FlashInfer batch decode tests (page_size parametrized)](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/attention/test_batch_decode_kernels.py)
- [mlx-lm #883 — wired KV kernel panic](https://github.com/ml-explore/mlx-lm/issues/883)
- [llama.cpp #20697 — `--cache-disk` for UMA](https://github.com/ggml-org/llama.cpp/issues/20697)
- [mlx-flash (SSD weight streaming + hybrid quantized KV)](https://github.com/matt-k-wong/mlx-flash)
- [MLX Metal memory management](https://ml-explore.github.io/mlx/build/html/python/metal.html)

**2026-04-15 industry research additions**:
- [vLLM KV Offloading Connector 2026-01-08 blog](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM KVConnectorBase_V1 API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/)
- [vLLM RFC: KV cache offloading (#19854)](https://github.com/vllm-project/vllm/issues/19854)
- [llm-d KV cache wins blog](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [llm-d Tiered Prefix Cache — CPU guide](https://llm-d.ai/docs/guide/Installation/tiered-prefix-cache/cpu)
- [SGLang HiCache design doc](https://docs.sglang.io/advanced_features/hicache_design.html)
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [LMCache docs root](https://docs.lmcache.ai/)
- [CacheBlend EuroSys'25 blog](https://blog.lmcache.ai/2025-03-31-eurosys/)
- [Mooncake arXiv 2407.00079](https://arxiv.org/abs/2407.00079)
- [Dynamo GitHub README](https://github.com/ai-dynamo/dynamo)
- [Dynamo KVBM component docs](https://docs.nvidia.com/dynamo/components/kvbm)
- [TensorRT-LLM KV cache system docs](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- [TensorRT-LLM KV cache reuse docs](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
- [5× faster TTFT with early KV reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)

---

## 13 · Revision log

### 2026-04-15 revision (post-survey + industry research)

The 2026-04-13 design was structurally sound but had three corrections that
the 2026-04-15 internal survey + 7-system industry comparison forced:

1. **`RadixCache` and `TierDirectory` merge into one data structure.** The
   original "radix tree → directory resolve" two-layer architecture was
   not industry-proven; 7 of 7 surveyed systems merge them. The project
   also never implemented the resolve call in code, so merging is both
   industry-aligned and removes 322 lines of unused code.
   - Affected sections: §4 (diagram), §4.2 (invariants 1 & 3), §5.2
     (entirely rewritten), §6 M1 (execution path)
   - File impact: `infer/src/kv_tier/directory.rs` deleted; the fields
     move onto `RadixNode` in `infer/src/prefix_cache.rs`.

2. **`BlockId` unified to `u32`, `BlockFingerprint([u8; 16])` separate.** The
   original single-type `BlockId(u64)` content-hash design shipped as
   three incompatible types in code. Unification picks the u32 canonical
   (vLLM / SGLang / block_manager all use u32) and extracts content
   hashing to its own type only used when persistence or cross-node
   reuse is needed.
   - Affected sections: §1 (naming), §4.2 (invariant 4), §5.1
     (entirely rewritten)
   - File impact: `infer/src/kv_tier/id.rs` deleted; new canonical types
     in `infer/src/types.rs`.

3. **`page_size = 1 → 16` promoted from "P0 can start immediately" to
   M0.3 prerequisite.** Industry floor is 16 (vLLM), 64 (SGLang), 512
   (Mooncake). At `page_size=1`, small-block DMA transfers are bottlenecked
   by DMA engine launch overhead, not throughput — M1 tier-transfer
   benchmarks cannot pass with `page_size=1`.
   - Affected sections: §3 current state fact 3, §6 M0.3, §8 pitfall 3
   - File impact: CUDA pool, FlashInfer metadata, 3 model batch-decode
     callers, scheduler `decode.rs:193`. M0.3 blocks on the in-flight
     `infer-cuda-kernels` crate extraction landing.

Additional touches in the same revision:

- Tier numbering: T0/T2/T3/T4 → T0/T1/T2/T3 for industry alignment (§4.1)
- `KVTransport` trait §5.3 updated to match the shipped code shape
  (`type Op: Send` + explicit `poll`, not `type Completion: Future`)
- §4.3 dual residency elevated to a first-class section with an
  invariant, not just "M2 will do this"
- §4.4 coordinator threading model committed to OS thread + crossbeam
  (the task doc §3.3 course correction)
- §8 pitfalls 10–12 added (the three prefix_cache correctness bugs)
- §11 industry comparison added (entirely new)
- §6 phase plan: P0–P5 replaced with M0–M5. The new sequencing puts the
  scheduler wire and the directory merge together as one atomic M1 PR
  instead of stacking them as P1(a)+P1(b), because the midway state of
  the original split is uncompilable
- §12 sources: 14 new industry-research references added

Original 2026-04-13 P0–P5 phase plan is preserved only in the commit
history of this file; the live plan is §6 M0–M5.

### 2026-04-13 — original plan

The 2026-04-13 version of this doc was the first draft of the tiered KV
implementation plan, written as a reaction to the discovery that
`RadixCache` was built but orphaned and `kv_tier/` was skeleton-only. It
specified P0–P5 phases, the two-layer `RadixCache → TierDirectory`
topology, and a single `BlockId(u64)` content-hash type. The 2026-04-15
revision supersedes all three of those as documented above.

---

## 14 · Next PR

**M0.1 — `BlockId` unification.** It is the only milestone that is both
(a) safe to start immediately and (b) unblocks every subsequent milestone.
It does not conflict with the in-flight `infer-cuda-kernels` crate
extraction (which touches `infer/src/backend/cuda/*` and `infer/csrc/cuda/*`,
not `infer/src/prefix_cache.rs` / `infer/src/types.rs` / `infer/src/kv_tier/id.rs`).

M0.2 (three prefix_cache bug fixes) can also start in parallel with M0.1
because it only touches `infer/src/prefix_cache.rs`.

M0.3 (`page_size` lift) **blocks on** the in-flight extraction's `.cu`
moves landing first.

M1 blocks on all three of M0.1 / M0.2 / M0.3.

---

*Live at commit efcc991; last revised 2026-04-15.*
