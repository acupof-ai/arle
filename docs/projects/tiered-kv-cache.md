# Tiered KV Cache — hierarchical KV with auto offload and forward-compat RDMA

**Status**: Active — opened 2026-04-13 as the implementation plan for turning
the agent-infer KV cache into a coordinator-managed hierarchy.

**Goal**: Every agent session reuses prior work across requests, survives
memory pressure without OOM, survives process restarts without paying the
cold-prefill tax, and — when we are ready to build multi-node serving —
extends to cross-node KV migration as a new transport impl, not as a rewrite.

This doc is the **implementation spec** for work items A1, B1, and B3 in
[`agent-first-architecture.md`](agent-first-architecture.md). Those three
items share one code topology and must be built against one data-structure
contract; splitting them into independent designs is how we end up with three
incompatible caches. This doc owns that contract.

This doc operates under the Phase-1 PR discipline in
[`../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md`](../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md):
one main topic per PR, structure-before-behavior, no mixed kernel+scheduler+
workspace diffs in the same review.

---

## 1 · Naming

We use the generic industry term, not a branded one. SGLang's "HiCache",
LMCache, and Mooncake Store are products; reusing any of those names would
confuse readers about which project they are looking at.

- **Code**: module `infer/src/kv_tier/` (flat layout: `kv_tier.rs` +
  `kv_tier/`). Types: `TieredKvCache`, `TierDirectory`, `Tier`, `BlockId`,
  `BlockDescriptor`, `KVTransport`, `EvictionPolicy`.
- **User-facing**: "Tiered KV Cache" or "hierarchical KV cache" — matches the
  vLLM/SGLang vocabulary without claiming implementation parity.
- **Not** `kv_fabric` — collides with `libfabric`/OpenFabrics, which is a
  concrete backend we may call through NIXL later.

---

## 2 · Non-goals

- **New attention kernels.** page_size is a pool/bookkeeping change, not a
  kernel rewrite; everything under `infer/csrc/cuda/` that touches paged KV
  already parameterizes page_size.
- **Metal hierarchy.** MLX unified memory makes T0↔T2 a self-memcpy; Metal
  only joins in Phase 3 for T3 (disk), and only because the wired-memory
  kernel panic in mlx-lm #883 forces us to bound the KV pool somehow.
- **Multi-node scheduling.** P5 designs the `KVTransport` trait so RDMA fits,
  but cross-node prefill/decode disagg is a separate project that lives on
  top of this one.
- **New storage backends beyond local disk and NIXL.** Mooncake Store, 3FS,
  S3, Valkey are all legitimate Phase 6 work; not in scope here.
- **Replacing the block manager on CPU backend.** `infer/src/cpu_backend.rs`
  remains a development stub.

---

## 3 · Current state (2026-04-13)

| Area | State | File:line |
|---|---|---|
| CUDA paged pool | `page_size = 1`, token-granular LIFO slot allocator | `infer/src/paged_kv.rs:7,760` |
| CUDA contiguous legacy KV | Has `k_host/v_host` CPU shadow buffers with `OFFLOAD_BLOCK_SIZE=64`, **default off** | `infer/src/model/kv_cache.rs:130-168` |
| RadixCache | 552 lines, leaf-LRU, refcount pinning, **no production consumer** | `infer/src/prefix_cache.rs` |
| CUDA scheduler prefix logic | `cached_prompts: Vec<Vec<u32>>` per-slot linear compare | `infer/src/scheduler/cuda/runtime.rs:75-151` |
| A2 session_id plumbing | Field lands in `IncomingRequest` and HTTP requests; scheduler does not yet consume it | `infer/src/scheduler/types.rs:113-128`, `infer/src/http_server/openai_v1.rs:21-54,222-248` |
| Metal KV pool | `SlotLedger` refcount-only, MLX unified memory, no tier concept | `infer/src/metal_kv_pool.rs` |
| Metal prefix cache | Wraps RadixCache, not wired into scheduler | `infer/src/metal_prefix_cache.rs` |
| Storage deps | None in any `Cargo.toml`; B1 session_store is paper-only | — |
| `infer::scheduler::policy` | `SchedulerSignals { prefix_hit_tokens, session_affinity_slot, turn_depth }` + `AdmissionPolicy` + `ChunkingPolicy` + `PrefixAwareAdmission` | `infer/src/scheduler/policy.rs` |

Three facts shape everything below:

1. The production data path is **single-tier** today. No CPU offload, no disk.
   The `KVCache::k_host` path exists but is not on in serving mode.
2. `RadixCache` is built but orphaned. Any tiered cache must go through it,
   not around it. Building the tier machinery before wiring radix would mean
   double-work.
3. `page_size = 1` is a choice, not a FlashInfer constraint. FlashInfer's
   `paged_kv_t::page_size` is a runtime `uint_fastdiv` field; upstream tests
   cover `{1, 5, 8, 16}`. SGLang and vLLM default to 16.

---

## 4 · Target architecture

```text
                   ┌──────────────────────────────────────────┐
                   │       RadixCache (block IDs only)        │
                   │   · session_id tags · recursive evict    │
                   └───────────────────┬──────────────────────┘
                                       │ lookup(tokens) → Vec<BlockId>
                   ┌───────────────────▼──────────────────────┐
                   │            TierDirectory                 │
                   │  BlockId → BlockDescriptor { tier,       │
                   │   location, hash, rc, last_access, sid } │
                   └───────────────────┬──────────────────────┘
        ┌──────────────┬───────────────┼───────────────┬───────────────┐
        ▼              ▼               ▼               ▼               ▼
    ┌───────┐     ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  T0   │     │   T2     │   │   T3     │   │   T4     │   │ (Metal)  │
    │ GPU   │     │ Pinned   │   │  NVMe    │   │ Remote   │   │ T0 + T3  │
    │ HBM   │     │  DRAM    │   │  SSD     │   │ (NIXL)   │   │   only   │
    └───────┘     └──────────┘   └──────────┘   └──────────┘   └──────────┘
        ▲              ▲              ▲              ▲
        │  cudaMemcpy  │   io_uring   │    NIXL      │
        │  D2H / H2D   │  tokio::fs   │   put/get    │
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                   kv_tier::Coordinator
                (tokio task, dedicated CUDA copy stream)
```

### 4.1 Tier semantics

| Tier | Medium | Latency class | Who reads/writes |
|---|---|---|---|
| **T0** | GPU HBM | ~0 (kernel direct) | Attention kernels |
| **T2** | Host pinned DRAM | ~10 µs via PCIe copy engine | Coordinator only (never direct kernel access) |
| **T3** | NVMe SSD | 10–100 µs via io_uring / `O_DIRECT` | Coordinator only |
| **T4** | Remote node | 1–50 µs over RDMA via NIXL | Coordinator only; Phase 5+ |

T1 (GPU-warm, HBM but not radix-active) is **intentionally cut**. It adds a
tier with the same hardware as T0, which is complexity for no new capacity.

T2 is meaningful only on CUDA. On Apple Silicon (MLX / `MTLStorageModeShared`),
CPU and GPU share one physical DRAM region; "offloading to host" is a self-
memcpy that buys nothing. The Metal backend treats T2 as a compile-time no-op
(see §10) and only gets a T0+T3 configuration in Phase 3.

### 4.2 Invariants

1. **RadixCache nodes carry BlockId, not slots.** The radix tree never touches
   physical memory. Slot resolution goes through `TierDirectory::resolve`.
2. **Only the coordinator moves blocks between tiers.** The scheduler emits
   intents (`Demote`, `Promote`, `Pin`, `Unpin`); the coordinator owns the
   copy stream, the in-flight IO queue, and tier transitions. No tier
   transitions on the scheduler main loop.
3. **Directory is the single source of truth for `tier`.** A block's tier
   changes atomically at directory commit; the pool, prefix cache, and
   transport never maintain their own tier bookkeeping.
4. **BlockId is stable across restarts and across nodes.** It must be a
   content hash, not a monotonic counter. Exact formula: see §5.1.
5. **MR registration stability.** T2 pinned regions are allocated once at
   pool init and never reallocated; this is a precondition for NIXL MR
   registration in Phase 5.

---

## 5 · Data structures (the contract)

### 5.1 Block identity

```rust
// infer/src/kv_tier/id.rs

/// Content-addressable identifier for a KV block. Stable across processes and
/// across nodes. Two nodes that independently prefill the same prefix produce
/// the same BlockId; that is the foundation for Phase 5 remote tier reuse.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BlockId(u64);

impl BlockId {
    /// Deterministic derivation used by RadixCache insertion.
    /// Hash inputs, in order:
    ///   1. model identity fingerprint (arch + weight digest + numeric profile)
    ///   2. layer index
    ///   3. kv format (bf16 / fp8e4m3 / int8 / turboquant-2/3/4)
    ///   4. parent block hash (chains the radix path into the hash)
    ///   5. token ids of THIS block, in order
    /// Collisions must never silently alias distinct content; we use blake3
    /// reduced to u64 and additionally verify `parent_hash` on radix insert.
    pub fn derive(ctx: &BlockHashCtx, tokens: &[u32]) -> Self { /* ... */ }
}

pub struct BlockHashCtx {
    pub model_fingerprint: u64,
    pub layer_idx: u16,
    pub kv_format: KvFormat,
    pub parent_hash: u64,
}
```

The `parent_hash` chain is what lets us detect radix-path divergence at block
granularity without walking the tree; it also gives the Phase 5 remote tier a
path-independent reuse key.

### 5.2 Directory and descriptors

```rust
// infer/src/kv_tier/directory.rs

pub enum Tier { Gpu, HostPinned, Disk, Remote }

pub enum BlockLocation {
    Gpu { slot: u32 },
    HostPinned { offset: u64 },          // offset within the pinned region
    Disk { file_id: u32, offset: u64 },
    Remote { node: NodeId, desc: OpaqueRemoteDesc },
}

pub struct BlockDescriptor {
    pub id: BlockId,
    pub tier: Tier,
    pub location: BlockLocation,
    pub byte_len: u32,                   // includes quantization scales
    pub ref_count: u32,                  // radix refcount + active requests
    pub last_access: u64,                // monotonic tick
    pub session_id: Option<SessionId>,   // from A2; None = cross-session block
    pub pin_until: Option<u64>,          // optional soft pin (keepalive)
}

/// Single source of truth. Lock-granularity decision: one RwLock over the
/// descriptor map for Phase 1/2 (single scheduler thread + one coordinator);
/// shard the map in Phase 5 when the coordinator pipeline becomes plural.
pub struct TierDirectory {
    blocks: DashMap<BlockId, BlockDescriptor>,
    // reverse indexes (tier → block ids, session → block ids) built lazily
}

impl TierDirectory {
    pub fn resolve(&self, id: BlockId) -> Option<BlockDescriptor>;
    pub fn insert(&self, desc: BlockDescriptor) -> Result<()>;
    pub fn promote(&self, id: BlockId, to: Tier, loc: BlockLocation) -> Result<()>;
    pub fn demote(&self, id: BlockId, to: Tier, loc: BlockLocation) -> Result<()>;
    pub fn touch(&self, id: BlockId, now: u64);
    pub fn pin(&self, id: BlockId); pub fn unpin(&self, id: BlockId);
}
```

### 5.3 KVTransport trait (forward-compat for RDMA)

```rust
// infer/src/kv_tier/transport.rs

pub enum MemKind { HostPinned, CudaDevice, CudaManaged, MetalUnified }

pub struct TransferOp {
    pub src: BlockLocation,
    pub dst: BlockLocation,
    pub len: u32,
}

pub trait KVTransport: Send + Sync {
    type Region;      // registered memory region handle (MR)
    type Completion;  // poll/await friendly

    // Registration — UCX, NIXL, Mooncake all require pre-registered MRs.
    // Must be called on allocation-stable regions only. See §4.2 invariant 5.
    fn register(&self, ptr: *mut u8, len: usize, kind: MemKind) -> Result<Self::Region>;
    fn deregister(&self, region: Self::Region) -> Result<()>;

    // Batched one-sided transfer. Agent traces dump O(layers × blocks) per
    // migration; per-block syscalls are not viable.
    fn put_batch(&self, ops: &[TransferOp]) -> Self::Completion;
    fn get_batch(&self, ops: &[TransferOp]) -> Self::Completion;
}

/// Remote descriptor content is opaque; NIXL serializes agent metadata,
/// Mooncake serializes segment handles, raw verbs serialize (rkey, addr, qpn).
/// Cross-backend code never parses it.
pub struct OpaqueRemoteDesc(pub smallvec::SmallVec<[u8; 32]>);
```

Transport implementations, in order of planned delivery:

- **`LocalCudaTransport`** (Phase 2). `cudaMemcpyAsync` on a dedicated copy
  stream. Covers T0↔T2. Registration is a nop beyond storing the base ptr.
- **`DiskTransport`** (Phase 3). `io_uring` (Linux) / `tokio::fs` fallback,
  `O_DIRECT` where possible. Covers T2↔T3.
- **`NixlTransport`** (Phase 5, `#[cfg(feature = "rdma-nixl")]`). Links
  `nixl-sys` from crates.io; uses its `stub-api` feature on CI builds without
  the native lib. Covers T4. NIXL in turn delegates to UCX, GDS, or Mooncake.

### 5.4 EvictionPolicy (lives in `infer::scheduler::policy`)

```rust
// infer/src/scheduler/policy.rs — sibling to AdmissionPolicy / ChunkingPolicy

pub enum SessionState { Active, Keepalive, Cold }

pub struct EvictionCandidate {
    pub last_access: u64,
    pub ref_count: u32,
    pub block_count: u32,
    pub session_state: SessionState,
    pub session_id: Option<SessionId>,
}

pub trait EvictionPolicy: Send + Sync {
    /// Score per candidate; lower = evict sooner. Pure function of candidate
    /// and signals; must not mutate state.
    fn score(&self, c: &EvictionCandidate, sig: &SchedulerSignals) -> i64;
}

/// Default policy. LRU over last_access, with session-aware bucketing.
/// Matches SGLang's recursive leaf eviction plus a keepalive window that the
/// KVFlow paper credits for the 1.83–2.19× agent-workload win over HiRadix LRU.
pub struct SessionBiasedLru {
    pub active_weight: i64,     // e.g. -1_000_000 (push way back)
    pub keepalive_weight: i64,  // e.g. -100_000
    pub keepalive_ticks: u64,   // default 30s worth of ticks
}
```

---

## 6 · Phase plan

Every phase below is the skeleton of a single PR (or a short stacked series).
Exit criteria are observable: test passes, benchmark delta, or the explicit
code-path appearance of a named type. No phase is "done when it feels done".

### P0 — page_size = 16 (pure refactor, zero behavior change)

**What**: Raise `TokenKVPool::page_size` from 1 to 16. Rewrite the pool as a
two-level allocator (allocate a new page when `seq_len % page_size == 0`,
else append to the tail page). Rewrite the FlashInfer metadata incremental
update to distinguish "spills to new page" from "bumps last_page_len".

**Why this goes first**: It is the only phase that is 100% structural with a
numerical-parity exit criterion (`greedy_consistency` + `e2e_qwen35`). It
halves the FlashInfer indices traffic at 32k context × 8-batch, improves
HND coalescing in `paged_kv_append.cu`, and aligns us with SGLang/vLLM
defaults so subsequent cross-system comparison is apples-to-apples.

**Files**:
- `infer/src/paged_kv.rs:7,76-87,482-511,549-562,614,760` — `page_size`
  field, two-level `alloc_tokens`, `build_flashinfer_metadata` /
  `build_last_page_lens` rewrite, `migrate_from_contiguous` literal `1` → `page_size`.
- `infer/src/flashinfer_metadata.rs:125,161` — incremental update: new token
  lands inside last page (bump `last_page_len`) vs spills to new page
  (append page id); comment at `:161` must be rewritten.
- `infer/src/model/qwen3/batch_decode.rs:384`,
  `infer/src/model/qwen35/batch_decode.rs:724`,
  `infer/src/model/glm4/batch_decode.rs:338` — drop `let page_size = 1;`
  locals, read from pool.
- `infer/src/scheduler/cuda/decode.rs:193` — literal `1` → `pool.page_size`.

No kernel changes. `infer/csrc/cuda/paged_kv_append.cu`,
`decode_prep_paged.cu`, `decode_prep_paged_hd256.cu`, `kv_cache_to_paged.cu`
all already compute `pos / page_size` correctly.

**Exit**:
1. `cargo test --release --test e2e` and `--test e2e_qwen35` pass unchanged.
2. `greedy_consistency` passes unchanged.
3. `scripts/bench_throughput_sweep.py --label page16` recorded vs a pre-PR
   `--label page1` baseline in `docs/experience/wins/`.
4. FlashInfer split-KV scheduler does not lose parallelism on short-context
   single-request benches (watch the tail in the sweep).

**Risk**: FlashInfer's split-KV fans out by `max_num_pages_per_batch`; at
very short contexts and batch=1 a larger page size can reduce fan-out. The
bench gate is there to catch it. Mitigation if real: keep page_size=1 on a
short-context fast path and page_size=16 otherwise — but do not build this
pre-emptively.

### P1 — wire RadixCache + introduce TierDirectory + session tags

**Scope**: Item A1 in `agent-first-architecture.md` (radix cache into CUDA
scheduler) is **folded in entirely**. A1 no longer exists as a separate work
item; its code ships here.

**What**: Delete `cached_prompts: Vec<Vec<u32>>`. Add `infer/src/kv_tier/`
with `id.rs`, `directory.rs`, and a `TieredKvCache` facade that owns a
`TierDirectory` and the T0 `TokenKVPool`. RadixCache nodes carry
`session_id` and a `parent_hash` for the block id chain; eviction walks up
to parents when a leaf dies (SGLang-style recursive). On request admission,
the scheduler queries the radix, receives `Vec<BlockId>`, and the
`TieredKvCache` resolves each to a physical slot (all T0 in this phase).

**Split into two stacked PRs**:
1. **Structural PR** — add `infer/src/kv_tier/`, derive `Serialize/Deserialize`
   on `RadixCache` / `Node`, add `session_id` + `parent_hash` fields, add
   recursive parent eviction, add `TieredKvCache` façade. No scheduler
   changes. All new types exercised by unit tests only.
2. **Behavior PR** — swap `scheduler/cuda/runtime.rs::best_prefix_slot` to
   query `TieredKvCache`; delete `cached_prompts`; scheduler consumes
   `IncomingRequest::session_id` and passes it to insertion. Exit criterion
   is a cross-session prefix-hit benchmark.

**Files (structural)**:
- `infer/src/prefix_cache.rs` — add fields, serde derives, recursive evict,
  change `lookup` return to `(matched_len, Vec<BlockId>)`.
- `infer/src/kv_tier.rs` + `infer/src/kv_tier/` — new module (flat layout).
- `infer/src/lib.rs` — declare the module.
- `infer/src/paged_kv.rs` — expose `alloc_slot/free_slot/read_into/write_from`
  as the T0 physical layer; the pool becomes the backing store that
  `TieredKvCache` drives.

**Files (behavior)**:
- `infer/src/scheduler/cuda/core.rs:114-141` — hold `Arc<TieredKvCache>`;
  remove `cached_prompts`.
- `infer/src/scheduler/cuda/runtime.rs:75-151` — admission rewrite. Radix
  lookup → block ref grant → emit prefill chunk for the suffix only.
- `infer/src/scheduler/cuda/request.rs` — `ActiveRequest` holds the
  `Vec<BlockId>` it is building up; on finish, commit to directory and
  release refcounts.
- `infer/src/server_engine.rs:437-475` — drop the per-engine
  `cached_prompt` second-source-of-truth.

**Exit**:
1. `grep -r cached_prompts infer/src/` returns empty.
2. `grep -r RadixCache infer/src/scheduler/` returns non-empty (the
   regression gate called out in `agent-first-architecture.md §5`).
3. A cross-session concurrent benchmark replayed by
   `scripts/bench_agent_trace.py` (renamed from the C6 proposal's
   `bench_agent.py` to avoid colliding with the existing binary-subprocess
   benchmark of that name; item C6 is folded into this phase's scoreboard)
   shows ≥70% prefix hit rate on a 2-session alternating trace.
4. README's "radix-tree prefix cache" claim is now substantiated and the
   diagram updated.

### P2 — T2 host pinned tier + coordinator + auto offload

**Scope**: This is the "automatic offload" phase the user asked for. Item B3
(prefix/session policy signals) is folded in: its `SessionBiasedLru` is the
`EvictionPolicy` default added here.

**What**: Add the pinned host pool (stable base pointer, allocated once at
engine init). Add `kv_tier::Coordinator` as a dedicated tokio task owning a
dedicated CUDA copy stream. Scheduler emits `Demote`/`Promote` intents; the
coordinator batches them and overlaps with compute. Add high/low watermarks
on T0 usage; the coordinator triggers demotion when `T0 > high_watermark`.
Delete the dormant `KVCache::k_host` path — it is strictly inferior and
confuses the offload story.

**Split into two stacked PRs**:
1. **Structural PR** — add `LocalCudaTransport`, `Coordinator`, pinned pool,
   watermark types. `TieredKvCache` gains `demote`/`promote` methods, but
   scheduler does not call them yet. `EvictionPolicy` trait lands in
   internal module `infer::scheduler::policy` with the `SessionBiasedLru`
   default. No observable
   behavior change (watermarks set so T0 never fills).
2. **Behavior PR** — scheduler calls `TieredKvCache::evict_if_needed(sig)`
   at admission and post-decode; coordinator actually moves blocks. Delete
   `infer/src/model/kv_cache.rs:130-168` CPU offload code after confirming
   zero production callers.

**Files (structural)**:
- `infer/src/kv_tier/transport.rs`, `infer/src/kv_tier/transport/local_cuda.rs`
- `infer/src/kv_tier/coordinator.rs` — tokio task, copy-stream ownership,
  in-flight IO tracking, cancel-safe.
- `infer/src/kv_tier/host_pool.rs` — stable-pointer pinned pool (backed by
  `cudaHostAlloc`).
- `infer/src/scheduler/policy.rs` — `EvictionPolicy` trait +
  `SessionBiasedLru` + `EvictionCandidate`.

**Files (behavior)**:
- `infer/src/scheduler/cuda/runtime.rs` — eviction hook at admission.
- `infer/src/scheduler/cuda/core.rs` — post-finish, keepalive-stamp any block
  whose session is still active.
- `infer/src/model/kv_cache.rs:130-168` — **delete** `k_host`, `v_host`,
  `ensure_on_gpu`, `offload_if_needed`. Diff-before-delete: confirm
  `grep -r 'offload_if_needed\|ensure_on_gpu' infer/src/` returns only this
  file and its tests before the PR lands.

**Exit**:
1. A long-agent-session benchmark (32k+ cumulative tokens, num_slots=4) that
   OOMs on current main runs to completion on this branch.
2. `scripts/bench_throughput_sweep.py --label tier-T2` recorded vs a
   pre-P2 baseline. Steady-state decode throughput regression ≤ 3% (auto
   offload is a capacity win, not a throughput win).
3. `cargo test --release --test e2e_qwen35` unchanged.

### P3 — T3 disk tier + session save/load + first Metal contact

**Scope**: Item B1 (session KV snapshot persistence) is folded in entirely.
B1 no longer exists as a separate work item. Metal backend joins the tiered
cache for the first time, but only at T0 + T3 (see §10).

**What**: Add `DiskTransport` (io_uring on Linux, `tokio::fs` fallback,
optional `O_DIRECT`). Add `POST /v1/sessions/{id}/save` and
`POST /v1/sessions/{id}/load` routes. Save serializes the radix subtree owned
by `session_id` plus its T0/T2 blocks through the disk transport; load
re-hydrates directly into T2 or T3, marks the radix subtree as resident, and
lets the coordinator promote on the next request touching it. Bind
`mlx.metal.set_wired_limit` and `get_active_memory` in `mlx-sys` so the
Metal backend has the telemetry it needs to enforce a bounded KV pool (the
mlx-lm #883 panic mitigation).

**Files (structural)**:
- `infer/src/kv_tier/transport/disk.rs` — io_uring + tokio::fs,
  allocation-stable region (one large file per node).
- `infer/src/http_server.rs:422-427` — new routes; re-use existing router.
- `infer/src/http_server/sessions.rs` — new, save/load handlers.
- `infer/mlx-sys/src/lib.rs` — bindings for `mlx_metal_set_wired_limit` and
  `mlx_metal_get_active_memory`.
- `infer/src/metal_kv_pool.rs` — read `set_wired_limit` / `get_active_memory`
  at init to cap `max_total_tokens`; no tier logic yet.
- `infer/src/metal_prefix_cache.rs` — disk tier hook via the same
  `TieredKvCache` façade.

**Files (behavior)**:
- `crates/infer-agent/src/lib.rs:166-188` — `save_to_path` / `load_from_path`
  gain an optional KV snapshot side-channel; JSON message persistence
  continues to work unchanged for clients that do not opt in.

**Exit**:
1. Restart smoke test: save a 30k-token system prompt session, kill the
   process, restart, reload, measure TTFT. Must land within 20% of the
   pre-restart warm-state baseline.
2. Metal backend with bounded `max_total_tokens` runs a long-context test
   without a `prepare count underflow` kernel panic (the mlx-lm #883 failure
   mode).
3. `grep -r session_store infer/` returns only references into `kv_tier`
   (B1's proposed standalone `session_store.rs` never lands; its
   functionality ships as `kv_tier::transport::disk` plus the HTTP handlers).

### P4 — reuse-distance prediction + cache-aware routing

**Scope**: Replace `SessionBiasedLru` default with a KVFlow-lite policy that
reads session turn history to predict next-access time, and make the
multi-slot scheduler prefer the slot whose radix subtree already holds the
most of the incoming prefix (Mooncake-style cache-aware routing).

**What**: Use A2's `turn_depth` and per-session turn-interarrival history to
compute `expected_next_access_time` per block; bias eviction by that instead
of raw `last_access`. Separately, in `scheduler/cuda/runtime.rs`, when
admitting a session request, rank candidate slots by radix-subtree overlap,
not just emptiness.

**Files**:
- `infer/src/scheduler/policy.rs` — new `ReuseDistancePolicy` impl; expose
  a feature flag or config knob to select it over `SessionBiasedLru`.
- `infer/src/scheduler/cuda/runtime.rs` — slot selection rewrite.
- `infer/src/kv_tier/directory.rs` — add per-session turn-arrival ring
  buffer.

**Exit**:
1. Cross-session prefix hit rate ≥ 85% on the same trace P1 hit ≥70% on.
2. KVFlow paper claim reproduction (hand-wavy target: ≥1.5× throughput on
   an agent-sequence benchmark vs `SessionBiasedLru`). If we cannot
   reproduce ≥1.2×, keep `SessionBiasedLru` as the default and ship the new
   policy behind a flag.

### P5 — KVTransport trait freeze + NixlTransport stub

**What**: Freeze the `KVTransport` trait surface. Add
`infer/src/kv_tier/transport/nixl.rs` behind `#[cfg(feature = "rdma-nixl")]`.
Depend on `nixl-sys = "0.10"` with the `stub-api` feature active in CI so
default builds compile without the native library. No routing changes.

**What this phase is NOT**: an actual RDMA serving path. It only guarantees
that when Phase 6 arrives, adding it is one transport impl, not a redesign.

**Files**:
- `infer/src/kv_tier/transport/nixl.rs` — `NixlTransport` skeleton, all
  methods wrap `nixl-sys` calls, returns `todo!("P6")` inside `put_batch`
  etc. but `register`/`deregister` are fully implemented because P5's gate
  is "can we register a pool region with NIXL without a runtime error".
- `infer/Cargo.toml` — `rdma-nixl = ["dep:nixl-sys"]` feature.
- `infer/src/events.rs` — new event kind for tier transitions so
  P6 can observe transfer latencies the same way.

**Exit**:
1. `cargo check --features rdma-nixl` compiles on a dev machine with
   `nixl-sys` stub-api. Not a CI gate; a manual smoke.
2. The trait signature has not changed from P2's draft — if it has, we go
   back and adjust P2 rather than fork the trait.

### P6 — real RDMA / cross-node KV migration

Separate project doc. Will depend on a multi-node test harness that does
not exist in this repo. Not scoped here.

---

## 7 · PR splitting discipline

| Phase | PRs | Structure / behavior |
|---|---|---|
| P0 | 1 | Pure structural; greedy parity gate |
| P1 | 2 stacked | (a) kv_tier module + radix serde/session/recursive evict — structural; (b) scheduler swap — behavior |
| P2 | 2 stacked | (a) coordinator + transport + pinned pool — structural; (b) watermark triggers + k_host deletion — behavior |
| P3 | 2 stacked | (a) disk transport + MLX bindings — structural; (b) session HTTP routes + save/load — behavior |
| P4 | 1 | Behavior (policy swap); gated on ≥1.2× reproduction |
| P5 | 1 | Structural (trait freeze + stub) |

No PR in this doc mixes kernel and scheduler changes. P0's kernel files are
untouched (already page-size aware); P1–P5 have zero kernel diffs.

---

## 8 · Pitfalls we already know about

These are the known sharp edges — the doc loses value if we discover them
again during implementation.

1. **MLX wired memory panic** (mlx-lm #883). MLX wires all allocations by
   default; an unbounded KV pool hits `prepare count underflow` in
   `IOGPUMemory` before the OS gets a chance to page. P3 must bind
   `set_wired_limit` and `get_active_memory` before enabling T3 on Metal,
   and `MetalKVPool::new` must cap `max_total_tokens` based on the live
   wired-memory budget.
2. **MR registration invalidation.** UCX, NIXL, and Mooncake all require
   pre-registered memory regions. If the T2 pinned pool ever reallocates
   or compacts, registered MRs become dangling. Allocate the pool once at
   engine init, never grow. If we need to grow, register the new region
   before freeing the old one.
3. **FlashInfer split-KV parallelism at short contexts.** P0's risk. The
   benchmark gate catches it; do not pre-emptively build a page_size=1
   fast path.
4. **`nvidia-peermem` vs old `nv_peer_mem`.** GDR needs the former; old
   docs and third-party crates still reference the latter. P5 must probe
   for `nvidia-peermem` and fall back to bounce buffer.
5. **NIXL stack requirements.** CUDA 12+, UCX 1.19/1.20, NIXL native lib
   at link time. `nixl-sys` stub-api feature is how we keep default CI
   green. Gate behind `rdma-nixl` feature.
6. **Mooncake metadata service.** If we add a Mooncake transport later,
   Mooncake Store needs etcd (or its own master). That is a deployment-
   story decision, not a transport-trait one. Keep the trait oblivious.
7. **Block id collision.** `BlockId::derive` uses blake3 reduced to u64.
   At 2^32 blocks the birthday bound is non-zero; radix insert must
   verify `parent_hash` chain to catch the pathological case.
8. **Scheduler single-threadedness.** Today the scheduler owns all KV
   under one thread and needs no locks. The coordinator is a second owner.
   P2's `TierDirectory` uses `DashMap` or `RwLock` and must be audited
   for cancel-safety at every `await` point.
9. **`metal_gdr.rs` is not GPUDirect RDMA.** The filename is misleading;
   it is the Qwen3.5 Gated Delta Rule linear-attention decoder. Do not
   reuse that module for transport work.

---

## 9 · Relationship to other docs

- [`agent-first-architecture.md`](agent-first-architecture.md) — owns A1,
  B1, B3. This doc **supersedes** those three items' implementation shape.
  When P1 lands, A1 moves to the Done section; when P2 lands, B3 moves;
  when P3 lands, B1 moves. `agent-first-architecture.md` gets an update
  pointer to this doc in the same PR series.
- [`kv-quantization-long-context.md`](kv-quantization-long-context.md) —
  KV quantization formats (FP8, INT8, TurboQuant) live inside T0 blocks.
  `BlockDescriptor.byte_len` must account for scale bytes. P0 must not
  regress the quantized fast paths; P2 coordinator must preserve format
  across tier transitions.
- [`mlx-backend-roadmap.md`](mlx-backend-roadmap.md) — Metal side. P3 is
  the first point of contact; the MLX roadmap should link back here once
  P3 enters execution.
- [`../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md`](../../infer/docs/projects/art-grade-architecture-for-long-agent-infer.md) —
  workspace crate topology and PR discipline. This doc operates under its
  rules; when P2 or later needs a promoted `infer-kv-tier` crate, that
  promotion goes through the crate-admission criteria there.

---

## 10 · Backend coverage summary

| Backend | P0 | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|
| CUDA | page16 refactor | radix wired, T0 directory | T0+T2 + coordinator | +T3 disk | reuse-dist policy | NIXL stub |
| Metal | n/a (MLX path untouched) | radix wired via `metal_prefix_cache` | **nop** (unified memory) | T0 bound + T3 disk | reuse-dist policy (shared) | n/a (no RDMA) |
| CPU backend | untouched (stub) | untouched | untouched | untouched | untouched | untouched |

The Metal column's P2 entry is a no-op intentionally; see §4.1.

---

## 11 · Open research items

None critical. The five questions that gated the design (FlashInfer
constraint, Metal tiering value, RDMA stack, eviction policy, B1 overlap)
were each investigated and answered; this doc encodes those answers. If any
of them turn out wrong during implementation, the correct response is to
update this doc first, then adjust the phase plan.

One deferred research question, relevant only for Phase 6:

- **Metadata service for multi-node deployments.** Mooncake needs etcd or
  its own master daemon; NIXL's agent-metadata model needs some discovery
  mechanism. That decision is outside the trait; it is a deployment-story
  decision that belongs to a Phase 6 project doc.

---

## 12 · Sources

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

---

## 13 · Next PR

**P0 — bump page_size to 16.** This is the only phase that can start
immediately; every other phase depends on P0's pool interface. Before
opening the PR:

1. Record `scripts/bench_throughput_sweep.py --label page1` as the
   baseline in `docs/experience/wins/2026-04-13-bench-page1.md`. This is
   immutable history per the benchmark rules.
2. Draft the two-level `TokenKVPool::alloc_tokens` against the seven file
   points listed in §6 P0.
3. Run the full test suite on both CUDA and Metal builds (Metal is
   unaffected but the test matrix must stay green).
4. Open the PR with a one-sentence title and the §6 P0 entry as the body.

When P0 lands, update this doc's "Current state" table and open P1.
