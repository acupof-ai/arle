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

## 3 · Current state (2026-04-15, post M1+M2a + Codex review)

Updated after M1a + M1b + M2a landed (commits `08718ad`, `323aee0`, `4402ab0`)
and the Codex 2026-04-15 review surfaced two issues the previous draft of
this section under-emphasised: the legacy `model/kv_cache.rs` CPU offload
as a parallel "second source of truth", and the `policy.rs` scoring trait
being bypassed by M2a's hand-rolled watermark. Both are flagged below.

| Area | State | File:line |
|---|---|---|
| CUDA paged pool | `page_size = 1`, token-granular LIFO + **per-page refcount via M2a** (`page_ref_count: Vec<u32>` + `retain_pages` / `release_pages` / refcount-aware `free_slot` — pages with refcount > 0 stay in limbo across `free_slot`) | `crates/infer-cuda-kernels/src/paged_kv.rs:71-83,416-440` |
| CUDA legacy contiguous KV offload | Separate code path: `k_host/v_host` shadow buffers with `OFFLOAD_BLOCK_SIZE = 64`, default off in serving but still compiles, still has tests, still shapes the model batch-decode call sites. **Must be retired before M3 or we end up with two truths for "where is layer L's KV right now"** — see §8 pitfall 13. | `infer/src/model/kv_cache.rs:29,517-590` |
| `infer/src/prefix_cache.rs` | 914 lines, leaf-LRU + cascaded eviction + ancestor-chain ref bumping (3 historical bugs fixed in `5da8b67`); **wired into scheduler as M2a authoritative pinned-page store**. Private `Node` still carries only `(tokens, block_id: Option<BlockId>, ref_count, last_access, children)` — **no `tier_location` / `fingerprint` / `session_id` / `byte_len` / `soft_pin_until` fields yet**, those are M2b/M3 work. | `infer/src/prefix_cache.rs:55-69` |
| CUDA scheduler prefix logic | **M1b shipped**: radix runs as a shadow observer, logs cross-slot hit length per admission. **M2a shipped**: refcount-aware `free_slot` + side map `block_to_pages: HashMap<BlockId, Vec<u32>>` + `evict_prefix_cache_if_pressured` (high=0.75, low=0.50). **Slot selection still goes through legacy `cached_prompts: Vec<Vec<u32>>` linear scan** — the M2b behavior flip has not landed. | `infer/src/scheduler/cuda/core.rs:42,333,430`, `infer/src/scheduler/cuda/runtime.rs:20-30,191` |
| `infer/src/kv_tier/` | `directory.rs` **deleted** in M1a (commit `08718ad`). Remaining: `tier.rs`, `transport.rs`, `transport/disk.rs`, `transport/nixl.rs`. `DiskStore` round-trips bytes; `NixlTransport` stub compiles under `rdma-nixl` (stub-api). **Zero scheduler consumers**, no coordinator yet. | `infer/src/kv_tier/**` |
| `BlockId` unification | **Done (M0.1).** `infer/src/types.rs:8` is canonical `BlockId(u32)`; `prefix_cache::BlockId` and `kv_tier::id::BlockId` re-export. `block_manager::BlockId` deleted. `BlockFingerprint([u8; 16])` lives next to it but **has zero call sites** — M4 session save/load needs to be the first consumer (see §5.1, §6 M4). | `infer/src/types.rs:8` |
| `infer::scheduler::policy` | Trait + 4 impls (`LruEviction`, `ReuseBiasedLru`, `HitCountLru`, `SessionBiasedLru`) plus `EvictionCandidate` data struct + `SchedulerSignals`. **Zero call sites — `evict_prefix_cache_if_pressured` is a hand-rolled high/low-watermark loop that bypasses `EvictionPolicy::score` entirely**. M3 must converge them; see §5.4. | `infer/src/scheduler/policy.rs:178,226,246,277,310` |
| A2 session_id plumbing | `IncomingRequest::session_id` populated from HTTP; scheduler does not consume it yet | `infer/src/scheduler/types.rs`, `infer/src/http_server/openai_v1.rs` |
| Metal KV pool | `SlotLedger` refcount-only, MLX unified memory, no tier concept. Untouched by M1/M2a. | `infer/src/backend/metal/kv_pool.rs` |
| Metal prefix cache | Wraps RadixCache, not wired into the Metal scheduler. | `infer/src/backend/metal/prefix_cache.rs` |
| Storage deps | `nixl-sys = "1.0"` optional, behind `rdma-nixl` (stub-api) and `rdma-nixl-real` features | `infer/Cargo.toml` |
| **Granularity mismatch** | Three different "block sizes" coexist: paged pool `page_size = 1` (current), planned `page_size = 16` (M0.3, BF16/FP16 only), legacy offload `OFFLOAD_BLOCK_SIZE = 64`. **No tier transfer can begin until exactly one is canonical for a given format×backend×tier triple** — see §8 pitfall 13. | `crates/infer-cuda-kernels/src/paged_kv.rs`, `infer/src/model/kv_cache.rs:29` |

Seven facts shape everything below (original fact 6 "P1(a) shipped, P1(b) never did" retired by M1b landing; replaced with the policy.rs / hand-rolled watermark divergence that Codex 2026-04-15 surfaced):

1. **Production data path is single-tier today, but T0 dual residency is half-built.** `paged_kv.rs:416-440` retains pages through `free_slot` when the radix holds them, so cross-request prefix bytes physically survive. The **selector** has not been switched to consume them — `assign_slots` still picks via `cached_prompts`. M2b is the behavior flip.
2. **`RadixCache` is wired but not yet load-bearing.** It is M2a's authoritative pinned-page store, but `runtime.rs:191` (`best_prefix_slot_for_cached_prompts`) still drives slot selection. The radix's `lookup` runs only as a shadow observer, logging the achievable cross-slot hit length so M2b's selector flip can be measured against it.
3. **`page_size = 1` is still the production path.** M0.3 (lift to 16 for BF16/FP16, keep 1 for INT8/FP8/quant) is unblocked and pending. **No tier work can begin until M0.3 lands** — `cudaMemcpyAsync` of 16 separate non-contiguous pages per "logical block" is bandwidth-hostile. See §6 M3a sequencing.
4. **`BlockId` unified, `BlockFingerprint` exists but unused.** When session save/load (M4) lands, **save format must address blocks by fingerprint, not by `BlockId`** — pool slot ids do not survive a restart. The serde derives on `RadixCache` are forward-compat scaffolding only; reload requires a fingerprint-based reconciliation pass that has not been written. Doc previously did not call this out; §6 M4 now does.
5. **Two parallel CPU offload code paths exist.** The new paged-pool tiering (§4) and the old contiguous-KV `model/kv_cache.rs` offload (`OFFLOAD_BLOCK_SIZE = 64`) cannot both grow. The old path is dormant in serving but still compiles, has tests, and shapes model call sites. **M3c must delete it before T0↔T1 transfer through the new path goes live**, otherwise we end up with two truths for "where is layer L's KV right now". See §8 pitfall 13.
6. **`policy.rs` scoring trait is shipped but bypassed.** `evict_prefix_cache_if_pressured` in `scheduler/cuda/core.rs:430` is a hand-rolled watermark loop that does not consult `EvictionPolicy::score`. M3 must converge eviction onto the policy trait — otherwise we have two competing eviction implementations the moment T1 lands. See §5.4 convergence note.
7. **`NixlTransport` trait shape is the right bet.** `type Op: Send` + explicit `poll()` + `abort()` — survives the Codex review unchanged. NIXL ↔ Mooncake plugin compatibility confirmed in 2026-04-15 industry research.

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
8. **Decode tail is never sealed mid-block.** Decode appends one token at
   a time. A page that holds fewer than `page_size` valid tokens is the
   request's hot tail and is **not eligible** for radix insertion, tier
   transfer, or fingerprint computation. Only when the tail page fills
   does the scheduler hand it to `RadixCache::insert` and to the
   coordinator as a candidate for demotion. Without this rule, the
   coordinator would either ship partial blocks across PCIe (wasting
   bandwidth) or maintain "in-progress block" state on every page (a
   second source of truth for tier location).
9. **Old contiguous CPU offload is mutually exclusive with paged-pool
   tiering.** `infer/src/model/kv_cache.rs:517-590` (`offload_if_needed`,
   `OFFLOAD_BLOCK_SIZE = 64`) and the new paged-pool T0↔T1 path cannot
   both be active for the same model. M3c retires the old path. Until
   then, every model batch-decode call site that consults `kv_cache.rs`
   must continue to assume single-tier; no model code may take a
   dependency on either path's tier metadata.
10. **Demote state machine on every pool page.** A page's lifecycle is
    `Free → Resident → Demoting → Free` (or `Demoting → Resident` if a
    cancel-on-hit overrides the demote). The Coordinator owns the
    `Demoting` transition, and the page is only released back to the
    pool's free list once the `cudaMemcpyAsync` to T1 has completed
    *and* refcount has dropped to 0 *and* no concurrent `lookup` has
    upgraded it back to `Resident`. Pages in `Demoting` state must not
    be reissued by `alloc_block`; they remain physically reachable for
    in-flight reads. Codex flagged the read-after-evict race here as
    M3's primary correctness risk; see §6 M3b.

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
and `crates/infer-cuda-kernels/src/paged_kv.rs::TokenKVPool`. No new
module. M2a already implemented the refcount half (`page_ref_count`,
`retain_pages`, `release_pages`, refcount-aware `free_slot`); M2b adds
the selector-flip half.

#### 4.3.1 Tombstone vs real entry (Codex review 2026-04-15)

Once T1/T2/T3 are in play, a `RadixCache` node may be in one of two
states:

- **Real entry** — node carries a live `block_id` resolvable to a
  `TierLocation` with bytes physically resident in some tier. `lookup`
  hit on a real entry returns the bytes' location; the scheduler can
  decode (or stage and decode).
- **Index-only tombstone** — node carries the prefix tokens and
  optionally a `BlockFingerprint`, but **no live `block_id`** and
  **no resident bytes**. The bytes were evicted from every tier; only
  the routing metadata remains. A `lookup` hit on a tombstone is a
  miss for the purpose of "can we reuse bytes", but it preserves
  statistics (the prefix is known to recur), and once `BlockFingerprint`
  is wired, it lets a future `M5` cross-node lookup find peers that
  still hold the bytes.

Tombstones exist for **statistics, future fingerprint-based recovery,
and to keep the radix shape stable** across evictions of leaf bytes.
They do not exist as a "free reuse" mechanism — actual byte reuse
**always** requires a hit on a real entry.

Tombstones are bounded by their own cap (default 100k entries; LMCache
parity), independent of the T0/T1/T2 byte budgets. When the tombstone
cap fires, the eviction policy drops whole entries (fingerprint +
metadata together), equivalent to "we never saw this prefix". This
prevents the radix tree itself from becoming an unbounded
metadata leak.

The current `RadixCache::evict` `infer/src/prefix_cache.rs:368` selects
victims with `block_id.is_some() && ref_count == 0` and so **never
cleans up pure tombstones today** — the byte-eviction pass leaves
`block_id = None` stubs behind. M3 must add a second tombstone-GC pass.
This is not in scope for M2b.

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

### 4.5 Lookup interface and recompute-vs-fetch fallback (added 2026-04-15)

Codex's review crystallised a question my analysis had also flagged: how
much should the scheduler know about staging? Two extreme answers — both
wrong:

- **Fully blind** ("scheduler just calls `lookup(tokens) → Vec<BlockId>`")
  — the scheduler cannot tell the difference between "T0 hit, decode now"
  and "T2 hit, stalled 30 ms on NVMe". Latency budgets blow up because
  the scheduler keeps starting decodes that immediately wait.
- **Tier-aware** ("scheduler enumerates T0 / T1 / T2 separately") —
  scheduler is now coupled to every backend; adding a tier means a
  scheduler diff. This is exactly what M5's NIXL deferral discipline
  rejects.

The right shape (matches NIXL's polling-completion model that the
`KVTransport` trait already enforces, §5.3):

```rust
// in RadixCache (or a thin façade above it)

pub enum HitKind {
    /// Bytes physically in T0 right now. Decode immediately.
    ReadyOnGpu,
    /// Bytes resident in T1. Coordinator has scheduled a copy onto a
    /// dedicated CUDA stream; poll the returned op to know when it
    /// lands in T0.
    StagingFromHost,
    /// Bytes resident in T2 (or T3 in M5+). Coordinator has scheduled
    /// the disk/RDMA fetch; same polling shape.
    StagingFromDisk,
    /// Index-only tombstone (or no entry at all). No reuse possible
    /// for these blocks — scheduler must prefill them itself.
    Miss,
}

pub struct LookupOutcome {
    pub blocks: Vec<(BlockId, HitKind)>,
    pub staging_op: Option<TransportOp>,   // populated iff any block is Staging*
    pub recompute_advised: bool,           // see §4.5.1
}

pub fn lookup_or_stage(&mut self, tokens: &[u32]) -> LookupOutcome { ... }
```

The scheduler does not know *which* tier holds a staging block, only
whether it is `ReadyOnGpu` (decode now) or `Staging*` (poll the op
before scheduling decode). This satisfies "scheduler is not blind" and
"scheduler is not tier-aware" simultaneously. It also lets the
coordinator move a block T2 → T1 → T0 in two hops without the scheduler
seeing two separate staging events — the outer poll just stays
`Pending` until both hops complete.

#### 4.5.1 Recompute-vs-fetch fallback

LMCache / CacheGen empirical result, confirmed by SGLang HiCache early
versions: **fetching a short prefix back from NVMe is slower than
re-prefilling it**. NVMe ≈ 3 GB/s single-stream; A100 BF16 prefill on
Qwen3-4B is roughly equivalent to ≥ 30 GB/s of "useful KV bytes per
second". For prefixes below ~256 tokens, recompute wins.

The decision is a single-line heuristic at lookup time:

```rust
// pseudocode, evaluated when any block is Staging*
let bytes_to_fetch = staging_blocks * page_size * bytes_per_token;
let recompute_advised =
    bytes_to_fetch as f32 / tier_bandwidth_bytes_per_sec
    >
    staging_tokens as f32 / prefill_tokens_per_sec;
```

The scheduler reads `LookupOutcome::recompute_advised` and, if true,
**discards the staging op** (transport `abort` is a no-op for this
case, the bytes will be dropped) and rebuilds those blocks via prefill.
The radix node **does not** become a tombstone — recompute does not
mean "we don't have it", it means "fetching is slower than rebuilding
right now". The next request might find the staging path is fast
enough.

This heuristic is cheap (5 lines, one `f32` divide), but it is the
single cheapest way to keep the M3+ tier integration from regressing
TTFT on workloads that spend most of their time in short prefixes. M3b
is the milestone that introduces it.

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

**Migration (shipped as M0.1)**:
1. Added `types::BlockId(u32)` and `types::BlockFingerprint([u8; 16])` as
   the canonical types in `infer/src/types.rs:8`.
2. `prefix_cache::BlockId` re-exports `types::BlockId`
   (`infer/src/prefix_cache.rs:53`).
3. `block_manager::BlockId` deleted; `block_manager.rs` now refers to
   `types::BlockId` directly.
4. `kv_tier::id.rs` reduced from a 56-line `BlockId(u64)` definition to
   a one-line re-export of `types::BlockId`.
5. `BlockFingerprint` exists but has **zero call sites today** — the
   first consumer is M4 session save/load, which must address blocks
   by fingerprint to survive a restart (§3 fact 4, §6 M4
   reconciliation note).

The radix-tree integration of `fingerprint: Option<BlockFingerprint>`
on the private `Node` struct is **deferred to M3a** (see §6 M3a
"`Node` field extension"); M1b shipped the radix integration without
the metadata extension because M2a did not need it.

This is M0.1 — **shipped**. See §6.

### 5.2 `Node` with tier metadata (merges the old `BlockDescriptor`)

The 2026-04-13 design had a separate `TierDirectory` holding
`BlockDescriptor { id, tier, location, byte_len, ref_count, last_access,
session_id, pin_until }`. The 2026-04-15 revision moves every field onto
the existing **private** `Node` struct inside `RadixCache` (at
`infer/src/prefix_cache.rs:55-69`). The struct stays private; no `pub`
leakage; consumers still go through `RadixCache` methods.

Note: the previous draft of this section called the struct `RadixNode`.
It is actually named `Node` and is `struct Node` (private) at the time
of writing. The rename to `RadixNode` is not required; keeping `Node`
matches the existing code.

**Status (2026-04-15, post M2a)**: M1b wired the radix into the
scheduler **without** adding any of these fields. The current `Node`
holds only `(tokens, block_id: Option<BlockId>, ref_count, last_access,
children)`. The tier metadata fields below are deferred to **M3a**
(§6) — they land alongside `LocalCudaTransport` and the
`HostPinnedPool`, which is the first consumer that actually needs to
read them. Adding them to a private struct with no consumers earlier
buys nothing.

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

#### 5.4.1 Convergence note (Codex review 2026-04-15)

M2a shipped a hand-rolled high/low-watermark eviction loop —
`evict_prefix_cache_if_pressured` at `infer/src/scheduler/cuda/core.rs:430`
— that **does not consult `EvictionPolicy::score`**. It computes the
number of blocks to free from the watermark hysteresis (`high = 0.75`,
`low = 0.50`) and calls `RadixCache::evict(n)`, which uses pure
last-access LRU on active leaves with `block_id.is_some() && ref_count == 0`.

This is fine for M2a (T0-only, no tiers, no session affinity in the
selector path). It is **not fine** for M3, because the moment the
coordinator joins, eviction needs three things the watermark loop
cannot provide:

1. **Pin protection** (`HitCountLru::hit_threshold`) — prevent the
   watermark from immediately re-evicting a block that was just
   promoted from T1.
2. **Session affinity** (`SessionBiasedLru::affinity_bonus`) — keep
   the active session's prefix above the eviction line during a long
   tool-call burst.
3. **Recency × hit-count weighting** (`ReuseBiasedLru`) — shipped as
   an opt-in for KVFlow-style temporal-locality.

M3b's exit criterion includes "`evict_prefix_cache_if_pressured` calls
the policy trait, not its own loop". Concretely:

- Add `evict_with_policy(&dyn EvictionPolicy, n: usize)` to
  `RadixCache` that walks all leaves, builds an `EvictionCandidate`
  per leaf, scores each, and removes the bottom-`n` (excluding
  `f32::INFINITY`).
- `evict_prefix_cache_if_pressured` becomes a thin wrapper that
  computes `n` from the watermarks and calls `evict_with_policy`.
- The default policy at startup is `SessionBiasedLru` (matches §11.1
  rationale and the existing `Default` impl).
- `RadixCache::evict(n)` (the policy-free version) stays for tests
  but its production callers move to `evict_with_policy`.

If we ship M3 without this convergence, the project ends up with two
parallel eviction implementations — the M2a watermark loop on T0 and
whatever the M3 coordinator builds on T1 — and they will disagree on
"is this block hot" the first time a session promotes a block from
T1 back to T0. This is precisely the "two truths" failure mode §3
fact 5 calls out for `kv_cache.rs`, replayed inside `kv_tier`.

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

#### M0.2 · Fix the three `prefix_cache.rs` correctness bugs — **already done**

**Status (2026-04-15)**: no-op. All three bugs listed in §8 (items 10–12)
were already fixed by commit `5da8b67 fix(prefix_cache): split must
inherit ref_count + evict must cascade` on the 2026-04-13 work batch.
The 2026-04-15 survey agent's report was stale on this point. Evidence:

```
$ cargo test -p infer --no-default-features --features cpu,no-cuda --release \
    --lib prefix_cache
test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 207 filtered out
```

The 22 tests include `split_node_inherits_ref_count_from_child`,
`lookup_bumps_every_block_bearing_node_on_path`, and three
`evict_cascade_*` tests covering orphan-parent iteration. See §8 items
10–12 for the fix locations in the current code.

No M0.2 PR is needed. M1 can proceed on M0.1 alone (plus M0.3 when the
extraction prereq is met).

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

### M1 — Wire `RadixCache` into scheduler, delete `TierDirectory` — **shipped**

The previous draft of this section described M1 as one atomic PR with an
optional 2-PR split. **Reality**: it shipped as a 2-PR split (M1a + M1b)
on 2026-04-14, and the `Node` field extension was **deferred** to M2b /
M3 to keep the M1 diff small enough to atomic-rollback. Recording what
actually landed:

- **M1a · `08718ad`** — delete `infer/src/kv_tier/directory.rs` (322
  lines) + retire its `kv_tier.rs` re-export + retire one `disk.rs` doc
  comment. Zero production callers; pure subtraction. **Done.**
- **M1b · `323aee0`** — wire `RadixCache` into `Scheduler<M>` as a
  **shadow observer**. New fields on `Scheduler<M>`:
  `prefix_cache: RadixCache` (block_size=16, owned outright since the
  scheduler runs on a single `std::thread`) + `next_prefix_block_id:
  u32` synthetic id counter. `cleanup` inserts the completed prompt
  into the radix; `assign_slots` runs `radix.lookup` before the legacy
  `best_prefix_slot_for_cached_prompts` linear scan and logs `"radix
  shadow: best cross-slot prefix hit = X/Y tokens"` before releasing
  the refs. **Behavior unchanged** — `cached_prompts` still drives
  actual KV reuse. **Done.**

**Not done in M1** (deferred to M2b / M3):
- `Node` field extension (`tier_location`, `session_id`,
  `soft_pin_until`, `byte_len`, `fingerprint`). Adding these to a
  private struct with no consumers buys nothing; they land when M2b
  (the selector flip) or M3 (the coordinator) needs to read them.
- `cached_prompts` deletion. Stays as the legacy ground truth until
  M2b's selector flip lands.

**M1 exit (achieved)**: `cargo test --release --test e2e_qwen35` golden
outputs unchanged on a CUDA host; shadow-observer logging present;
`grep -rn TierDirectory infer/src/` returns empty.

### M2 — Dual residency (T0 only, no new tiers)

The previous draft described M2 as one PR. **Reality**: it splits
cleanly into a data-model PR (M2a, shipped) and a behavior-flip PR
(M2b, pending). The split is exactly the "data model first, behavior
flip second" pattern that worked for M1, applied here because the
scheduler-side selector flip requires the pool's refcount discipline
to already exist.

#### M2a · refcount + watermark + side map — **shipped (`4402ab0`)**

- `TokenKVPool::page_ref_count: Vec<u32>` + `retain_pages` /
  `release_pages` / `retained_count` + a refcount-aware `free_slot`
  that leaves pinned pages in limbo (not in any slot, not in
  `free_slots`, still physically live in HBM). Implementation at
  `crates/infer-cuda-kernels/src/paged_kv.rs:71-83,416-440`.
- `Scheduler::publish_to_prefix_cache` (`infer/src/scheduler/cuda/core.rs:333`)
  takes a `slot_idx`, snapshots `token_indices(slot_idx)`, inserts
  **real** physical page indices into the radix as `BlockId`s, and
  calls `retain_pages` across the full `num_blocks × block_size` span.
- New side map `block_to_pages: HashMap<BlockId, Vec<u32>>` records
  each radix block's complete (non-contiguous after a few alloc/free
  cycles) page span.
- `evict_prefix_cache_if_pressured` (`infer/src/scheduler/cuda/core.rs:430`)
  watermark loop: `high = 0.75`, `low = 0.50`; runs at the end of
  `cleanup` and releases radix-held pages back to the pool when the
  retained fraction crosses the high mark. **Hand-rolled, bypasses
  `policy.rs`** — see §5.4.1 for the M3 convergence requirement.
- 4 new `MockPool`-based refcount unit tests in
  `crates/infer-cuda-kernels`. **Done.**

#### M2b · selector flip + retain hard cap + tombstone GC — **pending**

This is the **first** milestone where prefix reuse becomes
load-bearing. Scope:

1. **Replace `best_prefix_slot_for_cached_prompts`
   (`infer/src/scheduler/cuda/runtime.rs:20-30,191`) with a
   radix-aware selector** that uses the M1b shadow lookup as the
   ground truth. Slots that map to a non-empty radix hit win over
   slots whose `cached_prompts` happens to match — and `cached_prompts`
   is then deleted in the same PR.
2. **Add the resurrect read path in `step_new`** that, when
   `radix.lookup` returns a hit, copies the matched pages' state into
   the new slot (or, more cheaply, advances the slot's
   `token_indices` to alias the matched pages directly across slot
   boundaries) and skips `forward_prefill` over the matched prefix.
   This is the change that delivers the actual TTFT win — M1b+M2a
   alone are net-neutral on single-request benches.
3. **`alloc_tokens` OOM → `evict_prefix_cache_if_pressured` retry.**
   Today `alloc_tokens` returns `Err` on OOM with no retry path. The
   amortised cleanup-time eviction holds under steady-state load but
   an adversarial burst of long-prompt admissions can drain the free
   list between two `cleanup` calls. M2b adds the synchronous retry.
4. **Retain hard cap.** M2a's `0.75 / 0.50` watermark is an *evict
   trigger*, not a *retain cap*. If a single long prefix is held by
   100 concurrent requests, `ref_count > 0` skips it during eviction
   and the pool can saturate. Add `retained_count <
   max_total_tokens × 0.9` as a hard cap: above it, lookups
   intentionally do not retain — the hit is reported but the bytes
   are not pinned, so the requester pays prefill instead of starving
   admission. This is the "fail open" path under pressure.
5. **Tombstone-GC scaffolding**: `RadixCache::evict_with_policy` adds
   a second pass that prunes `block_id == None && ref_count == 0`
   stubs left behind by the byte eviction. Without this pass the
   radix tree grows unboundedly across long sessions. (Still no
   tombstone *cap* in M2b — that lands in M3 alongside the policy
   trait wire.)
6. **Delete `cached_prompts`**. After (1) replaces its only consumer,
   the field is dead.

**Exit**:
1. `grep -rn cached_prompts infer/src/` returns empty
2. Two-session alternating trace: second visit to the same system
   prompt achieves **≥ 95% prefix hit rate** (was ~0% pre-M2b)
3. `scripts/bench_agent_trace.py` shows TTFT drop on the
   agent-workload benchmark vs M2a baseline
4. `cargo test --release --test e2e` / `e2e_qwen35` golden outputs
   unchanged
5. Stress test: 100 concurrent requests sharing one 4 k-token prefix
   does not cause `alloc_tokens` to error out (the retain hard cap
   protects pool free-list)

### M3 — T1 host pinned tier + coordinator (stacked PR series)

Scope expanded by Codex review 2026-04-15: the coordinator must own a
**three-state page lifecycle** (§4.2 invariant 10), must converge
eviction onto `EvictionPolicy::score` (§5.4.1), and must add the
**recompute-vs-fetch fallback** (§4.5.1). M3c also retires the legacy
`model/kv_cache.rs` CPU offload (§3 fact 5).

**Hard prereq: M0.3 must land first.** `cudaMemcpyAsync` on
`page_size=1` issues 16 separate non-contiguous transfers per logical
block; DMA launch overhead dominates and the transport benchmarks
will be unrepresentative.

**Sub-PRs (in order)**:

- **M3a · `HostPinnedPool` + `LocalCudaTransport`.** One-way T0→T1
  transfer over a dedicated CUDA copy stream. Coordinator is called
  manually (no watermark trigger yet). No behavior change in the
  scheduler — it gets a new API to say "move this block to T1". Adds
  the `Node` fields the previous M1 draft promised (`tier_location:
  Cell<TierLocation>`, `byte_len: u32`, optional `fingerprint`,
  `soft_pin_until`); they were deferred from M1 because M2a did not
  need them and adding them to a private struct with no consumers
  buys nothing.
- **M3b · Coordinator OS thread + policy convergence + recompute
  fallback.** Coordinator owns: (1) the `crossbeam_channel::bounded`
  intent queue, (2) the dedicated `CudaStream`, (3) the `Free →
  Resident → Demoting → Free` page-lifecycle state machine on every
  pool page (§4.2 invariant 10 — this is the M3 correctness centre,
  not the cudaMemcpyAsync itself). `evict_prefix_cache_if_pressured`
  rewires from its hand-rolled watermark loop to the policy trait
  via a new `RadixCache::evict_with_policy` (§5.4.1). The
  `lookup_or_stage` interface (§4.5) lands here. The
  recompute-vs-fetch fallback heuristic (§4.5.1) lands here. Default
  policy is `SessionBiasedLru`. Watermarks: `T0_high = 0.85`,
  `T0_low = 0.70`, matching vLLM / SGLang defaults.
- **M3c · T1→T0 promotion path + delete legacy CPU offload.**
  Promotion on radix lookup hit when the matched node's
  `tier_location` is `HostPinned`. **Delete
  `infer/src/model/kv_cache.rs:517-590` (`offload_if_needed` +
  CPU shadow buffers) and the `OFFLOAD_BLOCK_SIZE = 64` constant.**
  This is the §8 pitfall 13 cleanup — confirm zero production
  callers via `grep -rn 'k_host\|v_host\|offload_if_needed' infer/src/`
  before deleting. Audit all model batch-decode files for any
  remaining assumption that the CPU shadow buffers exist.

**Exit**:
1. A long-agent-session benchmark (32k+ cumulative tokens,
   num_slots=4) that OOMs on pre-M3 runs to completion on M3c
2. `scripts/bench_throughput_sweep.py --label tier-T1` recorded vs a
   pre-M3 baseline. Steady-state decode throughput regression ≤ 3%.
3. `cargo test --release --test e2e_qwen35` unchanged
4. `grep -rn 'evict_prefix_cache_if_pressured.*evict(' infer/src/`
   returns empty (the watermark loop calls `evict_with_policy`, not
   the policy-free `evict`)
5. `grep -rn 'OFFLOAD_BLOCK_SIZE\|offload_if_needed\|k_host\|v_host'
   infer/src/` returns empty (legacy CPU offload deleted)
6. Demote/cancel-on-hit race test: synthetic test that fires a
   `lookup` against a block currently in the `Demoting` state and
   confirms the lookup either (a) waits and returns the block, or
   (b) cancels the demote and returns the T0 location — never
   returns stale bytes

### M4 — T2 disk tier + session save/load + first Metal contact

**What**: Add the real coordinator path for T1→T2 spill under watermark.
Change `DiskStore` wire format from raw-bytes dump to postcard header +
blake3-hash filename (task doc §4.2 spec). Add
`POST /v1/sessions/{id}/save` and `POST /v1/sessions/{id}/load` routes
for session persistence.

**Cross-restart identity must be `BlockFingerprint`, not `BlockId`**
(Codex review 2026-04-15). `BlockId(u32)` is a pool slot index and
**does not survive a restart** — the post-reload `TokenKVPool` allocates
its slots from scratch and any saved-but-not-yet-consumed `BlockId`
becomes a dangling reference. Save format on disk addresses every block
by its `BlockFingerprint([u8; 16])`. Reload runs a **reconciliation
pass**:

1. Walk the saved radix snapshot, mint a fresh `BlockId` from the new
   pool for each entry that has bytes on disk, populate the radix node
   with the new id and the fingerprint.
2. Tombstone entries (saved fingerprint, no bytes) round-trip as
   tombstones — they preserve cache-hit statistics across the restart
   without claiming reusable bytes.
3. The `serde::Deserialize` derive on `RadixCache` already exists
   (`infer/src/prefix_cache.rs:96`) but **today only round-trips raw
   `BlockId`s**, which means the current `Deserialize` path is broken
   for any non-trivial pool. M4 must either replace it with a
   fingerprint-based reconciliation API or fence the existing impl
   behind `#[cfg(test)]`. The previous draft of this section did not
   call this out.

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

Updated to reflect what actually shipped (M0.1, M1a, M1b, M2a) vs the
original plan, and the M2/M3 expansions added by Codex 2026-04-15.

| Milestone | PRs | Structure / behavior | Status |
|---|---|---|---|
| M0.1 | 1 | Type rename + `use` path update (structural) | **shipped** |
| M0.2 | 0 | No-op — three prefix_cache bugs already fixed in `5da8b67` | **shipped** |
| M0.3 | 1 | page_size lift with per-format dispatch (structural, benchmark gate) | **pending** (unblocked) |
| M1a | 1 | Delete `kv_tier/directory.rs` (pure subtraction) | **shipped (`08718ad`)** |
| M1b | 1 | RadixCache shadow observer wired into Scheduler (behavior-neutral) | **shipped (`323aee0`)** |
| M2a | 1 | Per-page refcount + side map + watermark eviction loop (data model) | **shipped (`4402ab0`)** |
| M2b | 1 | Selector flip + retain hard cap + tombstone GC + delete `cached_prompts` (behavior, benchmark gate) | **pending** |
| M3a | 1 | `HostPinnedPool` + `LocalCudaTransport` + `Node` field extension (structural) | not started |
| M3b | 1 | Coordinator OS thread + page lifecycle state machine + policy convergence + recompute fallback + `lookup_or_stage` interface (behavior, benchmark gate) | not started |
| M3c | 1 | T1→T0 promotion + delete legacy `model/kv_cache.rs` CPU offload (structural cleanup) | not started |
| M4a | 1 | Disk format change + MLX wired-memory bindings (structural) | not started |
| M4b | 1 | HTTP session save/load routes + fingerprint-reconciliation reload (behavior) | not started |
| M5 | deferred | triggered, not scheduled — see §6 M5 | deferred |

No PR in this doc mixes kernel and scheduler changes. M0.3's kernel
files are already page-size-aware internally; only the dispatch
locals change. M1–M4 have zero kernel diffs.

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

13. **Two parallel CPU offload code paths.** `infer/src/model/kv_cache.rs`
    has a contiguous-buffer `OFFLOAD_BLOCK_SIZE = 64` offload that predates
    the paged pool; it is dormant in production but still compiles and
    tests. Once M3 introduces T0↔T1 transfer through the **paged** path,
    the two paths are mutually exclusive — having both means two truths
    for "where is layer L's KV right now". M3c **must** delete the legacy
    path. Do not extend it for any reason in the meantime; do not try to
    bridge the two granularities (16 vs 64). See §6 M3c exit gate.
14. **Decode tail must not be tier-transferred.** Decode appends one
    token at a time; the last page of a request is the hot tail and
    holds fewer than `page_size` valid tokens until it fills. Tier
    transfer of an unsealed tail wastes PCIe bandwidth and creates a
    second source of truth for "is this block complete?". The seal
    happens when `seq_len % page_size == 0`; until then the tail is
    invisible to `RadixCache::insert` and the coordinator. See §4.2
    invariant 8.
15. **Index-only tombstones can leak the radix tree.** Once tier
    eviction runs, leaf nodes lose their `block_id` but the node
    itself stays as a fingerprint stub. Without a tombstone GC pass
    the radix grows unboundedly across long sessions. M2b adds the
    `block_id == None && ref_count == 0` cleanup pass; M3 adds an
    explicit tombstone *cap* (default 100k entries, LMCache parity)
    so even fingerprint-only nodes have a bounded memory cost. See
    §4.3.1.
16. **Retain hard cap vs evict watermark.** M2a's `0.75 / 0.50`
    watermark is an *evict trigger*, not a *retain cap*. Adversarial
    workload: 100 concurrent requests all hit one 4 k-token shared
    prefix. Every page in the prefix has `ref_count > 0`, so eviction
    skips them, and the pool's free-list drains. M2b adds
    `retained_count < max_total_tokens × 0.9` as a hard cap — above
    it, lookups intentionally do not retain (the hit is reported but
    not pinned). This is the "fail open" path. See §6 M2b.
17. **Demote / read-after-evict race.** Block X is being copied T0→T1.
    Mid-copy, a new request `lookup`s X. If the coordinator releases X
    from T0 the moment `cudaMemcpyAsync` is *posted*, the new request
    reads garbage. Correct fix: three-state page lifecycle
    `Free | Resident | Demoting` on every pool page; release only when
    the copy completes *and* refcount hits 0 *and* no `lookup` has
    upgraded the page back to `Resident`. M3b owns this state machine
    — it is the M3 correctness centre, not the cudaMemcpyAsync itself.
    See §4.2 invariant 10.

### 2026-04-15 bugs flagged by the internal survey — **all three already fixed**

When the 2026-04-15 survey agent read the task doc it found three
`prefix_cache.rs` correctness bugs flagged in the 2026-04-13 research
(§3.2) and assumed they were still open. Actual git log:

```text
5da8b67 fix(prefix_cache): split must inherit ref_count + evict must cascade
```

That commit landed before the 2026-04-15 revision and resolves all
three items. `cargo test -p infer --lib prefix_cache` runs 22 tests
green, including:

- `split_node_inherits_ref_count_from_child`
- `lookup_bumps_every_block_bearing_node_on_path`
- `evict_cascades_through_orphaned_parent_chain`
- `evict_cascade_respects_limit_n`
- `evict_cascade_respects_ref_count`

The items are preserved in the list below with their original
descriptions for history, but **M0.2 is a no-op** — its scope is
already shipped, and M1 can proceed on M0.1 alone.

10. **~~`_split_node` does not inherit child's `ref_count`.~~** *Fixed
    in `5da8b67`*. When the radix tree splits a node on insertion,
    the new parent-like node inherits the splitting child's
    `ref_count` — matches SGLang's `new_node.lock_ref = child.lock_ref`
    pattern. See `infer/src/prefix_cache.rs:258-275`.

11. **~~`lookup` does not walk the ancestor chain.~~** *Fixed in
    `5da8b67`*. Lookup walks root → leaf and increments `ref_count`
    on every node on the matched path, not just the leaf. See
    `infer/src/prefix_cache.rs:146-200`.

12. **~~`evict()` does not iterate orphan parents.~~** *Fixed in
    `5da8b67`*. Eviction runs an iterative outer loop (`while
    freed.len() < n`) and re-scans for active-leaf candidates each
    pass; a parent becomes an active leaf the moment its last
    non-evicted child joins `evicted_set`. See
    `infer/src/prefix_cache.rs:368-423`.

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
- **Pure LFU or LRU+LFU hybrid eviction** — explicitly **not** chosen
  for M3. vLLM, SGLang, and LMCache all use plain LRU on radix nodes;
  frequency is implicitly encoded in "which node sits on a hot
  prefix path" because lookups bump every ancestor's `last_access`.
  LFU adds per-node hit counters that must be aged (otherwise early
  hot prefixes are pinned forever), and the bookkeeping bug surface is
  not justified until a real trace shows pure LRU + 2-hit floor
  pathology. M3 ships with `SessionBiasedLru` (or `HitCountLru` with
  threshold 2 — the SGLang `write_through_threshold` analog). LFU is
  reconsidered only if M3 production traces show ≥ 5% of evictions
  removing a prefix that gets re-prefilled within the next few
  requests. The Codex review 2026-04-15 specifically argued against
  jumping to LFU prematurely; this entry codifies that.

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

### 2026-04-15 second revision (post-Codex review + M1+M2a landed)

After M1a / M1b / M2a shipped (commits `08718ad` / `323aee0` / `4402ab0`)
a Codex independent review caught two issues the first 2026-04-15
revision under-emphasised, plus a handful of architectural primitives
that needed to be elevated to first-class doc sections:

1. **Old `model/kv_cache.rs` CPU offload as parallel "second source of
   truth".** The first revision mentioned the legacy contiguous offload
   only as a §3 row and a §6 M3c "delete after confirming zero
   callers". Codex argued — correctly — that this is a higher-order
   risk than the first revision treated it as: it is dormant in
   serving but still compiles, still has tests, still shapes model
   call sites, and uses `OFFLOAD_BLOCK_SIZE = 64` which is incompatible
   with the paged pool's `page_size = 16` plan. Elevated to §3 fact 5,
   §4.2 invariant 9, and §8 pitfall 13. The "do not extend" rule is
   now explicit.
2. **`policy.rs` scoring trait bypassed by M2a's hand-rolled watermark.**
   M2a shipped `evict_prefix_cache_if_pressured` (`scheduler/cuda/core.rs:430`)
   as a hand-rolled high/low watermark loop that does not consult
   `EvictionPolicy::score`. Fine for T0-only, but the moment M3
   introduces T1 the project ends up with two parallel eviction
   implementations. New §5.4.1 codifies the convergence requirement;
   M3b's exit gate enforces it.
3. **`lookup_or_stage` interface and recompute-vs-fetch fallback.** The
   first revision had a thin `lookup(tokens) → Vec<BlockId>` shape that
   left "scheduler is blind vs scheduler is tier-aware" as an unresolved
   tension. Codex (and my own analysis independently) crystallised the
   right answer: scheduler sees a `HitKind` enum (`ReadyOnGpu /
   StagingFromHost / StagingFromDisk / Miss`) but never sees *which*
   tier holds a staging block, plus a single-line recompute heuristic
   for short-prefix workloads. New §4.5 captures both.
4. **Demote / read-after-evict race + page lifecycle state machine.**
   First revision's §4.4 said "OS thread + crossbeam" but did not
   mention the three-state page lifecycle that M3 must implement. New
   §4.2 invariant 10 + §8 pitfall 17 + §6 M3b language now treat this
   as the M3 correctness centre, not the cudaMemcpyAsync itself.
5. **Decode tail seal-on-block-fill.** First revision implicitly
   assumed it but never spelled it out. New §4.2 invariant 8 + §8
   pitfall 14 make it explicit: a partial tail page is invisible to
   `RadixCache::insert` and to the coordinator.
6. **Tombstone vs real entry framing for "index-only" radix nodes.**
   First revision left `block_id == None && ref_count == 0` stubs as
   an unhandled edge case in `RadixCache::evict`. New §4.3.1 names them
   "tombstones", caps them, and gives M2b/M3 explicit GC scope.
7. **Retain hard cap as distinct from evict watermark.** First revision
   conflated them. New §8 pitfall 16 + §6 M2b scope (4) split them
   into "evict trigger" and "fail-open retain cap".
8. **`BlockFingerprint` reload reconciliation.** First revision
   defined `BlockFingerprint` but never specified that **M4 reload
   must address blocks by fingerprint, not BlockId**, since pool slot
   ids do not survive a restart. New §6 M4 language + §3 fact 4
   capture this.
9. **LFU explicitly deferred.** First revision left the LRU vs
   LRU+LFU question open. New §11.1 entry codifies "ship LRU with
   2-hit floor, do not jump to LFU until trace shows ≥ 5% pathology".
10. **§6 M1, M2 rewritten to reflect what actually shipped.** First
    revision described M1 as "one atomic PR maybe split"; reality
    shipped as M1a + M1b. M2 is now decomposed into M2a (shipped) +
    M2b (pending) with explicit scope for each.
11. **§7 PR table** updated with shipped status per row.

Every change in this revision is additive or replaces stale prose with
the true post-M2a state; no architectural decisions from the first
2026-04-15 revision are reversed.

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

**Shipped (do not redo)**:
- **M0.1** — `BlockId` unification. `types::BlockId(u32)` canonical.
- **M0.2** — no-op (the three prefix_cache bugs were already fixed in
  `5da8b67` before the first 2026-04-15 revision was even written).
- **M1a** — `08718ad`. `kv_tier/directory.rs` deleted.
- **M1b** — `323aee0`. RadixCache wired as scheduler shadow observer.
- **M2a** — `4402ab0`. Per-page refcount + side map + watermark
  eviction loop. Pages survive `free_slot` when the radix retains them.

**Next, in dependency order**:

1. **M2b · selector flip** (next target). Replace
   `best_prefix_slot_for_cached_prompts`
   (`infer/src/scheduler/cuda/runtime.rs:20-30,191`) with a radix-aware
   selector + add the resurrect read path in `step_new` that copies
   matched pages' state into the new slot and skips prefill for the
   matched prefix. Wire `alloc_tokens` OOM → `evict_prefix_cache_if_pressured`
   retry. Add the retain hard cap (`retained_count <
   max_total_tokens × 0.9`). Add the tombstone GC pass (`block_id ==
   None && ref_count == 0` cleanup). Delete `cached_prompts`.
   Validation: `bench_serving matrix --concurrency 4 --prompt-lens
   128,512 --output-lens 64`, expect measurable TTFT win on repeated
   prompts. M2a's `MockPool` refcount tests stay as the regression
   gate. **No tier work in M2b — still T0-only.**
2. **M0.3 · `page_size = 1 → 16`** (parallel track, hard prereq for
   M3). Per-format dispatch — keep `page_size=1` for INT8/FP8/quant
   paths, lift to 16 for BF16/FP16. Kernel files at
   `crates/infer-cuda-kernels/src/paged_kv.rs` +
   `crates/infer-cuda-kernels/csrc/kv/{kv_cache_to_paged, kv_quant,
   scatter_kv}.cu`. M0.3 and M2b can sequence in either order; the
   only hard rule is **M0.3 lands before M3a**.
3. **M3a · `HostPinnedPool` + `LocalCudaTransport` + `Node` field
   extension**. Adds `tier_location: Cell<TierLocation>`, `byte_len:
   u32`, optional `fingerprint`, `soft_pin_until` to `Node`. One-way
   T0→T1 transfer over a dedicated copy stream. Coordinator is called
   manually — no watermark trigger yet.
4. **M3b · coordinator + page lifecycle + policy convergence + recompute
   fallback**. The M3 correctness centre. Three-state page lifecycle
   (`Free | Resident | Demoting`), policy-trait wire (`evict_with_policy`),
   `lookup_or_stage` interface (§4.5), recompute-vs-fetch fallback
   heuristic (§4.5.1). Default policy `SessionBiasedLru`. Watermarks
   `T0_high = 0.85`, `T0_low = 0.70`.
5. **M3c · T1→T0 promotion path + delete legacy `model/kv_cache.rs`
   CPU offload**. The §8 pitfall 13 cleanup.
6. **M4a / M4b** — disk tier + session save/load with
   fingerprint-reconciliation reload. Waits on M3 for the coordinator
   shape.
7. **M5** — NIXL real-RDMA path. Trigger-gated, post-project work
   unless one of the §6 M5 triggers fires.

---

*Live at commit `02407f2`; last revised 2026-04-15 (second revision,
post-M2a + Codex review integration).*
