//! Inference execution seam traits.
//!
//! The seam is the engine's COST CONTRACT: a submit/poll core loop whose costs
//! are parameterized by [`StepLimits`] and whose optional behaviors are explicit
//! capability accessors ([`PrefixReuse`], [`KvPageTier`], [`KvSlotTier`],
//! [`DeviceKvFit`], [`WeightResidency`], [`MultimodalGenerate`]). The seam is
//! host-only: [`ForwardPlan`], [`StepOutput`], and [`KvPool`] expose slots, page
//! ids, token ids, and lengths. Device tensors, collectives, sampling, and the
//! model forward all live inside the backend executors ([`BackendExecutor`]),
//! never crossing this seam. A model family whose decode is no longer
//! submit-poll shaped forks the loop (precedent: `diffusion_executor.rs`)
//! rather than bending this trait.

use std::cell::Cell;

use infer_plan::{
    DiffusionGenerateOutput, ForwardPlan, MultimodalImage, MultimodalKind, SamplingParams,
    StepOutput,
};

#[path = "allocator.rs"]
mod allocator;
#[path = "diffusion_executor.rs"]
mod diffusion_executor;
#[path = "host_paged_kv_pool.rs"]
mod host_paged_kv_pool;
#[path = "kv.rs"]
mod kv;
#[path = "kv_batch.rs"]
mod kv_batch;
#[path = "kv_budget.rs"]
mod kv_budget;
#[path = "kv_dtype.rs"]
mod kv_dtype;
#[path = "kv_query.rs"]
mod kv_query;
#[path = "prefix_store.rs"]
mod prefix_store;
#[path = "resource.rs"]
mod resource;
#[path = "runtime_flags.rs"]
mod runtime_flags;

pub use allocator::KvAllocator;
pub use diffusion_executor::BufferedDiffusionExecutor;
pub use host_paged_kv_pool::{EVICTED_PAGE, HostPagedKvPool, ShardSpec};
pub use kv::KvPool;
pub use kv_batch::{KvBatchDescriptor, KvBatchRow, KvBatchRowKind};
pub use kv_budget::KvTierBudget;
pub use kv_dtype::KvCacheDtype;
pub use kv_query::KvQuery;
pub use prefix_store::KvPrefixStore;
pub use resource::{
    DramTierPolicy, NvmeTierPolicy, PROFILE_KV_TOKENS_FLOOR, SlotBudget, clamp_mem_fraction_static,
    clamp_to_affordable, dram_l2_budget, nvme_l3_budget, profile_kv_pool_tokens,
};
pub use runtime_flags::{CommBackend, CudaRuntimeFlags, MetalRuntimeFlags};

#[derive(Debug, Clone)]
pub enum PollResult<I> {
    Ready(StepOutput),
    NotReady(I),
}

/// Bits of a tier key below the backend store's namespace byte; content keys
/// are masked to this width so they double as tier keys.
pub const TIER_KEY_BITS: u32 = 56;

/// Content key of one prefix block: FNV-1a over the parent block's key and the
/// block's tokens. Stable across processes, so a durable tier store can be
/// probed for a prompt's blocks without any in-memory index.
#[must_use]
pub fn prefix_block_content_key(parent: u64, block: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in parent
        .to_le_bytes()
        .into_iter()
        .chain(block.iter().flat_map(|t| t.to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash & ((1u64 << TIER_KEY_BITS) - 1)
}

/// Chained content keys for every full block of `tokens`.
#[must_use]
pub fn prefix_content_keys(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut parent = 0u64;
    tokens
        .chunks_exact(block_size.max(1))
        .map(|block| {
            parent = prefix_block_content_key(parent, block);
            parent
        })
        .collect()
}

/// One prefix-cache block offered to a backend for restore-boundary selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixBlock {
    /// Block is already device-resident under this host page id.
    ResidentPage(u32),
    /// Block is demoted to the backend tier store under this key.
    DemotedKey(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTierLocation {
    HostDemoted,
    Disk,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvTierReadHits {
    pub host_demoted: u64,
    pub disk: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTierIoMode {
    #[default]
    Disabled,
    Mmap,
    Direct,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvTierIoStats {
    pub mode: KvTierIoMode,
    pub useful_read_bytes: u64,
    pub useful_write_bytes: u64,
    pub submitted_read_bytes: u64,
    pub submitted_write_bytes: u64,
    pub metadata_write_bytes: u64,
    pub failures: u64,
    pub completion_wait_ns: u64,
}

/// Count the largest leading prefix that is complete for a pages-only KV
/// restore contract.
///
/// Resident pages are already attachable. Demoted pages are attachable only
/// when the backend tier store can materialize the key into a resident page
/// before attach. Attention kernels never consume `PrefixBlock` or demoted
/// keys; they consume the backend's resident page table after this lowering.
/// The first missing demoted key truncates the prefix; the engine re-prefills
/// the tail.
pub fn pages_only_reusable_prefix_blocks(
    blocks: &[PrefixBlock],
    mut demoted_available: impl FnMut(u64) -> bool,
) -> usize {
    let mut reusable = 0usize;
    for block in blocks {
        match *block {
            PrefixBlock::ResidentPage(_) => reusable += 1,
            PrefixBlock::DemotedKey(key) if demoted_available(key) => reusable += 1,
            PrefixBlock::DemotedKey(_) => break,
        }
    }
    reusable
}

/// Cumulative speculative-decode counters (host-side integers, monotonic since
/// engine start; all zero when spec decode is off). Fed by the backend from its
/// accept-commit path — no device sync involved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecDecodeStats {
    pub chains: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub rejected: u64,
    /// Chains drafted from a partial (base > 0) draft context — the
    /// partial-ctx share after a rebase/prefix-restore gap.
    pub partial_ctx_chains: u64,
}

/// One implementation's cumulative dispatch count. IDs are stable registry
/// keys, never request shapes or other unbounded labels.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OperatorImplementationHits {
    pub implementation_id: String,
    pub hits: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OperatorDispatchStats {
    pub policy_hash: String,
    pub implementation_hits: Vec<OperatorImplementationHits>,
    pub fallback_count: u64,
}

/// One operation's cumulative CUDA timing. Names are stable operation keys
/// (e.g. "attention", "mlp", "embedding"); per-layer entries use
/// "{name}_layer{idx}".
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpTimingEntry {
    pub name: String,
    pub total_micros: u64,
    pub count: u64,
}

/// Backend-accumulated per-operation CUDA timing, sorted by total_micros desc.
/// Empty when the backend does not implement per-op profiling or the profiling
/// env var is unset.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpTimingStats {
    pub ops: Vec<OpTimingEntry>,
}

/// Backend-owned build artifacts reported only at a stats request boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendArtifactIdentity {
    pub kernel_bundle_id: String,
}

/// One planned row's device-pool demand for [`DeviceKvFit::kv_device_fit`],
/// in plan order (decode rows first, then prefill chunks).
#[derive(Debug, Clone, Copy)]
pub struct DeviceRowDemand {
    pub slot: usize,
    /// The row's KNOWN final logical span: prefill = the full prompt,
    /// decode = seq_len + 1. Backends with shape-dependent growth (DSv4
    /// bands) project their own page need from this.
    pub target_tokens: usize,
    /// The engine's worst-case page projection (1 per decode row, chunk
    /// pages + 1 per prefill chunk), for backends whose device pool mirrors
    /// the host accounting granularity (Qwen3.6 `full_attn_kv`).
    pub pages_hint: usize,
}

/// Per-step cost parameters of a backend executor: the engine sizes plans and
/// admission against these. `Default` is the unconstrained backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepLimits {
    /// Maximum number of plan rows the backend can execute in one scheduler
    /// step. Backend-neutral schedulers use this as a capability, not a type
    /// dependency: batched backends keep the unbounded default, while scalar
    /// backends (Metal/MLX today) report `1` so core never submits a plan shape
    /// the executor must reject.
    pub max_rows_per_step: usize,
    /// Maximum total plan tokens (decode rows + prefill chunk tokens) per
    /// forward. Backends with a hard per-forward token limit (e.g. the
    /// deepep_ll NVSHMEM dispatch buffer) report it so core never builds a
    /// forward the executor must reject; default unbounded.
    pub max_tokens_per_step: usize,
    /// Maximum number of live frontend requests this backend wants the serve
    /// layer to allow at once. Batched/server backends keep the unbounded
    /// default. Desktop scalar backends can return `1` so the frontend rejects a
    /// second request instead of queueing it and pretending concurrency is
    /// supported.
    pub max_live_requests: usize,
    /// Largest single prefill forward this executor accepts, in tokens; also
    /// the restore-snapshot grain when > restore alignment (a forward
    /// snapshots boundary state only at its own end, so a bigger chunk
    /// coarsens the boundary grid). Default: unbounded.
    pub max_prefill_chunk: usize,
    /// Extra chunk-END alignment beyond KV pages, in tokens: the planner
    /// aligns each prefill chunk's end position to `lcm(page_size, this)`, so
    /// backend side state that snapshots at forward-call ends (e.g. DSv4's
    /// ring) lands on restorable boundaries. End alignment only — chunk SIZE
    /// is bounded by `max_prefill_chunk`. Default `1`: no constraint.
    pub prefill_restore_boundary_alignment: usize,
    /// Total KV tokens one decode row may reach in a single
    /// [`BackendExecutor::submit`]: the committed token plus any speculative
    /// chain the executor drafts and verifies in that step. The engine budgets
    /// and pre-allocates this many tokens per decode row through its
    /// reclaim/preempt path, so a speculative executor never grows the KV pool
    /// inside `submit` — where an alloc failure is engine-fatal. Default 1:
    /// non-speculative executors append one token.
    pub spec_row_tokens: usize,
}

impl Default for StepLimits {
    fn default() -> Self {
        Self {
            max_rows_per_step: usize::MAX,
            max_tokens_per_step: usize::MAX,
            max_live_requests: usize::MAX,
            max_prefill_chunk: usize::MAX,
            prefill_restore_boundary_alignment: 1,
            spec_row_tokens: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuDeviceSample {
    pub gpu_index: u8,
    pub util_pct: u8,
    pub memory_used_mb: u32,
    pub memory_total_mb: u32,
    pub temp_c: u8,
    pub power_w: u16,
}

/// Fixed-capacity multi-GPU sample (up to 8 devices).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuSample {
    pub devices: [GpuDeviceSample; 8],
    pub device_count: u8,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Backend-reported counters and identity, read at stats-request boundaries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BackendStats {
    /// Cumulative speculative-decode counters from the accept-commit path.
    pub spec_decode: SpecDecodeStats,
    /// Operator-policy identity and cumulative dispatch counters.
    pub operator_dispatch: OperatorDispatchStats,
    /// Per-operation CUDA timing accumulated since process start. Backends that
    /// implement per-op profiling return their stats; the default is empty.
    pub op_timing: OpTimingStats,
    /// Exact backend build-artifact identity, if the build verified one.
    pub artifact: BackendArtifactIdentity,
    /// GPU utilization/VRAM/temp/power from the backend's background sampler.
    /// `None` for backends without one (Metal, placeholder, non-zero TP rank).
    pub gpu: Option<GpuSample>,
}

/// Host-only engine-core to backend-executor seam: the core submit/poll loop,
/// its cost parameters ([`StepLimits`]), and explicit capability accessors.
///
/// The plan and step output contain only host data. Any device tensors needed
/// for KV, logits, collectives, graphs, or sampling remain inside the executor.
///
/// A capability accessor returning `None` (the default) is a written opt-out:
/// the engine substitutes the documented inert behavior at each call site.
pub trait BackendExecutor {
    /// Opaque backend-owned in-flight handle.
    type Inflight;

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool)
    -> anyhow::Result<Self::Inflight>;

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>>;

    /// Advance completed backend work that is independent of the active
    /// forward. Engine-core calls this before request admission.
    fn poll_background(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn warmup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Model-default stop token ids (EOS + configured stop ids).
    ///
    /// Engine-core uses these as the fallback stop set for a request that did
    /// not supply its own `stop_token_ids`. The default is empty so backends
    /// that do not expose model defaults keep their existing behavior.
    fn model_stop_token_ids(&self) -> Vec<u32> {
        Vec::new()
    }

    /// Per-step cost parameters. Backends override by returning a filled
    /// struct; the default is unconstrained.
    fn step_limits(&self) -> StepLimits {
        StepLimits::default()
    }

    /// Drop any captured decode graph so the next step re-captures. Called
    /// inside the re-merge control closure after a live weight update: a
    /// graph that baked the old weight pointers would replay stale data
    /// (C1 graph-pointer staleness, #97). Default no-op for backends without
    /// a captured decode graph.
    fn invalidate_decode_graph(&mut self) {}

    fn stats(&self) -> BackendStats {
        BackendStats::default()
    }

    /// The engine freed `slot`'s host pages (finish/preempt/abort): release
    /// any backend device-side per-slot KV the executor allocates on demand
    /// (#154 Phase 3b: DSv4's FlashMLA band pages return to the layer pools
    /// here — waiting for the next occupant would starve the free lists).
    /// Default no-op for backends whose per-slot device KV is slot-fixed.
    fn release_kv_slot(&mut self, _slot: usize) {}

    /// Cross-TP-rank min-reduce of a per-rank-local scalar. Single-rank/no-TP
    /// backends (default) return `local` unchanged. TP backends MUST call
    /// this symmetrically on every rank, every tick that reaches it.
    /// Skipping the sync on one rank while another still calls it desyncs
    /// the admission collective permanently (2026-07-05 TP=4 admission
    /// livelock — see
    /// docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
    fn tp_sync_min(&self, local: usize) -> anyhow::Result<usize> {
        Ok(local)
    }

    /// This rank's (cp_rank, cp_size) for the host pool's CP sequence-shard
    /// filter, or `None` when 2D (attn_tp × cp) is not engaged. The engine
    /// divides page budgeting by `cp_size` and recovers the rank directly —
    /// no pool probe needed. Default `None` (no sharding).
    fn kv_shard_spec(&self) -> Option<(usize, usize)> {
        None
    }

    /// Prefix-cache restore/publish coordination below the seam.
    fn prefix_reuse(&mut self) -> Option<&mut dyn PrefixReuse> {
        None
    }

    fn kv_page_tier(&mut self) -> Option<&mut dyn KvPageTier> {
        None
    }

    /// Read-only [`Self::kv_page_tier`] counterpart for telemetry reads;
    /// presence MUST match, or tier counters silently read as zero.
    fn kv_page_tier_view(&self) -> Option<&dyn KvPageTier> {
        None
    }

    /// Whole-slot KV tier store, for models whose restore state is not
    /// page-addressable. Presence gates the oversubscription park path.
    fn kv_slot_tier(&mut self) -> Option<&mut dyn KvSlotTier> {
        None
    }

    /// Device KV pool(s) separate from the engine's accounting pool, so plan
    /// rows must clear [`DeviceKvFit::kv_device_fit`] before submit. `None`
    /// (the default) skips the gate — the engine builds no demand rows at all.
    fn device_kv_fit(&self) -> Option<&dyn DeviceKvFit> {
        None
    }

    /// OPD weight/scratch/KV-pool VRAM residency control.
    fn weight_residency(&mut self) -> Option<&mut dyn WeightResidency> {
        None
    }

    /// Direct multimodal generation path. Returning `Some` IS the "this is a
    /// VLM backend" statement; `None` (the default) = text-only backend.
    fn multimodal(&mut self) -> Option<&mut dyn MultimodalGenerate> {
        None
    }
}

/// Prefix-cache restore/publish capability: restore boundaries, sidecar side
/// state, and position-0 whole-prefix snapshots. A backend without this
/// capability publishes nothing and restores nothing (fail-closed).
pub trait PrefixReuse {
    /// How many leading prefix-cache blocks are complete restore boundaries
    /// that the executor can materialize and attach to a slot.
    ///
    /// The host radix can match any page boundary, resident or demoted. A
    /// backend may need more than page bytes to resume from a boundary:
    /// recurrent state, ring cursors, compressor metadata, mirrored snapshots,
    /// or other backend-owned side state. Returning fewer blocks tells
    /// engine-core not to promote or attach the tail. A nonzero return value is
    /// a promise that after promote/attach, the backend's attention path can
    /// consume the resulting resident page table without any missing side state.
    /// Pages-only executors explicitly opt in with
    /// [`pages_only_reusable_prefix_blocks`].
    fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize;

    /// Reuse length for admitting a REQUEST, when the backend may reuse content
    /// that extends PAST the radix page-match and must verify it against the
    /// request's prompt `tokens`. DSv4 finish-write-through caches a sub-page
    /// tail whose content the radix key does not cover — it reuses that tail
    /// only for a verified continuation (`tokens` contains the cached sequence
    /// through the finish position), else falls back to the strict page-aligned
    /// count. Backends without such content defer to
    /// [`Self::reusable_prefix_blocks`], ignoring `tokens`.
    fn reusable_prefix_blocks_for_prompt(&self, blocks: &[PrefixBlock], tokens: &[u32]) -> usize;

    /// Notify the backend that host prefix-cache pages were evicted from the
    /// radix cache and released by the host KV pool. Backends that mirror page
    /// contents or restore-boundary side state below the seam can drop those
    /// mirrors here.
    fn release_prefix_pages(&mut self, pages: &[u32]);

    /// Notify the backend that a slot's pages are returning to the free pool
    /// (slot free/abort/preempt). Backends drop any NON-confirmed (write-only,
    /// never radix-published) prefix state keyed to these page ids — a freed
    /// id recycles, and a stale provisional entry under a recycled id could
    /// later be confirmed as if it were the new occupant's content. Confirmed
    /// entries ride the radix lifetime ([`Self::release_prefix_pages`]) and
    /// are untouched.
    fn release_provisional_prefix_pages(&mut self, pages: &[u32]);

    /// Restore the sidecar side state for `slot` when reusing a page-radix prefix
    /// of length `matched_len`, returning the ABSOLUTE token length actually
    /// restored — the position the engine sets `prefill_start_pos` to. Called by
    /// `attach_prefix_to_request` after `kv.attach_pages()` succeeds. Returning
    /// `matched_len` is byte-identical to a page-aligned restore (the engine
    /// prefills from `matched_len`) — the conservative correct answer for
    /// full-attention-only backends (CUDA Qwen dense, Metal).
    ///
    /// Two backends restore a length OTHER than `matched_len`:
    /// - **DSv4 finish-write-through** restores PAST the 64-block-aligned
    ///   `matched_len` to the finished turn's exact length (its sub-page tail
    ///   block lives in the pool entry's pending section) — returns
    ///   `matched_len + tail` (`tail < 64`); the engine grows the slot's own
    ///   partial page and prefills only from the extended length.
    /// - **Qwen3.5/3.6 hybrid** may restore a periodic snapshot boundary `B ≤
    ///   matched_len` when no sidecar exists exactly at `matched_len` (a
    ///   cross-conversation prefix hit) — returns `B`; the engine truncates the
    ///   attached pages back to `B` and re-prefills `[B..prompt]`.
    ///
    /// `prefix_pages` are the physical host-pool page ids already attached to the
    /// slot — the hybrid override uses them to sync the device KV pool seq_len.
    fn restore_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<usize>;

    /// Write the finishing slot's full frontier state THROUGH to the backend's
    /// content-keyed prefix store, so a later turn restores to the exact finish
    /// position. Called by `finish_slot` BEFORE `publish_prefix_blocks` (whose
    /// confirm/repair reconciles the provisional entries this publishes) and
    /// before `free_slot_pages`, at the eager finish sync point (graph-safe).
    /// `slot_pages` is the slot's own host page chain. Best-effort: a failure
    /// only forfeits future reuse, never the finish. Only DSv4 captures;
    /// others return `Ok`.
    fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        slot_pages: &[u32],
    ) -> anyhow::Result<()>;

    /// Capture the sidecar restore-boundary side state for `slot` at the
    /// just-published radix prefix `tokens[..matched_len]`, keyed so a later
    /// [`Self::restore_prefix_sidecar`] at the same boundary hits it. Called by
    /// `publish_prefix_blocks` right after the radix insert, while the slot's
    /// device state is still resident. `prefix_pages` are the resident host-pool
    /// pages the published prefix covers — the backend keys eviction
    /// coordination ([`Self::release_prefix_pages`]) off them so the sidecar's
    /// lifetime rides the radix blocks. `newly_cached` is the subset the radix
    /// insert actually adopted (and retained) this call — pages the radix
    /// deduped away will be freed and their ids recycled, so a backend must
    /// never key durable state to anything outside `newly_cached`.
    /// `slot_pages` is the slot's OWN page chain over the same token range —
    /// where radix dedup diverged it from `prefix_pages`, a backend can repair
    /// canonical-keyed state from the slot's freshly recomputed (content-
    /// identical) pages. A no-op `Ok` for full-attention-only backends;
    /// Qwen3.5/3.6 hybrid and DSv4 capture here.
    fn save_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
        slot_pages: &[u32],
        newly_cached: &[u32],
    ) -> anyhow::Result<()>;
}

pub trait KvPageTier {
    /// Number of KV pages the backend's host-demoted store can hold. `0`
    /// means the tier store is currently unbudgeted and the engine never calls
    /// the demote/promote hooks — the baseline eviction path stays
    /// byte-for-byte unchanged.
    fn kv_tier_capacity_pages(&self) -> usize;

    fn kv_tier_page_bytes(&self) -> usize;

    fn kv_tier_host_demoted_pages(&self) -> usize;

    fn kv_tier_disk_pages(&self) -> usize;

    fn kv_tier_read_hits(&self) -> KvTierReadHits;

    fn kv_tier_io_stats(&self) -> KvTierIoStats;

    fn kv_tier_transfer_is_zero_copy(&self) -> bool;

    fn kv_tier_location(&self, key: u64) -> Option<KvTierLocation>;

    /// Copy the contents of device KV pages into the backend's host tier
    /// store, keyed by the engine-assigned tier keys.
    ///
    /// The copy MUST be complete (host copy durable, no in-flight device
    /// reads pending) before this returns: the engine frees each accepted
    /// page immediately after, and a later allocation may overwrite it.
    /// Returns how many *leading* entries were accepted; entries past that
    /// count were rejected (store full) and the engine falls back to plain
    /// eviction for them.
    fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> anyhow::Result<usize>;

    /// Copy tier-store entries back into freshly allocated device KV pages
    /// (`(tier_key, dst_page)` pairs).
    ///
    /// The copy MUST be complete before this returns: the engine attaches the
    /// destination pages to a slot immediately after and the next forward
    /// step reads them. Promoted entries stay in the store until the engine
    /// drops them via [`Self::drop_kv_tier_entries`]. Only called
    /// when [`Self::kv_tier_capacity_pages`] is nonzero.
    fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> anyhow::Result<()>;

    /// Drop tier-store entries whose radix nodes were severed or restored to
    /// device residency.
    fn drop_kv_tier_entries(&mut self, keys: &[u64]);
}

/// Whole-slot KV tier capability: demote/promote a slot's complete restore
/// state as one image when page-addressed restore is not sufficient for that
/// model. Accessor presence gates the engine's oversubscription park path.
pub trait KvSlotTier {
    /// Snapshot the complete restore state of `slot` into the backend host
    /// store under `key`: KV at its exact positions plus every backend-owned
    /// sidecar needed to resume from the materialized sequence position. The
    /// copy MUST be complete before returning — the engine frees the slot
    /// immediately after. Returns `false` when the store has no room (the
    /// engine falls back to plain recompute).
    fn demote_slot(&mut self, slot: usize, key: u64) -> anyhow::Result<bool>;

    /// Restore a whole-slot snapshot into `slot`. The engine resumes from the
    /// exact demoted materialized position right after, so every byte of
    /// required backend state MUST be restored before returning.
    fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> anyhow::Result<()>;

    /// Drop whole-slot store entries (promoted, cancelled, or abandoned).
    fn drop_kv_slot_entries(&mut self, keys: &[u64]);
}

/// Device-pool fit gate capability, for backends whose device KV pool(s) are
/// separate from the engine's accounting pool.
pub trait DeviceKvFit {
    /// Per-row fit of the planned rows (decode rows first, then prefill
    /// chunks — the engine's shed-priority order) against the backend's own
    /// device KV pool(s). Push the index of EVERY row that does not fit into
    /// `unfit` (ascending); a fitting row debits the remaining headroom
    /// cumulatively, an unfit row debits nothing so later smaller rows are
    /// still tested — shedding exactly the unfit rows, never the tail (a
    /// stuck low-index row must not starve later fitting rows). The engine
    /// parks/sheds the pushed rows so device-side exhaustion degrades (#162)
    /// instead of failing an alloc inside [`BackendExecutor::submit`] — which is
    /// engine-fatal (#164). Backends with heterogeneous pools (DSv4
    /// per-layer FlashMLA bands, #160) must pair need and headroom PER POOL
    /// — a saturated pool with zero incremental need is not exhaustion. Host
    /// bookkeeping only — never a device readback (CUDA-graph hot loop).
    fn kv_device_fit(&self, rows: &[DeviceRowDemand], unfit: &mut Vec<usize>);
}

/// OPD VRAM residency capability: weight offload/reload plus scratch and KV
/// pool release for the co-resident writeback bracket.
pub trait WeightResidency {
    /// Move the model's device weights to host RAM and free the VRAM (OPD teacher
    /// time-share), returning the device bytes freed.
    /// After a successful offload the executor must NOT run a forward step until
    /// [`Self::reload_weights`].
    fn offload_weights(&mut self) -> anyhow::Result<usize>;

    /// Restore the model's device weights from the host snapshot (OPD teacher
    /// time-share). Idempotent if no offload is pending.
    fn reload_weights(&mut self) -> anyhow::Result<()>;

    /// Release the inference forward scratch (workspace / batched-decode scratch /
    /// captured graphs) WITHOUT offloading weights or evicting KV, so a co-resident
    /// OPD writeback reuses the VRAM (the OPD rollout->writeback path never
    /// offloads). The scratch rebuilds lazily on the next step.
    fn release_inference_scratch(&mut self) -> anyhow::Result<()>;

    /// Drop the engine's KV pool WITHOUT offloading weights, freeing its HBM for a
    /// co-resident OPD writeback whose fresh autograd forward does NOT use this
    /// engine's KV cache (so the pool is dead during the writeback). Paired with
    /// [`Self::ensure_kv_pool`], which re-acquires it before the next rollout.
    fn release_kv_pool(&mut self) -> anyhow::Result<()>;

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the next
    /// rollout. Idempotent if the pool is already resident.
    fn ensure_kv_pool(&mut self) -> anyhow::Result<()>;
}

/// Direct multimodal generation capability for scalar backends whose
/// image/text fusion happens inside the backend-owned model wrapper.
pub trait MultimodalGenerate {
    fn generate_multimodal(
        &mut self,
        prompt_tokens: &[u32],
        images: &[MultimodalImage],
        max_tokens: usize,
        sampling: &SamplingParams,
    ) -> anyhow::Result<Option<DiffusionGenerateOutput>>;

    /// Which VLM image-preprocessing/marker convention this backend expects.
    /// The serving layer dispatches preprocessing on this so a second VLM
    /// (DeepSeek-OCR) has its own resize/marker logic.
    fn multimodal_kind(&self) -> Option<MultimodalKind>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionVerdict {
    Admit,
    /// Admit nothing this tick (memory pressure, foreground contention, …).
    Hold,
}

/// Per-tick GPU work budget the engine must respect to stay an OS good citizen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBudget {
    pub max_tokens: usize,
    pub max_micros: u64,
}

impl StepBudget {
    /// Unbounded budget — the server-style "use all the hardware" default.
    pub const UNBOUNDED: StepBudget = StepBudget {
        max_tokens: usize::MAX,
        max_micros: u64::MAX,
    };
}

/// OS-citizen resource governance (AI PC north-star).
///
/// Engine-core consults the governor at the admission boundary and at step
/// boundaries so the engine never degrades the user's interactive OS use:
/// no busy-spin, bounded memory, yield to the foreground. This is the one
/// seam the AI-PC pivot adds; it is host-side and backend-neutral. Backends
/// supply the OS-signal readers (Metal: macOS memory-pressure + wired-limit
/// headroom + foreground/battery; CUDA: nvml free VRAM; AMD APU: unified-memory
/// pressure).
pub trait ResourceGovernor {
    fn admission_gate(&self) -> AdmissionVerdict;

    fn step_budget(&self) -> StepBudget;

    fn should_yield(&self) -> bool;
}

/// Permissive default: admit freely, unbounded budget, never yield.
///
/// This is the server-style baseline and the engine-core default until a
/// backend installs a real governor that reads OS pressure signals.
#[derive(Debug, Clone, Copy, Default)]
pub struct PermissiveGovernor;

impl ResourceGovernor for PermissiveGovernor {
    fn admission_gate(&self) -> AdmissionVerdict {
        AdmissionVerdict::Admit
    }

    fn step_budget(&self) -> StepBudget {
        StepBudget::UNBOUNDED
    }

    fn should_yield(&self) -> bool {
        false
    }
}

/// Static cooperative governor for local, low-impact serving.
///
/// It is intentionally backend-neutral: the caller chooses a token budget and
/// optional cooperative yield cadence, while engine-core enforces those knobs at
/// scheduler boundaries. OS-signal readers can wrap the same trait later without
/// changing service or scheduler code.
#[derive(Debug)]
pub struct CooperativeGovernor {
    admission: AdmissionVerdict,
    budget: StepBudget,
    yield_every_ticks: usize,
    tick: Cell<usize>,
}

impl CooperativeGovernor {
    #[must_use]
    pub fn new(budget: StepBudget) -> Self {
        Self {
            admission: AdmissionVerdict::Admit,
            budget,
            yield_every_ticks: 0,
            tick: Cell::new(0),
        }
    }

    /// Yield every `n` scheduler ticks. `0` disables periodic yield.
    #[must_use]
    pub fn with_yield_every_ticks(mut self, n: usize) -> Self {
        self.yield_every_ticks = n;
        self
    }
}

impl ResourceGovernor for CooperativeGovernor {
    fn admission_gate(&self) -> AdmissionVerdict {
        self.admission
    }

    fn step_budget(&self) -> StepBudget {
        self.budget
    }

    fn should_yield(&self) -> bool {
        if self.yield_every_ticks == 0 {
            return false;
        }
        let tick = self.tick.get().wrapping_add(1);
        self.tick.set(tick);
        tick.is_multiple_of(self.yield_every_ticks)
    }
}
