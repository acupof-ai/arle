//! Inference execution seam traits.
//!
//! The engine-core facing seam is host-only: [`ForwardPlan`], [`StepOutput`],
//! and [`KvPool`] expose slots, page ids, token ids, and lengths. Device
//! tensors, collectives, sampling, and the model forward all live inside the
//! backend executors ([`BackendExecutor`]), never crossing this seam.

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
#[path = "kv_dtype.rs"]
mod kv_dtype;
#[path = "kv_query.rs"]
mod kv_query;
#[path = "prefix_store.rs"]
mod prefix_store;
#[path = "resource.rs"]
mod resource;

pub use allocator::KvAllocator;
pub use diffusion_executor::BufferedDiffusionExecutor;
pub use host_paged_kv_pool::{EVICTED_PAGE, HostPagedKvPool};
pub use kv::KvPool;
pub use kv_batch::{KvBatchDescriptor, KvBatchRow, KvBatchRowKind};
pub use kv_dtype::KvCacheDtype;
pub use kv_query::KvQuery;
pub use prefix_store::KvPrefixStore;
pub use resource::{
    DramTierPolicy, NvmeTierPolicy, PROFILE_KV_TOKENS_FLOOR, SlotBudget, clamp_mem_fraction_static,
    clamp_to_affordable, dram_l2_budget, nvme_l3_budget, profile_kv_pool_tokens,
};

/// Result of polling a submitted executor step.
#[derive(Debug, Clone)]
pub enum PollResult<I> {
    /// The executor step finished and produced host-visible tokens.
    Ready(StepOutput),
    /// The executor step is still in flight and should be polled again.
    NotReady(I),
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

/// Host-only engine-core to backend-executor seam.
///
/// The plan and step output contain only host data. Any device tensors needed
/// for KV, logits, collectives, graphs, or sampling remain inside the executor.
pub trait BackendExecutor {
    /// Opaque backend-owned in-flight handle.
    type Inflight;

    /// Submit a forward plan for asynchronous or synchronous execution.
    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool)
    -> anyhow::Result<Self::Inflight>;

    /// Poll an in-flight forward step for completion.
    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>>;

    /// Perform optional backend warmup before serving.
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

    /// Optional direct multimodal generation path for scalar backends whose
    /// image/text fusion happens inside the backend-owned model wrapper.
    fn generate_multimodal(
        &mut self,
        _prompt_tokens: &[u32],
        _images: &[MultimodalImage],
        _max_tokens: usize,
        _sampling: &SamplingParams,
    ) -> anyhow::Result<Option<DiffusionGenerateOutput>> {
        Ok(None)
    }

    /// Which VLM image-preprocessing/marker convention this backend expects.
    /// The serving layer dispatches preprocessing on this so a second VLM
    /// (DeepSeek-OCR) doesn't run Gemma4's resize/marker logic. Default `None`
    /// = text-only backend.
    fn multimodal_kind(&self) -> Option<MultimodalKind> {
        None
    }

    /// Maximum number of plan rows the backend can execute in one scheduler
    /// step. Backend-neutral schedulers use this as a capability, not a type
    /// dependency: batched backends keep the unbounded default, while scalar
    /// backends (Metal/MLX today) report `1` so core never submits a plan shape
    /// the executor must reject.
    fn max_rows_per_step(&self) -> usize {
        usize::MAX
    }

    /// Maximum total plan tokens (decode rows + prefill chunk tokens) per
    /// forward. Backends with a hard per-forward token limit (e.g. the
    /// deepep_ll NVSHMEM dispatch buffer) report it so core never builds a
    /// forward the executor must reject; default unbounded.
    fn max_tokens_per_step(&self) -> usize {
        usize::MAX
    }

    /// Maximum number of live frontend requests this backend wants the serve
    /// layer to allow at once. Batched/server backends keep the unbounded
    /// default. Desktop scalar backends can return `1` so the frontend rejects a
    /// second request instead of queueing it and pretending concurrency is
    /// supported.
    fn max_live_requests(&self) -> usize {
        usize::MAX
    }

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
    /// The default is fail-closed; pages-only executors explicitly opt in with
    /// [`pages_only_reusable_prefix_blocks`].
    fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        let _ = blocks;
        0
    }

    /// Notify the backend that host prefix-cache pages were evicted from the
    /// radix cache and released by the host KV pool. Backends that mirror page
    /// contents or restore-boundary side state below the seam can drop those
    /// mirrors here; the default is a no-op for executors with no such mirrors.
    fn release_prefix_pages(&mut self, _pages: &[u32]) {}

    /// Number of KV pages the backend's host-demoted store can
    /// hold. `0` (the default) means the backend has no tier store and the
    /// engine never calls the demote/promote hooks — the baseline eviction
    /// path stays byte-for-byte unchanged.
    fn kv_tier_capacity_pages(&self) -> usize {
        0
    }

    fn kv_tier_page_bytes(&self) -> usize {
        0
    }

    fn kv_tier_host_demoted_pages(&self) -> usize {
        0
    }

    fn kv_tier_disk_pages(&self) -> usize {
        0
    }

    fn kv_tier_transfer_is_zero_copy(&self) -> bool {
        false
    }

    fn kv_tier_location(&self, _key: u64) -> Option<KvTierLocation> {
        None
    }

    /// Copy the contents of device KV pages into the backend's host tier
    /// store, keyed by the engine-assigned tier keys.
    ///
    /// The copy MUST be complete (host copy durable, no in-flight device
    /// reads pending) before this returns: the engine frees each accepted
    /// page immediately after, and a later allocation may overwrite it.
    /// Returns how many *leading* entries were accepted; entries past that
    /// count were rejected (store full) and the engine falls back to plain
    /// eviction for them. The default backend has no tier store and accepts
    /// nothing.
    fn demote_prefix_pages(&mut self, _entries: &[(u32, u64)]) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Copy tier-store entries back into freshly allocated device KV pages
    /// (`(tier_key, dst_page)` pairs).
    ///
    /// The copy MUST be complete before this returns: the engine attaches the
    /// destination pages to a slot immediately after and the next forward
    /// step reads them. Promoted entries stay in the store until the engine
    /// drops them via [`BackendExecutor::drop_kv_tier_entries`]. Only called
    /// when [`BackendExecutor::kv_tier_capacity_pages`] is nonzero.
    fn promote_prefix_pages(&mut self, _entries: &[(u64, u32)]) -> anyhow::Result<()> {
        anyhow::bail!("backend has no KV tier store")
    }

    /// Drop tier-store entries whose radix nodes were severed or restored to
    /// device residency. The default is a no-op for backends without a tier.
    fn drop_kv_tier_entries(&mut self, _keys: &[u64]) {}

    /// Whether the backend can demote/promote a whole slot's complete restore
    /// state as one image when page-addressed restore is not sufficient for
    /// that model. Default: no.
    fn kv_slot_tier_enabled(&self) -> bool {
        false
    }

    /// Snapshot the complete restore state of `slot` into the backend host
    /// store under `key`: KV at its exact positions plus every backend-owned
    /// sidecar needed to resume from the materialized sequence position. The
    /// copy MUST be complete before returning — the engine frees the slot
    /// immediately after. Returns `false` when the store has no room (the
    /// engine falls back to plain recompute).
    fn demote_slot(&mut self, _slot: usize, _key: u64) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Restore a whole-slot snapshot into `slot`. The engine resumes from the
    /// exact demoted materialized position right after, so every byte of
    /// required backend state MUST be restored before returning. Only called
    /// when [`BackendExecutor::kv_slot_tier_enabled`] is `true`.
    fn promote_slot(&mut self, _key: u64, _slot: usize, _slot_pages: &[u32]) -> anyhow::Result<()> {
        anyhow::bail!("backend has no whole-slot KV tier store")
    }

    /// Drop whole-slot store entries (promoted, cancelled, or abandoned).
    fn drop_kv_slot_entries(&mut self, _keys: &[u64]) {}

    /// Length of the longest leading prefix of `tokens` for which the backend
    /// holds a position-0-anchored cached KV snapshot it can restore into a fresh
    /// slot.
    ///
    /// This is the cross-request prefix-reuse seam for backends whose KV cannot
    /// be page-reattached at arbitrary positions (DSv4's RoPE-rotated K, the
    /// sliding-window ring indexed by `abs_pos % window`, and the DSA indexer
    /// keys are all position-locked). The only safe reuse is a prefix captured
    /// at absolute positions `[0, len)` and reattached as the leading prefix of
    /// a new request that also starts at position 0. The default `0` means the
    /// backend has no such store and the engine never calls the capture/restore
    /// hooks — the page-radix reuse path stays byte-for-byte unchanged.
    fn cached_prefix_match_len(&self, _tokens: &[u32]) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Capture `slot`'s complete restore image into the backend's
    /// position-0-anchored prefix store, keyed by `tokens`.
    ///
    /// Called on request finish ONLY when the request's prefill started at
    /// absolute position 0, so the captured KV is exactly the materialization of
    /// `tokens` at positions `[0, tokens.len())`. The copy MUST be complete
    /// before returning (the engine frees the slot right after). The default is
    /// a no-op for backends without the store.
    fn capture_cached_prefix(&mut self, _slot: usize, _tokens: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }

    /// Restore the cached position-0 prefix snapshot for `tokens[..matched_len]`
    /// into `slot`, setting the slot's materialized length to `matched_len`.
    ///
    /// `matched_len` is the value returned by [`Self::cached_prefix_match_len`].
    /// The engine has already allocated `matched_len` tokens of host KV pages on
    /// `slot` and resumes prefill from absolute position `matched_len` right
    /// after, so every byte of restored KV/side state MUST land before
    /// returning. Only called when `cached_prefix_match_len > 0`.
    fn restore_cached_prefix(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _slot_pages: &[u32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("backend has no position-0 prefix store")
    }

    /// Restore the sidecar recurrent state for `slot` when reusing a page-radix
    /// prefix of length `matched_len`. Called by `attach_prefix_to_request` after
    /// `kv.attach_pages()` succeeds. Default no-op for full-attention-only backends
    /// (CUDA Qwen dense, Metal); only Qwen3.5/3.6 hybrid overrides this.
    ///
    /// `prefix_pages` are the physical host-pool page ids already attached to the
    /// slot — the hybrid override uses them to sync the device KV pool seq_len.
    fn restore_prefix_sidecar(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _prefix_pages: &[u32],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Move the model's device weights to host RAM and free the VRAM (OPD teacher
    /// time-share), returning the device bytes freed. The default is a no-op
    /// (returns 0) so backends that do not support weight offload are unaffected.
    /// After a successful offload the executor must NOT run a forward step until
    /// [`BackendExecutor::reload_weights`].
    fn offload_weights(&mut self) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Restore the model's device weights from the host snapshot (OPD teacher
    /// time-share). The default is a no-op so non-offloading backends are
    /// unaffected; idempotent if no offload is pending.
    fn reload_weights(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Release the inference forward scratch (workspace / batched-decode scratch /
    /// captured graphs) WITHOUT offloading weights or evicting KV, so a co-resident
    /// OPD writeback reuses the VRAM (the OPD rollout->writeback path never offloads).
    /// The default is a no-op (returns `Ok`) so backends without an inference
    /// scratch surface are unaffected; the scratch rebuilds lazily on the next step.
    fn release_inference_scratch(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Drop the engine's KV pool WITHOUT offloading weights, freeing its HBM for a
    /// co-resident OPD writeback whose fresh autograd forward does NOT use this
    /// engine's KV cache (so the pool is dead during the writeback). Default no-op
    /// (backends without a droppable pool are unaffected). Paired with
    /// [`Self::ensure_kv_pool`], which re-acquires it before the next rollout.
    fn release_kv_pool(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the next
    /// rollout. Default no-op; idempotent if the pool is already resident.
    fn ensure_kv_pool(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixBlock, pages_only_reusable_prefix_blocks};

    #[test]
    fn pages_only_counts_resident_and_available_demoted_prefix() {
        let blocks = [
            PrefixBlock::ResidentPage(1),
            PrefixBlock::DemotedKey(7),
            PrefixBlock::ResidentPage(2),
            PrefixBlock::DemotedKey(8),
        ];
        assert_eq!(
            pages_only_reusable_prefix_blocks(&blocks, |key| key == 7),
            3
        );
    }

    #[test]
    fn pages_only_truncates_at_first_missing_demoted_key() {
        let blocks = [
            PrefixBlock::ResidentPage(1),
            PrefixBlock::DemotedKey(7),
            PrefixBlock::ResidentPage(2),
        ];
        assert_eq!(pages_only_reusable_prefix_blocks(&blocks, |_| false), 1);
    }
}

/// Verdict returned by the resource governor at the admission boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionVerdict {
    /// Admit waiting work normally.
    Admit,
    /// Admit nothing this tick (memory pressure, foreground contention, …).
    Hold,
    /// Shed running work down to at most `n` concurrent requests.
    ShedTo(usize),
}

/// Per-tick GPU work budget the engine must respect to stay an OS good citizen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBudget {
    /// Upper bound on tokens to process this tick.
    pub max_tokens: usize,
    /// Soft wall-clock budget for this tick, in microseconds.
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
/// pressure). See docs/projects/2026-06-03-aipc-pivot-and-northstar.md.
pub trait ResourceGovernor {
    /// May the engine admit more waiting work right now?
    fn admission_gate(&self) -> AdmissionVerdict;

    /// How much GPU work may this tick do without harming foreground UX?
    fn step_budget(&self) -> StepBudget;

    /// Should the engine back off this tick to keep the OS responsive?
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
    /// Build a governor that admits normally, enforces `budget`, and never
    /// voluntarily yields unless [`Self::with_yield_every_ticks`] is used.
    #[must_use]
    pub fn new(budget: StepBudget) -> Self {
        Self {
            admission: AdmissionVerdict::Admit,
            budget,
            yield_every_ticks: 0,
            tick: Cell::new(0),
        }
    }

    /// Override the admission verdict returned at scheduler admission.
    #[must_use]
    pub fn with_admission(mut self, admission: AdmissionVerdict) -> Self {
        self.admission = admission;
        self
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
