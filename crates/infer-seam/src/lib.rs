//! Inference execution seam traits.
//!
//! The engine-core facing seam is host-only: [`ForwardPlan`], [`StepOutput`],
//! and [`KvPool`] expose slots, page ids, token ids, and lengths. Device
//! tensors, collectives, sampling, and the model forward all live inside the
//! backend executors ([`BackendExecutor`]), never crossing this seam.

use std::cell::Cell;

use infer_plan::{ForwardPlan, StepOutput};

#[path = "allocator.rs"]
mod allocator;
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
pub use host_paged_kv_pool::HostPagedKvPool;
pub use kv::KvPool;
pub use kv_batch::{KvBatchDescriptor, KvBatchRow, KvBatchRowKind};
pub use kv_dtype::KvCacheDtype;
pub use kv_query::KvQuery;
pub use prefix_store::KvPrefixStore;
pub use resource::{
    HostTierBudget, HostTierPolicy, SlotBudget, clamp_to_affordable, split_host_tiers,
};

/// Result of polling a submitted executor step.
#[derive(Debug, Clone)]
pub enum PollResult<I> {
    /// The executor step finished and produced host-visible tokens.
    Ready(StepOutput),
    /// The executor step is still in flight and should be polled again.
    NotReady(I),
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

    /// Maximum number of plan rows the backend can execute in one scheduler
    /// step. Backend-neutral schedulers use this as a capability, not a type
    /// dependency: batched backends keep the unbounded default, while scalar
    /// backends (Metal/MLX today) report `1` so core never submits a plan shape
    /// the executor must reject.
    fn max_rows_per_step(&self) -> usize {
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

    /// How many leading prefix-cache pages of `block_ids` (in prompt order) the
    /// executor can actually reuse when attaching this prefix to a slot.
    ///
    /// The host radix caches a block at every page boundary, but a backend
    /// whose layers carry prefix-wide recurrent/conv state (e.g. Metal
    /// linear-attention "GDR" layers) can only attach at boundaries where it
    /// snapshotted that state during a forward pass — chunked prefill skips the
    /// interior boundaries. Returning fewer pages than `block_ids.len()` tells
    /// engine-core to re-prefill the unsnapshotted tail instead of asking the
    /// executor for a boundary it cannot serve (which would error on attach).
    /// The default reuses everything the radix offers, which is correct for
    /// fully page-sliceable KV (paged attention).
    fn reusable_prefix_pages(&self, block_ids: &[u32]) -> usize {
        block_ids.len()
    }

    /// Notify the backend that host prefix-cache pages were evicted from the
    /// radix cache and released by the host KV pool. Backends that mirror page
    /// contents or prefix snapshots below the seam can drop those mirrors here;
    /// the default is a no-op for page-sliceable or stateless executors.
    fn release_prefix_pages(&mut self, _pages: &[u32]) {}

    /// Number of KV pages the backend's host-side tier store (T1 DRAM) can
    /// hold. `0` (the default) means the backend has no tier store and the
    /// engine never calls the demote/promote hooks — the baseline eviction
    /// path stays byte-for-byte unchanged.
    fn kv_tier_capacity_pages(&self) -> usize {
        0
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

    /// Whether the backend can demote/promote a whole slot's device state as
    /// one image — the tier route for models whose KV is NOT page-addressable
    /// (recurrent / ring / compressed-arena state, e.g. DSv4). Default: no.
    fn kv_slot_tier_enabled(&self) -> bool {
        false
    }

    /// Snapshot the entire device state of `slot` (KV at its exact positions
    /// plus every recurrent/ring/compressor sidecar) into the backend host
    /// store under `key`. The copy MUST be complete before returning — the
    /// engine frees the slot immediately after. Returns `false` when the
    /// store has no room (the engine falls back to plain recompute).
    fn demote_slot(&mut self, _slot: usize, _key: u64) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Restore a whole-slot image into `slot`. The engine resumes decode at
    /// the exact demoted position right after, so the copy MUST be complete
    /// before returning. Only called when
    /// [`BackendExecutor::kv_slot_tier_enabled`] is `true`.
    fn promote_slot(&mut self, _key: u64, _slot: usize) -> anyhow::Result<()> {
        anyhow::bail!("backend has no whole-slot KV tier store")
    }

    /// Drop whole-slot store entries (promoted, cancelled, or abandoned).
    fn drop_kv_slot_entries(&mut self, _keys: &[u64]) {}

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
