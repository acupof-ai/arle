//! Inference execution seam traits.
//!
//! The engine-core facing seam is host-only: [`ForwardPlan`], [`StepOutput`],
//! and [`KvPool`] expose slots, page ids, token ids, and lengths. Device
//! tensors, collectives, sampling, and the model forward all live inside the
//! backend executors ([`BackendExecutor`]), never crossing this seam.

use infer_plan::{ForwardPlan, StepOutput};

#[path = "allocator.rs"]
mod allocator;
#[path = "kv.rs"]
mod kv;
#[path = "kv_batch.rs"]
mod kv_batch;
#[path = "kv_query.rs"]
mod kv_query;
#[path = "prefix_store.rs"]
mod prefix_store;

pub use allocator::KvAllocator;
pub use kv::KvPool;
pub use kv_batch::{KvBatchDescriptor, KvBatchRow, KvBatchRowKind};
pub use kv_query::KvQuery;
pub use prefix_store::KvPrefixStore;

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
