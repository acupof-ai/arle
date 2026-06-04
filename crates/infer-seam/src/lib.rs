//! Inference execution seam traits.
//!
//! The engine-core facing seam is host-only: [`ForwardPlan`], [`StepOutput`],
//! and [`KvPool`] expose slots, page ids, token ids, and lengths. Device
//! tensors remain inside backend executors, model implementations, and the
//! lower-seam traits in this crate.

use infer_plan::{ForwardPlan, SamplingParams, StepOutput};

#[path = "allocator.rs"]
mod allocator;
#[path = "kv.rs"]
mod kv;
#[path = "kv_query.rs"]
mod kv_query;
#[path = "prefix_store.rs"]
mod prefix_store;

pub use allocator::KvAllocator;
pub use kv::KvPool;
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
/// for KV, logits, collectives, graphs, or sampling remain inside the executor
/// and lower-seam implementations.
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
}

/// Backend-internal collective communication seam.
///
/// `Tensor` is deliberately associated with the backend implementation and is
/// never exposed through the engine-core executor boundary.
pub trait Communicator {
    /// Backend tensor type used by collectives.
    type Tensor;

    /// Run an in-place all-reduce over a tensor-parallel group.
    fn all_reduce(&self, tensor: &mut Self::Tensor);

    /// Run all-to-all dispatch/combine between expert-parallel ranks.
    fn all_to_all(&self, send: &Self::Tensor, recv: &mut Self::Tensor);

    /// Run a pipeline-stage point-to-point send/recv exchange.
    fn send_recv(&self, stage: u32, tensor: &mut Self::Tensor);
}

/// Backend-internal sampling seam.
pub trait Sampler {
    /// Backend logits representation consumed by the sampler.
    type Logits;

    /// Sample one token per logits row using the provided per-row parameters.
    fn sample(&mut self, logits: &Self::Logits, params: &[SamplingParams]) -> Vec<u32>;
}

/// Backend-internal model architecture seam.
///
/// Model implementations use backend tensors and communicators below the
/// executor seam. The only engine-core input is the host-only [`ForwardPlan`]
/// plus a host-indexed [`KvPool`] handle.
pub trait ModelArch {
    /// Backend tensor type used for intermediate activations.
    type Tensor;

    /// Backend logits representation returned by the model.
    type Logits;

    /// Backend communicator implementation used by this model.
    type Comm: Communicator<Tensor = Self::Tensor>;

    /// Execute model forward for `plan` using backend-owned tensors.
    fn forward(
        &mut self,
        plan: &ForwardPlan,
        kv: &mut dyn KvPool,
        comm: &Self::Comm,
    ) -> anyhow::Result<Self::Logits>;
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
