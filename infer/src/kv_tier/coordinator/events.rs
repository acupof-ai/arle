//! Command and event payloads exchanged between scheduler and coordinator.
//! Plain data — no I/O, no thread state.

use crate::kv_tier::tier::BlockLocation;
use crate::types::BlockId;

use super::types::{FailureClass, FetchTicket, PlanTicket, StoreTicket};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreTarget {
    Disk,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreRequest {
    pub block_id: BlockId,
    pub fingerprint: crate::types::BlockFingerprint,
    pub kv_format_tag: u8,
    pub host_pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
    pub host_region: crate::kv_tier::host_pool::HostPinnedRegion,
    pub target: StoreTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchPlanRequest {
    pub block_id: BlockId,
    pub source: Option<BlockLocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchAction {
    ReadyOnGpu,
    PromoteFromHost,
    FetchFromDisk,
    FetchFromRemote,
    Recompute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchPlan {
    pub block_id: BlockId,
    pub action: PrefetchAction,
}

/// Request handed to the coordinator for a T1/T2 → T0 prefetch preparation.
///
/// The coordinator always materializes the result into a host-pinned region so
/// the scheduler can run one canonical `host -> gpu` promote path afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRequest {
    pub block_id: BlockId,
    pub source: BlockLocation,
    pub byte_len: usize,
    pub host_pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedBlock {
    pub block_id: BlockId,
    pub host_region: crate::kv_tier::host_pool::HostPinnedRegion,
    pub byte_len: usize,
    /// True when the coordinator allocated the region for this fetch and the
    /// scheduler should release it after promotion. False when the block was
    /// already resident in T1 and the region is the canonical host location.
    pub release_after_promote: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorCommand {
    /// **Reserved for future distributed-scheduler centralization** — see
    /// [`super::super::Coordinator::handle_plan`] for the rationale. Not wired
    /// into the live single-scheduler readmission path. Tests cover the
    /// API contract so the surface stays buildable; production callers
    /// should NOT add a `submit_prefetch_plan → wait PlanCompleted →
    /// submit_fetch` round-trip today.
    Plan {
        ticket: PlanTicket,
        blocks: Vec<PrefetchPlanRequest>,
    },
    Store {
        ticket: StoreTicket,
        blocks: Vec<StoreRequest>,
    },
    /// Prepare staged blocks for local readmission. Host-pinned sources are
    /// reported back as-is; disk sources are fetched into temporary host
    /// regions first.
    Fetch {
        ticket: FetchTicket,
        blocks: Vec<FetchRequest>,
    },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorEvent {
    StoreQueued {
        ticket: StoreTicket,
        block_count: usize,
    },
    StoreCompleted {
        ticket: StoreTicket,
        locations: Vec<(BlockId, BlockLocation)>,
    },
    StoreFailed {
        ticket: StoreTicket,
        failed_block: BlockId,
        /// Typed cancel-vs-failure classification. Lets consumers distinguish
        /// cooperative cancellation from hard failure without parsing
        /// `reason`. See [`FailureClass`].
        class: FailureClass,
        reason: String,
    },
    FetchQueued {
        ticket: FetchTicket,
        block_count: usize,
    },
    FetchCompleted {
        ticket: FetchTicket,
        blocks: Vec<FetchedBlock>,
    },
    FetchFailed {
        ticket: FetchTicket,
        failed_block: BlockId,
        /// Typed cancel-vs-failure classification. See [`FailureClass`].
        class: FailureClass,
        reason: String,
    },
    /// Reserved for the M5+ distributed-scheduler use case. See the doc on
    /// [`CoordinatorCommand::Plan`] and on
    /// [`super::super::Coordinator::handle_plan`]. The single-scheduler
    /// runtime emits these only when tests drive `submit_prefetch_plan`
    /// directly.
    PlanQueued {
        ticket: PlanTicket,
        block_count: usize,
    },
    /// See [`CoordinatorEvent::PlanQueued`] — reserved API.
    PlanCompleted {
        ticket: PlanTicket,
        plans: Vec<PrefetchPlan>,
    },
    /// See [`CoordinatorEvent::PlanQueued`] — reserved API.
    PlanFailed {
        ticket: PlanTicket,
        failed_block: BlockId,
        /// Typed cancel-vs-failure classification. See [`FailureClass`].
        class: FailureClass,
        reason: String,
    },
}
