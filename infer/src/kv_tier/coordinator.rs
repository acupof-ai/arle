//! Tiered KV coordinator for the shipped local store/readmission path.
//!
//! The live local runtime uses the coordinator for three concrete behaviors:
//! - persist host-pinned blocks into `DiskStore`
//! - fetch staged T1/T2 blocks back into host-pinned regions
//! - expose explicit `plan/fetch/store` queue vocabulary for the next
//!   readmission tranche
//!
//! The scheduler never performs disk I/O directly on the read path. It builds
//! `ReadmissionPlan`s, submits `FetchRequest`s, waits in `Phase::WaitingFetch`,
//! and resumes once the coordinator reports `FetchCompleted`.

use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, bounded};

use crate::types::BlockId;

use super::backend::ClusterSharedBackend;
use super::chunk::{KVBlock, KVHandle, KVSpanId, LayerRange, TokenRange};
use super::io::{KVBackendCompletion, KVBackendFetch, KVBackendStore, KVPayload};
use super::tier::BlockLocation;
use super::transport::disk::{DiskBlockLocation, DiskStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FetchTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoreTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    Plan,
    Fetch,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueTicket {
    Plan(PlanTicket),
    Fetch(FetchTicket),
    Store(StoreTicket),
}

impl QueueTicket {
    pub fn kind(self) -> QueueKind {
        match self {
            Self::Plan(_) => QueueKind::Plan,
            Self::Fetch(_) => QueueKind::Fetch,
            Self::Store(_) => QueueKind::Store,
        }
    }
}

impl From<PlanTicket> for QueueTicket {
    fn from(value: PlanTicket) -> Self {
        Self::Plan(value)
    }
}

impl From<FetchTicket> for QueueTicket {
    fn from(value: FetchTicket) -> Self {
        Self::Fetch(value)
    }
}

impl From<StoreTicket> for QueueTicket {
    fn from(value: StoreTicket) -> Self {
        Self::Store(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueBackpressure {
    Normal,
    Elevated,
    Saturated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueControlStats {
    pub capacity: usize,
    pub queued: usize,
    pub in_flight: usize,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub rejected: u64,
}

impl QueueControlStats {
    pub fn active(&self) -> usize {
        self.queued.saturating_add(self.in_flight)
    }

    pub fn soft_saturated(&self, threshold: f64) -> bool {
        if self.capacity == 0 {
            return false;
        }
        (self.active() as f64 / self.capacity as f64) >= threshold
    }

    pub fn backpressure(&self) -> QueueBackpressure {
        pressure_from_load(self.active(), self.capacity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorQueueStats {
    pub capacity: usize,
    pub total_queued: usize,
    pub total_in_flight: usize,
    pub plan: QueueControlStats,
    pub fetch: QueueControlStats,
    pub store: QueueControlStats,
    /// Filled by scheduler post-snapshot via [`Self::with_fetch_waiters`];
    /// the coordinator itself does not track scheduler-side waiters.
    /// See `scheduler/cuda/core.rs`.
    pub fetch_waiters: usize,
}

impl CoordinatorQueueStats {
    pub fn active(&self) -> usize {
        self.total_queued.saturating_add(self.total_in_flight)
    }

    #[must_use]
    pub fn with_fetch_waiters(mut self, fetch_waiters: usize) -> Self {
        self.fetch_waiters = fetch_waiters;
        self
    }

    pub fn queue_capacity(&self) -> usize {
        self.capacity
    }

    pub fn fetch_queue_depth(&self) -> usize {
        self.fetch.active()
    }

    pub fn store_queue_depth(&self) -> usize {
        self.store.active()
    }

    pub fn fetch_backpressured(&self) -> bool {
        self.fetch.backpressure() != QueueBackpressure::Normal
    }

    pub fn store_backpressured(&self) -> bool {
        self.store.backpressure() != QueueBackpressure::Normal
    }

    pub fn backpressure(&self) -> QueueBackpressure {
        pressure_from_load(self.active(), self.capacity)
    }
}

fn pressure_from_load(active: usize, capacity: usize) -> QueueBackpressure {
    if active >= capacity {
        QueueBackpressure::Saturated
    } else if active.saturating_mul(4) >= capacity.saturating_mul(3) {
        QueueBackpressure::Elevated
    } else {
        QueueBackpressure::Normal
    }
}

#[derive(Debug)]
struct QueueCounters {
    queued: AtomicUsize,
    in_flight: AtomicUsize,
    submitted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    rejected: AtomicU64,
}

impl QueueCounters {
    const fn new() -> Self {
        Self {
            queued: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, capacity: usize) -> QueueControlStats {
        QueueControlStats {
            capacity,
            queued: self.queued.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct CoordinatorControl {
    capacity: usize,
    plan: QueueCounters,
    fetch: QueueCounters,
    store: QueueCounters,
    live: Mutex<HashSet<QueueTicket>>,
    cancelled: Mutex<HashSet<QueueTicket>>,
}

impl CoordinatorControl {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            plan: QueueCounters::new(),
            fetch: QueueCounters::new(),
            store: QueueCounters::new(),
            live: Mutex::new(HashSet::new()),
            cancelled: Mutex::new(HashSet::new()),
        }
    }

    fn queue(&self, kind: QueueKind) -> &QueueCounters {
        match kind {
            QueueKind::Plan => &self.plan,
            QueueKind::Fetch => &self.fetch,
            QueueKind::Store => &self.store,
        }
    }

    fn on_submit(&self, ticket: QueueTicket) {
        let queue = self.queue(ticket.kind());
        queue.submitted.fetch_add(1, Ordering::Relaxed);
        queue.queued.fetch_add(1, Ordering::Relaxed);
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.insert(ticket);
    }

    fn on_reject(&self, kind: QueueKind) {
        self.queue(kind).rejected.fetch_add(1, Ordering::Relaxed);
    }

    fn on_start(&self, ticket: QueueTicket) {
        let queue = self.queue(ticket.kind());
        queue.queued.fetch_sub(1, Ordering::Relaxed);
        queue.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    fn on_finish(&self, ticket: QueueTicket, outcome: QueueOutcome) {
        let queue = self.queue(ticket.kind());
        queue.in_flight.fetch_sub(1, Ordering::Relaxed);
        match outcome {
            QueueOutcome::Completed => {
                queue.completed.fetch_add(1, Ordering::Relaxed);
            }
            QueueOutcome::Failed => {
                queue.failed.fetch_add(1, Ordering::Relaxed);
            }
            QueueOutcome::Cancelled => {
                queue.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.remove(&ticket);
        let mut cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.remove(&ticket);
    }

    fn cancel(&self, ticket: QueueTicket) -> bool {
        let live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !live.contains(&ticket) {
            return false;
        }
        drop(live);
        let mut cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.insert(ticket)
    }

    fn is_cancelled(&self, ticket: QueueTicket) -> bool {
        let cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.contains(&ticket)
    }

    fn stats(&self) -> CoordinatorQueueStats {
        let plan = self.plan.snapshot(self.capacity);
        let fetch = self.fetch.snapshot(self.capacity);
        let store = self.store.snapshot(self.capacity);
        CoordinatorQueueStats {
            capacity: self.capacity,
            total_queued: plan
                .queued
                .saturating_add(fetch.queued)
                .saturating_add(store.queued),
            total_in_flight: plan
                .in_flight
                .saturating_add(fetch.in_flight)
                .saturating_add(store.in_flight),
            plan,
            fetch,
            store,
            fetch_waiters: 0,
        }
    }

    fn queue_stats(&self, kind: QueueKind) -> QueueControlStats {
        self.queue(kind).snapshot(self.capacity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueOutcome {
    Completed,
    Failed,
    Cancelled,
}

/// Typed classification of how a queue task failed. Each call site already
/// knows whether it is reporting a cooperative cancel (saw `is_cancelled`
/// return true) or a hard failure — pass the typed variant instead of
/// shipping the distinction through reason-string substring matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    Cancelled,
    Failed,
}

impl FailureClass {
    fn into_outcome(self) -> QueueOutcome {
        match self {
            Self::Cancelled => QueueOutcome::Cancelled,
            Self::Failed => QueueOutcome::Failed,
        }
    }
}

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
    CommandQueued(CoordinatorCommand),
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
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorEvent {
    PlanQueued {
        ticket: PlanTicket,
        block_count: usize,
    },
    PlanCompleted {
        ticket: PlanTicket,
        plans: Vec<PrefetchPlan>,
    },
    FetchQueued {
        ticket: FetchTicket,
        block_count: usize,
    },
    FetchCompleted {
        ticket: FetchTicket,
        blocks: Vec<FetchedBlock>,
    },
    StoreQueued {
        ticket: StoreTicket,
        block_count: usize,
    },
    StoreCompleted {
        ticket: StoreTicket,
        locations: Vec<(BlockId, BlockLocation)>,
    },
    TaskFailed {
        queue: QueueKind,
        ticket: QueueTicket,
        failed_block: BlockId,
        reason: String,
    },
}

#[derive(Clone)]
pub struct CoordinatorHandle {
    tx: Sender<CoordinatorCommand>,
    next_ticket: Arc<AtomicU64>,
    control: Arc<CoordinatorControl>,
}

impl CoordinatorHandle {
    pub fn send(&self, cmd: CoordinatorCommand) -> Result<()> {
        self.tx
            .send(cmd)
            .map_err(|e| anyhow!("coordinator send failed: {e}"))
    }

    /// Best-effort non-blocking `Shutdown` signal. Safe to call from a
    /// `Drop` path where a blocking `send` on a full bounded channel
    /// would deadlock. If the command channel is already full the
    /// signal is dropped; the caller should then rely on *channel
    /// disconnect* (dropping every `CoordinatorHandle` clone plus the
    /// events receiver) to terminate the coordinator thread.
    pub fn try_send_shutdown(&self) {
        let _ = self.tx.try_send(CoordinatorCommand::Shutdown);
    }

    pub fn stats(&self) -> CoordinatorQueueStats {
        self.control.stats()
    }

    pub fn queue_stats(&self, kind: QueueKind) -> QueueControlStats {
        self.control.queue_stats(kind)
    }

    pub fn backpressure(&self) -> QueueBackpressure {
        self.stats().backpressure()
    }

    pub fn cancel(&self, ticket: QueueTicket) -> bool {
        self.control.cancel(ticket)
    }

    pub fn cancel_plan(&self, ticket: PlanTicket) -> bool {
        self.cancel(ticket.into())
    }

    pub fn cancel_fetch(&self, ticket: FetchTicket) -> bool {
        self.cancel(ticket.into())
    }

    pub fn cancel_store(&self, ticket: StoreTicket) -> bool {
        self.cancel(ticket.into())
    }

    fn try_send_ticket(&self, ticket: QueueTicket, cmd: CoordinatorCommand) -> Result<()> {
        self.tx
            .try_send(cmd)
            .map(|()| {
                self.control.on_submit(ticket);
            })
            .map_err(|err| match err {
                TrySendError::Full(_) => {
                    self.control.on_reject(ticket.kind());
                    anyhow!("coordinator queue full")
                }
                TrySendError::Disconnected(_) => anyhow!("coordinator disconnected"),
            })
    }

    /// Mint a fresh `FetchTicket` and enqueue a non-blocking readmission fetch.
    pub fn submit_fetch(&self, blocks: Vec<FetchRequest>) -> Option<FetchTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = FetchTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send_ticket(ticket.into(), CoordinatorCommand::Fetch { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    pub fn submit_prefetch_plan(&self, blocks: Vec<PrefetchPlanRequest>) -> Option<PlanTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = PlanTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send_ticket(ticket.into(), CoordinatorCommand::Plan { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    pub fn submit_store(&self, blocks: Vec<StoreRequest>) -> Option<StoreTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = StoreTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send_ticket(ticket.into(), CoordinatorCommand::Store { ticket, blocks })
            .ok()?;
        Some(ticket)
    }
}

pub struct Coordinator {
    rx: Receiver<CoordinatorCommand>,
    events: Sender<CoordinatorEvent>,
    orchestrator_events: Option<Sender<OrchestratorEvent>>,
    disk_store: Option<Arc<DiskStore>>,
    cluster_shared_backend: Option<ClusterSharedBackend>,
    control: Arc<CoordinatorControl>,
}

/// RAII guard for host-pinned regions allocated mid-fetch that must be
/// released back to the pool unless ownership transfers to the caller via
/// the success path.
///
/// Replaces the prior pattern of calling a static `release_allocated_regions`
/// helper at every `return self.fetch_failed(...)` branch — easy to forget
/// when adding a new branch and the source of one historical leak audit.
/// The fetch happy-path explicitly calls [`Self::commit`] at the end, which
/// no-ops the destructor; every error path simply returns and Drop releases.
struct AllocatedRegions {
    regions: Vec<(
        crate::kv_tier::host_pool::SharedHostPinnedPool,
        crate::kv_tier::host_pool::HostPinnedRegion,
    )>,
    committed: bool,
}

impl AllocatedRegions {
    fn new() -> Self {
        Self {
            regions: Vec::new(),
            committed: false,
        }
    }

    fn push(
        &mut self,
        pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
        region: crate::kv_tier::host_pool::HostPinnedRegion,
    ) {
        self.regions.push((pool, region));
    }

    /// Mark every accumulated region as ownership-transferred to the caller.
    /// After commit, `Drop` is a no-op.
    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for AllocatedRegions {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for (pool, region) in &self.regions {
            if let Err(err) = pool.release_region(*region) {
                log::warn!(
                    "coordinator AllocatedRegions drop: release_region failed offset={} len={}: {}",
                    region.offset,
                    region.len,
                    err
                );
            }
        }
    }
}

/// Fluent builder for [`Coordinator`]. Replaced the previous family of
/// telescoping constructors (`Coordinator::new`, `new_with_disk_store`,
/// `new_with_backends`, `new_with_orchestrator_events`) so that adding a new
/// optional backend or channel does not multiply constructor surface.
pub struct CoordinatorBuilder {
    queue_capacity: usize,
    disk_store: Option<Arc<DiskStore>>,
    cluster_shared_backend: Option<ClusterSharedBackend>,
    with_orchestrator_events: bool,
}

impl CoordinatorBuilder {
    /// New builder with the given bounded-channel capacity. Capacity is
    /// clamped to `>= 1` at `build*` time.
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue_capacity,
            disk_store: None,
            cluster_shared_backend: None,
            with_orchestrator_events: false,
        }
    }

    /// Attach a node-local disk-tier store (T2). Required for any disk-backed
    /// store/fetch; absent → coordinator returns "disk store not configured".
    #[must_use]
    pub fn disk_store(mut self, store: Arc<DiskStore>) -> Self {
        self.disk_store = Some(store);
        self
    }

    /// Attach a cluster-shared backend (T3). Required for any remote-target
    /// store/fetch; absent → coordinator returns "remote store not configured".
    #[must_use]
    pub fn cluster_shared_backend(mut self, backend: ClusterSharedBackend) -> Self {
        self.cluster_shared_backend = Some(backend);
        self
    }

    /// Enable the secondary orchestrator-events channel. When set, `build` is
    /// not callable — use [`Self::build_with_orchestrator`] instead so the
    /// extra `Receiver<OrchestratorEvent>` is returned.
    #[must_use]
    pub fn with_orchestrator_events(mut self) -> Self {
        self.with_orchestrator_events = true;
        self
    }

    /// Build a coordinator without the orchestrator-events channel.
    /// Panics if [`Self::with_orchestrator_events`] was set; that path must
    /// use [`Self::build_with_orchestrator`].
    pub fn build(self) -> (Coordinator, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        assert!(
            !self.with_orchestrator_events,
            "CoordinatorBuilder::build called with orchestrator events enabled; use build_with_orchestrator",
        );
        let (coord, handle, events, _) = self.build_inner(false);
        (coord, handle, events)
    }

    /// Build a coordinator with both the legacy event channel and the
    /// orchestrator-events channel.
    pub fn build_with_orchestrator(
        mut self,
    ) -> (
        Coordinator,
        CoordinatorHandle,
        Receiver<CoordinatorEvent>,
        Receiver<OrchestratorEvent>,
    ) {
        self.with_orchestrator_events = true;
        let (coord, handle, events, orch) = self.build_inner(true);
        (
            coord,
            handle,
            events,
            orch.expect("orchestrator events requested"),
        )
    }

    fn build_inner(
        self,
        with_orchestrator: bool,
    ) -> (
        Coordinator,
        CoordinatorHandle,
        Receiver<CoordinatorEvent>,
        Option<Receiver<OrchestratorEvent>>,
    ) {
        let capacity = self.queue_capacity.max(1);
        let (tx, rx) = bounded(capacity);
        let (event_tx, event_rx) = bounded(capacity);
        let (orchestrator_tx, orchestrator_rx) = if with_orchestrator {
            let (tx, rx) = bounded::<OrchestratorEvent>(capacity);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let control = Arc::new(CoordinatorControl::new(capacity));
        (
            Coordinator {
                rx,
                events: event_tx,
                orchestrator_events: orchestrator_tx,
                disk_store: self.disk_store,
                cluster_shared_backend: self.cluster_shared_backend,
                control: control.clone(),
            },
            CoordinatorHandle {
                tx,
                next_ticket: Arc::new(AtomicU64::new(1)),
                control,
            },
            event_rx,
            orchestrator_rx,
        )
    }
}

impl Coordinator {
    fn emit_event(&self, event: CoordinatorEvent) -> Result<()> {
        self.events
            .send(event)
            .map_err(|e| anyhow!("coordinator event send failed: {e}"))
    }

    fn emit_orchestrator_event(&self, event: OrchestratorEvent) {
        if let Some(tx) = &self.orchestrator_events {
            let _ = tx.send(event);
        }
    }

    fn is_cancelled(&self, ticket: QueueTicket) -> bool {
        self.control.is_cancelled(ticket)
    }

    /// Typed classification of how a queued task ended in non-success.
    /// Replaces fragile substring matching on the failure reason string —
    /// each call site already knows whether it is reporting a cancel
    /// (cooperative `is_cancelled` check returned true) or a hard failure.
    fn report_failure(
        &self,
        ticket: QueueTicket,
        failed_block: BlockId,
        class: FailureClass,
        reason: impl Into<String>,
    ) -> Result<QueueOutcome> {
        let reason = reason.into();
        // Per-queue legacy `CoordinatorEvent` emission. Plan has no
        // `CoordinatorEvent::PlanFailed` variant — surfacing plan failures
        // is exclusively the orchestrator channel's job.
        match ticket {
            QueueTicket::Plan(_) => {}
            QueueTicket::Store(store_ticket) => {
                let _ = self.emit_event(CoordinatorEvent::StoreFailed {
                    ticket: store_ticket,
                    failed_block,
                    reason: reason.clone(),
                });
            }
            QueueTicket::Fetch(fetch_ticket) => {
                self.emit_event(CoordinatorEvent::FetchFailed {
                    ticket: fetch_ticket,
                    failed_block,
                    reason: reason.clone(),
                })?;
            }
        }
        self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
            queue: ticket.kind(),
            ticket,
            failed_block,
            reason,
        });
        Ok(class.into_outcome())
    }

    fn handle_plan(
        &self,
        ticket: PlanTicket,
        blocks: &[PrefetchPlanRequest],
    ) -> Result<QueueOutcome> {
        self.emit_orchestrator_event(OrchestratorEvent::PlanQueued {
            ticket,
            block_count: blocks.len(),
        });
        if self.is_cancelled(ticket.into()) {
            let failed_block = blocks.first().map_or(BlockId(0), |block| block.block_id);
            return self.report_failure(
                ticket.into(),
                failed_block,
                FailureClass::Cancelled,
                "plan cancelled",
            );
        }
        let plans = blocks
            .iter()
            .map(|block| PrefetchPlan {
                block_id: block.block_id,
                action: match block.source {
                    Some(BlockLocation::Gpu { .. }) => PrefetchAction::ReadyOnGpu,
                    Some(BlockLocation::HostPinned { .. }) => PrefetchAction::PromoteFromHost,
                    Some(BlockLocation::Disk { .. }) => PrefetchAction::FetchFromDisk,
                    Some(BlockLocation::Remote { .. }) => PrefetchAction::FetchFromRemote,
                    None => PrefetchAction::Recompute,
                },
            })
            .collect();
        self.emit_orchestrator_event(OrchestratorEvent::PlanCompleted { ticket, plans });
        Ok(QueueOutcome::Completed)
    }

    fn handle_store(&self, ticket: StoreTicket, blocks: &[StoreRequest]) -> Result<QueueOutcome> {
        let _ = self.emit_event(CoordinatorEvent::StoreQueued {
            ticket,
            block_count: blocks.len(),
        });
        self.emit_orchestrator_event(OrchestratorEvent::StoreQueued {
            ticket,
            block_count: blocks.len(),
        });
        let queue_ticket: QueueTicket = ticket.into();
        if self.is_cancelled(queue_ticket) {
            let failed_block = blocks.first().map_or(BlockId(0), |block| block.block_id);
            return self.report_failure(
                queue_ticket,
                failed_block,
                FailureClass::Cancelled,
                "store cancelled",
            );
        }

        let disk_store = self.disk_store.as_ref();
        let cluster_shared_backend = self.cluster_shared_backend.as_ref();

        let mut locations = Vec::with_capacity(blocks.len());
        for block in blocks {
            if self.is_cancelled(queue_ticket) {
                return self.report_failure(
                    queue_ticket,
                    block.block_id,
                    FailureClass::Cancelled,
                    "store cancelled",
                );
            }
            let payload = match block.host_pool.read_region(block.host_region) {
                Ok(payload) => payload,
                Err(err) => {
                    return self.report_failure(
                        queue_ticket,
                        block.block_id,
                        FailureClass::Failed,
                        err.to_string(),
                    );
                }
            };

            match block.target {
                StoreTarget::Disk => {
                    let Some(disk_store) = disk_store else {
                        return self.report_failure(
                            queue_ticket,
                            block.block_id,
                            FailureClass::Failed,
                            "coordinator disk store not configured",
                        );
                    };
                    match disk_store.put_block(block.fingerprint, block.kv_format_tag, &payload) {
                        Ok(location) => locations.push((
                            block.block_id,
                            BlockLocation::Disk {
                                fingerprint: location.fingerprint,
                                payload_len: location.payload_len,
                            },
                        )),
                        Err(err) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                err.to_string(),
                            );
                        }
                    }
                }
                StoreTarget::Remote => {
                    let Some(cluster_shared_backend) = cluster_shared_backend else {
                        return self.report_failure(
                            queue_ticket,
                            block.block_id,
                            FailureClass::Failed,
                            "coordinator remote store not configured",
                        );
                    };
                    let payload_len = payload.len() as u64;
                    let location = match cluster_shared_backend
                        .remote_location_for(block.fingerprint, payload_len)
                    {
                        Ok(location) => location,
                        Err(err) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                err.to_string(),
                            );
                        }
                    };
                    let handle = KVHandle::new(
                        KVSpanId(u64::from(block.block_id.0)),
                        block.block_id,
                        location.clone(),
                        0,
                        payload_len,
                    );
                    match cluster_shared_backend.exists(&handle) {
                        Ok(true) => {
                            locations.push((block.block_id, location));
                            continue;
                        }
                        Ok(false) => {}
                        Err(err) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                err.to_string(),
                            );
                        }
                    }
                    let block_meta = KVBlock::new(
                        block.block_id,
                        LayerRange::new(0, 0),
                        TokenRange::new(0, 0),
                        payload_len,
                    )
                    .with_fingerprint(block.fingerprint);
                    let mut op = match cluster_shared_backend.store(KVBackendStore {
                        handle: handle.clone(),
                        block: block_meta,
                        kv_format_tag: block.kv_format_tag,
                        payload: KVPayload::from_vec(payload),
                    }) {
                        Ok(op) => op,
                        Err(err) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                err.to_string(),
                            );
                        }
                    };
                    match cluster_shared_backend.poll(&mut op) {
                        std::task::Poll::Ready(Ok(KVBackendCompletion::Stored(handle))) => {
                            locations.push((block.block_id, handle.location));
                        }
                        std::task::Poll::Ready(Ok(other)) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                format!("unexpected remote store completion: {other:?}"),
                            );
                        }
                        std::task::Poll::Ready(Err(err)) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                err.to_string(),
                            );
                        }
                        std::task::Poll::Pending => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                "remote store unexpectedly returned Pending",
                            );
                        }
                    }
                }
            }
        }

        let _ = self.emit_event(CoordinatorEvent::StoreCompleted {
            ticket,
            locations: locations.clone(),
        });
        self.emit_orchestrator_event(OrchestratorEvent::StoreCompleted { ticket, locations });
        Ok(QueueOutcome::Completed)
    }

    fn handle_fetch(&self, ticket: FetchTicket, blocks: &[FetchRequest]) -> Result<QueueOutcome> {
        self.emit_event(CoordinatorEvent::FetchQueued {
            ticket,
            block_count: blocks.len(),
        })?;
        self.emit_orchestrator_event(OrchestratorEvent::FetchQueued {
            ticket,
            block_count: blocks.len(),
        });
        let queue_ticket: QueueTicket = ticket.into();
        if self.is_cancelled(queue_ticket) {
            let failed_block = blocks.first().map_or(BlockId(0), |block| block.block_id);
            return self.report_failure(
                queue_ticket,
                failed_block,
                FailureClass::Cancelled,
                "fetch cancelled",
            );
        }

        let mut fetched = Vec::with_capacity(blocks.len());
        // RAII: every region pushed here is released on early-return. The
        // happy path at the end of this function calls `regions.commit()`
        // before returning, which no-ops the destructor and transfers
        // ownership to the scheduler via `FetchedBlock.release_after_promote`.
        let mut regions = AllocatedRegions::new();
        for block in blocks {
            if self.is_cancelled(queue_ticket) {
                return self.report_failure(
                    queue_ticket,
                    block.block_id,
                    FailureClass::Cancelled,
                    "fetch cancelled",
                );
            }
            match &block.source {
                BlockLocation::HostPinned { offset } => {
                    fetched.push(FetchedBlock {
                        block_id: block.block_id,
                        host_region: crate::kv_tier::host_pool::HostPinnedRegion {
                            offset: *offset,
                            len: block.byte_len,
                        },
                        byte_len: block.byte_len,
                        release_after_promote: false,
                    });
                }
                BlockLocation::Disk {
                    fingerprint,
                    payload_len,
                } => {
                    let payload = match self.fetch_disk_payload(*fingerprint, *payload_len) {
                        Ok(payload) => payload,
                        Err(reason) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                reason,
                            );
                        }
                    };
                    let region = match Self::stage_into_host_pool(&block.host_pool, &payload) {
                        Ok(region) => region,
                        Err(reason) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                reason,
                            );
                        }
                    };
                    regions.push(block.host_pool.clone(), region);
                    fetched.push(FetchedBlock {
                        block_id: block.block_id,
                        host_region: region,
                        byte_len: payload.len(),
                        release_after_promote: true,
                    });
                }
                BlockLocation::Remote { .. } => {
                    let payload = match self.fetch_remote_payload(block) {
                        Ok(payload) => payload,
                        Err(reason) => {
                            return self.report_failure(
                                queue_ticket,
                                block.block_id,
                                FailureClass::Failed,
                                reason,
                            );
                        }
                    };
                    let region =
                        match Self::stage_into_host_pool(&block.host_pool, payload.as_slice()) {
                            Ok(region) => region,
                            Err(reason) => {
                                return self.report_failure(
                                    queue_ticket,
                                    block.block_id,
                                    FailureClass::Failed,
                                    reason,
                                );
                            }
                        };
                    regions.push(block.host_pool.clone(), region);
                    fetched.push(FetchedBlock {
                        block_id: block.block_id,
                        host_region: region,
                        byte_len: payload.len(),
                        release_after_promote: true,
                    });
                }
                BlockLocation::Gpu { .. } => {
                    return self.report_failure(
                        queue_ticket,
                        block.block_id,
                        FailureClass::Failed,
                        "gpu source should not go through fetch queue",
                    );
                }
            }
        }

        regions.commit();
        self.emit_orchestrator_event(OrchestratorEvent::FetchCompleted {
            ticket,
            blocks: fetched.clone(),
        });
        self.emit_event(CoordinatorEvent::FetchCompleted {
            ticket,
            blocks: fetched,
        })?;
        Ok(QueueOutcome::Completed)
    }

    /// Read a disk-backed block payload. `Err(String)` carries a
    /// human-readable reason to feed straight into `report_failure`.
    fn fetch_disk_payload(
        &self,
        fingerprint: crate::types::BlockFingerprint,
        payload_len: u64,
    ) -> std::result::Result<Vec<u8>, String> {
        let disk_store = self
            .disk_store
            .as_ref()
            .ok_or_else(|| "coordinator disk store not configured".to_string())?;
        let location = DiskBlockLocation {
            path: disk_store
                .block_path_for(fingerprint)
                .map_err(|e| e.to_string())?,
            payload_len,
            fingerprint,
        };
        disk_store
            .get_block(&location, Some(fingerprint))
            .map_err(|e| e.to_string())
    }

    /// Read a remote-backed block payload via the cluster-shared backend.
    fn fetch_remote_payload(&self, block: &FetchRequest) -> std::result::Result<KVPayload, String> {
        let backend = self
            .cluster_shared_backend
            .as_ref()
            .ok_or_else(|| "coordinator remote store not configured".to_string())?;
        let handle = KVHandle::new(
            KVSpanId(u64::from(block.block_id.0)),
            block.block_id,
            block.source.clone(),
            0,
            block.byte_len as u64,
        );
        let mut op = backend
            .fetch(KVBackendFetch { handle })
            .map_err(|e| e.to_string())?;
        match backend.poll(&mut op) {
            std::task::Poll::Ready(Ok(KVBackendCompletion::Loaded { payload, .. })) => Ok(payload),
            std::task::Poll::Ready(Ok(other)) => {
                Err(format!("unexpected remote fetch completion: {other:?}"))
            }
            std::task::Poll::Ready(Err(err)) => Err(err.to_string()),
            std::task::Poll::Pending => Err("remote fetch unexpectedly returned Pending".into()),
        }
    }

    /// Reserve a host-pinned region and copy `payload` into it. Returns the
    /// region on success; the caller is responsible for either pushing it
    /// into [`AllocatedRegions`] (so Drop releases on later failure) or
    /// keeping it owned.
    fn stage_into_host_pool(
        host_pool: &crate::kv_tier::host_pool::SharedHostPinnedPool,
        payload: &[u8],
    ) -> std::result::Result<crate::kv_tier::host_pool::HostPinnedRegion, String> {
        let mut pool = host_pool.lock().map_err(|e| e.to_string())?;
        let region = pool
            .reserve(payload.len())
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "host pinned pool exhausted".to_string())?;
        pool.as_mut_slice(region).copy_from_slice(payload);
        Ok(region)
    }

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                let queue_ticket = match &cmd {
                    CoordinatorCommand::Plan { ticket, .. } => Some((*ticket).into()),
                    CoordinatorCommand::Store { ticket, .. } => Some((*ticket).into()),
                    CoordinatorCommand::Fetch { ticket, .. } => Some((*ticket).into()),
                    CoordinatorCommand::Shutdown => None,
                };
                if let Some(ticket) = queue_ticket {
                    self.control.on_start(ticket);
                }
                let outcome = match &cmd {
                    CoordinatorCommand::Plan { ticket, blocks } => {
                        Some(self.handle_plan(*ticket, blocks)?)
                    }
                    CoordinatorCommand::Store { ticket, blocks } => {
                        Some(self.handle_store(*ticket, blocks)?)
                    }
                    CoordinatorCommand::Fetch { ticket, blocks } => {
                        Some(self.handle_fetch(*ticket, blocks)?)
                    }
                    CoordinatorCommand::Shutdown => {
                        self.events
                            .send(CoordinatorEvent::CommandQueued(cmd.clone()))
                            .map_err(|e| anyhow!("coordinator event send failed: {e}"))?;
                        None
                    }
                };
                if let (Some(ticket), Some(outcome)) = (queue_ticket, outcome) {
                    self.control.on_finish(ticket, outcome);
                }
                Ok(!matches!(cmd, CoordinatorCommand::Shutdown))
            }
            Err(RecvTimeoutError::Timeout) => Ok(true),
            Err(RecvTimeoutError::Disconnected) => Ok(false),
        }
    }

    pub fn spawn(self, name: &'static str) -> thread::JoinHandle<Result<()>> {
        thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while self.run_once()? {}
                Ok(())
            })
            .expect("coordinator thread spawn")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_tier::backend::ClusterSharedBackend;
    use crate::kv_tier::transport::shared_fs::{SharedFsBlockLocation, SharedFsStore};
    use crate::types::BlockFingerprint;
    use tempfile::tempdir;

    #[test]
    fn coordinator_receives_commands() {
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        handle.send(CoordinatorCommand::Shutdown).unwrap();
        assert!(!coordinator.run_once().unwrap());
        let evt = events.recv().unwrap();
        assert_eq!(
            evt,
            CoordinatorEvent::CommandQueued(CoordinatorCommand::Shutdown)
        );
    }

    #[test]
    fn coordinator_shutdown_joins_thread_cleanly() {
        let (coordinator, handle, _events) = CoordinatorBuilder::new(4).build();
        let join_handle = coordinator.spawn("infer-tiered-kv-coord-test");
        handle.send(CoordinatorCommand::Shutdown).unwrap();

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(join_handle.join());
        });

        let join_result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("coordinator join timed out");
        assert!(matches!(join_result, Ok(Ok(()))));
    }

    #[test]
    fn store_roundtrip_through_disk_store() {
        let dir = tempdir().unwrap();
        let disk_store = Arc::new(DiskStore::new(dir.path()));
        let (coordinator, handle, events) = CoordinatorBuilder::new(4)
            .disk_store(disk_store.clone())
            .build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(256).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(6).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"abcdef");
            region
        };
        let fingerprint = BlockFingerprint([0x2A; 16]);
        let ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(7),
                fingerprint,
                kv_format_tag: 3,
                host_pool: host_pool.clone(),
                host_region: region,
                target: StoreTarget::Disk,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StoreQueued {
                ticket,
                block_count: 1,
            }
        );
        let location = match events.recv().unwrap() {
            CoordinatorEvent::StoreCompleted {
                ticket: done,
                locations,
            } => {
                assert_eq!(done, ticket);
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].0, BlockId(7));
                match locations[0].1.clone() {
                    BlockLocation::Disk {
                        fingerprint,
                        payload_len,
                    } => crate::kv_tier::transport::disk::DiskBlockLocation {
                        path: disk_store.block_path_for(fingerprint).unwrap(),
                        fingerprint,
                        payload_len,
                    },
                    other => panic!("expected disk location, got {other:?}"),
                }
            }
            other => panic!("unexpected store event: {other:?}"),
        };

        let payload = host_pool.read_region(region).unwrap();
        assert_eq!(payload, b"abcdef");
        let reloaded = disk_store.get_block(&location, Some(fingerprint)).unwrap();
        assert_eq!(reloaded, b"abcdef");
    }

    #[test]
    fn store_to_disk_fails_without_configured_disk_store() {
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(3),
                fingerprint: BlockFingerprint([0x44; 16]),
                kv_format_tag: 1,
                host_pool,
                host_region: region,
                target: StoreTarget::Disk,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StoreQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::StoreFailed {
                ticket: failed_ticket,
                failed_block,
                reason,
            } => {
                assert_eq!(failed_ticket, ticket);
                assert_eq!(failed_block, BlockId(3));
                assert!(reason.contains("disk store not configured"));
            }
            other => panic!("unexpected store failure event: {other:?}"),
        }
    }

    #[test]
    fn fetch_from_host_passthrough_keeps_existing_region() {
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(128).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(5).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"hello");
            region
        };

        let ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(9),
                source: BlockLocation::HostPinned {
                    offset: region.offset,
                },
                byte_len: region.len,
                host_pool: host_pool.clone(),
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::FetchQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::FetchCompleted {
                ticket: done,
                blocks,
            } => {
                assert_eq!(done, ticket);
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].block_id, BlockId(9));
                assert_eq!(blocks[0].host_region, region);
                assert_eq!(blocks[0].byte_len, region.len);
                assert!(!blocks[0].release_after_promote);
                assert_eq!(
                    host_pool.read_region(blocks[0].host_region).unwrap(),
                    b"hello"
                );
            }
            other => panic!("unexpected fetch event: {other:?}"),
        }
    }

    #[test]
    fn fetch_from_disk_materializes_temp_host_region() {
        let dir = tempdir().unwrap();
        let disk_store = Arc::new(DiskStore::new(dir.path()));
        let (coordinator, handle, events) = CoordinatorBuilder::new(4)
            .disk_store(disk_store.clone())
            .build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(256).unwrap(),
        );
        let fingerprint = BlockFingerprint([0x7A; 16]);
        let location = disk_store.put_block(fingerprint, 5, b"disk-bytes").unwrap();

        let ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(11),
                source: BlockLocation::Disk {
                    fingerprint,
                    payload_len: location.payload_len,
                },
                byte_len: usize::try_from(location.payload_len).unwrap(),
                host_pool: host_pool.clone(),
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::FetchQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::FetchCompleted {
                ticket: done,
                blocks,
            } => {
                assert_eq!(done, ticket);
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].block_id, BlockId(11));
                assert!(blocks[0].release_after_promote);
                assert_eq!(
                    host_pool.read_region(blocks[0].host_region).unwrap(),
                    b"disk-bytes"
                );
                host_pool.release_region(blocks[0].host_region).unwrap();
            }
            other => panic!("unexpected fetch event: {other:?}"),
        }
    }

    #[test]
    fn fetch_fails_for_gpu_source() {
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(13),
                source: BlockLocation::Gpu { slot: 0 },
                byte_len: 16,
                host_pool,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::FetchQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::FetchFailed {
                ticket: failed_ticket,
                failed_block,
                reason,
            } => {
                assert_eq!(failed_ticket, ticket);
                assert_eq!(failed_block, BlockId(13));
                assert!(reason.contains("gpu source"));
            }
            other => panic!("unexpected fetch failure event: {other:?}"),
        }
    }

    #[test]
    fn submit_store_is_non_blocking_when_queue_is_full() {
        let (_coordinator, handle, _events) = CoordinatorBuilder::new(1).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let first = handle.submit_store(vec![StoreRequest {
            block_id: BlockId(1),
            fingerprint: BlockFingerprint([0x11; 16]),
            kv_format_tag: 1,
            host_pool: host_pool.clone(),
            host_region: region,
            target: StoreTarget::Disk,
        }]);
        assert!(first.is_some());

        let second = handle.submit_store(vec![StoreRequest {
            block_id: BlockId(2),
            fingerprint: BlockFingerprint([0x22; 16]),
            kv_format_tag: 1,
            host_pool,
            host_region: region,
            target: StoreTarget::Disk,
        }]);
        assert!(second.is_none(), "full queue should not block store submit");
    }

    #[test]
    fn queue_stats_report_backpressure_and_rejections() {
        let (_coordinator, handle, _events) = CoordinatorBuilder::new(1).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let first = handle.submit_fetch(vec![FetchRequest {
            block_id: BlockId(21),
            source: BlockLocation::HostPinned {
                offset: region.offset,
            },
            byte_len: region.len,
            host_pool: host_pool.clone(),
        }]);
        assert!(first.is_some());
        let stats = handle.stats();
        assert_eq!(stats.fetch.queued, 1);
        assert_eq!(stats.total_queued, 1);
        assert_eq!(stats.backpressure(), QueueBackpressure::Saturated);
        assert_eq!(
            handle.queue_stats(QueueKind::Fetch).backpressure(),
            QueueBackpressure::Saturated
        );

        let second = handle.submit_fetch(vec![FetchRequest {
            block_id: BlockId(22),
            source: BlockLocation::HostPinned {
                offset: region.offset,
            },
            byte_len: region.len,
            host_pool,
        }]);
        assert!(second.is_none());
        let stats = handle.stats();
        assert_eq!(stats.fetch.submitted, 1);
        assert_eq!(stats.fetch.rejected, 1);
    }

    #[test]
    fn cancelled_fetch_updates_stats_and_reports_cancel_reason() {
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(128).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(5).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"hello");
            region
        };

        let ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(23),
                source: BlockLocation::HostPinned {
                    offset: region.offset,
                },
                byte_len: region.len,
                host_pool,
            }])
            .unwrap();
        assert!(handle.cancel_fetch(ticket));

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::FetchQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::FetchFailed {
                ticket: failed_ticket,
                failed_block,
                reason,
            } => {
                assert_eq!(failed_ticket, ticket);
                assert_eq!(failed_block, BlockId(23));
                assert!(reason.contains("cancelled"));
            }
            other => panic!("unexpected fetch cancel event: {other:?}"),
        }

        let stats = handle.stats();
        assert_eq!(stats.fetch.cancelled, 1);
        assert_eq!(stats.fetch.failed, 0);
        assert_eq!(stats.fetch.completed, 0);
        assert_eq!(stats.fetch.queued, 0);
        assert_eq!(stats.fetch.in_flight, 0);
        assert!(!handle.cancel_fetch(ticket));
    }

    #[test]
    fn orchestrator_plan_classifies_tiers_without_touching_legacy_events() {
        let (coordinator, handle, _events, orchestrator_events) =
            CoordinatorBuilder::new(4).build_with_orchestrator();
        let ticket = handle
            .submit_prefetch_plan(vec![
                PrefetchPlanRequest {
                    block_id: BlockId(1),
                    source: Some(BlockLocation::Gpu { slot: 0 }),
                },
                PrefetchPlanRequest {
                    block_id: BlockId(2),
                    source: Some(BlockLocation::HostPinned { offset: 1024 }),
                },
                PrefetchPlanRequest {
                    block_id: BlockId(3),
                    source: Some(BlockLocation::Disk {
                        fingerprint: BlockFingerprint([0x33; 16]),
                        payload_len: 4096,
                    }),
                },
                PrefetchPlanRequest {
                    block_id: BlockId(4),
                    source: Some(BlockLocation::Remote {
                        desc: crate::kv_tier::RemoteBlockDesc {
                            transport: crate::kv_tier::TransportId::Nixl,
                            payload: vec![1, 2, 3],
                        },
                    }),
                },
                PrefetchPlanRequest {
                    block_id: BlockId(5),
                    source: None,
                },
            ])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            orchestrator_events.recv().unwrap(),
            OrchestratorEvent::PlanQueued {
                ticket,
                block_count: 5,
            }
        );
        match orchestrator_events.recv().unwrap() {
            OrchestratorEvent::PlanCompleted {
                ticket: done,
                plans,
            } => {
                assert_eq!(done, ticket);
                assert_eq!(
                    plans,
                    vec![
                        PrefetchPlan {
                            block_id: BlockId(1),
                            action: PrefetchAction::ReadyOnGpu,
                        },
                        PrefetchPlan {
                            block_id: BlockId(2),
                            action: PrefetchAction::PromoteFromHost,
                        },
                        PrefetchPlan {
                            block_id: BlockId(3),
                            action: PrefetchAction::FetchFromDisk,
                        },
                        PrefetchPlan {
                            block_id: BlockId(4),
                            action: PrefetchAction::FetchFromRemote,
                        },
                        PrefetchPlan {
                            block_id: BlockId(5),
                            action: PrefetchAction::Recompute,
                        },
                    ]
                );
            }
            other => panic!("unexpected orchestrator event: {other:?}"),
        }
    }

    #[test]
    fn orchestrator_store_reports_remote_stub_failure() {
        let (coordinator, handle, _events, orchestrator_events) =
            CoordinatorBuilder::new(4).build_with_orchestrator();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(7),
                fingerprint: BlockFingerprint([0x77; 16]),
                kv_format_tag: 1,
                host_pool,
                host_region: region,
                target: StoreTarget::Remote,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            orchestrator_events.recv().unwrap(),
            OrchestratorEvent::StoreQueued {
                ticket,
                block_count: 1,
            }
        );
        match orchestrator_events.recv().unwrap() {
            OrchestratorEvent::TaskFailed {
                queue,
                ticket: failed_ticket,
                failed_block,
                reason,
            } => {
                assert_eq!(queue, QueueKind::Store);
                assert_eq!(failed_ticket, QueueTicket::Store(ticket));
                assert_eq!(failed_block, BlockId(7));
                assert!(reason.contains("remote store not configured"));
            }
            other => panic!("unexpected orchestrator event: {other:?}"),
        }
    }

    #[test]
    fn remote_store_and_fetch_roundtrip_through_shared_fs_backend() {
        let dir = tempdir().unwrap();
        let remote_store = Arc::new(SharedFsStore::new(dir.path()));
        let (coordinator, handle, events) = CoordinatorBuilder::new(4)
            .cluster_shared_backend(ClusterSharedBackend::SharedFs(remote_store.clone()))
            .build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(256).unwrap(),
        );
        let fingerprint = BlockFingerprint([0x31; 16]);
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(12).unwrap().unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"remote-bytes");
            region
        };

        let store_ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(31),
                fingerprint,
                kv_format_tag: 2,
                host_pool: host_pool.clone(),
                host_region: region,
                target: StoreTarget::Remote,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StoreQueued {
                ticket: store_ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::StoreCompleted { ticket, locations } => {
                assert_eq!(ticket, store_ticket);
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].0, BlockId(31));
            }
            other => panic!("unexpected store event: {other:?}"),
        }
        let stored = remote_store
            .get_block(
                SharedFsBlockLocation::new(fingerprint, 12),
                Some(fingerprint),
            )
            .unwrap();
        assert_eq!(stored, b"remote-bytes");
        let second_store_ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(31),
                fingerprint,
                kv_format_tag: 2,
                host_pool: host_pool.clone(),
                host_region: region,
                target: StoreTarget::Remote,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StoreQueued {
                ticket: second_store_ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::StoreCompleted { ticket, locations } => {
                assert_eq!(ticket, second_store_ticket);
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].0, BlockId(31));
                assert_eq!(
                    locations[0].1,
                    SharedFsBlockLocation::new(fingerprint, 12)
                        .into_block_location()
                        .unwrap()
                );
            }
            other => panic!("unexpected second store event: {other:?}"),
        }
        let remote_location = SharedFsBlockLocation::new(fingerprint, 12)
            .into_block_location()
            .unwrap();
        assert_eq!(store_ticket.0, 1);

        let fetch_ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(31),
                source: remote_location,
                byte_len: 12,
                host_pool: host_pool.clone(),
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::FetchQueued {
                ticket: fetch_ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::FetchCompleted { ticket, blocks } => {
                assert_eq!(ticket, fetch_ticket);
                assert_eq!(blocks.len(), 1);
                assert!(blocks[0].release_after_promote);
                assert_eq!(
                    host_pool.read_region(blocks[0].host_region).unwrap(),
                    b"remote-bytes"
                );
                host_pool.release_region(blocks[0].host_region).unwrap();
            }
            other => panic!("unexpected remote fetch event: {other:?}"),
        }
    }

    #[test]
    fn coordinator_builder_constructs_with_no_backends() {
        // Smoke test: the new builder reaches a working coordinator with
        // zero optional backends configured (replaces the implicit shape
        // assertion that lived inside the old `Coordinator::new` path).
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        assert_eq!(handle.stats().capacity, 4);
        handle.send(CoordinatorCommand::Shutdown).unwrap();
        assert!(!coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::CommandQueued(CoordinatorCommand::Shutdown),
        );
    }

    #[test]
    fn report_failure_classifies_cancel_vs_fail() {
        // Drive the typed `FailureClass` end-to-end and verify both legs
        // surface their queue-specific event shape; this is the main
        // protection against silent drift back to substring matching.
        let (coordinator, handle, events) = CoordinatorBuilder::new(4).build();
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );

        // Cancel path.
        let ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(101),
                source: BlockLocation::HostPinned { offset: 0 },
                byte_len: 4,
                host_pool: host_pool.clone(),
            }])
            .unwrap();
        assert!(handle.cancel_fetch(ticket));
        assert!(coordinator.run_once().unwrap());
        let _queued = events.recv().unwrap();
        match events.recv().unwrap() {
            CoordinatorEvent::FetchFailed { reason, .. } => {
                assert!(reason.contains("cancelled"));
            }
            other => panic!("expected FetchFailed cancel, got {other:?}"),
        }
        assert_eq!(handle.stats().fetch.cancelled, 1);
        assert_eq!(handle.stats().fetch.failed, 0);

        // Hard-fail path (no remote backend configured for a remote fetch).
        let ticket2 = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(102),
                source: BlockLocation::Remote {
                    desc: crate::kv_tier::RemoteBlockDesc {
                        transport: crate::kv_tier::TransportId::Nixl,
                        payload: vec![0],
                    },
                },
                byte_len: 4,
                host_pool,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        let _queued2 = events.recv().unwrap();
        match events.recv().unwrap() {
            CoordinatorEvent::FetchFailed { reason, .. } => {
                assert!(reason.contains("remote store not configured"));
                assert!(!reason.contains("cancelled"));
            }
            other => panic!("expected FetchFailed hard error, got {other:?}"),
        }
        let stats = handle.stats();
        assert_eq!(stats.fetch.cancelled, 1, "cancel count untouched");
        assert_eq!(stats.fetch.failed, 1, "hard fail recorded");
        let _ = ticket2;
    }

    #[test]
    fn allocated_regions_releases_on_drop_when_uncommitted() {
        // Reserve a region, hand it to AllocatedRegions, drop without
        // committing — the region must be released and the pool ready
        // to reserve the same span again.
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            pool.reserve(32).unwrap().unwrap()
        };
        {
            let mut regions = AllocatedRegions::new();
            regions.push(host_pool.clone(), region);
            // Dropped here without commit.
        }
        // After drop, the pool should accept another full reservation.
        let mut pool = host_pool.lock().unwrap();
        let r2 = pool
            .reserve(32)
            .unwrap()
            .expect("pool should have free space after drop release");
        // Release for cleanliness (region tracker may otherwise leak in
        // the test-scoped pool).
        pool.release(r2).unwrap();
    }

    #[test]
    fn allocated_regions_no_release_when_committed() {
        // Commit before drop → region stays reserved; a follow-up full
        // reserve must fail because no slot is free.
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(32).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            pool.reserve(32).unwrap().unwrap()
        };
        {
            let mut regions = AllocatedRegions::new();
            regions.push(host_pool.clone(), region);
            regions.commit();
        }
        // Pool is full; a new full reservation must be None (or the
        // reserve returns Err if the implementation rejects oversize
        // requests up front). Both indicate the committed region was
        // not released.
        let mut pool = host_pool.lock().unwrap();
        let outcome = pool.reserve(32).unwrap();
        assert!(
            outcome.is_none(),
            "committed region should NOT have been released",
        );
        // Cleanup — release the original region so the pool is tidy.
        pool.release(region).unwrap();
    }
}
