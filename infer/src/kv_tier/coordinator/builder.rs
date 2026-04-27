//! Builder, public client handle, and the RAII region guard used by the
//! fetch happy path.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};

use crate::kv_tier::backend::ClusterSharedBackend;
use crate::kv_tier::transport::disk::DiskStore;

use super::Coordinator;
use super::control::CoordinatorControl;
use super::events::{
    CoordinatorCommand, CoordinatorEvent, FetchRequest, PrefetchPlanRequest, StoreRequest,
};
use super::types::{
    CoordinatorQueueStats, FetchTicket, PlanTicket, QueueBackpressure, QueueControlStats,
    QueueKind, QueueTicket, StoreTicket,
};

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

    /// **Reserved for future distributed-scheduler centralization.** Not
    /// called by the live single-scheduler CUDA runtime — see the doc on
    /// `Coordinator::handle_plan` for the rationale (the scheduler already
    /// has everything this method would surface, via `lookup_or_stage` +
    /// inline `TieredKvPolicy::allow_prefetch`). Production callers should
    /// not add a `submit_prefetch_plan → wait PlanCompleted → submit_fetch`
    /// round-trip today; tests cover the API contract for the M5+ use case.
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

/// RAII guard for host-pinned regions allocated mid-fetch that must be
/// released back to the pool unless ownership transfers to the caller via
/// the success path.
///
/// Replaces the prior pattern of calling a static `release_allocated_regions`
/// helper at every `return self.fetch_failed(...)` branch — easy to forget
/// when adding a new branch and the source of one historical leak audit.
/// The fetch happy-path explicitly calls [`Self::commit`] at the end, which
/// no-ops the destructor; every error path simply returns and Drop releases.
pub(super) struct AllocatedRegions {
    regions: Vec<(
        crate::kv_tier::host_pool::SharedHostPinnedPool,
        crate::kv_tier::host_pool::HostPinnedRegion,
    )>,
    committed: bool,
}

impl AllocatedRegions {
    pub(super) fn new() -> Self {
        Self {
            regions: Vec::new(),
            committed: false,
        }
    }

    pub(super) fn push(
        &mut self,
        pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
        region: crate::kv_tier::host_pool::HostPinnedRegion,
    ) {
        self.regions.push((pool, region));
    }

    /// Mark every accumulated region as ownership-transferred to the caller.
    /// After commit, `Drop` is a no-op.
    pub(super) fn commit(&mut self) {
        self.committed = true;
    }

    /// Eagerly release every accumulated region back to its pool, then mark
    /// the guard committed so `Drop` is a no-op. Idempotent.
    ///
    /// Use this on error paths that do potentially-blocking work *after*
    /// staging some regions — e.g. emitting on a bounded event channel whose
    /// receiver may be slow. Without `release_now`, the implicit `Drop` only
    /// fires after the blocked send returns, holding host-pinned-pool capacity
    /// reserved for the duration of the stall.
    pub(super) fn release_now(&mut self) {
        if self.committed {
            return;
        }
        self.drain_and_release();
        self.committed = true;
    }

    /// Drain the staged region vec and return each entry to its pool, logging
    /// (but otherwise ignoring) per-region release errors. Shared between
    /// [`Self::release_now`] and the `Drop` impl so the cleanup loop has one
    /// canonical body.
    fn drain_and_release(&mut self) {
        for (pool, region) in self.regions.drain(..) {
            if let Err(err) = pool.release_region(region) {
                log::warn!(
                    "coordinator AllocatedRegions release failed offset={} len={}: {}",
                    region.offset,
                    region.len,
                    err,
                );
            }
        }
    }
}

impl Drop for AllocatedRegions {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.drain_and_release();
    }
}

/// Fluent builder for [`Coordinator`]. Replaced the previous family of
/// telescoping constructors (`Coordinator::new`, `new_with_disk_store`,
/// `new_with_backends`) so that adding a new optional backend does not
/// multiply constructor surface.
pub struct CoordinatorBuilder {
    queue_capacity: usize,
    disk_store: Option<Arc<DiskStore>>,
    cluster_shared_backend: Option<ClusterSharedBackend>,
}

impl CoordinatorBuilder {
    /// New builder with the given bounded-channel capacity. Capacity is
    /// clamped to `>= 1` at `build` time.
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue_capacity,
            disk_store: None,
            cluster_shared_backend: None,
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

    /// Build a coordinator and its single unified event channel. All event
    /// shapes (Store*, Fetch*, Plan*) flow through this one stream.
    pub fn build(self) -> (Coordinator, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        let capacity = self.queue_capacity.max(1);
        let (tx, rx) = bounded(capacity);
        let (event_tx, event_rx) = bounded(capacity);
        let control = Arc::new(CoordinatorControl::new(capacity));
        (
            Coordinator::assemble(
                rx,
                event_tx,
                self.disk_store,
                self.cluster_shared_backend,
                control.clone(),
            ),
            CoordinatorHandle {
                tx,
                next_ticket: Arc::new(AtomicU64::new(1)),
                control,
            },
            event_rx,
        )
    }
}
