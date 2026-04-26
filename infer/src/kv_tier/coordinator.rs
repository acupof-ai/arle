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
//!
//! # Submodule layout
//!
//! Flat-layout split (2026-04-26) — the coordinator outgrew a single file:
//! - [`types`]   — ticket / queue-stat / failure-class value types.
//! - [`control`] — atomic `QueueCounters` + the `live`/`cancelled` ticket sets.
//! - [`events`]  — request / command / event payloads on the channels.
//! - [`builder`] — `CoordinatorHandle`, `CoordinatorBuilder`, RAII region guard.
//! - [`tests`]   — all `#[cfg(test)]` integration tests.
//!
//! This file (the module root) keeps the [`Coordinator`] struct and its
//! handler loop (`handle_plan` / `handle_store` / `handle_fetch` /
//! `report_failure` / `run_once` / `spawn`).

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

use crate::types::BlockId;

use super::backend::ClusterSharedBackend;
use super::chunk::{KVBlock, KVHandle, LayerRange, TokenRange};
use super::io::{KVBackendCompletion, KVBackendFetch, KVBackendStore, KVPayload};
use super::tier::BlockLocation;
use super::transport::disk::{DiskBlockLocation, DiskStore};

#[path = "coordinator/builder.rs"]
pub mod builder;
#[path = "coordinator/control.rs"]
mod control;
#[path = "coordinator/events.rs"]
pub mod events;
#[path = "coordinator/types.rs"]
pub mod types;

pub use builder::{CoordinatorBuilder, CoordinatorHandle, NoOrchestrator, WithOrchestrator};
pub use events::{
    CoordinatorCommand, CoordinatorEvent, FetchRequest, FetchedBlock, OrchestratorEvent,
    PrefetchAction, PrefetchPlan, PrefetchPlanRequest, StoreRequest, StoreTarget,
};
pub use types::{
    CoordinatorQueueStats, FailureClass, FetchTicket, PlanTicket, QueueBackpressure,
    QueueControlStats, QueueKind, QueueTicket, StoreTicket,
};

use builder::AllocatedRegions;
use control::CoordinatorControl;
use types::QueueOutcome;

pub struct Coordinator {
    rx: Receiver<CoordinatorCommand>,
    events: Sender<CoordinatorEvent>,
    orchestrator_events: Option<Sender<OrchestratorEvent>>,
    disk_store: Option<Arc<DiskStore>>,
    cluster_shared_backend: Option<ClusterSharedBackend>,
    control: Arc<CoordinatorControl>,
}

impl Coordinator {
    /// Construct a `Coordinator` from parts assembled by `builder::build_inner`.
    /// Internal to the coordinator subtree — external callers must go through
    /// the builder.
    pub(in crate::kv_tier::coordinator) fn assemble(
        rx: Receiver<CoordinatorCommand>,
        events: Sender<CoordinatorEvent>,
        orchestrator_events: Option<Sender<OrchestratorEvent>>,
        disk_store: Option<Arc<DiskStore>>,
        cluster_shared_backend: Option<ClusterSharedBackend>,
        control: Arc<CoordinatorControl>,
    ) -> Self {
        Self {
            rx,
            events,
            orchestrator_events,
            disk_store,
            cluster_shared_backend,
            control,
        }
    }

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
                    let handle =
                        KVHandle::new(None, block.block_id, location.clone(), 0, payload_len);
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
        // Every report_failure inside this loop must `regions.release_now()`
        // first: report_failure sends on a bounded event channel, and a slow
        // receiver would otherwise keep the staged host-pool regions reserved
        // for the duration of the stall (relying on the implicit Drop is too
        // late). See codex review comment 2026-04-26.
        for block in blocks {
            if self.is_cancelled(queue_ticket) {
                regions.release_now();
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
                            regions.release_now();
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
                            regions.release_now();
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
                            regions.release_now();
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
                                regions.release_now();
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
                    regions.release_now();
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
            None,
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
                    CoordinatorCommand::Shutdown => None,
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
#[path = "coordinator/tests.rs"]
mod tests;
