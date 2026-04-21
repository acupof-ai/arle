//! Tiered KV coordinator for the shipped local spill/readmission path.
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

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, bounded};

use crate::types::BlockId;

use super::tier::BlockLocation;
use super::transport::disk::{DiskBlockLocation, DiskStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FetchTicket(pub u64);

pub type StoreTicket = SpillTicket;

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

/// Request handed to the coordinator for a T1 → T2 spill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillRequest {
    pub block_id: BlockId,
    pub fingerprint: crate::types::BlockFingerprint,
    pub kv_format_tag: u8,
    pub host_pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
    pub host_region: crate::kv_tier::host_pool::HostPinnedRegion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl From<SpillRequest> for StoreRequest {
    fn from(request: SpillRequest) -> Self {
        Self {
            block_id: request.block_id,
            fingerprint: request.fingerprint,
            kv_format_tag: request.kv_format_tag,
            host_pool: request.host_pool,
            host_region: request.host_region,
            target: StoreTarget::Disk,
        }
    }
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
    /// T1 → T2 spill ticket. The coordinator routes each `SpillRequest`
    /// through `DiskStore::put_block` and emits `SpillCompleted` with the
    /// resulting disk locations on success.
    Spill {
        ticket: SpillTicket,
        blocks: Vec<SpillRequest>,
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
    /// T1 → T2 spill enqueued. Informational — the real work happens
    /// asynchronously on the coordinator's disk I/O thread / worker pool.
    SpillQueued {
        ticket: SpillTicket,
        block_count: usize,
    },
    /// T1 → T2 spill finished. `locations` gives the scheduler the
    /// canonical disk locations for every block that was persisted so
    /// the radix can flip `tier_location` to `BlockLocation::Disk`.
    SpillCompleted {
        ticket: SpillTicket,
        locations: Vec<(BlockId, DiskBlockLocation)>,
    },
    /// T1 → T2 spill failed. `failed_block` identifies the first block
    /// that did not persist successfully; any preceding blocks in the
    /// ticket were persisted, so the scheduler has to decide whether
    /// to roll them back or accept a partial spill.
    SpillFailed {
        ticket: SpillTicket,
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

    fn try_send(&self, cmd: CoordinatorCommand) -> Result<()> {
        self.tx.try_send(cmd).map_err(|err| match err {
            TrySendError::Full(_) => anyhow!("coordinator queue full"),
            TrySendError::Disconnected(_) => anyhow!("coordinator disconnected"),
        })
    }

    /// Mint a fresh `SpillTicket` for a T1 → T2 spill batch and enqueue
    /// the corresponding `Spill` command. Returns `None` if `blocks`
    /// is empty or the command channel is full.
    pub fn submit_spill(&self, blocks: Vec<SpillRequest>) -> Option<SpillTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = SpillTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Spill { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    /// Mint a fresh `FetchTicket` and enqueue a non-blocking readmission fetch.
    pub fn submit_fetch(&self, blocks: Vec<FetchRequest>) -> Option<FetchTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = FetchTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Fetch { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    pub fn submit_prefetch_plan(&self, blocks: Vec<PrefetchPlanRequest>) -> Option<PlanTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = PlanTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Plan { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    pub fn submit_store(&self, blocks: Vec<StoreRequest>) -> Option<StoreTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = SpillTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Store { ticket, blocks })
            .ok()?;
        Some(ticket)
    }
}

pub struct Coordinator {
    rx: Receiver<CoordinatorCommand>,
    events: Sender<CoordinatorEvent>,
    orchestrator_events: Option<Sender<OrchestratorEvent>>,
    disk_store: Option<Arc<DiskStore>>,
}

impl Coordinator {
    pub fn new(queue_capacity: usize) -> (Self, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        Self::new_with_optional_disk_store(queue_capacity, None)
    }

    pub fn new_with_disk_store(
        queue_capacity: usize,
        disk_store: Arc<DiskStore>,
    ) -> (Self, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        Self::new_with_optional_disk_store(queue_capacity, Some(disk_store))
    }

    pub fn new_with_orchestrator_events(
        queue_capacity: usize,
        disk_store: Option<Arc<DiskStore>>,
    ) -> (
        Self,
        CoordinatorHandle,
        Receiver<CoordinatorEvent>,
        Receiver<OrchestratorEvent>,
    ) {
        let (tx, rx) = bounded(queue_capacity.max(1));
        let (event_tx, event_rx) = bounded(queue_capacity.max(1));
        let (orchestrator_tx, orchestrator_rx) = bounded(queue_capacity.max(1));
        (
            Self {
                rx,
                events: event_tx,
                orchestrator_events: Some(orchestrator_tx),
                disk_store,
            },
            CoordinatorHandle {
                tx,
                next_ticket: Arc::new(AtomicU64::new(1)),
            },
            event_rx,
            orchestrator_rx,
        )
    }

    fn new_with_optional_disk_store(
        queue_capacity: usize,
        disk_store: Option<Arc<DiskStore>>,
    ) -> (Self, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        let (tx, rx) = bounded(queue_capacity.max(1));
        let (event_tx, event_rx) = bounded(queue_capacity.max(1));
        (
            Self {
                rx,
                events: event_tx,
                orchestrator_events: None,
                disk_store,
            },
            CoordinatorHandle {
                tx,
                next_ticket: Arc::new(AtomicU64::new(1)),
            },
            event_rx,
        )
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

    fn handle_plan(&self, ticket: PlanTicket, blocks: &[PrefetchPlanRequest]) {
        self.emit_orchestrator_event(OrchestratorEvent::PlanQueued {
            ticket,
            block_count: blocks.len(),
        });
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
    }

    fn handle_spill(&self, ticket: SpillTicket, blocks: &[SpillRequest]) -> Result<()> {
        self.emit_event(CoordinatorEvent::SpillQueued {
            ticket,
            block_count: blocks.len(),
        })?;

        let Some(disk_store) = &self.disk_store else {
            if let Some(first) = blocks.first() {
                self.emit_event(CoordinatorEvent::SpillFailed {
                    ticket,
                    failed_block: first.block_id,
                    reason: "coordinator disk store not configured".to_string(),
                })?;
            } else {
                self.emit_event(CoordinatorEvent::SpillCompleted {
                    ticket,
                    locations: Vec::new(),
                })?;
            }
            return Ok(());
        };

        let mut locations = Vec::with_capacity(blocks.len());
        for block in blocks {
            let payload = match block.host_pool.read_region(block.host_region) {
                Ok(payload) => payload,
                Err(err) => {
                    self.emit_event(CoordinatorEvent::SpillFailed {
                        ticket,
                        failed_block: block.block_id,
                        reason: err.to_string(),
                    })?;
                    return Ok(());
                }
            };

            match disk_store.put_block(block.fingerprint, block.kv_format_tag, &payload) {
                Ok(location) => locations.push((block.block_id, location)),
                Err(err) => {
                    self.emit_event(CoordinatorEvent::SpillFailed {
                        ticket,
                        failed_block: block.block_id,
                        reason: err.to_string(),
                    })?;
                    return Ok(());
                }
            }
        }

        self.emit_event(CoordinatorEvent::SpillCompleted { ticket, locations })
    }

    fn handle_store(&self, ticket: StoreTicket, blocks: &[StoreRequest]) {
        self.emit_orchestrator_event(OrchestratorEvent::StoreQueued {
            ticket,
            block_count: blocks.len(),
        });

        let disk_store = self.disk_store.as_ref();

        let mut locations = Vec::with_capacity(blocks.len());
        for block in blocks {
            let payload = match block.host_pool.read_region(block.host_region) {
                Ok(payload) => payload,
                Err(err) => {
                    self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                        queue: QueueKind::Store,
                        ticket: QueueTicket::Store(ticket),
                        failed_block: block.block_id,
                        reason: err.to_string(),
                    });
                    return;
                }
            };

            match block.target {
                StoreTarget::Disk => {
                    let Some(disk_store) = disk_store else {
                        self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                            queue: QueueKind::Store,
                            ticket: QueueTicket::Store(ticket),
                            failed_block: block.block_id,
                            reason: "coordinator disk store not configured".to_string(),
                        });
                        return;
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
                            self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                                queue: QueueKind::Store,
                                ticket: QueueTicket::Store(ticket),
                                failed_block: block.block_id,
                                reason: err.to_string(),
                            });
                            return;
                        }
                    }
                }
                StoreTarget::Remote => {
                    self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                        queue: QueueKind::Store,
                        ticket: QueueTicket::Store(ticket),
                        failed_block: block.block_id,
                        reason: "remote store not implemented".to_string(),
                    });
                    return;
                }
            }
        }

        self.emit_orchestrator_event(OrchestratorEvent::StoreCompleted { ticket, locations });
    }

    fn handle_fetch(&self, ticket: FetchTicket, blocks: &[FetchRequest]) -> Result<()> {
        self.emit_event(CoordinatorEvent::FetchQueued {
            ticket,
            block_count: blocks.len(),
        })?;
        self.emit_orchestrator_event(OrchestratorEvent::FetchQueued {
            ticket,
            block_count: blocks.len(),
        });

        let mut fetched = Vec::with_capacity(blocks.len());
        for block in blocks {
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
                    let Some(disk_store) = &self.disk_store else {
                        self.emit_event(CoordinatorEvent::FetchFailed {
                            ticket,
                            failed_block: block.block_id,
                            reason: "coordinator disk store not configured".to_string(),
                        })?;
                        self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                            queue: QueueKind::Fetch,
                            ticket: QueueTicket::Fetch(ticket),
                            failed_block: block.block_id,
                            reason: "coordinator disk store not configured".to_string(),
                        });
                        return Ok(());
                    };
                    let location = DiskBlockLocation {
                        path: match disk_store.block_path_for(*fingerprint) {
                            Ok(path) => path,
                            Err(err) => {
                                self.emit_event(CoordinatorEvent::FetchFailed {
                                    ticket,
                                    failed_block: block.block_id,
                                    reason: err.to_string(),
                                })?;
                                self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                                    queue: QueueKind::Fetch,
                                    ticket: QueueTicket::Fetch(ticket),
                                    failed_block: block.block_id,
                                    reason: err.to_string(),
                                });
                                return Ok(());
                            }
                        },
                        payload_len: *payload_len,
                        fingerprint: *fingerprint,
                    };
                    let payload = match disk_store.get_block(&location, Some(*fingerprint)) {
                        Ok(payload) => payload,
                        Err(err) => {
                            self.emit_event(CoordinatorEvent::FetchFailed {
                                ticket,
                                failed_block: block.block_id,
                                reason: err.to_string(),
                            })?;
                            self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                                queue: QueueKind::Fetch,
                                ticket: QueueTicket::Fetch(ticket),
                                failed_block: block.block_id,
                                reason: err.to_string(),
                            });
                            return Ok(());
                        }
                    };
                    let region = {
                        let mut pool = match block.host_pool.lock() {
                            Ok(pool) => pool,
                            Err(err) => {
                                self.emit_event(CoordinatorEvent::FetchFailed {
                                    ticket,
                                    failed_block: block.block_id,
                                    reason: err.to_string(),
                                })?;
                                self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                                    queue: QueueKind::Fetch,
                                    ticket: QueueTicket::Fetch(ticket),
                                    failed_block: block.block_id,
                                    reason: err.to_string(),
                                });
                                return Ok(());
                            }
                        };
                        let Some(region) = pool.reserve(payload.len()) else {
                            self.emit_event(CoordinatorEvent::FetchFailed {
                                ticket,
                                failed_block: block.block_id,
                                reason: "host pinned pool exhausted".to_string(),
                            })?;
                            self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                                queue: QueueKind::Fetch,
                                ticket: QueueTicket::Fetch(ticket),
                                failed_block: block.block_id,
                                reason: "host pinned pool exhausted".to_string(),
                            });
                            return Ok(());
                        };
                        pool.as_mut_slice(region).copy_from_slice(&payload);
                        region
                    };
                    fetched.push(FetchedBlock {
                        block_id: block.block_id,
                        host_region: region,
                        byte_len: payload.len(),
                        release_after_promote: true,
                    });
                }
                BlockLocation::Remote { .. } => {
                    self.emit_event(CoordinatorEvent::FetchFailed {
                        ticket,
                        failed_block: block.block_id,
                        reason: "remote staging not implemented".to_string(),
                    })?;
                    self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                        queue: QueueKind::Fetch,
                        ticket: QueueTicket::Fetch(ticket),
                        failed_block: block.block_id,
                        reason: "remote staging not implemented".to_string(),
                    });
                    return Ok(());
                }
                BlockLocation::Gpu { .. } => {
                    self.emit_event(CoordinatorEvent::FetchFailed {
                        ticket,
                        failed_block: block.block_id,
                        reason: "gpu source should not go through fetch queue".to_string(),
                    })?;
                    self.emit_orchestrator_event(OrchestratorEvent::TaskFailed {
                        queue: QueueKind::Fetch,
                        ticket: QueueTicket::Fetch(ticket),
                        failed_block: block.block_id,
                        reason: "gpu source should not go through fetch queue".to_string(),
                    });
                    return Ok(());
                }
            }
        }

        self.emit_orchestrator_event(OrchestratorEvent::FetchCompleted {
            ticket,
            blocks: fetched.clone(),
        });
        self.emit_event(CoordinatorEvent::FetchCompleted {
            ticket,
            blocks: fetched,
        })?;
        Ok(())
    }

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                match &cmd {
                    CoordinatorCommand::Plan { ticket, blocks } => {
                        self.handle_plan(*ticket, blocks);
                    }
                    CoordinatorCommand::Spill { ticket, blocks } => {
                        self.handle_spill(*ticket, blocks)?;
                    }
                    CoordinatorCommand::Store { ticket, blocks } => {
                        self.handle_store(*ticket, blocks);
                    }
                    CoordinatorCommand::Fetch { ticket, blocks } => {
                        self.handle_fetch(*ticket, blocks)?;
                    }
                    CoordinatorCommand::Shutdown => {
                        self.events
                            .send(CoordinatorEvent::CommandQueued(cmd.clone()))
                            .map_err(|e| anyhow!("coordinator event send failed: {e}"))?;
                    }
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
    use crate::types::BlockFingerprint;
    use tempfile::tempdir;

    #[test]
    fn coordinator_receives_commands() {
        let (coordinator, handle, events) = Coordinator::new(4);
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
        let (coordinator, handle, _events) = Coordinator::new(4);
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
    fn spill_roundtrip_through_disk_store() {
        let dir = tempdir().unwrap();
        let disk_store = Arc::new(DiskStore::new(dir.path()));
        let (coordinator, handle, events) = Coordinator::new_with_disk_store(4, disk_store.clone());
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(256).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(6).unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"abcdef");
            region
        };
        let fingerprint = BlockFingerprint([0x2A; 16]);
        let ticket = handle
            .submit_spill(vec![SpillRequest {
                block_id: BlockId(7),
                fingerprint,
                kv_format_tag: 3,
                host_pool: host_pool.clone(),
                host_region: region,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::SpillQueued {
                ticket,
                block_count: 1,
            }
        );
        let location = match events.recv().unwrap() {
            CoordinatorEvent::SpillCompleted {
                ticket: done,
                locations,
            } => {
                assert_eq!(done, ticket);
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].0, BlockId(7));
                locations[0].1.clone()
            }
            other => panic!("unexpected spill event: {other:?}"),
        };

        let payload = host_pool.read_region(region).unwrap();
        assert_eq!(payload, b"abcdef");
        let reloaded = disk_store.get_block(&location, Some(fingerprint)).unwrap();
        assert_eq!(reloaded, b"abcdef");
    }

    #[test]
    fn spill_fails_without_configured_disk_store() {
        let (coordinator, handle, events) = Coordinator::new(4);
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let ticket = handle
            .submit_spill(vec![SpillRequest {
                block_id: BlockId(3),
                fingerprint: BlockFingerprint([0x44; 16]),
                kv_format_tag: 1,
                host_pool,
                host_region: region,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::SpillQueued {
                ticket,
                block_count: 1,
            }
        );
        match events.recv().unwrap() {
            CoordinatorEvent::SpillFailed {
                ticket: failed_ticket,
                failed_block,
                reason,
            } => {
                assert_eq!(failed_ticket, ticket);
                assert_eq!(failed_block, BlockId(3));
                assert!(reason.contains("disk store not configured"));
            }
            other => panic!("unexpected spill failure event: {other:?}"),
        }
    }

    #[test]
    fn fetch_from_host_passthrough_keeps_existing_region() {
        let (coordinator, handle, events) = Coordinator::new(4);
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(128).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(5).unwrap();
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
        let (coordinator, handle, events) = Coordinator::new_with_disk_store(4, disk_store.clone());
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
                host_pool.lock().unwrap().release(blocks[0].host_region);
            }
            other => panic!("unexpected fetch event: {other:?}"),
        }
    }

    #[test]
    fn fetch_fails_for_gpu_source() {
        let (coordinator, handle, events) = Coordinator::new(4);
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
    fn submit_spill_is_non_blocking_when_queue_is_full() {
        let (_coordinator, handle, _events) = Coordinator::new(1);
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"test");
            region
        };

        let first = handle.submit_spill(vec![SpillRequest {
            block_id: BlockId(1),
            fingerprint: BlockFingerprint([0x11; 16]),
            kv_format_tag: 1,
            host_pool: host_pool.clone(),
            host_region: region,
        }]);
        assert!(first.is_some());

        let second = handle.submit_spill(vec![SpillRequest {
            block_id: BlockId(2),
            fingerprint: BlockFingerprint([0x22; 16]),
            kv_format_tag: 1,
            host_pool,
            host_region: region,
        }]);
        assert!(second.is_none(), "full queue should not block spill submit");
    }

    #[test]
    fn orchestrator_plan_classifies_tiers_without_touching_legacy_events() {
        let (coordinator, handle, _events, orchestrator_events) =
            Coordinator::new_with_orchestrator_events(4, None);
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
            Coordinator::new_with_orchestrator_events(4, None);
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(64).unwrap(),
        );
        let region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(4).unwrap();
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
                assert!(reason.contains("remote store not implemented"));
            }
            other => panic!("unexpected orchestrator event: {other:?}"),
        }
    }
}
