//! Tiered KV coordinator skeleton for M3a.
//!
//! The real coordinator owns dedicated CUDA copy streams and drains commands on
//! an OS thread. This local skeleton keeps the command channel and lifecycle
//! boundaries explicit without pretending the CUDA path is already wired.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};

use crate::types::BlockId;

use super::tier::BlockLocation;
use super::transport::disk::DiskStore;
use super::{StagePlanner, StageRequest, StageTicket};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageLifecycleState {
    Free,
    Resident,
    Demoting {
        ticket: StageTicket,
        target: BlockLocation,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageLifecycleError {
    UnknownPage {
        page: usize,
    },
    InvalidTransition {
        page: usize,
        from: PageLifecycleState,
        to: PageLifecycleState,
    },
}

impl std::fmt::Display for PageLifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageLifecycleError::UnknownPage { page } => {
                write!(f, "unknown page {page}")
            }
            PageLifecycleError::InvalidTransition { page, from, to } => {
                write!(f, "invalid page transition on {page}: {from:?} -> {to:?}")
            }
        }
    }
}

impl std::error::Error for PageLifecycleError {}

#[derive(Debug, Clone)]
pub struct PageLifecycle {
    states: Vec<PageLifecycleState>,
}

impl PageLifecycle {
    pub fn new(page_count: usize) -> Self {
        Self {
            states: vec![PageLifecycleState::Free; page_count],
        }
    }

    pub fn state(&self, page: usize) -> Option<PageLifecycleState> {
        self.states.get(page).cloned()
    }

    pub fn mark_resident(&mut self, page: usize) -> Result<(), PageLifecycleError> {
        match self
            .states
            .get_mut(page)
            .ok_or(PageLifecycleError::UnknownPage { page })?
        {
            state @ PageLifecycleState::Free => {
                *state = PageLifecycleState::Resident;
                Ok(())
            }
            state @ PageLifecycleState::Demoting { .. } => {
                *state = PageLifecycleState::Resident;
                Ok(())
            }
            current @ PageLifecycleState::Resident => Err(PageLifecycleError::InvalidTransition {
                page,
                from: current.clone(),
                to: PageLifecycleState::Resident,
            }),
        }
    }

    pub fn begin_demote(
        &mut self,
        page: usize,
        ticket: StageTicket,
        target: BlockLocation,
    ) -> Result<(), PageLifecycleError> {
        let next = PageLifecycleState::Demoting { ticket, target };
        match self
            .states
            .get_mut(page)
            .ok_or(PageLifecycleError::UnknownPage { page })?
        {
            state @ PageLifecycleState::Resident => {
                *state = next;
                Ok(())
            }
            current => Err(PageLifecycleError::InvalidTransition {
                page,
                from: current.clone(),
                to: next,
            }),
        }
    }

    pub fn finish_demote(
        &mut self,
        page: usize,
        ticket: StageTicket,
    ) -> Result<(), PageLifecycleError> {
        let next = PageLifecycleState::Free;
        let state = self
            .states
            .get_mut(page)
            .ok_or(PageLifecycleError::UnknownPage { page })?;
        match state {
            PageLifecycleState::Demoting {
                ticket: current, ..
            } => {
                if *current == ticket {
                    *state = next;
                    Ok(())
                } else {
                    Err(PageLifecycleError::InvalidTransition {
                        page,
                        from: state.clone(),
                        to: next,
                    })
                }
            }
            current => Err(PageLifecycleError::InvalidTransition {
                page,
                from: current.clone(),
                to: next,
            }),
        }
    }
}

/// Request handed to the coordinator for a T1 → T2 spill.
///
/// Shipped as part of the "coordinator real byte path" batch: spills
/// LRU host-pinned blocks out to `DiskStore` when the host pinned
/// retained fraction crosses `SchedulerConfig::t1_host_pinned_high_water`.
/// The block's live host bytes are packaged by the scheduler and handed
/// to the coordinator via `host_region`; the coordinator calls
/// `DiskStore::put_block` and emits `SpillCompleted` with the
/// resulting `DiskBlockLocation` so the scheduler can point the radix
/// node at T2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillRequest {
    pub block_id: BlockId,
    pub fingerprint: crate::types::BlockFingerprint,
    pub kv_format_tag: u8,
    pub host_pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
    pub host_region: crate::kv_tier::host_pool::HostPinnedRegion,
}

/// Inverse of [`SpillRequest`]: read a block's bytes back from the disk
/// tier into a freshly-allocated host-pinned region. The scheduler
/// passes a `host_region` that already has the right shape; the
/// coordinator does the `DiskStore::get_block` read and writes into
/// that region, then emits `RehydrateCompleted`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RehydrateRequest {
    pub block_id: BlockId,
    pub fingerprint: crate::types::BlockFingerprint,
    pub disk_location: crate::kv_tier::transport::disk::DiskBlockLocation,
    pub host_pool: crate::kv_tier::host_pool::SharedHostPinnedPool,
    pub host_region: crate::kv_tier::host_pool::HostPinnedRegion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorCommand {
    Demote {
        block: BlockId,
        from: BlockLocation,
        to: BlockLocation,
    },
    Promote {
        block: BlockId,
        from: BlockLocation,
        to: BlockLocation,
    },
    Stage {
        ticket: StageTicket,
        requests: Vec<StageRequest>,
    },
    /// T1 → T2 spill ticket. The coordinator routes each `SpillRequest`
    /// through `DiskStore::put_block` and emits `SpillCompleted` with
    /// the resulting disk locations on success.
    Spill {
        ticket: StageTicket,
        blocks: Vec<SpillRequest>,
    },
    /// T2 → T1 rehydrate ticket. The coordinator reads each block from
    /// `DiskStore::get_block`, writes into the caller-provided host
    /// region, and emits `RehydrateCompleted`.
    Rehydrate {
        ticket: StageTicket,
        blocks: Vec<RehydrateRequest>,
    },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorEvent {
    CommandQueued(CoordinatorCommand),
    StagingQueued {
        ticket: StageTicket,
        request_count: usize,
    },
    /// Stub completes immediately after `StagingQueued` on the local lane; real
    /// CUDA path must emit this only after the copy stream reports done.
    StagingCompleted {
        ticket: StageTicket,
    },
    /// T1 → T2 spill enqueued. Informational — the real work happens
    /// asynchronously on the coordinator's disk I/O thread / worker pool.
    SpillQueued {
        ticket: StageTicket,
        block_count: usize,
    },
    /// T1 → T2 spill finished. `locations` gives the scheduler the
    /// canonical disk locations for every block that was persisted so
    /// the radix can flip `tier_location` to `BlockLocation::Disk`.
    SpillCompleted {
        ticket: StageTicket,
        locations: Vec<(BlockId, crate::kv_tier::transport::disk::DiskBlockLocation)>,
    },
    /// T1 → T2 spill failed. `failed_block` identifies the first block
    /// that did not persist successfully; any preceding blocks in the
    /// ticket were persisted, so the scheduler has to decide whether
    /// to roll them back or accept a partial spill.
    SpillFailed {
        ticket: StageTicket,
        failed_block: BlockId,
        reason: String,
    },
    /// T2 → T1 rehydrate enqueued. Informational.
    RehydrateQueued {
        ticket: StageTicket,
        block_count: usize,
    },
    /// T2 → T1 rehydrate finished. The scheduler flips the radix
    /// node's `tier_location` back to `BlockLocation::HostPinned`.
    RehydrateCompleted {
        ticket: StageTicket,
        rehydrated_blocks: Vec<BlockId>,
    },
    /// T2 → T1 rehydrate failed. Same partial-failure semantics as
    /// `SpillFailed`.
    RehydrateFailed {
        ticket: StageTicket,
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

    /// Mint a fresh `StageTicket` for a T1 → T2 spill batch and enqueue
    /// the corresponding `Spill` command. Returns `None` if `blocks`
    /// is empty or the command channel is full.
    pub fn submit_spill(&self, blocks: Vec<SpillRequest>) -> Option<StageTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = StageTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Spill { ticket, blocks })
            .ok()?;
        Some(ticket)
    }

    /// Mint a fresh `StageTicket` for a T2 → T1 rehydrate batch and
    /// enqueue the corresponding `Rehydrate` command. Returns `None`
    /// if `blocks` is empty or the command channel is full.
    pub fn submit_rehydrate(&self, blocks: Vec<RehydrateRequest>) -> Option<StageTicket> {
        if blocks.is_empty() {
            return None;
        }
        let ticket = StageTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Rehydrate { ticket, blocks })
            .ok()?;
        Some(ticket)
    }
}

impl StagePlanner for CoordinatorHandle {
    fn stage(&self, requests: &[StageRequest]) -> Option<StageTicket> {
        if requests.is_empty() {
            return None;
        }

        let ticket = StageTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.send(CoordinatorCommand::Stage {
            ticket,
            requests: requests.to_vec(),
        })
        .ok()?;
        Some(ticket)
    }
}

pub struct Coordinator {
    rx: Receiver<CoordinatorCommand>,
    events: Sender<CoordinatorEvent>,
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

    fn handle_spill(&self, ticket: StageTicket, blocks: &[SpillRequest]) -> Result<()> {
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

    fn handle_rehydrate(&self, ticket: StageTicket, blocks: &[RehydrateRequest]) -> Result<()> {
        self.emit_event(CoordinatorEvent::RehydrateQueued {
            ticket,
            block_count: blocks.len(),
        })?;

        let Some(disk_store) = &self.disk_store else {
            if let Some(first) = blocks.first() {
                self.emit_event(CoordinatorEvent::RehydrateFailed {
                    ticket,
                    failed_block: first.block_id,
                    reason: "coordinator disk store not configured".to_string(),
                })?;
            } else {
                self.emit_event(CoordinatorEvent::RehydrateCompleted {
                    ticket,
                    rehydrated_blocks: Vec::new(),
                })?;
            }
            return Ok(());
        };

        let mut rehydrated_blocks = Vec::with_capacity(blocks.len());
        for block in blocks {
            let payload = match disk_store.get_block(&block.disk_location, Some(block.fingerprint))
            {
                Ok(payload) => payload,
                Err(err) => {
                    self.emit_event(CoordinatorEvent::RehydrateFailed {
                        ticket,
                        failed_block: block.block_id,
                        reason: err.to_string(),
                    })?;
                    return Ok(());
                }
            };

            if payload.len() != block.host_region.len {
                self.emit_event(CoordinatorEvent::RehydrateFailed {
                    ticket,
                    failed_block: block.block_id,
                    reason: format!(
                        "rehydrate byte length mismatch: disk={} host_region={}",
                        payload.len(),
                        block.host_region.len
                    ),
                })?;
                return Ok(());
            }

            if let Err(err) = block.host_pool.write_region(block.host_region, &payload) {
                self.emit_event(CoordinatorEvent::RehydrateFailed {
                    ticket,
                    failed_block: block.block_id,
                    reason: err.to_string(),
                })?;
                return Ok(());
            }
            rehydrated_blocks.push(block.block_id);
        }

        self.emit_event(CoordinatorEvent::RehydrateCompleted {
            ticket,
            rehydrated_blocks,
        })
    }

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                match &cmd {
                    CoordinatorCommand::Stage { ticket, requests } => {
                        self.events
                            .send(CoordinatorEvent::StagingQueued {
                                ticket: *ticket,
                                request_count: requests.len(),
                            })
                            .map_err(|e| anyhow!("coordinator event send failed: {e}"))?;
                        self.events
                            .send(CoordinatorEvent::StagingCompleted { ticket: *ticket })
                            .map_err(|e| anyhow!("coordinator event send failed: {e}"))?;
                    }
                    CoordinatorCommand::Spill { ticket, blocks } => {
                        self.handle_spill(*ticket, blocks)?;
                    }
                    CoordinatorCommand::Rehydrate { ticket, blocks } => {
                        self.handle_rehydrate(*ticket, blocks)?;
                    }
                    _ => {
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
    use crate::prefix_cache::RadixCache;
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
    fn coordinator_stage_planner_emits_ticketed_event() {
        let (coordinator, handle, events) = Coordinator::new(4);
        let ticket = handle
            .stage(&[StageRequest {
                block_id: BlockId(7),
                from: BlockLocation::HostPinned { offset: 4096 },
                byte_len: 8192,
            }])
            .expect("stage ticket");
        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingQueued {
                ticket,
                request_count: 1,
            }
        );
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingCompleted { ticket }
        );
    }

    #[test]
    fn stage_command_emits_queued_then_completed_events() {
        let (coordinator, handle, events) = Coordinator::new(4);
        let ticket = StageTicket(17);
        handle
            .send(CoordinatorCommand::Stage {
                ticket,
                requests: vec![StageRequest {
                    block_id: BlockId(9),
                    from: BlockLocation::HostPinned { offset: 0 },
                    byte_len: 4096,
                }],
            })
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingQueued {
                ticket,
                request_count: 1,
            }
        );
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingCompleted { ticket }
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
    fn lookup_or_stage_with_coordinator_emits_queued_then_completed_events() {
        let mut cache = RadixCache::with_soft_pin_keepalive(4, 64);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        cache.insert(&tokens, &[BlockId(10), BlockId(20)]);
        assert!(cache.set_block_location(BlockId(10), BlockLocation::HostPinned { offset: 0 }));
        assert!(cache.set_block_byte_len(BlockId(10), 8192));

        let (coordinator, handle, events) = Coordinator::new(4);
        let outcome = cache.lookup_or_stage(
            &tokens,
            crate::kv_tier::LookupHeuristics::default(),
            Some(&handle),
        );
        let ticket = outcome.staging_ticket.expect("staging ticket");

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingQueued {
                ticket,
                request_count: 1,
            }
        );
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingCompleted { ticket }
        );
    }

    #[test]
    fn page_lifecycle_supports_cancel_on_hit() {
        let mut lifecycle = PageLifecycle::new(2);
        let ticket = StageTicket(3);
        lifecycle.mark_resident(0).unwrap();
        lifecycle
            .begin_demote(0, ticket, BlockLocation::HostPinned { offset: 128 })
            .unwrap();
        lifecycle.mark_resident(0).unwrap();
        assert_eq!(lifecycle.state(0), Some(PageLifecycleState::Resident));
    }

    #[test]
    fn page_lifecycle_finishes_demote_to_free() {
        let mut lifecycle = PageLifecycle::new(1);
        let ticket = StageTicket(11);
        lifecycle.mark_resident(0).unwrap();
        lifecycle
            .begin_demote(0, ticket, BlockLocation::HostPinned { offset: 256 })
            .unwrap();
        lifecycle.finish_demote(0, ticket).unwrap();
        assert_eq!(lifecycle.state(0), Some(PageLifecycleState::Free));
    }

    #[test]
    fn page_lifecycle_rejects_invalid_transition() {
        let mut lifecycle = PageLifecycle::new(1);
        let err = lifecycle
            .begin_demote(0, StageTicket(1), BlockLocation::HostPinned { offset: 0 })
            .expect_err("free pages cannot enter demoting directly");
        assert!(matches!(
            err,
            PageLifecycleError::InvalidTransition {
                from: PageLifecycleState::Free,
                ..
            }
        ));
    }

    #[test]
    fn spill_and_rehydrate_roundtrip_through_disk_store() {
        let dir = tempdir().unwrap();
        let disk_store = Arc::new(crate::kv_tier::transport::disk::DiskStore::new(dir.path()));
        let (coordinator, handle, events) = Coordinator::new_with_disk_store(4, disk_store);
        let host_pool = crate::kv_tier::host_pool::SharedHostPinnedPool::new(
            crate::kv_tier::HostPinnedPool::new(256).unwrap(),
        );
        let spill_region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(6).unwrap();
            pool.as_mut_slice(region).copy_from_slice(b"abcdef");
            region
        };
        let fingerprint = BlockFingerprint([0x2A; 16]);
        let spill_ticket = handle
            .submit_spill(vec![SpillRequest {
                block_id: BlockId(7),
                fingerprint,
                kv_format_tag: 3,
                host_pool: host_pool.clone(),
                host_region: spill_region,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::SpillQueued {
                ticket: spill_ticket,
                block_count: 1,
            }
        );
        let spill_location = match events.recv().unwrap() {
            CoordinatorEvent::SpillCompleted { ticket, locations } => {
                assert_eq!(ticket, spill_ticket);
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].0, BlockId(7));
                locations[0].1.clone()
            }
            other => panic!("unexpected spill event: {other:?}"),
        };

        let rehydrate_region = {
            let mut pool = host_pool.lock().unwrap();
            let region = pool.reserve(6).unwrap();
            pool.as_mut_slice(region).fill(0);
            region
        };
        let rehydrate_ticket = handle
            .submit_rehydrate(vec![RehydrateRequest {
                block_id: BlockId(7),
                fingerprint,
                disk_location: spill_location,
                host_pool: host_pool.clone(),
                host_region: rehydrate_region,
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::RehydrateQueued {
                ticket: rehydrate_ticket,
                block_count: 1,
            }
        );
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::RehydrateCompleted {
                ticket: rehydrate_ticket,
                rehydrated_blocks: vec![BlockId(7)],
            }
        );

        let pool = host_pool.lock().unwrap();
        assert_eq!(pool.as_slice(rehydrate_region), b"abcdef");
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
}
