//! Tiered KV coordinator skeleton for M3a.
//!
//! The real coordinator owns dedicated CUDA copy streams and drains commands on
//! an OS thread. This local skeleton keeps the command channel and lifecycle
//! boundaries explicit without pretending the CUDA path is already wired.

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
use super::transport::disk::DiskStore;
use super::{StagePlanner, StageRequest, StageTicket};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorCommand {
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
    /// Stage request failed before bytes became available in T1 for the
    /// scheduler to promote back into T0.
    StagingFailed {
        ticket: StageTicket,
        failed_block: BlockId,
        reason: String,
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

}

impl StagePlanner for CoordinatorHandle {
    fn stage(&self, requests: &[StageRequest]) -> Option<StageTicket> {
        if requests.is_empty() {
            return None;
        }

        let ticket = StageTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed));
        self.try_send(CoordinatorCommand::Stage {
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

    fn handle_stage(&self, ticket: StageTicket, requests: &[StageRequest]) -> Result<()> {
        self.emit_event(CoordinatorEvent::StagingQueued {
            ticket,
            request_count: requests.len(),
        })?;

        for request in requests {
            match &request.from {
                BlockLocation::HostPinned { .. } | BlockLocation::Gpu { .. } => {}
                BlockLocation::Disk {
                    fingerprint,
                    payload_len,
                } => {
                    let Some(disk_store) = &self.disk_store else {
                        self.emit_event(CoordinatorEvent::StagingFailed {
                            ticket,
                            failed_block: request.block_id,
                            reason: "coordinator disk store not configured".to_string(),
                        })?;
                        return Ok(());
                    };
                    let (Some(host_pool), Some(host_region)) =
                        (&request.host_pool, request.host_region)
                    else {
                        self.emit_event(CoordinatorEvent::StagingFailed {
                            ticket,
                            failed_block: request.block_id,
                            reason: "disk stage request missing host staging region".to_string(),
                        })?;
                        return Ok(());
                    };

                    let disk_location = crate::kv_tier::transport::disk::DiskBlockLocation {
                        path: disk_store.block_path_for(*fingerprint)?,
                        payload_len: *payload_len,
                        fingerprint: *fingerprint,
                    };
                    let payload = match disk_store.get_block(&disk_location, Some(*fingerprint)) {
                        Ok(payload) => payload,
                        Err(err) => {
                            self.emit_event(CoordinatorEvent::StagingFailed {
                                ticket,
                                failed_block: request.block_id,
                                reason: err.to_string(),
                            })?;
                            return Ok(());
                        }
                    };
                    if payload.len() != host_region.len {
                        self.emit_event(CoordinatorEvent::StagingFailed {
                            ticket,
                            failed_block: request.block_id,
                            reason: format!(
                                "stage byte length mismatch: disk={} host_region={}",
                                payload.len(),
                                host_region.len
                            ),
                        })?;
                        return Ok(());
                    }
                    if let Err(err) = host_pool.write_region(host_region, &payload) {
                        self.emit_event(CoordinatorEvent::StagingFailed {
                            ticket,
                            failed_block: request.block_id,
                            reason: err.to_string(),
                        })?;
                        return Ok(());
                    }
                }
                BlockLocation::Remote { .. } => {
                    self.emit_event(CoordinatorEvent::StagingFailed {
                        ticket,
                        failed_block: request.block_id,
                        reason: "remote staging not implemented".to_string(),
                    })?;
                    return Ok(());
                }
            }
        }

        self.emit_event(CoordinatorEvent::StagingCompleted { ticket })
    }

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                match &cmd {
                    CoordinatorCommand::Stage { ticket, requests } => {
                        self.handle_stage(*ticket, requests)?;
                    }
                    CoordinatorCommand::Spill { ticket, blocks } => {
                        self.handle_spill(*ticket, blocks)?;
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
                host_pool: None,
                host_region: None,
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
                    host_pool: None,
                    host_region: None,
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
    fn spill_and_stage_roundtrip_through_disk_store() {
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
        let stage_ticket = handle
            .stage(&[StageRequest {
                block_id: BlockId(7),
                from: BlockLocation::Disk {
                    fingerprint,
                    payload_len: spill_location.payload_len,
                },
                byte_len: spill_location.payload_len as u32,
                host_pool: Some(host_pool.clone()),
                host_region: Some(rehydrate_region),
            }])
            .unwrap();

        assert!(coordinator.run_once().unwrap());
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingQueued {
                ticket: stage_ticket,
                request_count: 1,
            }
        );
        assert_eq!(
            events.recv().unwrap(),
            CoordinatorEvent::StagingCompleted {
                ticket: stage_ticket,
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

    #[test]
    fn stage_is_non_blocking_when_queue_is_full() {
        let (_coordinator, handle, _events) = Coordinator::new(1);
        let first = handle.stage(&[StageRequest {
            block_id: BlockId(1),
            from: BlockLocation::HostPinned { offset: 0 },
            byte_len: 4096,
            host_pool: None,
            host_region: None,
        }]);
        assert!(first.is_some());

        let second = handle.stage(&[StageRequest {
            block_id: BlockId(2),
            from: BlockLocation::HostPinned { offset: 4096 },
            byte_len: 4096,
            host_pool: None,
            host_region: None,
        }]);
        assert!(second.is_none(), "full queue should not block stage submit");
    }
}
