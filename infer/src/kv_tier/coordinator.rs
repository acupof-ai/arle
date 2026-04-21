//! Tiered KV coordinator skeleton for the shipped local spill path.
//!
//! The live local runtime uses the coordinator for one real behavior only:
//! persisting host-pinned blocks into `DiskStore` and reporting the canonical
//! disk locations back to the scheduler. Live staged readmission was removed
//! from the hot path until the runtime grows a real attach/ownership model.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError, bounded};

use crate::types::BlockId;

use super::transport::disk::{DiskBlockLocation, DiskStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillTicket(pub u64);

/// Request handed to the coordinator for a T1 → T2 spill.
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
    /// T1 → T2 spill ticket. The coordinator routes each `SpillRequest`
    /// through `DiskStore::put_block` and emits `SpillCompleted` with the
    /// resulting disk locations on success.
    Spill {
        ticket: SpillTicket,
        blocks: Vec<SpillRequest>,
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

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                match &cmd {
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
