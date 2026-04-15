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
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};

use crate::types::BlockId;

use super::tier::BlockLocation;
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
}

impl Coordinator {
    pub fn new(queue_capacity: usize) -> (Self, CoordinatorHandle, Receiver<CoordinatorEvent>) {
        let (tx, rx) = bounded(queue_capacity.max(1));
        let (event_tx, event_rx) = bounded(queue_capacity.max(1));
        (
            Self {
                rx,
                events: event_tx,
            },
            CoordinatorHandle {
                tx,
                next_ticket: Arc::new(AtomicU64::new(1)),
            },
            event_rx,
        )
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
        let join_handle = coordinator.spawn("pegainfer-tiered-kv-coord-test");
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
}
