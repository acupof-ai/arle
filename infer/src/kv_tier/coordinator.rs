//! Tiered KV coordinator skeleton for M3a.
//!
//! The real coordinator owns dedicated CUDA copy streams and drains commands on
//! an OS thread. This local skeleton keeps the command channel and lifecycle
//! boundaries explicit without pretending the CUDA path is already wired.

use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};

use crate::types::BlockId;

use super::tier::BlockLocation;

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
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorEvent {
    CommandQueued(CoordinatorCommand),
}

#[derive(Clone)]
pub struct CoordinatorHandle {
    tx: Sender<CoordinatorCommand>,
}

impl CoordinatorHandle {
    pub fn send(&self, cmd: CoordinatorCommand) -> Result<()> {
        self.tx
            .send(cmd)
            .map_err(|e| anyhow!("coordinator send failed: {e}"))
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
            CoordinatorHandle { tx },
            event_rx,
        )
    }

    pub fn run_once(&self) -> Result<bool> {
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(cmd) => {
                self.events
                    .send(CoordinatorEvent::CommandQueued(cmd.clone()))
                    .map_err(|e| anyhow!("coordinator event send failed: {e}"))?;
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
}
