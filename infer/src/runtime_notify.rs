//! Low-overhead runtime notification primitives.
//!
//! The first primitive is a one-shot broadcast gate for coordinating runtime
//! startup across worker threads. It is process-local today; the public API is
//! intentionally small so it can grow an eventfd/UDS/shared-memory backend if
//! we need cross-process notification later.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

const PENDING: u8 = 0;
const RELEASED: u8 = 1;
const CANCELLED: u8 = 2;

#[derive(Clone)]
pub struct RuntimeNotifyGate {
    inner: Arc<RuntimeNotifyGateInner>,
}

struct RuntimeNotifyGateInner {
    state: AtomicU8,
    wait_lock: Mutex<()>,
    cv: Condvar,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeNotifyOutcome {
    Released,
    Cancelled,
}

impl RuntimeNotifyGate {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RuntimeNotifyGateInner {
                state: AtomicU8::new(PENDING),
                wait_lock: Mutex::new(()),
                cv: Condvar::new(),
            }),
        }
    }

    /// Release all current and future waiters. The first terminal transition
    /// wins, so a later `cancel()` cannot override a release.
    pub fn release(&self) -> RuntimeNotifyOutcome {
        self.complete(RELEASED)
    }

    /// Cancel all current and future waiters. The first terminal transition
    /// wins, so a later `release()` cannot override a cancel.
    pub fn cancel(&self) -> RuntimeNotifyOutcome {
        self.complete(CANCELLED)
    }

    /// Wait until the gate reaches a terminal state. After the gate is
    /// released or cancelled, this is an atomic load fast path.
    pub fn wait(&self) -> RuntimeNotifyOutcome {
        loop {
            if let Some(outcome) = decode_terminal(self.inner.state.load(Ordering::Acquire)) {
                return outcome;
            }

            let guard = self.lock_waiters();
            let _guard = self
                .inner
                .cv
                .wait_while(guard, |_| {
                    self.inner.state.load(Ordering::Acquire) == PENDING
                })
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub fn outcome(&self) -> Option<RuntimeNotifyOutcome> {
        decode_terminal(self.inner.state.load(Ordering::Acquire))
    }

    fn complete(&self, next: u8) -> RuntimeNotifyOutcome {
        let _guard = self.lock_waiters();
        let current = self.inner.state.load(Ordering::Acquire);
        if let Some(outcome) = decode_terminal(current) {
            return outcome;
        }
        self.inner.state.store(next, Ordering::Release);
        self.inner.cv.notify_all();
        decode_terminal(next).expect("runtime notify terminal state")
    }

    fn lock_waiters(&self) -> MutexGuard<'_, ()> {
        self.inner
            .wait_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for RuntimeNotifyGate {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RuntimeNotifyGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeNotifyGate")
            .field("outcome", &self.outcome())
            .finish()
    }
}

fn decode_terminal(state: u8) -> Option<RuntimeNotifyOutcome> {
    match state {
        RELEASED => Some(RuntimeNotifyOutcome::Released),
        CANCELLED => Some(RuntimeNotifyOutcome::Cancelled),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{RuntimeNotifyGate, RuntimeNotifyOutcome};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn release_wakes_waiter() {
        let gate = RuntimeNotifyGate::new();
        let waiter = gate.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            waiter.wait()
        });

        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(gate.release(), RuntimeNotifyOutcome::Released);
        assert_eq!(thread.join().unwrap(), RuntimeNotifyOutcome::Released);
    }

    #[test]
    fn cancel_wakes_waiter() {
        let gate = RuntimeNotifyGate::new();
        let waiter = gate.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            waiter.wait()
        });

        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(gate.cancel(), RuntimeNotifyOutcome::Cancelled);
        assert_eq!(thread.join().unwrap(), RuntimeNotifyOutcome::Cancelled);
    }

    #[test]
    fn first_terminal_transition_wins() {
        let gate = RuntimeNotifyGate::new();

        assert_eq!(gate.cancel(), RuntimeNotifyOutcome::Cancelled);
        assert_eq!(gate.release(), RuntimeNotifyOutcome::Cancelled);
        assert_eq!(gate.wait(), RuntimeNotifyOutcome::Cancelled);
    }
}
