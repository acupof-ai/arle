//! Ticket / queue / backpressure value types for the tiered KV coordinator.
//!
//! Plain data types only — no I/O, no thread state. Sibling modules carry
//! the runtime: `control` owns the atomics + cancel-set, `events` owns the
//! command/event enums, `builder` owns construction, and `coordinator.rs`
//! owns the handler loop.

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

pub(super) fn pressure_from_load(active: usize, capacity: usize) -> QueueBackpressure {
    if active >= capacity {
        QueueBackpressure::Saturated
    } else if active.saturating_mul(4) >= capacity.saturating_mul(3) {
        QueueBackpressure::Elevated
    } else {
        QueueBackpressure::Normal
    }
}

/// Outcome reported back to [`super::control::CoordinatorControl::on_finish`]
/// after a queued task settles. Internal to the coordinator subtree — call
/// sites should produce one of these via [`FailureClass::into_outcome`] or
/// return [`QueueOutcome::Completed`] directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueueOutcome {
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
    pub(super) fn into_outcome(self) -> QueueOutcome {
        match self {
            Self::Cancelled => QueueOutcome::Cancelled,
            Self::Failed => QueueOutcome::Failed,
        }
    }
}
