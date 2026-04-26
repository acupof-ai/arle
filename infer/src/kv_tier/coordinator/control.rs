//! Atomic queue counters and the per-coordinator `live` / `cancelled`
//! ticket sets. Internal to the coordinator subtree — the public surface
//! is the snapshot types in [`super::types`].

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::types::{
    CoordinatorQueueStats, QueueControlStats, QueueKind, QueueOutcome, QueueTicket,
};

#[derive(Debug)]
pub(super) struct QueueCounters {
    queued: AtomicUsize,
    in_flight: AtomicUsize,
    submitted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    rejected: AtomicU64,
}

impl QueueCounters {
    const fn new() -> Self {
        Self {
            queued: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, capacity: usize) -> QueueControlStats {
        QueueControlStats {
            capacity,
            queued: self.queued.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
pub(super) struct CoordinatorControl {
    capacity: usize,
    plan: QueueCounters,
    fetch: QueueCounters,
    store: QueueCounters,
    live: Mutex<HashSet<QueueTicket>>,
    cancelled: Mutex<HashSet<QueueTicket>>,
}

impl CoordinatorControl {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            plan: QueueCounters::new(),
            fetch: QueueCounters::new(),
            store: QueueCounters::new(),
            live: Mutex::new(HashSet::new()),
            cancelled: Mutex::new(HashSet::new()),
        }
    }

    fn queue(&self, kind: QueueKind) -> &QueueCounters {
        match kind {
            QueueKind::Plan => &self.plan,
            QueueKind::Fetch => &self.fetch,
            QueueKind::Store => &self.store,
        }
    }

    pub(super) fn on_submit(&self, ticket: QueueTicket) {
        let queue = self.queue(ticket.kind());
        queue.submitted.fetch_add(1, Ordering::Relaxed);
        queue.queued.fetch_add(1, Ordering::Relaxed);
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.insert(ticket);
    }

    pub(super) fn on_reject(&self, kind: QueueKind) {
        self.queue(kind).rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn on_start(&self, ticket: QueueTicket) {
        let queue = self.queue(ticket.kind());
        queue.queued.fetch_sub(1, Ordering::Relaxed);
        queue.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn on_finish(&self, ticket: QueueTicket, outcome: QueueOutcome) {
        let queue = self.queue(ticket.kind());
        queue.in_flight.fetch_sub(1, Ordering::Relaxed);
        match outcome {
            QueueOutcome::Completed => {
                queue.completed.fetch_add(1, Ordering::Relaxed);
            }
            QueueOutcome::Failed => {
                queue.failed.fetch_add(1, Ordering::Relaxed);
            }
            QueueOutcome::Cancelled => {
                queue.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.remove(&ticket);
        let mut cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.remove(&ticket);
    }

    pub(super) fn cancel(&self, ticket: QueueTicket) -> bool {
        let live = self
            .live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !live.contains(&ticket) {
            return false;
        }
        drop(live);
        let mut cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.insert(ticket)
    }

    pub(super) fn is_cancelled(&self, ticket: QueueTicket) -> bool {
        let cancelled = self
            .cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancelled.contains(&ticket)
    }

    pub(super) fn stats(&self) -> CoordinatorQueueStats {
        let plan = self.plan.snapshot(self.capacity);
        let fetch = self.fetch.snapshot(self.capacity);
        let store = self.store.snapshot(self.capacity);
        CoordinatorQueueStats {
            capacity: self.capacity,
            total_queued: plan
                .queued
                .saturating_add(fetch.queued)
                .saturating_add(store.queued),
            total_in_flight: plan
                .in_flight
                .saturating_add(fetch.in_flight)
                .saturating_add(store.in_flight),
            plan,
            fetch,
            store,
            fetch_waiters: 0,
        }
    }

    pub(super) fn queue_stats(&self, kind: QueueKind) -> QueueControlStats {
        self.queue(kind).snapshot(self.capacity)
    }
}
