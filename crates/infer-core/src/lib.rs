//! Core domain types shared across infer workspace crates.

/// Stable request identifier across scheduler/runtime boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// Canonical inference mode used by control/data-plane orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    Prefill,
    Decode,
}

/// Lifecycle action emitted by scheduler-like components.
///
/// These events are intentionally action-oriented instead of state-oriented so
/// consumers can distinguish initial admission from preemption/requeue, and
/// lifecycle transitions from chunked work units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestEventKind {
    Enqueued,
    Requeued,
    PrefillStarted,
    DecodeStep,
    Evicted,
    Completed,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_is_hashable_and_copy() {
        let a = RequestId(7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_event_kind_progression_example() {
        let events = [
            RequestEventKind::Enqueued,
            RequestEventKind::Requeued,
            RequestEventKind::PrefillStarted,
            RequestEventKind::DecodeStep,
            RequestEventKind::Evicted,
            RequestEventKind::Completed,
            RequestEventKind::Cancelled,
        ];
        assert_eq!(events.len(), 7);
    }
}
