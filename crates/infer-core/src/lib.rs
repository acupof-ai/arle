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

/// Lifecycle state for an inference request in scheduler-like components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    Queued,
    Prefill,
    Decode,
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
    fn request_phase_progression_example() {
        let phases = [
            RequestPhase::Queued,
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::Completed,
        ];
        assert_eq!(phases.len(), 4);
    }
}
