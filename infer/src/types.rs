//! Core domain types shared across infer workspace crates.

use std::sync::Arc;

/// Stable request identifier across scheduler/runtime boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// Client-supplied session/conversation identifier used by the scheduler to
/// route subsequent turns of the same agent session to the slot or radix
/// subtree that already holds their KV prefix.
///
/// The value is opaque to the engine. Callers should treat it as a stable
/// per-conversation key (typically a UUID or hash of the conversation array).
/// It is the protocol-level plumbing for
/// `docs/projects/agent-first-architecture.md::A2` (session-sticky routing);
/// admission logic consumes it once `A1` wires the RadixCache into the
/// CUDA/Metal schedulers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(Arc<str>);

impl SessionId {
    /// Wrap an arbitrary string as a session id. Empty strings are rejected by
    /// callers; this type does not enforce non-emptiness because higher layers
    /// decide what to do with invalid input.
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

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

    #[test]
    fn session_id_round_trips_from_str_and_string() {
        let from_str = SessionId::from("abc-123");
        let from_string = SessionId::from(String::from("abc-123"));
        assert_eq!(from_str, from_string);
        assert_eq!(from_str.as_str(), "abc-123");
        assert_eq!(from_str.to_string(), "abc-123");
    }

    #[test]
    fn session_id_is_hashable_and_cheap_clone() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let a = SessionId::new("agent-session-1");
        let b = a.clone();
        set.insert(a);
        assert!(set.contains(&b));
    }
}
