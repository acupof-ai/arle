//! Shared observability schema across engine components.

use infer_core::{InferenceMode, RequestEventKind, RequestId};

/// Minimal event model for request lifecycle tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineEvent {
    pub request_id: RequestId,
    pub kind: RequestEventKind,
    pub mode: Option<InferenceMode>,
}

/// Event sink abstraction; allows plugging metrics/tracing backends.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: &EngineEvent);
}

/// Default sink used when observability is not externally configured.
#[derive(Debug, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    fn emit(&self, _event: &EngineEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct VecSink {
        events: Mutex<Vec<EngineEvent>>,
    }

    impl EventSink for VecSink {
        fn emit(&self, event: &EngineEvent) {
            self.events.lock().expect("poisoned").push(event.clone());
        }
    }

    #[test]
    fn sink_receives_event() {
        let sink = VecSink::default();
        let event = EngineEvent {
            request_id: RequestId(1),
            kind: RequestEventKind::PrefillStarted,
            mode: Some(InferenceMode::Prefill),
        };
        sink.emit(&event);
        let stored = sink.events.lock().expect("poisoned");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0], event);
    }
}
