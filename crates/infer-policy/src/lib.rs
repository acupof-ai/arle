//! Policy traits for admission and chunking decisions.

use infer_core::InferenceMode;

/// Read-only scheduler signal snapshot for policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerSignals {
    pub queued_requests: usize,
    pub active_decodes: usize,
}

/// A policy that decides whether new work can enter the system.
pub trait AdmissionPolicy: Send + Sync {
    fn allow(&self, signals: SchedulerSignals) -> bool;
}

/// Default policy: allow requests while queue pressure is bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueBoundAdmission {
    pub max_queued_requests: usize,
}

impl AdmissionPolicy for QueueBoundAdmission {
    fn allow(&self, signals: SchedulerSignals) -> bool {
        signals.queued_requests < self.max_queued_requests
    }
}

/// Policy to determine next prefill chunk size.
pub trait ChunkingPolicy: Send + Sync {
    fn next_chunk_size(&self, mode: InferenceMode, signals: SchedulerSignals) -> usize;
}

/// Decode-aware chunking policy with a conservative fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeAwareChunking {
    pub decode_active_chunk: usize,
    pub idle_chunk: usize,
}

impl ChunkingPolicy for DecodeAwareChunking {
    fn next_chunk_size(&self, mode: InferenceMode, signals: SchedulerSignals) -> usize {
        match mode {
            InferenceMode::Decode => self.decode_active_chunk,
            InferenceMode::Prefill if signals.active_decodes > 0 => self.decode_active_chunk,
            InferenceMode::Prefill => self.idle_chunk,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_bound_admission_blocks_when_full() {
        let policy = QueueBoundAdmission {
            max_queued_requests: 2,
        };
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 1,
            active_decodes: 0,
        }));
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 2,
            active_decodes: 0,
        }));
    }

    #[test]
    fn decode_aware_chunking_switches_chunk_size() {
        let policy = DecodeAwareChunking {
            decode_active_chunk: 64,
            idle_chunk: 512,
        };

        assert_eq!(
            policy.next_chunk_size(
                InferenceMode::Prefill,
                SchedulerSignals {
                    queued_requests: 4,
                    active_decodes: 0,
                }
            ),
            512
        );

        assert_eq!(
            policy.next_chunk_size(
                InferenceMode::Prefill,
                SchedulerSignals {
                    queued_requests: 4,
                    active_decodes: 2,
                }
            ),
            64
        );
    }
}
