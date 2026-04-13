//! Policy traits for admission and chunking decisions.

use infer_core::InferenceMode;

/// Read-only scheduler signal snapshot for policy decisions.
///
/// Fields come in two groups:
/// - **Global state** (`queued_requests`, `active_decodes`): describes the
///   scheduler as a whole and is what chunking-policy decisions read.
/// - **Per-request hints** (`prefix_hit_tokens`, `session_affinity_slot`,
///   `turn_depth`): describe the *incoming request* being considered for
///   admission. Callers that only need chunking decisions leave these at
///   their default (`0`/`None`) via struct-update syntax.
///
/// The per-request fields are the agent-aware surface
/// `docs/projects/agent-first-architecture.md::B3` asks for; they become
/// meaningful once `A1` wires the RadixCache into the schedulers and can
/// actually compute prefix hits before calling `AdmissionPolicy::allow`.
/// Until then, existing call sites pass them as defaults and the legacy
/// [`QueueBoundAdmission`] policy ignores them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerSignals {
    /// Requests currently waiting in the scheduler queue.
    pub queued_requests: usize,
    /// Requests currently in the decode phase.
    pub active_decodes: usize,
    /// Tokens the incoming request matches against any cached prefix.
    ///
    /// `0` means the incoming request is cold (no reusable KV). Populated
    /// by the scheduler's prefix-cache lookup immediately before admission.
    pub prefix_hit_tokens: usize,
    /// Slot or radix subtree that already holds the incoming request's
    /// session prefix, when known. `None` for cold first-turn requests or
    /// when the caller has no session routing information.
    pub session_affinity_slot: Option<usize>,
    /// Nth turn of the incoming request's conversation, where `0` means
    /// "no prior turns observed" (a cold first turn or a session-less
    /// request). Turn counters are owned by the scheduler — the policy
    /// only reads them.
    pub turn_depth: u32,
}

impl SchedulerSignals {
    /// Convenience constructor for chunking-policy call sites that only care
    /// about global state. Matches the pre-B3 ergonomics: no per-request
    /// hints, everything else defaults to zero.
    pub fn queue_state(queued_requests: usize, active_decodes: usize) -> Self {
        Self {
            queued_requests,
            active_decodes,
            ..Self::default()
        }
    }

    /// True when the incoming request has no reusable KV prefix and no
    /// session affinity — i.e. the admission path should treat it as a
    /// cold first-turn request.
    pub fn is_cold_request(&self) -> bool {
        self.prefix_hit_tokens == 0 && self.session_affinity_slot.is_none() && self.turn_depth == 0
    }
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

/// Agent-aware admission policy that prefers warm (session-continuation)
/// requests over cold ones once queue pressure crosses a soft threshold.
///
/// Semantics:
/// 1. **Hard cap**: always reject when `queued_requests >= hard_cap`
///    (matches [`QueueBoundAdmission`] so existing backpressure still
///    triggers).
/// 2. **Cold throttle**: when `queued_requests >= cold_soft_cap` and the
///    incoming request is cold ([`SchedulerSignals::is_cold_request`]),
///    reject. Warm requests (prefix hit, session affinity, or non-zero
///    turn depth) stay admissible until the hard cap.
///
/// The exit criterion from `docs/projects/agent-first-architecture.md::B3`
/// is "warm (session-continuation) requests do not get starved behind
/// bursts of cold requests". This policy enforces that by only letting
/// cold requests join the queue while there is still headroom beneath the
/// soft cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixAwareAdmission {
    /// Absolute cap; mirrors `QueueBoundAdmission::max_queued_requests`.
    pub hard_cap: usize,
    /// Soft cap above which cold requests are rejected. Must satisfy
    /// `cold_soft_cap <= hard_cap`; values above the hard cap are clamped
    /// at call time.
    pub cold_soft_cap: usize,
}

impl PrefixAwareAdmission {
    /// Build a policy where cold requests get `cold_headroom` slots of
    /// breathing room beneath the hard cap.
    pub fn with_cold_headroom(hard_cap: usize, cold_headroom: usize) -> Self {
        let cold_soft_cap = hard_cap.saturating_sub(cold_headroom);
        Self {
            hard_cap,
            cold_soft_cap,
        }
    }
}

impl AdmissionPolicy for PrefixAwareAdmission {
    fn allow(&self, signals: SchedulerSignals) -> bool {
        if signals.queued_requests >= self.hard_cap {
            return false;
        }
        let effective_cold_cap = self.cold_soft_cap.min(self.hard_cap);
        if signals.is_cold_request() && signals.queued_requests >= effective_cold_cap {
            return false;
        }
        true
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
        assert!(policy.allow(SchedulerSignals::queue_state(1, 0)));
        assert!(!policy.allow(SchedulerSignals::queue_state(2, 0)));
    }

    #[test]
    fn decode_aware_chunking_switches_chunk_size() {
        let policy = DecodeAwareChunking {
            decode_active_chunk: 64,
            idle_chunk: 512,
        };

        assert_eq!(
            policy.next_chunk_size(InferenceMode::Prefill, SchedulerSignals::queue_state(4, 0)),
            512
        );

        assert_eq!(
            policy.next_chunk_size(InferenceMode::Prefill, SchedulerSignals::queue_state(4, 2)),
            64
        );
    }

    #[test]
    fn scheduler_signals_default_is_cold() {
        assert!(SchedulerSignals::default().is_cold_request());
    }

    #[test]
    fn scheduler_signals_warm_on_prefix_hit() {
        let signals = SchedulerSignals {
            prefix_hit_tokens: 128,
            ..SchedulerSignals::default()
        };
        assert!(!signals.is_cold_request());
    }

    #[test]
    fn scheduler_signals_warm_on_session_affinity() {
        let signals = SchedulerSignals {
            session_affinity_slot: Some(3),
            ..SchedulerSignals::default()
        };
        assert!(!signals.is_cold_request());
    }

    #[test]
    fn scheduler_signals_warm_on_turn_depth() {
        let signals = SchedulerSignals {
            turn_depth: 2,
            ..SchedulerSignals::default()
        };
        assert!(!signals.is_cold_request());
    }

    #[test]
    fn prefix_aware_admission_rejects_cold_above_soft_cap() {
        let policy = PrefixAwareAdmission {
            hard_cap: 10,
            cold_soft_cap: 6,
        };

        // Cold request below soft cap: admitted.
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 5,
            ..SchedulerSignals::default()
        }));

        // Cold request at soft cap: rejected.
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 6,
            ..SchedulerSignals::default()
        }));

        // Cold request above soft cap: rejected.
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 8,
            ..SchedulerSignals::default()
        }));
    }

    #[test]
    fn prefix_aware_admission_admits_warm_until_hard_cap() {
        let policy = PrefixAwareAdmission {
            hard_cap: 10,
            cold_soft_cap: 6,
        };

        // Warm request at 6 (cold would be rejected): admitted.
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 6,
            prefix_hit_tokens: 64,
            ..SchedulerSignals::default()
        }));

        // Warm request at 9 (one below hard cap): admitted.
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 9,
            turn_depth: 3,
            ..SchedulerSignals::default()
        }));

        // Warm request at hard cap: rejected.
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 10,
            session_affinity_slot: Some(1),
            ..SchedulerSignals::default()
        }));
    }

    #[test]
    fn prefix_aware_admission_handles_soft_cap_above_hard() {
        // Guard: nonsense config must not admit past the hard cap.
        let policy = PrefixAwareAdmission {
            hard_cap: 4,
            cold_soft_cap: 16,
        };
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 4,
            ..SchedulerSignals::default()
        }));
    }

    #[test]
    fn with_cold_headroom_reserves_slots_for_warm_requests() {
        let policy = PrefixAwareAdmission::with_cold_headroom(10, 4);
        assert_eq!(policy.hard_cap, 10);
        assert_eq!(policy.cold_soft_cap, 6);

        // Cold at 6 rejected, warm at 6 admitted.
        assert!(!policy.allow(SchedulerSignals {
            queued_requests: 6,
            ..SchedulerSignals::default()
        }));
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 6,
            prefix_hit_tokens: 32,
            ..SchedulerSignals::default()
        }));
    }

    #[test]
    fn with_cold_headroom_saturates_at_zero() {
        // Headroom larger than the cap must clamp instead of wrapping.
        let policy = PrefixAwareAdmission::with_cold_headroom(3, 10);
        assert_eq!(policy.hard_cap, 3);
        assert_eq!(policy.cold_soft_cap, 0);

        // Cold requests hit the soft cap immediately (0 >= 0) and are rejected
        // regardless of queue depth.
        assert!(!policy.allow(SchedulerSignals::queue_state(0, 0)));
        assert!(!policy.allow(SchedulerSignals::queue_state(2, 0)));

        // Warm requests are still admitted up to the hard cap.
        assert!(policy.allow(SchedulerSignals {
            queued_requests: 2,
            prefix_hit_tokens: 1,
            ..SchedulerSignals::default()
        }));
    }
}
