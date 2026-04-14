//! Policy traits for admission and chunking decisions.

use crate::types::InferenceMode;

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

/// Snapshot of one eviction candidate as the coordinator sees it. Pure data —
/// the coordinator builds these from its own bookkeeping and hands them to
/// [`EvictionPolicy::score`], which only scores. Sibling shape to
/// [`SchedulerSignals`]: small, [`Copy`], no allocations.
///
/// Field semantics:
/// - `slot`: coordinator-assigned slot index. Useful for [`SessionBiasedLru`]
///   (bonus when matching `signals.session_affinity_slot`) and for
///   deterministic tie-breaking.
/// - `tokens`: number of tokens this candidate represents (one block, one
///   slot, one radix node — depends on the granularity of the caller).
/// - `last_access_step`: monotonic scheduler tick. Higher = more recent. The
///   coordinator picks the unit (per-step counter, per-decode counter, etc.).
/// - `hit_count`: how many requests reused this candidate. Used by
///   [`HitCountLru`] to protect "warm" prefixes from immediate eviction.
/// - `prefix_depth`: radix-tree depth. Reserved for future policies that
///   prefer to keep deep shared prefixes; current default impls do not read
///   it but it costs nothing to surface.
/// - `pinned`: `true` ⇒ a request is currently using this candidate's KV
///   (refcount > 0 in the coordinator's bookkeeping). All shipped policies
///   return [`f32::INFINITY`] when this is set, so pinned candidates are
///   never selected for eviction. Maps to SGLang's `lock_ref > 0` and
///   vLLM's `KVCacheBlock::ref_cnt > 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvictionCandidate {
    pub slot: u32,
    pub tokens: u32,
    pub last_access_step: u64,
    pub hit_count: u32,
    pub prefix_depth: u32,
    pub pinned: bool,
}

impl EvictionCandidate {
    /// Convenience constructor for tests and policy-only call sites that do
    /// not need every field. Defaults: `pinned = false`, everything else `0`.
    pub fn new(slot: u32, last_access_step: u64) -> Self {
        Self {
            slot,
            tokens: 0,
            last_access_step,
            hit_count: 0,
            prefix_depth: 0,
            pinned: false,
        }
    }
}

/// A policy that ranks eviction candidates. The coordinator scores every
/// candidate it owns and evicts in ascending score order: **lower = evict
/// first**, [`f32::INFINITY`] = pinned (never evict).
///
/// This trait is a sibling of [`AdmissionPolicy`] and [`ChunkingPolicy`]:
/// pure scoring, `Send + Sync`, no IO, no mutable state in the hot path.
/// Stateful policies (e.g. the future `ReuseDistancePolicy` from KVFlow-lite)
/// keep their state inside the impl struct via `Mutex` / `RwLock` and still
/// expose this stateless interface.
pub trait EvictionPolicy: Send + Sync {
    /// Score a single candidate against the current scheduler signals.
    ///
    /// Lower scores are evicted first. Return [`f32::INFINITY`] for pinned
    /// candidates (refcount > 0, in-flight requests). Return values are
    /// finite floats elsewhere; policies should avoid NaN.
    fn score(&self, candidate: EvictionCandidate, signals: SchedulerSignals) -> f32;
}

/// Plain LRU: oldest `last_access_step` evicts first. Pinned candidates are
/// protected. Mirrors vLLM's `FreeKVCacheBlockQueue` recency-only ordering
/// (cached blocks tail-pushed at release; popped from the head). The
/// closest analogue in SGLang is the fallback path inside
/// `radix_cache.RadixCache.evict()` when no priority strategy is set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LruEviction;

impl EvictionPolicy for LruEviction {
    fn score(&self, candidate: EvictionCandidate, _signals: SchedulerSignals) -> f32 {
        if candidate.pinned {
            return f32::INFINITY;
        }
        // Lower last_access_step ⇒ older ⇒ smaller score ⇒ evict first.
        candidate.last_access_step as f32
    }
}

/// LRU biased by reuse count: blocks that were hit often survive longer
/// even when they are slightly older. Lifts vLLM's recency-only baseline
/// with KVFlow-style temporal-locality × hit-count weighting (KVFlow §3,
/// arXiv:2507.07400).
///
/// `recency_weight` and `hit_weight` are taste knobs. The default
/// `Default` impl uses `(1.0, 8.0)`: a hit is worth eight steps of recency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReuseBiasedLru {
    pub recency_weight: f32,
    pub hit_weight: f32,
}

impl Default for ReuseBiasedLru {
    fn default() -> Self {
        Self {
            recency_weight: 1.0,
            hit_weight: 8.0,
        }
    }
}

impl EvictionPolicy for ReuseBiasedLru {
    fn score(&self, candidate: EvictionCandidate, _signals: SchedulerSignals) -> f32 {
        if candidate.pinned {
            return f32::INFINITY;
        }
        let recency = candidate.last_access_step as f32 * self.recency_weight;
        let reuse = candidate.hit_count as f32 * self.hit_weight;
        recency + reuse
    }
}

/// Two-tier eviction: candidates with `hit_count >= hit_threshold` are
/// pinned (never evicted by this policy). Below the threshold, falls back
/// to plain LRU. Mirrors SGLang's `write_through_selective` heuristic
/// (`hiradix_cache.py`, `write_through_threshold` default = 2): only blocks
/// with at least two hits are considered "warm" and worth protecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HitCountLru {
    pub hit_threshold: u32,
}

impl Default for HitCountLru {
    fn default() -> Self {
        Self { hit_threshold: 2 }
    }
}

impl EvictionPolicy for HitCountLru {
    fn score(&self, candidate: EvictionCandidate, _signals: SchedulerSignals) -> f32 {
        if candidate.pinned {
            return f32::INFINITY;
        }
        if candidate.hit_count >= self.hit_threshold {
            return f32::INFINITY;
        }
        candidate.last_access_step as f32
    }
}

/// LRU with a session-affinity bonus: when the coordinator is scoring
/// candidates on behalf of an incoming request whose
/// [`SchedulerSignals::session_affinity_slot`] is `Some(slot)`, candidates
/// living in that slot get `affinity_bonus` added to their score so they
/// outrank cold candidates and survive eviction.
///
/// This is the default policy that ships with the Tiered KV Cache project
/// (P2): it satisfies `agent-first-architecture.md::B3`'s "warm
/// (session-continuation) requests do not get starved" requirement while
/// remaining a pure scoring function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionBiasedLru {
    pub affinity_bonus: f32,
}

impl Default for SessionBiasedLru {
    fn default() -> Self {
        // Empirically: should be larger than the largest realistic
        // last_access_step delta within a single decode burst (~10 ks) so a
        // matching slot wins regardless of how stale its access timestamp is.
        Self {
            affinity_bonus: 1.0e6,
        }
    }
}

impl EvictionPolicy for SessionBiasedLru {
    fn score(&self, candidate: EvictionCandidate, signals: SchedulerSignals) -> f32 {
        if candidate.pinned {
            return f32::INFINITY;
        }
        let mut score = candidate.last_access_step as f32;
        if let Some(affinity_slot) = signals.session_affinity_slot
            && candidate.slot as usize == affinity_slot
        {
            score += self.affinity_bonus;
        }
        score
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

    // ──────────────────────────────────────────────────────────────────
    // EvictionPolicy
    // ──────────────────────────────────────────────────────────────────

    fn cold_signals() -> SchedulerSignals {
        SchedulerSignals::default()
    }

    fn warm_signals(slot: usize) -> SchedulerSignals {
        SchedulerSignals {
            session_affinity_slot: Some(slot),
            ..SchedulerSignals::default()
        }
    }

    #[test]
    fn eviction_candidate_new_defaults() {
        let c = EvictionCandidate::new(7, 42);
        assert_eq!(c.slot, 7);
        assert_eq!(c.last_access_step, 42);
        assert_eq!(c.tokens, 0);
        assert_eq!(c.hit_count, 0);
        assert_eq!(c.prefix_depth, 0);
        assert!(!c.pinned);
    }

    #[test]
    fn lru_eviction_orders_by_recency() {
        // Three candidates: oldest first should evict first.
        let policy = LruEviction;
        let oldest = EvictionCandidate::new(0, 10);
        let middle = EvictionCandidate::new(1, 30);
        let newest = EvictionCandidate::new(2, 50);

        let s_old = policy.score(oldest, cold_signals());
        let s_mid = policy.score(middle, cold_signals());
        let s_new = policy.score(newest, cold_signals());

        assert!(s_old < s_mid);
        assert!(s_mid < s_new);
    }

    #[test]
    fn lru_eviction_pins_in_use_candidates() {
        // pinned == true → INFINITY → never evicted.
        let policy = LruEviction;
        let pinned = EvictionCandidate {
            pinned: true,
            ..EvictionCandidate::new(0, 0)
        };
        assert_eq!(policy.score(pinned, cold_signals()), f32::INFINITY);
    }

    #[test]
    fn all_default_policies_pin_in_use_candidates() {
        // Every shipped policy must honor the pinned invariant.
        let pinned = EvictionCandidate {
            pinned: true,
            slot: 0,
            tokens: 16,
            last_access_step: 0,
            hit_count: 0,
            prefix_depth: 0,
        };
        let signals = cold_signals();

        assert_eq!(LruEviction.score(pinned, signals), f32::INFINITY);
        assert_eq!(
            ReuseBiasedLru::default().score(pinned, signals),
            f32::INFINITY
        );
        assert_eq!(HitCountLru::default().score(pinned, signals), f32::INFINITY);
        assert_eq!(
            SessionBiasedLru::default().score(pinned, signals),
            f32::INFINITY
        );
    }

    #[test]
    fn reuse_biased_lru_protects_hot_blocks() {
        // Two candidates: one is older but hit many times; should outscore
        // the slightly newer cold one.
        let policy = ReuseBiasedLru::default(); // recency=1, hit=8
        let hot_old = EvictionCandidate {
            hit_count: 5, // adds 40 to score
            ..EvictionCandidate::new(0, 100)
        };
        let cold_new = EvictionCandidate::new(1, 120); // score = 120

        let s_hot = policy.score(hot_old, cold_signals());
        let s_cold = policy.score(cold_new, cold_signals());

        // hot_old: 100 + 40 = 140; cold_new: 120 + 0 = 120
        assert!(s_hot > s_cold, "hot {s_hot} should outscore cold {s_cold}");
    }

    #[test]
    fn hit_count_lru_pins_above_threshold() {
        // hit_count >= threshold → INFINITY (warm pin).
        // hit_count < threshold → falls back to LRU.
        let policy = HitCountLru { hit_threshold: 2 };
        let warm = EvictionCandidate {
            hit_count: 2,
            ..EvictionCandidate::new(0, 100)
        };
        let single_hit = EvictionCandidate {
            hit_count: 1,
            ..EvictionCandidate::new(1, 50)
        };
        let no_hits = EvictionCandidate::new(2, 30);

        assert_eq!(policy.score(warm, cold_signals()), f32::INFINITY);

        // Both below-threshold candidates fall back to LRU; older = lower.
        let s_hit = policy.score(single_hit, cold_signals());
        let s_none = policy.score(no_hits, cold_signals());
        assert!(s_none < s_hit);
        assert!(s_hit.is_finite());
        assert!(s_none.is_finite());
    }

    #[test]
    fn session_biased_lru_protects_affinity_slot() {
        // When signals.session_affinity_slot points at slot 3, candidates in
        // slot 3 get the affinity bonus and outscore older non-affinity ones.
        let policy = SessionBiasedLru::default();

        // Newer cold candidate in slot 7.
        let cold_new = EvictionCandidate::new(7, 1_000);
        // Older candidate in the affinity slot (3).
        let warm_old = EvictionCandidate::new(3, 100);

        let signals = warm_signals(3);
        let s_cold = policy.score(cold_new, signals);
        let s_warm = policy.score(warm_old, signals);

        // warm_old gets affinity_bonus (1e6) added → outscores cold_new.
        assert!(
            s_warm > s_cold,
            "session-biased warm {s_warm} should outscore cold {s_cold}"
        );
    }

    #[test]
    fn session_biased_lru_falls_back_to_lru_without_affinity() {
        // When signals.session_affinity_slot is None, this is just LRU.
        let policy = SessionBiasedLru::default();
        let older = EvictionCandidate::new(0, 10);
        let newer = EvictionCandidate::new(1, 50);

        let s_old = policy.score(older, cold_signals());
        let s_new = policy.score(newer, cold_signals());
        assert!(s_old < s_new);
    }
}
