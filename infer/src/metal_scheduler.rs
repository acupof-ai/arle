//! Pure CPU scheduling skeleton for the Metal backend.
//!
//! This module is intentionally self-contained so it can be tested on machines
//! without Apple GPU access. It models the phase-1 Metal scheduling policy:
//! decode-priority, serial interleaving, and chunked prefill.
//!
//! The module does not touch `MetalBackend` yet. It is a planning / accounting
//! layer that can be wired into the real Metal execution path later.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Request priority used by the Metal scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MetalRequestPriority {
    Low,
    Normal,
    High,
}

impl Default for MetalRequestPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Scheduler configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalSchedulerConfig {
    /// Maximum number of requests admitted into the active set at once.
    pub max_active_requests: usize,
    /// Maximum tokens processed in one prefill chunk.
    pub prefill_chunk_size: usize,
    /// Prefill chunk cap when decode work is already active.
    pub decode_active_prefill_cap: usize,
    /// Maximum requests waiting in the queue.
    pub max_waiting_requests: usize,
}

impl Default for MetalSchedulerConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 4,
            prefill_chunk_size: 512,
            decode_active_prefill_cap: 128,
            max_waiting_requests: 256,
        }
    }
}

impl MetalSchedulerConfig {
    /// Validate runtime configuration.
    pub fn validate(&self) -> Result<(), MetalSchedulerError> {
        if self.max_active_requests == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "max_active_requests must be >= 1".to_string(),
            ));
        }
        if self.prefill_chunk_size == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "prefill_chunk_size must be >= 1".to_string(),
            ));
        }
        if self.decode_active_prefill_cap == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "decode_active_prefill_cap must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Request lifecycle phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalRequestPhase {
    Waiting,
    Prefilling,
    Decoding,
}

/// Scheduling error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalSchedulerError {
    InvalidConfig(String),
    QueueFull,
    EmptyPrompt,
    UnknownRequest(u64),
    WrongPhase {
        req_id: u64,
        expected: MetalRequestPhase,
        actual: MetalRequestPhase,
    },
}

impl fmt::Display for MetalSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::QueueFull => write!(f, "waiting queue is full"),
            Self::EmptyPrompt => write!(f, "prompt_tokens must not be empty"),
            Self::UnknownRequest(req_id) => write!(f, "unknown request {req_id}"),
            Self::WrongPhase {
                req_id,
                expected,
                actual,
            } => write!(
                f,
                "request {req_id} is in phase {:?}, expected {:?}",
                actual, expected
            ),
        }
    }
}

impl std::error::Error for MetalSchedulerError {}

#[derive(Clone, Debug)]
struct MetalRequestState {
    req_id: u64,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    priority: MetalRequestPriority,
    arrival_order: u64,
    phase: MetalRequestPhase,
    prefill_progress: usize,
    generated_tokens: usize,
    last_token: Option<u32>,
}

impl MetalRequestState {
    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    fn decode_input_token(&self) -> u32 {
        self.last_token
            .unwrap_or_else(|| self.prompt_tokens.last().copied().unwrap_or(0))
    }
}

/// Decode batch emitted by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDecodeBatch {
    pub req_ids: Vec<u64>,
    pub input_tokens: Vec<u32>,
}

/// Prefill chunk emitted by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalPrefillChunk {
    pub req_id: u64,
    pub input_tokens: Vec<u32>,
    pub prompt_start: usize,
    pub prompt_end: usize,
    pub prompt_len: usize,
}

/// Scheduler decision for one step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalScheduleDecision {
    DecodeBatch(MetalDecodeBatch),
    PrefillChunk(MetalPrefillChunk),
    Mixed {
        decode: MetalDecodeBatch,
        prefill: MetalPrefillChunk,
    },
    Idle,
}

/// Pure CPU scheduling skeleton for the Metal backend.
pub struct MetalScheduler {
    config: MetalSchedulerConfig,
    next_req_id: u64,
    next_arrival_order: u64,
    waiting: Vec<u64>,
    requests: HashMap<u64, MetalRequestState>,
}

impl MetalScheduler {
    /// Create a new scheduler with the provided config.
    pub fn new(config: MetalSchedulerConfig) -> Result<Self, MetalSchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            next_req_id: 0,
            next_arrival_order: 0,
            waiting: Vec::new(),
            requests: HashMap::new(),
        })
    }

    /// Submit a new prompt into the waiting queue.
    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: MetalRequestPriority,
    ) -> Result<u64, MetalSchedulerError> {
        if prompt_tokens.is_empty() {
            return Err(MetalSchedulerError::EmptyPrompt);
        }
        if max_tokens == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "max_tokens must be >= 1".to_string(),
            ));
        }
        if self.waiting.len() >= self.config.max_waiting_requests {
            return Err(MetalSchedulerError::QueueFull);
        }

        let req_id = self.next_req_id;
        self.next_req_id += 1;
        let arrival_order = self.next_arrival_order;
        self.next_arrival_order += 1;

        self.requests.insert(
            req_id,
            MetalRequestState {
                req_id,
                prompt_tokens,
                max_tokens,
                priority,
                arrival_order,
                phase: MetalRequestPhase::Waiting,
                prefill_progress: 0,
                generated_tokens: 0,
                last_token: None,
            },
        );
        self.waiting.push(req_id);
        Ok(req_id)
    }

    /// Emit the next scheduling decision.
    ///
    /// The returned decision follows decode-priority semantics:
    /// decode work is always emitted before any new or continuing prefill work.
    /// Prefill is chunked so it can be interleaved with decode.
    pub fn step(&mut self) -> MetalScheduleDecision {
        let decode_batch = self.build_decode_batch();
        let prefill_chunk = self.build_prefill_chunk();

        match (decode_batch.req_ids.is_empty(), prefill_chunk) {
            (true, Some(prefill)) => MetalScheduleDecision::PrefillChunk(prefill),
            (false, Some(prefill)) => MetalScheduleDecision::Mixed {
                decode: decode_batch,
                prefill,
            },
            (false, None) => MetalScheduleDecision::DecodeBatch(decode_batch),
            (true, None) => MetalScheduleDecision::Idle,
        }
    }

    /// Commit a sampled decode token for one request.
    ///
    /// Returns `true` if the request reaches its `max_tokens` budget and is
    /// removed from the scheduler.
    pub fn advance_decode(
        &mut self,
        req_id: u64,
        sampled_token: u32,
    ) -> Result<bool, MetalSchedulerError> {
        let state = self
            .requests
            .get_mut(&req_id)
            .ok_or(MetalSchedulerError::UnknownRequest(req_id))?;

        if state.phase != MetalRequestPhase::Decoding {
            return Err(MetalSchedulerError::WrongPhase {
                req_id,
                expected: MetalRequestPhase::Decoding,
                actual: state.phase,
            });
        }

        state.last_token = Some(sampled_token);
        state.generated_tokens += 1;

        if state.generated_tokens >= state.max_tokens {
            self.remove_request(req_id);
            return Ok(true);
        }

        Ok(false)
    }

    /// Explicitly finish a request, releasing all scheduler bookkeeping.
    pub fn finish_request(&mut self, req_id: u64) -> bool {
        self.remove_request(req_id)
    }

    /// Get the current phase of a request.
    pub fn request_phase(&self, req_id: u64) -> Option<MetalRequestPhase> {
        self.requests.get(&req_id).map(|state| state.phase)
    }

    /// Number of waiting requests.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Number of active requests, including prefilling and decoding requests.
    pub fn active_len(&self) -> usize {
        self.requests
            .values()
            .filter(|state| {
                matches!(
                    state.phase,
                    MetalRequestPhase::Prefilling | MetalRequestPhase::Decoding
                )
            })
            .count()
    }

    /// Count requests currently decoding.
    pub fn decoding_len(&self) -> usize {
        self.requests
            .values()
            .filter(|state| state.phase == MetalRequestPhase::Decoding)
            .count()
    }

    /// Count requests currently prefilling.
    pub fn prefilling_len(&self) -> usize {
        self.requests
            .values()
            .filter(|state| state.phase == MetalRequestPhase::Prefilling)
            .count()
    }

    fn build_decode_batch(&self) -> MetalDecodeBatch {
        let mut states: Vec<&MetalRequestState> = self
            .requests
            .values()
            .filter(|state| state.phase == MetalRequestPhase::Decoding)
            .collect();
        states.sort_by(compare_decode_order);

        let req_ids = states.iter().map(|state| state.req_id).collect();
        let input_tokens = states
            .iter()
            .map(|state| state.decode_input_token())
            .collect();

        MetalDecodeBatch {
            req_ids,
            input_tokens,
        }
    }

    fn build_prefill_chunk(&mut self) -> Option<MetalPrefillChunk> {
        let req_id = if let Some(req_id) = self.find_prefilling_request() {
            req_id
        } else if self.active_len() < self.config.max_active_requests {
            self.admit_next_waiting_request()?
        } else {
            return None;
        };

        let has_decode_work = self.decoding_len() > 0;
        let chunk_cap = if has_decode_work {
            self.config
                .prefill_chunk_size
                .min(self.config.decode_active_prefill_cap)
        } else {
            self.config.prefill_chunk_size
        };

        let state = self.requests.get_mut(&req_id)?;
        if state.phase == MetalRequestPhase::Waiting {
            state.phase = MetalRequestPhase::Prefilling;
        }

        let prompt_len = state.prompt_len();
        let prompt_start = state.prefill_progress;
        let prompt_end = (prompt_start + chunk_cap).min(prompt_len);
        let input_tokens = state.prompt_tokens[prompt_start..prompt_end].to_vec();
        state.prefill_progress = prompt_end;

        if state.prefill_progress >= prompt_len {
            state.phase = MetalRequestPhase::Decoding;
            state.last_token = state.prompt_tokens.last().copied();
        } else {
            state.phase = MetalRequestPhase::Prefilling;
        }

        Some(MetalPrefillChunk {
            req_id,
            input_tokens,
            prompt_start,
            prompt_end,
            prompt_len,
        })
    }

    fn find_prefilling_request(&self) -> Option<u64> {
        self.requests
            .values()
            .filter(|state| state.phase == MetalRequestPhase::Prefilling)
            .min_by(compare_prefill_order)
            .map(|state| state.req_id)
    }

    fn admit_next_waiting_request(&mut self) -> Option<u64> {
        let best_idx = self
            .waiting
            .iter()
            .enumerate()
            .filter_map(|(idx, req_id)| self.requests.get(req_id).map(|state| (idx, state)))
            .min_by(|(_, a), (_, b)| compare_priority_waiting(a, b))
            .map(|(idx, _)| idx)?;

        let req_id = self.waiting.remove(best_idx);
        let state = self.requests.get_mut(&req_id)?;
        state.phase = MetalRequestPhase::Prefilling;
        Some(req_id)
    }

    fn remove_request(&mut self, req_id: u64) -> bool {
        let removed = self.requests.remove(&req_id).is_some();
        if removed {
            if let Some(pos) = self.waiting.iter().position(|id| *id == req_id) {
                self.waiting.remove(pos);
            }
        }
        removed
    }
}

fn compare_priority_waiting(a: &MetalRequestState, b: &MetalRequestState) -> Ordering {
    match a.priority.cmp(&b.priority).reverse() {
        Ordering::Equal => a.arrival_order.cmp(&b.arrival_order),
        other => other,
    }
}

fn compare_prefill_order(a: &&MetalRequestState, b: &&MetalRequestState) -> Ordering {
    match a.arrival_order.cmp(&b.arrival_order) {
        Ordering::Equal => a.req_id.cmp(&b.req_id),
        other => other,
    }
}

fn compare_decode_order(a: &&MetalRequestState, b: &&MetalRequestState) -> Ordering {
    match a.priority.cmp(&b.priority).reverse() {
        Ordering::Equal => match a.arrival_order.cmp(&b.arrival_order) {
            Ordering::Equal => a.req_id.cmp(&b.req_id),
            other => other,
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler(
        max_active_requests: usize,
        chunk: usize,
        decode_cap: usize,
    ) -> MetalScheduler {
        MetalScheduler::new(MetalSchedulerConfig {
            max_active_requests,
            prefill_chunk_size: chunk,
            decode_active_prefill_cap: decode_cap,
            max_waiting_requests: 16,
        })
        .expect("config should be valid")
    }

    #[test]
    fn decode_beats_new_prefill_and_prefill_is_capped_when_decode_is_active() {
        let mut sched = make_scheduler(2, 4, 2);
        let req0 = sched
            .submit(vec![1, 2, 3, 4], 8, MetalRequestPriority::Normal)
            .expect("submit req0");
        let req1 = sched
            .submit(vec![5, 6, 7, 8, 9], 8, MetalRequestPriority::Normal)
            .expect("submit req1");

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req0);
                assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
                assert_eq!(sched.request_phase(req0), Some(MetalRequestPhase::Decoding));
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        assert!(!sched.advance_decode(req0, 11).expect("decode commit"));
        assert_eq!(sched.request_phase(req0), Some(MetalRequestPhase::Decoding));

        match sched.step() {
            MetalScheduleDecision::Mixed { decode, prefill } => {
                assert_eq!(decode.req_ids, vec![req0]);
                assert_eq!(decode.input_tokens, vec![11]);
                assert_eq!(prefill.req_id, req1);
                assert_eq!(prefill.input_tokens, vec![5, 6]);
                assert_eq!(prefill.prompt_start, 0);
                assert_eq!(prefill.prompt_end, 2);
            }
            other => panic!("expected mixed decode+prefill, got {other:?}"),
        }
    }

    #[test]
    fn prefills_are_chunked_until_complete() {
        let mut sched = make_scheduler(1, 3, 3);
        let req = sched
            .submit(
                vec![10, 11, 12, 13, 14, 15, 16, 17],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![10, 11, 12]);
                assert_eq!(prefill.prompt_start, 0);
                assert_eq!(prefill.prompt_end, 3);
                assert_eq!(
                    sched.request_phase(req),
                    Some(MetalRequestPhase::Prefilling)
                );
            }
            other => panic!("expected chunk 1, got {other:?}"),
        }

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.input_tokens, vec![13, 14, 15]);
                assert_eq!(prefill.prompt_start, 3);
                assert_eq!(prefill.prompt_end, 6);
                assert_eq!(
                    sched.request_phase(req),
                    Some(MetalRequestPhase::Prefilling)
                );
            }
            other => panic!("expected chunk 2, got {other:?}"),
        }

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.input_tokens, vec![16, 17]);
                assert_eq!(prefill.prompt_start, 6);
                assert_eq!(prefill.prompt_end, 8);
                assert_eq!(sched.request_phase(req), Some(MetalRequestPhase::Decoding));
            }
            other => panic!("expected chunk 3, got {other:?}"),
        }

        assert_eq!(sched.prefilling_len(), 0);
        assert_eq!(sched.decoding_len(), 1);
    }

    #[test]
    fn finishing_a_request_releases_the_slot_for_waiting_work() {
        let mut sched = make_scheduler(1, 4, 4);
        let req0 = sched
            .submit(vec![1, 2, 3, 4], 1, MetalRequestPriority::Normal)
            .expect("submit req0");
        let req1 = sched
            .submit(vec![5, 6, 7, 8], 1, MetalRequestPriority::Normal)
            .expect("submit req1");

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req0);
                assert_eq!(sched.request_phase(req0), Some(MetalRequestPhase::Decoding));
            }
            other => panic!("expected req0 prefill, got {other:?}"),
        }

        assert!(sched.advance_decode(req0, 99).expect("req0 should finish"));
        assert_eq!(sched.request_phase(req0), None);
        assert_eq!(sched.active_len(), 0);
        assert_eq!(sched.waiting_len(), 1);

        match sched.step() {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req1);
                assert_eq!(prefill.input_tokens, vec![5, 6, 7, 8]);
                assert_eq!(sched.request_phase(req1), Some(MetalRequestPhase::Decoding));
            }
            other => panic!("expected req1 to start after req0 finished, got {other:?}"),
        }
    }

    #[test]
    fn waiting_queue_rejects_when_full() {
        let mut sched = make_scheduler(1, 4, 4);
        sched.config.max_waiting_requests = 1;
        sched
            .submit(vec![1, 2, 3, 4], 1, MetalRequestPriority::Normal)
            .expect("first submit");
        assert_eq!(
            sched.submit(vec![5, 6, 7, 8], 1, MetalRequestPriority::Normal),
            Err(MetalSchedulerError::QueueFull)
        );
    }
}
