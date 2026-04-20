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
use std::sync::Arc;

use crate::events::{EngineEvent, EventSink, NoopEventSink};
use crate::scheduler::policy::{
    AdmissionPolicy, ChunkingPolicy, DecodeAwareChunking, QueueBoundAdmission, SchedulerSignals,
};
use crate::types::{InferenceMode, RequestEventKind, RequestId};

/// Request priority used by the Metal scheduler.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum MetalRequestPriority {
    Low,
    #[default]
    Normal,
    High,
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
    /// When true, disable DFlash whenever >1 slot is open (legacy behavior,
    /// retained for A/B bench and kill-switch). When false (default),
    /// concurrent DFlash-enabled rows are batched through
    /// `execute_qwen35_dflash_packed_batch`; rows that must plain-decode
    /// this tick still serialize via `execute_decode_single`.
    ///
    /// Flag flipped to false 2026-04-19 after bench-proving concurrent
    /// DFlash beats the legacy downgrade at c≥2 (see
    /// `docs/experience/wins/2026-04-19-metal-qwen35-concurrent-dflash-default-on.md`).
    /// Admission was flipped in parallel at
    /// `runtime::admit_request` — all Qwen3.5 requests now acquire DFlash
    /// state regardless of `active` occupancy.
    pub metal_dflash_concurrency_off: bool,
}

impl Default for MetalSchedulerConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 4,
            prefill_chunk_size: 512,
            decode_active_prefill_cap: 128,
            max_waiting_requests: 256,
            metal_dflash_concurrency_off: false,
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

/// Runtime-authored request lifecycle snapshot used by the scheduler when it
/// needs to pick the next chunk. The scheduler treats this as read-only
/// authority for phase/progress and keeps only queueing/ordering state itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalRuntimeRequestState {
    pub req_id: RequestId,
    pub phase: MetalRequestPhase,
    pub prompt_progress: usize,
    pub last_token: Option<u32>,
}

/// Scheduling error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalSchedulerError {
    InvalidConfig(String),
    QueueFull,
    EmptyPrompt,
    UnknownRequest(RequestId),
}

impl fmt::Display for MetalSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::QueueFull => write!(f, "waiting queue is full"),
            Self::EmptyPrompt => write!(f, "prompt_tokens must not be empty"),
            Self::UnknownRequest(req_id) => write!(f, "unknown request {req_id:?}"),
        }
    }
}

impl std::error::Error for MetalSchedulerError {}

#[derive(Clone, Debug)]
struct MetalRequestState {
    req_id: RequestId,
    prompt_tokens: Vec<u32>,
    priority: MetalRequestPriority,
    arrival_order: u64,
    admitted: bool,
}

impl MetalRequestState {
    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

/// Decode batch emitted by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDecodeBatch {
    pub req_ids: Vec<RequestId>,
    pub input_tokens: Vec<u32>,
}

/// Prefill chunk emitted by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalPrefillChunk {
    pub req_id: RequestId,
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
    waiting: Vec<RequestId>,
    requests: HashMap<RequestId, MetalRequestState>,
    event_sink: Arc<dyn EventSink>,
}

impl MetalScheduler {
    fn emit_event(&self, req_id: RequestId, kind: RequestEventKind, mode: Option<InferenceMode>) {
        self.event_sink.emit(&EngineEvent {
            request_id: req_id,
            kind,
            mode,
        });
    }

    fn admission_allows(&self, queued_requests: usize) -> bool {
        if self.config.max_waiting_requests == 0 {
            return true;
        }

        QueueBoundAdmission {
            max_queued_requests: self.config.max_waiting_requests,
        }
        .allow(SchedulerSignals::queue_state(
            queued_requests,
            self.requests.values().filter(|req| req.admitted).count(),
        ))
    }

    fn prefill_chunk_budget(&self, decode_count: usize) -> usize {
        DecodeAwareChunking {
            decode_active_chunk: self.config.decode_active_prefill_cap,
            idle_chunk: self.config.prefill_chunk_size,
        }
        .next_chunk_size(
            InferenceMode::Prefill,
            SchedulerSignals::queue_state(self.waiting.len(), decode_count),
        )
        .max(1)
        .min(self.config.prefill_chunk_size)
    }

    /// Create a new scheduler with the provided config.
    pub fn new(config: MetalSchedulerConfig) -> Result<Self, MetalSchedulerError> {
        Self::with_event_sink(config, Arc::new(NoopEventSink))
    }

    /// Create a new scheduler with an explicit observability sink.
    pub fn with_event_sink(
        config: MetalSchedulerConfig,
        event_sink: Arc<dyn EventSink>,
    ) -> Result<Self, MetalSchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            next_req_id: 0,
            next_arrival_order: 0,
            waiting: Vec::new(),
            requests: HashMap::new(),
            event_sink,
        })
    }

    /// Submit a new prompt into the waiting queue.
    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: MetalRequestPriority,
    ) -> Result<RequestId, MetalSchedulerError> {
        if prompt_tokens.is_empty() {
            return Err(MetalSchedulerError::EmptyPrompt);
        }
        if max_tokens == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "max_tokens must be >= 1".to_string(),
            ));
        }
        if !self.admission_allows(self.waiting.len()) {
            return Err(MetalSchedulerError::QueueFull);
        }

        let req_id = RequestId(self.next_req_id);
        self.next_req_id += 1;
        let arrival_order = self.next_arrival_order;
        self.next_arrival_order += 1;

        self.requests.insert(
            req_id,
            MetalRequestState {
                req_id,
                prompt_tokens,
                priority,
                arrival_order,
                admitted: false,
            },
        );
        self.waiting.push(req_id);
        self.emit_event(req_id, RequestEventKind::Enqueued, None);
        Ok(req_id)
    }

    /// Emit the next scheduling decision.
    ///
    /// The returned decision follows decode-priority semantics:
    /// decode work is always emitted before any new or continuing prefill work.
    /// Prefill is chunked so it can be interleaved with decode.
    pub fn step(&mut self, runtime_states: &[MetalRuntimeRequestState]) -> MetalScheduleDecision {
        let decode_batch = self.build_decode_batch(runtime_states);
        let prefill_chunk = self.build_prefill_chunk(runtime_states);

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

    /// Explicitly finish a request, releasing all scheduler bookkeeping.
    pub fn finish_request(&mut self, req_id: RequestId, mode: Option<InferenceMode>) -> bool {
        let removed = self.remove_request(req_id);
        if removed.is_some() {
            self.emit_event(req_id, RequestEventKind::Completed, mode);
            true
        } else {
            false
        }
    }

    /// Whether the request is still queued and has not been admitted yet.
    pub fn is_waiting(&self, req_id: RequestId) -> bool {
        self.requests
            .get(&req_id)
            .is_some_and(|state| !state.admitted)
    }

    /// Number of waiting requests.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Number of active requests, including prefilling and decoding requests.
    pub fn active_len(&self) -> usize {
        self.requests
            .values()
            .filter(|state| state.admitted)
            .count()
    }

    pub fn rewrite_waiting_prompt(
        &mut self,
        req_id: RequestId,
        prompt_tokens: Vec<u32>,
    ) -> Result<(), MetalSchedulerError> {
        if prompt_tokens.is_empty() {
            return Err(MetalSchedulerError::EmptyPrompt);
        }

        let state = self
            .requests
            .get_mut(&req_id)
            .ok_or(MetalSchedulerError::UnknownRequest(req_id))?;

        state.prompt_tokens = prompt_tokens;
        Ok(())
    }

    fn build_decode_batch(&self, runtime_states: &[MetalRuntimeRequestState]) -> MetalDecodeBatch {
        let mut states: Vec<(&MetalRequestState, u32)> = self
            .requests
            .values()
            .filter_map(|state| {
                let runtime = runtime_state(runtime_states, state.req_id)?;
                (state.admitted
                    && runtime.phase == MetalRequestPhase::Decoding
                    && runtime.last_token.is_some())
                .then_some((state, runtime.last_token?))
            })
            .collect();
        states.sort_by(|(a, _), (b, _)| compare_decode_order(a, b));

        let req_ids = states.iter().map(|(state, _)| state.req_id).collect();
        let input_tokens = states.into_iter().map(|(_, token)| token).collect();

        MetalDecodeBatch {
            req_ids,
            input_tokens,
        }
    }

    fn build_prefill_chunk(
        &mut self,
        runtime_states: &[MetalRuntimeRequestState],
    ) -> Option<MetalPrefillChunk> {
        let req_id = if let Some(req_id) = self.find_prefilling_request(runtime_states) {
            req_id
        } else if self.active_len() < self.config.max_active_requests {
            self.admit_next_waiting_request()?
        } else {
            return None;
        };

        let chunk_cap = self.prefill_chunk_budget(self.count_decode_runtime(runtime_states));

        let (prompt_len, prompt_start, prompt_end, input_tokens, emit_prefill_started) = {
            let state = self.requests.get_mut(&req_id)?;
            let prompt_len = state.prompt_len();
            let prompt_start = runtime_state(runtime_states, req_id)
                .map(|runtime| runtime.prompt_progress)
                .unwrap_or(0)
                .min(prompt_len);
            let prompt_end = (prompt_start + chunk_cap).min(prompt_len);
            let input_tokens = state.prompt_tokens[prompt_start..prompt_end].to_vec();

            (
                prompt_len,
                prompt_start,
                prompt_end,
                input_tokens,
                prompt_start == 0,
            )
        };

        if emit_prefill_started {
            self.emit_event(
                req_id,
                RequestEventKind::PrefillStarted,
                Some(InferenceMode::Prefill),
            );
        }

        Some(MetalPrefillChunk {
            req_id,
            input_tokens,
            prompt_start,
            prompt_end,
            prompt_len,
        })
    }

    fn find_prefilling_request(
        &self,
        runtime_states: &[MetalRuntimeRequestState],
    ) -> Option<RequestId> {
        self.requests
            .values()
            .filter(|state| {
                state.admitted
                    && runtime_state(runtime_states, state.req_id)
                        .is_some_and(|runtime| runtime.phase == MetalRequestPhase::Prefilling)
            })
            .min_by(|a, b| compare_prefill_order(a, b))
            .map(|state| state.req_id)
    }

    fn admit_next_waiting_request(&mut self) -> Option<RequestId> {
        let best_idx = self
            .waiting
            .iter()
            .enumerate()
            .filter_map(|(idx, req_id)| self.requests.get(req_id).map(|state| (idx, state)))
            .min_by(|(_, a), (_, b)| compare_priority_waiting(a, b))
            .map(|(idx, _)| idx)?;

        let req_id = self.waiting.remove(best_idx);
        let state = self.requests.get_mut(&req_id)?;
        state.admitted = true;
        Some(req_id)
    }

    fn remove_request(&mut self, req_id: RequestId) -> Option<MetalRequestState> {
        let removed = self.requests.remove(&req_id);
        if removed.is_some() {
            if let Some(pos) = self.waiting.iter().position(|id| *id == req_id) {
                self.waiting.remove(pos);
            }
        }
        removed
    }

    fn count_decode_runtime(&self, runtime_states: &[MetalRuntimeRequestState]) -> usize {
        runtime_states
            .iter()
            .filter(|runtime| runtime.phase == MetalRequestPhase::Decoding)
            .count()
    }
}

fn runtime_state<'a>(
    runtime_states: &'a [MetalRuntimeRequestState],
    req_id: RequestId,
) -> Option<&'a MetalRuntimeRequestState> {
    runtime_states.iter().find(|state| state.req_id == req_id)
}

fn compare_priority_waiting(a: &MetalRequestState, b: &MetalRequestState) -> Ordering {
    match a.priority.cmp(&b.priority).reverse() {
        Ordering::Equal => a.arrival_order.cmp(&b.arrival_order),
        other => other,
    }
}

fn compare_prefill_order(a: &MetalRequestState, b: &MetalRequestState) -> Ordering {
    match a.arrival_order.cmp(&b.arrival_order) {
        Ordering::Equal => a.req_id.cmp(&b.req_id),
        other => other,
    }
}

fn compare_decode_order(a: &MetalRequestState, b: &MetalRequestState) -> Ordering {
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
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingEventSink {
        events: Mutex<Vec<EngineEvent>>,
    }

    impl EventSink for RecordingEventSink {
        fn emit(&self, event: &EngineEvent) {
            self.events.lock().expect("poisoned").push(event.clone());
        }
    }

    fn recorded_events(sink: &RecordingEventSink) -> Vec<EngineEvent> {
        sink.events.lock().expect("poisoned").clone()
    }

    fn rt(
        req_id: RequestId,
        phase: MetalRequestPhase,
        prompt_progress: usize,
        last_token: Option<u32>,
    ) -> MetalRuntimeRequestState {
        MetalRuntimeRequestState {
            req_id,
            phase,
            prompt_progress,
            last_token,
        }
    }

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
            metal_dflash_concurrency_off: false,
        })
        .expect("config should be valid")
    }

    fn make_scheduler_with_event_sink(
        max_active_requests: usize,
        chunk: usize,
        decode_cap: usize,
        event_sink: Arc<dyn EventSink>,
    ) -> MetalScheduler {
        MetalScheduler::with_event_sink(
            MetalSchedulerConfig {
                max_active_requests,
                prefill_chunk_size: chunk,
                decode_active_prefill_cap: decode_cap,
                max_waiting_requests: 16,
                metal_dflash_concurrency_off: false,
            },
            event_sink,
        )
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

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req0);
                assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
                assert_eq!(prefill.prompt_start, 0);
                assert_eq!(prefill.prompt_end, 4);
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        match sched.step(&[rt(req0, MetalRequestPhase::Decoding, 4, Some(11))]) {
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

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![10, 11, 12]);
                assert_eq!(prefill.prompt_start, 0);
                assert_eq!(prefill.prompt_end, 3);
            }
            other => panic!("expected chunk 1, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Prefilling, 3, None)]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.input_tokens, vec![13, 14, 15]);
                assert_eq!(prefill.prompt_start, 3);
                assert_eq!(prefill.prompt_end, 6);
            }
            other => panic!("expected chunk 2, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Prefilling, 6, None)]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.input_tokens, vec![16, 17]);
                assert_eq!(prefill.prompt_start, 6);
                assert_eq!(prefill.prompt_end, 8);
                assert_eq!(prefill.prompt_len, 8);
            }
            other => panic!("expected chunk 3, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Decoding, 8, Some(77))]) {
            MetalScheduleDecision::DecodeBatch(batch) => {
                assert_eq!(batch.req_ids, vec![req]);
                assert_eq!(batch.input_tokens, vec![77]);
            }
            other => panic!("expected decode batch after terminal prefill, got {other:?}"),
        }
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

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req0);
                assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
            }
            other => panic!("expected req0 prefill, got {other:?}"),
        }

        assert!(sched.finish_request(req0, Some(InferenceMode::Decode)));
        assert_eq!(sched.active_len(), 0);
        assert_eq!(sched.waiting_len(), 1);

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req1);
                assert_eq!(prefill.input_tokens, vec![5, 6, 7, 8]);
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

    #[test]
    fn metal_scheduler_emits_lifecycle_events_for_successful_request() {
        let sink = Arc::new(RecordingEventSink::default());
        let mut sched = make_scheduler_with_event_sink(1, 4, 4, sink.clone());
        let req = sched
            .submit(vec![1, 2, 3, 4], 1, MetalRequestPriority::Normal)
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        assert!(sched.finish_request(req, Some(InferenceMode::Decode)));

        assert_eq!(
            recorded_events(sink.as_ref()),
            vec![
                EngineEvent {
                    request_id: req,
                    kind: RequestEventKind::Enqueued,
                    mode: None,
                },
                EngineEvent {
                    request_id: req,
                    kind: RequestEventKind::PrefillStarted,
                    mode: Some(InferenceMode::Prefill),
                },
                EngineEvent {
                    request_id: req,
                    kind: RequestEventKind::Completed,
                    mode: Some(InferenceMode::Decode),
                },
            ]
        );
    }

    #[test]
    fn terminal_prefill_records_first_token_before_decode_loop() {
        let mut sched = make_scheduler(1, 4, 4);
        let req = sched
            .submit(vec![1, 2, 3, 4], 3, MetalRequestPriority::Normal)
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.prompt_end, 4);
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Decoding, 4, Some(77))]) {
            MetalScheduleDecision::DecodeBatch(batch) => {
                assert_eq!(batch.req_ids, vec![req]);
                assert_eq!(batch.input_tokens, vec![77]);
            }
            other => panic!("expected decode batch, got {other:?}"),
        }
    }

    #[test]
    fn decode_batch_uses_runtime_last_token_for_input() {
        let mut sched = make_scheduler(1, 4, 4);
        let prompt = vec![1u32, 2, 3, 4];
        let req = sched
            .submit(prompt.clone(), 3, MetalRequestPriority::Normal)
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.prompt_end, 4);
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Decoding, 4, Some(99))]) {
            MetalScheduleDecision::DecodeBatch(batch) => {
                assert_eq!(batch.req_ids, vec![req]);
                assert_eq!(batch.input_tokens, vec![99]);
            }
            other => panic!("expected decode batch, got {other:?}"),
        }
    }

    #[test]
    fn decode_batch_skips_requests_missing_runtime_last_token() {
        let mut sched = make_scheduler(1, 4, 4);
        let req = sched
            .submit(vec![10, 11, 12, 13], 3, MetalRequestPriority::Normal)
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.prompt_end, 4);
            }
            other => panic!("expected initial prefill, got {other:?}"),
        }

        assert_eq!(
            sched.step(&[rt(req, MetalRequestPhase::Decoding, 4, None)]),
            MetalScheduleDecision::Idle
        );
    }

    #[test]
    fn waiting_request_may_be_rewritten_before_admission() {
        let mut sched = make_scheduler(1, 4, 4);
        let req = sched
            .submit(vec![1, 2, 3, 4, 5, 6], 2, MetalRequestPriority::Normal)
            .expect("submit");

        sched
            .rewrite_waiting_prompt(req, vec![7, 8, 9, 10])
            .expect("rewrite waiting prompt");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![7, 8, 9, 10]);
            }
            other => panic!("expected rewritten prefill, got {other:?}"),
        }
    }

    #[test]
    fn metal_scheduler_emits_prefill_started_once_for_chunked_request() {
        let sink = Arc::new(RecordingEventSink::default());
        let mut sched = make_scheduler_with_event_sink(1, 3, 3, sink.clone());
        let req = sched
            .submit(
                vec![10, 11, 12, 13, 14, 15, 16, 17],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![10, 11, 12]);
            }
            other => panic!("expected prefill chunk, got {other:?}"),
        }
        match sched.step(&[rt(req, MetalRequestPhase::Prefilling, 3, None)]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![13, 14, 15]);
            }
            other => panic!("expected prefill chunk, got {other:?}"),
        }
        match sched.step(&[rt(req, MetalRequestPhase::Prefilling, 6, None)]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.input_tokens, vec![16, 17]);
            }
            other => panic!("expected prefill chunk, got {other:?}"),
        }

        assert_eq!(
            recorded_events(sink.as_ref()),
            vec![
                EngineEvent {
                    request_id: req,
                    kind: RequestEventKind::Enqueued,
                    mode: None,
                },
                EngineEvent {
                    request_id: req,
                    kind: RequestEventKind::PrefillStarted,
                    mode: Some(InferenceMode::Prefill),
                },
            ]
        );
    }

    #[test]
    fn runtime_prefill_progress_controls_chunk_selection() {
        let mut sched = make_scheduler(1, 4, 4);
        let req = sched
            .submit(
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        match sched.step(&[]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.prompt_start, 0);
                assert_eq!(prefill.prompt_end, 4);
            }
            other => panic!("expected first prefill, got {other:?}"),
        }

        match sched.step(&[rt(req, MetalRequestPhase::Prefilling, 4, None)]) {
            MetalScheduleDecision::PrefillChunk(prefill) => {
                assert_eq!(prefill.req_id, req);
                assert_eq!(prefill.prompt_start, 4);
                assert_eq!(prefill.prompt_end, 8);
            }
            other => panic!("expected second prefill, got {other:?}"),
        }
    }
}
