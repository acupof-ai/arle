//! Pure CPU scheduling policy for the Metal backend hot path.
//!
//! This module stays self-contained so it can be tested on machines without
//! Apple GPU access, but the runtime executes its output directly: one
//! decode-first `MetalScheduleStep` per tick plus an optional prefill chunk.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::events::{EngineEvent, EventSink, NoopEventSink};
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
    /// Maximum number of requests admitted into the running set at once.
    pub max_running_requests: usize,
    /// Shared token budget per scheduler tick across decode and prefill work.
    pub max_batch_tokens: usize,
}

impl Default for MetalSchedulerConfig {
    fn default() -> Self {
        Self {
            max_running_requests: 4,
            max_batch_tokens: 512,
        }
    }
}

impl MetalSchedulerConfig {
    /// Validate runtime configuration.
    pub fn validate(&self) -> Result<(), MetalSchedulerError> {
        if self.max_running_requests == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "max_running_requests must be >= 1".to_string(),
            ));
        }
        if self.max_batch_tokens == 0 {
            return Err(MetalSchedulerError::InvalidConfig(
                "max_batch_tokens must be >= 1".to_string(),
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
    EmptyPrompt,
    UnknownRequest(RequestId),
}

impl fmt::Display for MetalSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
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

/// Scheduler work for one tick. Decode always runs before prefill when both
/// are present so runtime priority stays explicit and single-sourced here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalScheduleStep {
    pub decode: Option<MetalDecodeBatch>,
    pub prefill: Option<MetalPrefillChunk>,
}

impl MetalScheduleStep {
    pub fn is_idle(&self) -> bool {
        self.decode.is_none() && self.prefill.is_none()
    }
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

    fn prefill_chunk_budget(&self, decode_count: usize) -> usize {
        self.config.max_batch_tokens.saturating_sub(decode_count)
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

    /// Emit the next scheduling step.
    ///
    /// The returned step follows decode-priority semantics:
    /// decode work is always emitted before any new or continuing prefill work.
    /// Prefill is chunked so it can be interleaved with decode.
    pub fn step(&mut self, runtime_states: &[MetalRuntimeRequestState]) -> MetalScheduleStep {
        self.validate_runtime_snapshots(runtime_states);
        let decode_batch = self.build_decode_batch(runtime_states);
        let prefill_chunk = self.build_prefill_chunk(runtime_states);
        let decode = (!decode_batch.req_ids.is_empty()).then_some(decode_batch);
        MetalScheduleStep {
            decode,
            prefill: prefill_chunk,
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

    /// Number of running requests, including prefilling and decoding requests.
    pub fn running_len(&self) -> usize {
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
        let mut states: Vec<(&MetalRequestState, u32)> = Vec::new();
        for state in self.requests.values() {
            if !state.admitted {
                continue;
            }

            let runtime = runtime_state(runtime_states, state.req_id).unwrap_or_else(|| {
                panic!(
                    "Metal scheduler contract violation: admitted request {:?} missing runtime snapshot",
                    state.req_id
                )
            });

            if runtime.phase != MetalRequestPhase::Decoding {
                continue;
            }

            let last_token = runtime.last_token.unwrap_or_else(|| {
                panic!(
                    "Metal scheduler contract violation: admitted decode request {:?} missing last_token",
                    state.req_id
                )
            });
            states.push((state, last_token));
        }
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
        let decode_count = Self::count_decode_runtime(runtime_states);
        let chunk_cap = self.prefill_chunk_budget(decode_count);
        if chunk_cap == 0 {
            return None;
        }
        let (req_id, newly_admitted) =
            if let Some(req_id) = self.find_prefilling_request(runtime_states) {
                (req_id, false)
            } else if self.running_len() < self.config.max_running_requests {
                (self.admit_next_waiting_request()?, true)
            } else {
                return None;
            };

        let (prompt_len, prompt_start, prompt_end, input_tokens, emit_prefill_started) = {
            let state = self.requests.get_mut(&req_id)?;
            let prompt_len = state.prompt_len();
            let prompt_start = if newly_admitted {
                0
            } else {
                runtime_state(runtime_states, req_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "Metal scheduler contract violation: admitted request {:?} missing runtime snapshot",
                            req_id
                        )
                    })
                    .prompt_progress
                    .min(prompt_len)
            };
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

    fn validate_runtime_snapshots(&self, runtime_states: &[MetalRuntimeRequestState]) {
        for state in self.requests.values().filter(|state| state.admitted) {
            let runtime = runtime_state(runtime_states, state.req_id).unwrap_or_else(|| {
                panic!(
                    "Metal scheduler contract violation: admitted request {:?} missing runtime snapshot",
                    state.req_id
                )
            });

            assert!(
                !(runtime.phase == MetalRequestPhase::Decoding && runtime.last_token.is_none()),
                "Metal scheduler contract violation: admitted decode request {:?} missing last_token",
                state.req_id
            );
        }
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

    fn count_decode_runtime(runtime_states: &[MetalRuntimeRequestState]) -> usize {
        runtime_states
            .iter()
            .filter(|runtime| runtime.phase == MetalRequestPhase::Decoding)
            .count()
    }
}

fn runtime_state(
    runtime_states: &[MetalRuntimeRequestState],
    req_id: RequestId,
) -> Option<&MetalRuntimeRequestState> {
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

    fn make_scheduler(max_running_requests: usize, max_batch_tokens: usize) -> MetalScheduler {
        MetalScheduler::new(MetalSchedulerConfig {
            max_running_requests,
            max_batch_tokens,
        })
        .expect("config should be valid")
    }

    fn make_scheduler_with_event_sink(
        max_running_requests: usize,
        max_batch_tokens: usize,
        event_sink: Arc<dyn EventSink>,
    ) -> MetalScheduler {
        MetalScheduler::with_event_sink(
            MetalSchedulerConfig {
                max_running_requests,
                max_batch_tokens,
            },
            event_sink,
        )
        .expect("config should be valid")
    }

    fn expect_prefill_only(step: MetalScheduleStep) -> MetalPrefillChunk {
        assert!(
            step.decode.is_none(),
            "expected no decode work, got {step:?}"
        );
        step.prefill.expect("expected prefill work")
    }

    fn expect_decode_only(step: MetalScheduleStep) -> MetalDecodeBatch {
        assert!(
            step.prefill.is_none(),
            "expected no prefill work, got {step:?}"
        );
        step.decode.expect("expected decode work")
    }

    fn expect_decode_then_prefill(
        step: MetalScheduleStep,
    ) -> (MetalDecodeBatch, MetalPrefillChunk) {
        (
            step.decode.expect("expected decode work"),
            step.prefill.expect("expected prefill work"),
        )
    }

    #[test]
    fn decode_and_prefill_share_the_same_token_budget() {
        let mut sched = make_scheduler(2, 4);
        let req0 = sched
            .submit(vec![1, 2, 3, 4], 8, MetalRequestPriority::Normal)
            .expect("submit req0");
        let req1 = sched
            .submit(vec![5, 6, 7, 8, 9], 8, MetalRequestPriority::Normal)
            .expect("submit req1");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req0);
        assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);
        assert_eq!(prefill.prompt_start, 0);
        assert_eq!(prefill.prompt_end, 4);

        let (decode, prefill) = expect_decode_then_prefill(sched.step(&[rt(
            req0,
            MetalRequestPhase::Decoding,
            4,
            Some(11),
        )]));
        assert_eq!(decode.req_ids, vec![req0]);
        assert_eq!(decode.input_tokens, vec![11]);
        assert_eq!(prefill.req_id, req1);
        assert_eq!(prefill.input_tokens, vec![5, 6, 7]);
        assert_eq!(prefill.prompt_start, 0);
        assert_eq!(prefill.prompt_end, 3);
    }

    #[test]
    fn decode_can_consume_the_entire_tick_budget() {
        let mut sched = make_scheduler(2, 1);
        let req0 = sched
            .submit(vec![1], 4, MetalRequestPriority::Normal)
            .expect("submit req0");
        let req1 = sched
            .submit(vec![5, 6, 7], 4, MetalRequestPriority::Normal)
            .expect("submit req1");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req0);

        let decode =
            expect_decode_only(sched.step(&[rt(req0, MetalRequestPhase::Decoding, 1, Some(11))]));
        assert_eq!(decode.req_ids, vec![req0]);
        assert_eq!(decode.input_tokens, vec![11]);
        assert!(sched.is_waiting(req1));
    }

    #[test]
    fn prefills_are_chunked_until_complete() {
        let mut sched = make_scheduler(1, 3);
        let req = sched
            .submit(
                vec![10, 11, 12, 13, 14, 15, 16, 17],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![10, 11, 12]);
        assert_eq!(prefill.prompt_start, 0);
        assert_eq!(prefill.prompt_end, 3);

        let prefill =
            expect_prefill_only(sched.step(&[rt(req, MetalRequestPhase::Prefilling, 3, None)]));
        assert_eq!(prefill.input_tokens, vec![13, 14, 15]);
        assert_eq!(prefill.prompt_start, 3);
        assert_eq!(prefill.prompt_end, 6);

        let prefill =
            expect_prefill_only(sched.step(&[rt(req, MetalRequestPhase::Prefilling, 6, None)]));
        assert_eq!(prefill.input_tokens, vec![16, 17]);
        assert_eq!(prefill.prompt_start, 6);
        assert_eq!(prefill.prompt_end, 8);
        assert_eq!(prefill.prompt_len, 8);

        let batch =
            expect_decode_only(sched.step(&[rt(req, MetalRequestPhase::Decoding, 8, Some(77))]));
        assert_eq!(batch.req_ids, vec![req]);
        assert_eq!(batch.input_tokens, vec![77]);
    }

    #[test]
    fn finishing_a_request_releases_the_slot_for_waiting_work() {
        let mut sched = make_scheduler(1, 4);
        let req0 = sched
            .submit(vec![1, 2, 3, 4], 1, MetalRequestPriority::Normal)
            .expect("submit req0");
        let req1 = sched
            .submit(vec![5, 6, 7, 8], 1, MetalRequestPriority::Normal)
            .expect("submit req1");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req0);
        assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);

        assert!(sched.finish_request(req0, Some(InferenceMode::Decode)));
        assert_eq!(sched.running_len(), 0);
        assert_eq!(sched.waiting_len(), 1);

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req1);
        assert_eq!(prefill.input_tokens, vec![5, 6, 7, 8]);
    }

    #[test]
    fn metal_scheduler_emits_lifecycle_events_for_successful_request() {
        let sink = Arc::new(RecordingEventSink::default());
        let mut sched = make_scheduler_with_event_sink(1, 4, sink.clone());
        let req = sched
            .submit(vec![1, 2, 3, 4], 1, MetalRequestPriority::Normal)
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![1, 2, 3, 4]);

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
        let mut sched = make_scheduler(1, 4);
        let req = sched
            .submit(vec![1, 2, 3, 4], 3, MetalRequestPriority::Normal)
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.prompt_end, 4);

        let batch =
            expect_decode_only(sched.step(&[rt(req, MetalRequestPhase::Decoding, 4, Some(77))]));
        assert_eq!(batch.req_ids, vec![req]);
        assert_eq!(batch.input_tokens, vec![77]);
    }

    #[test]
    fn decode_batch_uses_runtime_last_token_for_input() {
        let mut sched = make_scheduler(1, 4);
        let prompt = vec![1u32, 2, 3, 4];
        let req = sched
            .submit(prompt.clone(), 3, MetalRequestPriority::Normal)
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.prompt_end, 4);

        let batch =
            expect_decode_only(sched.step(&[rt(req, MetalRequestPhase::Decoding, 4, Some(99))]));
        assert_eq!(batch.req_ids, vec![req]);
        assert_eq!(batch.input_tokens, vec![99]);
    }

    #[test]
    #[should_panic(expected = "Metal scheduler contract violation: admitted decode request")]
    fn decode_batch_requires_runtime_last_token_for_admitted_requests() {
        let mut sched = make_scheduler(1, 4);
        let req = sched
            .submit(vec![10, 11, 12, 13], 3, MetalRequestPriority::Normal)
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.prompt_end, 4);

        assert!(
            sched
                .step(&[rt(req, MetalRequestPhase::Decoding, 4, None)])
                .is_idle()
        );
    }

    #[test]
    #[should_panic(expected = "Metal scheduler contract violation: admitted request")]
    fn admitted_requests_require_runtime_snapshots() {
        let mut sched = make_scheduler(1, 4);
        let req = sched
            .submit(vec![21, 22, 23, 24], 3, MetalRequestPriority::Normal)
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![21, 22, 23, 24]);

        let _ = sched.step(&[]);
    }

    #[test]
    fn waiting_request_may_be_rewritten_before_admission() {
        let mut sched = make_scheduler(1, 4);
        let req = sched
            .submit(vec![1, 2, 3, 4, 5, 6], 2, MetalRequestPriority::Normal)
            .expect("submit");

        sched
            .rewrite_waiting_prompt(req, vec![7, 8, 9, 10])
            .expect("rewrite waiting prompt");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![7, 8, 9, 10]);
    }

    #[test]
    fn metal_scheduler_emits_prefill_started_once_for_chunked_request() {
        let sink = Arc::new(RecordingEventSink::default());
        let mut sched = make_scheduler_with_event_sink(1, 3, sink.clone());
        let req = sched
            .submit(
                vec![10, 11, 12, 13, 14, 15, 16, 17],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![10, 11, 12]);
        let prefill =
            expect_prefill_only(sched.step(&[rt(req, MetalRequestPhase::Prefilling, 3, None)]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![13, 14, 15]);
        let prefill =
            expect_prefill_only(sched.step(&[rt(req, MetalRequestPhase::Prefilling, 6, None)]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.input_tokens, vec![16, 17]);

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
        let mut sched = make_scheduler(1, 4);
        let req = sched
            .submit(
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                4,
                MetalRequestPriority::Normal,
            )
            .expect("submit");

        let prefill = expect_prefill_only(sched.step(&[]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.prompt_start, 0);
        assert_eq!(prefill.prompt_end, 4);

        let prefill =
            expect_prefill_only(sched.step(&[rt(req, MetalRequestPhase::Prefilling, 4, None)]));
        assert_eq!(prefill.req_id, req);
        assert_eq!(prefill.prompt_start, 4);
        assert_eq!(prefill.prompt_end, 8);
    }
}
