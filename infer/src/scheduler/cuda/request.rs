use fastrace::Span;
use fastrace::collector::SpanContext;
use log::warn;

use super::{CompletionStreamDelta, FinishReason, RequestPriority, TokenUsage, Tokenizer, mpsc};

#[derive(Default)]
pub(crate) struct StreamDecodeState {
    pub(crate) full_decoded: String,
    /// Number of generated tokens already decoded or dispatched for streaming.
    pub(crate) decoded_token_count: usize,
    /// Number of characters already sent to the client.
    pub(crate) sent_len: usize,
    /// Cached byte length of the decoded prefix (tokens[0..safe_point]).
    pub(crate) prefix_byte_len: usize,
}

pub(crate) enum EmitOutcome {
    Continue,
    Finished,
}

#[derive(Default)]
pub(crate) struct EmitCursor {
    dispatched_token_count: usize,
}

impl StreamDecodeState {
    pub(crate) fn emit_delta(
        &mut self,
        generated_tokens: &[u32],
        tokenizer: &Tokenizer,
        delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
        logprob: Option<f32>,
        stops: Option<&[String]>,
        prompt_tokens: usize,
    ) -> EmitOutcome {
        let n = generated_tokens.len();
        if n == 0 {
            return EmitOutcome::Continue;
        }

        let overlap = 4;
        let safe_point = self.decoded_token_count.saturating_sub(overlap);
        let Ok(new_text) = tokenizer.decode(&generated_tokens[safe_point..]) else {
            return EmitOutcome::Continue;
        };

        if safe_point > 0 {
            let prefix_len = self
                .full_decoded
                .floor_char_boundary(self.prefix_byte_len.min(self.full_decoded.len()));
            self.full_decoded.truncate(prefix_len);
            self.full_decoded.push_str(&new_text);
        } else {
            self.full_decoded = new_text;
        }

        let new_safe = n.saturating_sub(overlap);
        if new_safe > 0 {
            let suffix = tokenizer
                .decode(&generated_tokens[new_safe..])
                .unwrap_or_default();
            let prefix_len = self.full_decoded.len().saturating_sub(suffix.len());
            self.prefix_byte_len = self.full_decoded.floor_char_boundary(prefix_len);
        } else {
            self.prefix_byte_len = 0;
        }

        self.decoded_token_count = n;

        if let Some(stops) = stops {
            match check_stop_sequences(&self.full_decoded, stops) {
                StopCheckResult::StopFound { stop_pos } => {
                    if stop_pos > self.sent_len {
                        let _ = delta_tx.send(CompletionStreamDelta {
                            text_delta: self.full_decoded[self.sent_len..stop_pos].to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob,
                            // TODO Phase-2 follow-up: per-delta token_ids
                            // would require correlating sent text byte
                            // ranges back to token boundaries through
                            // `decoded_token_count`. The cumulative list
                            // still rides on `send_finish` below.
                            token_ids: Vec::new(),
                        });
                    }
                    self.sent_len = stop_pos;
                    self.send_finish(
                        delta_tx,
                        prompt_tokens,
                        generated_tokens.len(),
                        FinishReason::Stop,
                        generated_tokens,
                    );
                    EmitOutcome::Finished
                }
                StopCheckResult::NoStop { safe_len } => {
                    if safe_len > self.sent_len {
                        let _ = delta_tx.send(CompletionStreamDelta {
                            text_delta: self.full_decoded[self.sent_len..safe_len].to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob,
                            // TODO Phase-2 follow-up: see above.
                            token_ids: Vec::new(),
                        });
                        self.sent_len = safe_len;
                    }
                    EmitOutcome::Continue
                }
            }
        } else {
            if self.full_decoded.len() > self.sent_len {
                let start = self.full_decoded.floor_char_boundary(self.sent_len);
                if start < self.full_decoded.len() {
                    let _ = delta_tx.send(CompletionStreamDelta {
                        text_delta: self.full_decoded[start..].to_string(),
                        finish_reason: None,
                        usage: None,
                        logprob,
                        // TODO Phase-2 follow-up: see above.
                        token_ids: Vec::new(),
                    });
                }
                self.sent_len = self.full_decoded.len();
            }
            EmitOutcome::Continue
        }
    }

    pub(crate) fn finish(
        &mut self,
        generated_tokens: &[u32],
        tokenizer: &Tokenizer,
        delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
        prompt_tokens: usize,
        reason: FinishReason,
        stops: Option<&[String]>,
    ) {
        let mut emitted_visible_text = false;
        if !generated_tokens.is_empty() {
            match tokenizer.decode(generated_tokens) {
                Ok(full_text) => {
                    let end = if let Some(stops) = stops {
                        match check_stop_sequences(&full_text, stops) {
                            StopCheckResult::StopFound { stop_pos } => stop_pos,
                            StopCheckResult::NoStop { .. } => full_text.len(),
                        }
                    } else {
                        full_text.len()
                    };
                    let start = full_text.floor_char_boundary(self.sent_len);
                    let end = full_text.floor_char_boundary(end);
                    emitted_visible_text = end > 0;
                    if end > start {
                        let _ = delta_tx.send(CompletionStreamDelta {
                            text_delta: full_text[start..end].to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                            // TODO Phase-2 follow-up: see emit_delta.
                            token_ids: Vec::new(),
                        });
                    }
                }
                Err(err) => {
                    warn!(
                        "failed to decode {} generated tokens during finish: {err}",
                        generated_tokens.len(),
                    );
                }
            }
        }

        if !generated_tokens.is_empty() && !emitted_visible_text {
            warn!(
                "finishing with {} generated tokens but no visible text delta",
                generated_tokens.len(),
            );
        }

        self.send_finish(
            delta_tx,
            prompt_tokens,
            generated_tokens.len(),
            reason,
            generated_tokens,
        );
    }

    pub(crate) fn send_finish(
        &self,
        delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
        prompt_tokens: usize,
        completion_tokens: usize,
        reason: FinishReason,
        generated_tokens: &[u32],
    ) {
        // Phase 2 trajectory: ride the cumulative response token IDs on
        // the terminator delta. Per-delta `token_ids` is left as an
        // empty Vec (TODO above) — the consumer's collator concatenates
        // every delta's `token_ids` into `response_token_ids`, and with
        // every text delta empty + the final delta carrying the full
        // list, that concatenation comes out correct.
        let _ = delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(reason),
            usage: Some(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            logprob: None,
            token_ids: generated_tokens.to_vec(),
        });
    }
}

impl EmitCursor {
    pub(crate) fn has_pending_emit(&self, generated_tokens_len: usize) -> bool {
        self.dispatched_token_count < generated_tokens_len
    }

    pub(crate) fn pending_tokens<'a>(&self, generated_tokens: &'a [u32]) -> &'a [u32] {
        let start = self.dispatched_token_count.min(generated_tokens.len());
        &generated_tokens[start..]
    }

    pub(crate) fn mark_dispatched(&mut self, generated_tokens_len: usize) {
        self.dispatched_token_count = generated_tokens_len;
    }

    pub(crate) fn is_initial_dispatch(&self) -> bool {
        self.dispatched_token_count == 0
    }
}

/// Newly assigned, needs prefix cache check.
pub(crate) enum Phase {
    /// Waiting for staged T1/T2 bytes to be fetched/promoted back into T0.
    WaitingFetch,
    /// Prefilling in chunks. Decode takes priority between chunks.
    Prefilling {
        effective_tokens: Vec<u32>,
        /// Progress within `effective_tokens`, not total slot seq_len.
        progress: usize,
    },
    /// Generating tokens.
    Decoding,
    /// Completed.
    Finished,
}

pub(crate) struct ActiveRequest {
    pub(crate) id: u64,
    pub(crate) admitted_at: std::time::Instant,
    pub(crate) first_token_at: Option<std::time::Instant>,
    /// Original prompt string (kept for preemption re-queue).
    pub(crate) prompt: String,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) priority: RequestPriority,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: crate::sampler::SamplingParams,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) speculative: Option<crate::scheduler::RequestSpecConfig>,
    /// Optional client session identifier forwarded from `IncomingRequest`.
    /// Preserved across preemption so requeued work stays session-sticky.
    pub(crate) session_id: Option<crate::types::SessionId>,
    /// Most recent trace context used to chain request-level spans across threads.
    pub(crate) trace_context: Option<SpanContext>,
    pub(crate) delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    /// Streaming emit dispatch bookkeeping.
    pub(crate) emit_cursor: EmitCursor,
    pub(crate) phase: Phase,
    /// Prompt length that is known to be fully materialized in this slot's state.
    /// Zero means the slot must not publish a cached prefix on cleanup.
    pub(crate) cacheable_prompt_len: usize,
    /// Latest token's log-probability (greedy only, set by scheduler decode step).
    pub(crate) latest_logprob: Option<f32>,
    /// Deferred terminal reason that must wait for a stop-sensitive emit gate
    /// result before the scheduler can finalize the request.
    pub(crate) pending_finish_reason: Option<FinishReason>,
    /// Block-aligned prefix length that was proven reusable at admission time.
    /// This is derived from the global radix lookup, not a slot-local token
    /// compare. Zero means the request should start cold.
    pub(crate) reusable_prefix_len: usize,
    /// Total prompt length currently materialized in the assigned slot's state
    /// at admission time. Lets `step_new()` preserve the exact-hit /
    /// prompt-prefix-of-cached / extendable-prefix distinctions without keeping
    /// a parallel `cached_prompts: Vec<Vec<u32>>` token store.
    pub(crate) reusable_cached_prompt_len: usize,
    /// Radix blocks whose refs remain pinned for the lifetime of this active
    /// request. Used by the direct paged-prefix attachment path so the radix
    /// will not evict blocks that still back a live slot.
    pub(crate) attached_prefix_blocks: Vec<crate::prefix_cache::BlockId>,
    /// Canonical staged-prefix fetch plan for this request while the prefix is
    /// being promoted back into T0. The scheduler thread owns the fetch queue;
    /// the request owns only its current plan and held radix refs.
    pub(crate) staged_prefix: Option<crate::kv_tier::ReadmissionPlan>,
}

impl ActiveRequest {
    pub(crate) fn begin_trace_span(&self, name: &'static str) -> Option<Span> {
        self.trace_context.map(|parent| Span::root(name, parent))
    }

    pub(crate) fn update_trace_context(&mut self, span: Option<&Span>) {
        if let Some(span) = span {
            self.trace_context = SpanContext::from_span(span).or(self.trace_context);
        }
    }

    pub(crate) fn has_pending_emit(&self) -> bool {
        self.emit_cursor
            .has_pending_emit(self.generated_tokens.len())
    }

    pub(crate) fn has_stop_sequences(&self) -> bool {
        self.stop
            .as_ref()
            .is_some_and(|stops| stops.iter().any(|stop| !stop.is_empty()))
    }

    pub(crate) fn stops_for_emit_dispatch(&self) -> Option<Vec<String>> {
        if self.emit_cursor.is_initial_dispatch() {
            self.stop.clone()
        } else {
            None
        }
    }

    pub(crate) fn pending_emit_tokens(&self) -> Vec<u32> {
        self.emit_cursor
            .pending_tokens(&self.generated_tokens)
            .to_vec()
    }

    pub(crate) fn mark_emit_dispatched(&mut self) {
        self.emit_cursor
            .mark_dispatched(self.generated_tokens.len());
    }

    pub(crate) fn mark_prompt_cacheable(&mut self) {
        self.cacheable_prompt_len = self.prompt_tokens.len();
    }

    pub(crate) fn cached_prompt_to_publish(&self) -> Option<&[u32]> {
        if self.cacheable_prompt_len == self.prompt_tokens.len() && !self.prompt.is_empty() {
            Some(&self.prompt_tokens)
        } else {
            None
        }
    }

    pub(crate) fn held_prefix_blocks(&self) -> Vec<crate::prefix_cache::BlockId> {
        let mut held = self.attached_prefix_blocks.clone();
        if let Some(plan) = &self.staged_prefix {
            held.extend(plan.block_ids());
        }
        held
    }
}

/// Result of scanning decoded text for stop sequences.
pub(crate) enum StopCheckResult {
    /// No stop found. Safe to emit text up to `safe_len` (holds back
    /// `max_stop_len` bytes to avoid splitting a partially-matched stop).
    NoStop { safe_len: usize },
    /// Stop sequence found; text up to `stop_pos` should be emitted, then
    /// the request should finish.
    StopFound { stop_pos: usize },
}

/// Scan `text` for any of the given stop sequences.
///
/// When no stop is found, returns `NoStop` with a safe emit length that
/// withholds `max_stop_len` bytes from the tail (in case a stop sequence
/// straddles the boundary of text decoded so far).
pub(crate) fn check_stop_sequences(text: &str, stops: &[String]) -> StopCheckResult {
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop.as_str()) {
            return StopCheckResult::StopFound { stop_pos: pos };
        }
    }
    let max_stop_len = stops
        .iter()
        .map(std::string::String::len)
        .max()
        .unwrap_or(0);
    let raw_safe = text.len().saturating_sub(max_stop_len);
    let safe_len = text.floor_char_boundary(raw_safe);
    StopCheckResult::NoStop { safe_len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SamplingParams;
    use tokio::sync::mpsc;

    fn test_request(prompt: &str, prompt_tokens: Vec<u32>) -> ActiveRequest {
        let (delta_tx, _delta_rx) = mpsc::unbounded_channel();
        ActiveRequest {
            id: 1,
            admitted_at: std::time::Instant::now(),
            first_token_at: None,
            prompt: prompt.to_string(),
            prompt_tokens,
            generated_tokens: Vec::new(),
            priority: RequestPriority::Normal,
            max_tokens: 16,
            sampling: SamplingParams::default(),
            stop: None,
            speculative: None,
            session_id: None,
            trace_context: None,
            delta_tx,
            emit_cursor: EmitCursor::default(),
            phase: Phase::Finished,
            cacheable_prompt_len: 0,
            latest_logprob: None,
            pending_finish_reason: None,
            reusable_prefix_len: 0,
            reusable_cached_prompt_len: 0,
            attached_prefix_blocks: Vec::new(),
            staged_prefix: None,
        }
    }

    #[test]
    fn publishes_cached_prompt_only_after_explicit_mark() {
        let mut req = test_request("hello", vec![1, 2, 3]);
        assert_eq!(req.cached_prompt_to_publish(), None);

        req.mark_prompt_cacheable();
        assert_eq!(req.cached_prompt_to_publish(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn does_not_publish_cached_prompt_for_preempted_request() {
        let mut req = test_request("hello", vec![1, 2, 3]);
        req.mark_prompt_cacheable();
        req.prompt.clear();

        assert_eq!(req.cached_prompt_to_publish(), None);
    }

    #[test]
    fn pending_emit_requires_new_tokens() {
        let mut req = test_request("hello", vec![1, 2, 3]);
        assert!(!req.has_pending_emit());

        req.generated_tokens.push(42);
        assert!(req.has_pending_emit());

        req.emit_cursor.dispatched_token_count = 1;
        assert!(!req.has_pending_emit());
    }

    #[test]
    fn stop_gate_only_for_non_empty_stop_sequences() {
        let mut req = test_request("hello", vec![1, 2, 3]);
        assert!(!req.has_stop_sequences());

        req.stop = Some(vec![String::new()]);
        assert!(!req.has_stop_sequences());

        req.stop = Some(vec!["</tool>".to_string()]);
        assert!(req.has_stop_sequences());
    }

    #[test]
    fn stop_sequences_are_only_cloned_for_initial_emit_dispatch() {
        let mut req = test_request("hello", vec![1, 2, 3]);
        req.stop = Some(vec!["</tool>".to_string()]);

        assert_eq!(req.stops_for_emit_dispatch(), req.stop);

        req.generated_tokens.push(42);
        req.mark_emit_dispatched();
        assert_eq!(req.stops_for_emit_dispatch(), None);
    }
}
