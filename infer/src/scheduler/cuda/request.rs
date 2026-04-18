use super::{CompletionStreamDelta, FinishReason, TokenUsage, Tokenizer, mpsc};

/// Newly assigned, needs prefix cache check.
pub(crate) enum Phase {
    New,
    /// Prefilling in chunks. Decode takes priority between chunks.
    Prefilling {
        effective_tokens: Vec<u32>,
        progress: usize,
    },
    /// Generating tokens.
    Decoding,
    /// Completed.
    Finished,
}

pub(crate) struct ActiveRequest {
    pub(crate) id: u64,
    pub(crate) slot_idx: usize,
    pub(crate) admitted_at: std::time::Instant,
    pub(crate) first_token_at: Option<std::time::Instant>,
    /// Original prompt string (kept for preemption re-queue).
    pub(crate) prompt: String,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: crate::sampler::SamplingParams,
    pub(crate) stop: Option<Vec<String>>,
    /// Optional client session identifier forwarded from `IncomingRequest`.
    /// Preserved across preemption so requeued work stays session-sticky.
    /// Consumed by slot-admission once A1 lands; currently informational.
    pub(crate) session_id: Option<crate::types::SessionId>,
    pub(crate) delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    /// Full decoded text, maintained incrementally.
    pub(crate) full_decoded: String,
    /// Number of tokens already decoded into full_decoded.
    pub(crate) decoded_token_count: usize,
    /// Number of characters already sent to the client.
    pub(crate) sent_len: usize,
    pub(crate) phase: Phase,
    /// Prompt length that is known to be fully materialized in this slot's state.
    /// Zero means the slot must not publish a cached prefix on cleanup.
    pub(crate) cacheable_prompt_len: usize,
    /// Cached byte length of the decoded prefix (tokens[0..safe_point]).
    /// Avoids O(N) re-decode of prefix in emit_delta.
    pub(crate) prefix_byte_len: usize,
    /// Latest token's log-probability (greedy only, set by scheduler decode step).
    pub(crate) latest_logprob: Option<f32>,
    /// Block-aligned prefix length that was proven reusable at admission time.
    /// This is derived from the global radix lookup, not a slot-local token
    /// compare. Zero means the request should start cold.
    pub(crate) reusable_prefix_len: usize,
    /// Total prompt length currently materialized in the assigned slot's state
    /// at admission time. Lets `step_new()` preserve the exact-hit /
    /// prompt-prefix-of-cached / extendable-prefix distinctions without keeping
    /// a parallel `cached_prompts: Vec<Vec<u32>>` token store.
    pub(crate) reusable_cached_prompt_len: usize,
}

impl ActiveRequest {
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

    /// Decode newly generated tokens and emit text deltas to the client.
    ///
    /// Uses incremental decode: only re-decodes a small suffix (4 tokens)
    /// to handle multi-byte character boundaries, instead of all tokens.
    /// Cost per call: O(1) instead of O(N) where N = generated token count.
    pub(crate) fn emit_delta(&mut self, tokenizer: &Tokenizer) {
        let n = self.generated_tokens.len();
        if n == 0 {
            return;
        }

        let overlap = 4;
        let safe_point = self.decoded_token_count.saturating_sub(overlap);
        let new_text = match tokenizer.decode(&self.generated_tokens[safe_point..]) {
            Ok(t) => t,
            Err(_) => return,
        };

        if safe_point > 0 {
            // Use cached prefix byte length instead of re-decoding all prefix tokens.
            // The cache is valid because safe_point == previous decoded_token_count - overlap,
            // which is exactly where the previous call cached the prefix length.
            self.full_decoded.truncate(self.prefix_byte_len);
            self.full_decoded.push_str(&new_text);
        } else {
            self.full_decoded = new_text;
        }

        // Cache prefix byte length for next call: the prefix is tokens[0..n-overlap],
        // which in full_decoded starts at byte 0 and has the length we just computed.
        let new_safe = n.saturating_sub(overlap);
        if new_safe > 0 {
            // prefix byte len = total decoded len - suffix len (suffix = tokens[new_safe..n])
            let suffix = tokenizer
                .decode(&self.generated_tokens[new_safe..])
                .unwrap_or_default();
            self.prefix_byte_len = self.full_decoded.len().saturating_sub(suffix.len());
        } else {
            self.prefix_byte_len = 0;
        }

        self.decoded_token_count = n;

        if let Some(ref stops) = self.stop {
            match check_stop_sequences(&self.full_decoded, stops) {
                StopCheckResult::StopFound { stop_pos } => {
                    if stop_pos > self.sent_len {
                        let _ = self.delta_tx.send(CompletionStreamDelta {
                            text_delta: self.full_decoded[self.sent_len..stop_pos].to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: self.latest_logprob,
                        });
                    }
                    self.sent_len = stop_pos;
                    self.phase = Phase::Finished;
                    self.send_finish(FinishReason::Stop);
                }
                StopCheckResult::NoStop { safe_len } => {
                    if safe_len > self.sent_len {
                        let _ = self.delta_tx.send(CompletionStreamDelta {
                            text_delta: self.full_decoded[self.sent_len..safe_len].to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: self.latest_logprob,
                        });
                        self.sent_len = safe_len;
                    }
                }
            }
        } else if self.full_decoded.len() > self.sent_len {
            // Snap to char boundary to avoid panic on multi-byte characters.
            let start = self.full_decoded.floor_char_boundary(self.sent_len);
            if start < self.full_decoded.len() {
                let _ = self.delta_tx.send(CompletionStreamDelta {
                    text_delta: self.full_decoded[start..].to_string(),
                    finish_reason: None,
                    usage: None,
                    logprob: self.latest_logprob,
                });
            }
            self.sent_len = self.full_decoded.len();
        }
    }

    /// Flush remaining buffered text and send the final finish delta.
    pub(crate) fn finish(&mut self, reason: FinishReason, tokenizer: &Tokenizer) {
        if matches!(self.phase, Phase::Finished) {
            return;
        }
        self.phase = Phase::Finished;

        if !self.generated_tokens.is_empty() {
            if let Ok(full_text) = tokenizer.decode(&self.generated_tokens) {
                let end = if let Some(ref stops) = self.stop {
                    match check_stop_sequences(&full_text, stops) {
                        StopCheckResult::StopFound { stop_pos } => stop_pos,
                        StopCheckResult::NoStop { .. } => full_text.len(),
                    }
                } else {
                    full_text.len()
                };
                let start = full_text.floor_char_boundary(self.sent_len);
                let end = full_text.floor_char_boundary(end);
                if end > start {
                    let _ = self.delta_tx.send(CompletionStreamDelta {
                        text_delta: full_text[start..end].to_string(),
                        finish_reason: None,
                        usage: None,
                        logprob: None,
                    });
                }
            }
        }

        self.send_finish(reason);
    }

    fn send_finish(&self, reason: FinishReason) {
        let _ = self.delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(reason),
            usage: Some(TokenUsage {
                prompt_tokens: self.prompt_tokens.len(),
                completion_tokens: self.generated_tokens.len(),
                total_tokens: self.prompt_tokens.len() + self.generated_tokens.len(),
            }),
            logprob: None,
        });
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
    // Snap to a char boundary so slicing never panics on multi-byte chars.
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
            slot_idx: 0,
            admitted_at: std::time::Instant::now(),
            first_token_at: None,
            prompt: prompt.to_string(),
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens: 16,
            sampling: SamplingParams::default(),
            stop: None,
            session_id: None,
            delta_tx,
            full_decoded: String::new(),
            decoded_token_count: 0,
            sent_len: 0,
            phase: Phase::New,
            cacheable_prompt_len: 0,
            prefix_byte_len: 0,
            latest_logprob: None,
            reusable_prefix_len: 0,
            reusable_cached_prompt_len: 0,
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
}
