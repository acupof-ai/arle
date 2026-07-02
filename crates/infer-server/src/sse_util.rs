//! Shared SSE formatting helpers used by both the in-process and coordinator paths.
use std::time::{SystemTime, UNIX_EPOCH};

use infer_plan::FinishReason;
use serde_json::json;

pub(crate) fn completion_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    finish: Option<&str>,
    usage: Option<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}],
        "usage": usage,
    })
}

pub(crate) fn chat_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish: Option<&str>,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish}],
        "usage": null,
    })
}

pub(crate) fn finish_reason(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) | None => "length",
        Some(FinishReason::Abort) => "abort",
    }
}

pub(crate) fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// One decoded chat-SSE text delta, routed to its wire field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatDelta {
    Reasoning(String),
    Content(String),
}

impl ChatDelta {
    pub(crate) fn into_delta(self) -> serde_json::Value {
        match self {
            Self::Reasoning(text) => json!({"reasoning_content": text}),
            Self::Content(text) => json!({"content": text}),
        }
    }
}

/// Incremental twin of `schema::split_reasoning` for chat SSE deltas: reasoning
/// until the first `</think>` (dropped; may span push boundaries), content after.
/// Disabled is a byte-identical passthrough — no scans, no trims.
pub(crate) struct StreamingReasoningSplitter {
    enabled: bool,
    in_reasoning: bool,
    /// Held-back text: a partial `<think>` opener at stream start, or a
    /// trailing partial `</think>` closer awaiting the next push.
    pending: String,
    at_start: bool,
    reasoning_started: bool,
    content_started: bool,
}

impl StreamingReasoningSplitter {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            in_reasoning: enabled,
            pending: String::new(),
            at_start: true,
            reasoning_started: false,
            content_started: false,
        }
    }

    /// Route one decoded delta batch; emits zero, one, or two deltas.
    pub(crate) fn push(&mut self, text: &str) -> Vec<ChatDelta> {
        if !self.enabled {
            return (!text.is_empty())
                .then(|| ChatDelta::Content(text.to_string()))
                .into_iter()
                .collect();
        }
        if !self.in_reasoning {
            return self.content(text).into_iter().collect();
        }
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(text);
        if self.at_start {
            // Hold until the opener is decidable; strip exactly one `<think>`.
            if buf.len() < THINK_START.len() && THINK_START.starts_with(&buf) {
                self.pending = buf;
                return Vec::new();
            }
            self.at_start = false;
            if let Some(rest) = buf.strip_prefix(THINK_START) {
                buf = rest.to_string();
            }
        }
        match buf.find(THINK_END) {
            Some(idx) => {
                self.in_reasoning = false;
                let after = buf.split_off(idx + THINK_END.len());
                buf.truncate(idx);
                self.reasoning(&buf)
                    .into_iter()
                    .chain(self.content(&after))
                    .collect()
            }
            None => {
                // Hold back the longest suffix that could grow into the closer.
                let held = (1..THINK_END.len())
                    .rev()
                    .map(|len| &THINK_END[..len])
                    .find(|prefix| buf.ends_with(prefix))
                    .map_or(0, str::len);
                self.pending = buf.split_off(buf.len() - held);
                self.reasoning(&buf).into_iter().collect()
            }
        }
    }

    /// Flush any held-back pending as reasoning (truncated-thinking case).
    pub(crate) fn finish(&mut self) -> Option<ChatDelta> {
        let pending = std::mem::take(&mut self.pending);
        (!pending.is_empty()).then_some(ChatDelta::Reasoning(pending))
    }

    /// One-time `trim_start` on the first non-empty emission of each phase,
    /// mirroring the non-streaming split's leading-whitespace policy.
    fn reasoning(&mut self, text: &str) -> Option<ChatDelta> {
        let text = trimmed_once(&mut self.reasoning_started, text)?;
        Some(ChatDelta::Reasoning(text.to_string()))
    }

    fn content(&mut self, text: &str) -> Option<ChatDelta> {
        let text = trimmed_once(&mut self.content_started, text)?;
        Some(ChatDelta::Content(text.to_string()))
    }
}

fn trimmed_once<'a>(started: &mut bool, text: &'a str) -> Option<&'a str> {
    let text = if *started { text } else { text.trim_start() };
    (!text.is_empty()).then(|| {
        *started = true;
        text
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasoning(text: &str) -> ChatDelta {
        ChatDelta::Reasoning(text.to_string())
    }

    fn content(text: &str) -> ChatDelta {
        ChatDelta::Content(text.to_string())
    }

    #[test]
    fn chat_stream_chunk_shape() {
        let chunk = chat_stream_chunk(
            "chatcmpl-x",
            123,
            "m",
            json!({"role": "assistant", "content": "hi"}),
            None,
        );
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["id"], "chatcmpl-x");
        assert_eq!(chunk["created"], 123);
        assert_eq!(chunk["model"], "m");
        let choice = &chunk["choices"][0];
        assert_eq!(choice["index"], 0);
        assert_eq!(choice["delta"]["role"], "assistant");
        assert_eq!(choice["delta"]["content"], "hi");
        assert!(choice["logprobs"].is_null());
        assert!(choice["finish_reason"].is_null());
        assert!(chunk["usage"].is_null());

        let done = chat_stream_chunk("chatcmpl-x", 123, "m", json!({}), Some("stop"));
        assert_eq!(done["choices"][0]["finish_reason"], "stop");
        assert_eq!(done["choices"][0]["delta"], json!({}));
    }

    #[test]
    fn splitter_disabled_is_byte_identical_passthrough() {
        let mut splitter = StreamingReasoningSplitter::new(false);
        assert_eq!(
            splitter.push("  a literal </think> tag "),
            vec![content("  a literal </think> tag ")]
        );
        assert_eq!(splitter.push("</th"), vec![content("</th")]);
        assert_eq!(splitter.finish(), None);
    }

    #[test]
    fn splitter_closer_within_one_push() {
        let mut splitter = StreamingReasoningSplitter::new(true);
        assert_eq!(
            splitter.push("reason</think>answer"),
            vec![reasoning("reason"), content("answer")]
        );
        // Post-closer pushes stay content, untrimmed after the one-time trim.
        assert_eq!(splitter.push(" more"), vec![content(" more")]);
        assert_eq!(splitter.finish(), None);
    }

    #[test]
    fn splitter_closer_split_across_pushes() {
        let mut splitter = StreamingReasoningSplitter::new(true);
        assert_eq!(
            splitter.push("thinking...</th"),
            vec![reasoning("thinking...")]
        );
        assert_eq!(splitter.push("ink>answer"), vec![content("answer")]);
        assert_eq!(splitter.finish(), None);
    }

    #[test]
    fn splitter_truncated_thinking_flushes_pending_as_reasoning() {
        let mut splitter = StreamingReasoningSplitter::new(true);
        assert_eq!(
            splitter.push("unfinished </th"),
            vec![reasoning("unfinished ")]
        );
        assert_eq!(splitter.finish(), Some(reasoning("</th")));
    }

    #[test]
    fn splitter_strips_leading_think_opener() {
        let mut splitter = StreamingReasoningSplitter::new(true);
        assert_eq!(splitter.push("<think>\nreason"), vec![reasoning("reason")]);
        assert_eq!(splitter.push("</think>answer"), vec![content("answer")]);
    }

    #[test]
    fn splitter_opener_split_across_pushes() {
        let mut splitter = StreamingReasoningSplitter::new(true);
        assert_eq!(splitter.push("<thi"), Vec::new());
        assert_eq!(
            splitter.push("nk>r</think>a"),
            vec![reasoning("r"), content("a")]
        );
    }
}
