//! Shared SSE formatting helpers used by both the in-process and coordinator paths.
use std::time::{SystemTime, UNIX_EPOCH};

use infer_plan::FinishReason;
use serde_json::json;

/// OpenAI `system_fingerprint` — must match `schema::SYSTEM_FINGERPRINT`.
const SYSTEM_FINGERPRINT: &str = "arle_fp_1";

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
        "system_fingerprint": SYSTEM_FINGERPRINT,
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
        "system_fingerprint": SYSTEM_FINGERPRINT,
    })
}

/// OpenAI `stream_options.include_usage` trailer: sent once, after the
/// finish-reason chunk and before `[DONE]`, with EMPTY `choices` and the
/// populated `usage` — mirrors vLLM/SGLang so clients that only look for
/// usage on the empty-choices chunk find it. `object` is
/// `"text_completion"` (completions) or `"chat.completion.chunk"` (chat).
pub(crate) fn stream_usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    object: &str,
    usage: serde_json::Value,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    })
}

/// OpenAI streaming error chunk: a `data:` frame carrying `{"error": {...}}`
/// (no `choices`). The error `type` follows the HTTP-status mapping
/// (`api_error` for 5xx, `invalid_request_error` for 4xx). Sent once,
/// immediately before `[DONE]`.
pub(crate) fn stream_error_chunk(
    id: &str,
    created: u64,
    model: &str,
    message: &str,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": "error",
        "created": created,
        "model": model,
        "error": {
            "message": message,
            "type": "api_error",
            "param": null,
            "code": null
        },
        "system_fingerprint": SYSTEM_FINGERPRINT,
    })
}

pub(crate) fn finish_reason(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) | None => "length",
        // OpenAI has no `abort` finish reason; map to `stop` (generation was
        // terminated) so strict OpenAI clients don't choke on an unknown value.
        Some(FinishReason::Abort) => "stop",
    }
}

pub(crate) fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

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

/// Incremental twin of `schema::split_reasoning` for chat SSE deltas.
/// State machine over `<think>` / `</think>` markers: reasoning inside a
/// thinking block, content outside. Handles multi-segment thinking
/// (`r1</think>c1<think>r2</think>c2`) by re-entering reasoning on `<think>`.
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

    pub(crate) fn push(&mut self, text: &str) -> Vec<ChatDelta> {
        if !self.enabled {
            // Thinking is off, but a reasoning-trained model (e.g. Qwen3.6) may
            // still emit a leading <think> block. Detect it on the first push
            // and strip it; reasoning is dropped (the caller gates emission).
            if self.at_start {
                let trimmed = text.trim_start();
                if trimmed.starts_with("<think>") {
                    self.enabled = true;
                    self.in_reasoning = true;
                    self.at_start = false;
                    let rest = trimmed.strip_prefix("<think>").unwrap_or(trimmed);
                    return self.push(rest);
                }
                self.at_start = false;
            }
            return (!text.is_empty())
                .then(|| ChatDelta::Content(text.to_string()))
                .into_iter()
                .collect();
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
        let mut deltas = Vec::new();
        loop {
            if self.in_reasoning {
                match buf.find(THINK_END) {
                    Some(idx) => {
                        let after = buf.split_off(idx + THINK_END.len());
                        buf.truncate(idx);
                        if let Some(d) = self.reasoning(&buf) {
                            deltas.push(d);
                        }
                        self.in_reasoning = false;
                        buf = after;
                    }
                    None => {
                        self.hold_partial(&mut buf, THINK_END);
                        if let Some(d) = self.reasoning(&buf) {
                            deltas.push(d);
                        }
                        break;
                    }
                }
            } else {
                match buf.find(THINK_START) {
                    Some(idx) => {
                        let after = buf.split_off(idx + THINK_START.len());
                        buf.truncate(idx);
                        if let Some(d) = self.content(&buf) {
                            deltas.push(d);
                        }
                        self.in_reasoning = true;
                        buf = after;
                    }
                    None => {
                        self.hold_partial(&mut buf, THINK_START);
                        if let Some(d) = self.content(&buf) {
                            deltas.push(d);
                        }
                        break;
                    }
                }
            }
        }
        deltas
    }

    /// Hold back the longest suffix of `buf` that could grow into `marker`.
    fn hold_partial(&mut self, buf: &mut String, marker: &str) {
        let held = (1..marker.len())
            .rev()
            .map(|len| &marker[..len])
            .find(|prefix| buf.ends_with(prefix))
            .map_or(0, str::len);
        self.pending = buf.split_off(buf.len() - held);
    }

    pub(crate) fn finish(&mut self) -> Option<ChatDelta> {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return None;
        }
        if self.in_reasoning {
            Some(ChatDelta::Reasoning(pending))
        } else {
            Some(ChatDelta::Content(pending))
        }
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

/// Converged decode pipeline shared by the OpenAI and Anthropic SSE paths —
/// the streaming twin of `coordinator::finalize_chat_content`.
///
/// Reasoning splits FIRST: the chat template pre-fills `<think>` into the
/// *prompt*, so thinking output arrives as `reasoning</think>answer` with no
/// opening tag — the tool stream's opening-tag-triggered hiding misses it and
/// leaks reasoning + a stray `</think>` into visible text. Only the content
/// half then feeds tool-call extraction (so reasoning that merely *mentions*
/// a tool block never parses as a call). Reasoning always reaches the wire:
/// `reasoning_content` deltas on OpenAI, `thinking` blocks on Anthropic —
/// including in tools mode, and including reasoning a model emits without the
/// request asking for it.
pub(crate) struct StreamPipeline {
    splitter: StreamingReasoningSplitter,
    tool_stream: Option<chat::StreamingToolCalls>,
}

impl StreamPipeline {
    pub(crate) fn new(thinking: bool, tools_active: bool) -> Self {
        Self {
            splitter: StreamingReasoningSplitter::new(thinking),
            tool_stream: tools_active.then(chat::StreamingToolCalls::default),
        }
    }

    fn route(
        &mut self,
        piece: ChatDelta,
        deltas: &mut Vec<ChatDelta>,
        calls: &mut Vec<chat::ToolCall>,
    ) {
        match piece {
            // Reasoning the model actually produced always reaches the client:
            // `reasoning_content` on OpenAI, `thinking` blocks on Anthropic.
            // Dropping it left the client with a silent multi-second stall.
            ChatDelta::Reasoning(text) => deltas.push(ChatDelta::Reasoning(text)),
            ChatDelta::Content(text) => match self.tool_stream.as_mut() {
                Some(stream) => {
                    let (visible, new_calls) = stream.push(&text);
                    calls.extend(new_calls);
                    if !visible.is_empty() {
                        deltas.push(ChatDelta::Content(visible));
                    }
                }
                None => deltas.push(ChatDelta::Content(text)),
            },
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> (Vec<ChatDelta>, Vec<chat::ToolCall>) {
        let mut deltas = Vec::new();
        let mut calls = Vec::new();
        for piece in self.splitter.push(text) {
            self.route(piece, &mut deltas, &mut calls);
        }
        (deltas, calls)
    }

    /// Flush both stages at end of stream (truncated thinking, buffered tool
    /// tail). Ordered splitter → tool stream, mirroring `push`.
    pub(crate) fn finish(&mut self) -> (Vec<ChatDelta>, Vec<chat::ToolCall>) {
        let mut deltas = Vec::new();
        let mut calls = Vec::new();
        if let Some(piece) = self.splitter.finish() {
            self.route(piece, &mut deltas, &mut calls);
        }
        if let Some(stream) = self.tool_stream.as_mut() {
            let (visible, new_calls) = stream.finish();
            calls.extend(new_calls);
            if !visible.is_empty() {
                deltas.push(ChatDelta::Content(visible));
            }
        }
        (deltas, calls)
    }
}
