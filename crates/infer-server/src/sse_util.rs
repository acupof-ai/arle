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

/// Converged decode pipeline shared by the OpenAI and Anthropic SSE paths —
/// the streaming twin of `coordinator::finalize_chat_content`.
///
/// Reasoning splits FIRST: the chat template pre-fills `<think>` into the
/// *prompt*, so thinking output arrives as `reasoning</think>answer` with no
/// opening tag — the tool stream's opening-tag-triggered hiding misses it and
/// leaks reasoning + a stray `</think>` into visible text. Only the content
/// half then feeds tool-call extraction (so reasoning that merely *mentions*
/// a tool block never parses as a call). In tools mode reasoning is dropped,
/// matching the non-streaming finalize; tool-less thinking keeps emitting
/// `Reasoning` deltas (the OpenAI `reasoning_content` lane — Anthropic has no
/// lane and drops them at the encoder).
pub(crate) struct StreamPipeline {
    splitter: StreamingReasoningSplitter,
    tool_stream: Option<chat::StreamingToolCalls>,
    /// Whether reasoning deltas are emitted on the wire. When false (thinking
    /// disabled), the splitter still strips a leading `<think>` block so it
    /// doesn't leak into content, but the reasoning text is dropped.
    emit_reasoning: bool,
    /// Whether reasoning is emitted even when tools are active. OpenAI drops
    /// reasoning in tools mode (no wire lane); Anthropic emits thinking blocks
    /// alongside tool_use blocks.
    emit_reasoning_in_tools: bool,
}

impl StreamPipeline {
    pub(crate) fn new(thinking: bool, tools_active: bool) -> Self {
        Self {
            splitter: StreamingReasoningSplitter::new(thinking),
            tool_stream: tools_active.then(chat::StreamingToolCalls::default),
            emit_reasoning: thinking,
            emit_reasoning_in_tools: false,
        }
    }

    /// Construct a pipeline that emits reasoning deltas even in tools mode —
    /// the Anthropic streaming path, where `thinking` blocks precede
    /// `tool_use` blocks.
    pub(crate) fn new_anthropic(thinking: bool, tools_active: bool) -> Self {
        Self {
            splitter: StreamingReasoningSplitter::new(thinking),
            tool_stream: tools_active.then(chat::StreamingToolCalls::default),
            emit_reasoning: thinking,
            emit_reasoning_in_tools: true,
        }
    }

    fn route(
        &mut self,
        piece: ChatDelta,
        deltas: &mut Vec<ChatDelta>,
        calls: &mut Vec<chat::ToolCall>,
    ) {
        match piece {
            // Tools mode: reasoning has no wire lane for OpenAI (dropped), but
            // Anthropic emits it as `thinking` blocks.
            ChatDelta::Reasoning(_)
                if self.tool_stream.is_some() && !self.emit_reasoning_in_tools => {}
            ChatDelta::Reasoning(text) => {
                if self.emit_reasoning {
                    deltas.push(ChatDelta::Reasoning(text));
                }
            }
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

    /// Feed one decoded delta; returns routed deltas + newly completed calls.
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

    /// `stream_options.include_usage` trailer: empty `choices` (so it can't be
    /// mistaken for a content delta), populated `usage`, `object` matching the
    /// endpoint that sent it.
    #[test]
    fn stream_usage_chunk_shape() {
        let chunk = stream_usage_chunk(
            "chatcmpl-x",
            123,
            "m",
            "chat.completion.chunk",
            json!({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        );
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"], json!([]));
        assert_eq!(chunk["usage"]["total_tokens"], 7);
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

    /// Run chunks through the converged pipeline and flatten the results.
    fn pipeline_run(
        thinking: bool,
        tools: bool,
        chunks: &[&str],
    ) -> (Vec<ChatDelta>, Vec<chat::ToolCall>) {
        let mut pipeline = StreamPipeline::new(thinking, tools);
        let (mut deltas, mut calls) = (Vec::new(), Vec::new());
        for chunk in chunks {
            let (d, c) = pipeline.push(chunk);
            deltas.extend(d);
            calls.extend(c);
        }
        let (d, c) = pipeline.finish();
        deltas.extend(d);
        calls.extend(c);
        (deltas, calls)
    }

    fn visible_text(deltas: &[ChatDelta]) -> String {
        deltas
            .iter()
            .filter_map(|delta| match delta {
                ChatDelta::Content(text) => Some(text.as_str()),
                ChatDelta::Reasoning(_) => None,
            })
            .collect()
    }

    /// The `</think>` leak matrix, streaming path (tools active + thinking):
    /// reasoning must never surface as content — including the prompt-prefilled
    /// form with NO opening tag, which the tool stream's opening-tag-triggered
    /// hiding misses.
    #[test]
    fn pipeline_reasoning_never_leaks_into_content() {
        // (a) bare closer (opening <think> lives in the PROMPT), split mid-tag.
        let (deltas, calls) = pipeline_run(
            true,
            true,
            &["The user asks... Let's call the tool.\n</th", "ink>answer"],
        );
        assert_eq!(visible_text(&deltas), "answer");
        assert!(deltas.iter().all(|d| matches!(d, ChatDelta::Content(_))));
        assert!(calls.is_empty());
        // (b) full <think>...</think> pair.
        let (deltas, _) = pipeline_run(true, true, &["<think>r</think>ans", "wer"]);
        assert_eq!(visible_text(&deltas), "answer");
        // (c) unclosed think (generation cut mid-reasoning) → nothing visible.
        let (deltas, calls) = pipeline_run(true, true, &["<think>only reasoning, no closer"]);
        assert_eq!(visible_text(&deltas), "");
        assert!(calls.is_empty());
        // (d) reasoning then a tool call, no text answer.
        let (deltas, calls) = pipeline_run(
            true,
            true,
            &[
                "I need the weather.</think>",
                "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
            ],
        );
        assert_eq!(visible_text(&deltas).trim(), "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        // (e) plain answer, thinking off → byte-identical passthrough.
        let (deltas, calls) = pipeline_run(false, true, &["plain answer"]);
        assert_eq!(visible_text(&deltas), "plain answer");
        assert!(calls.is_empty());
    }

    /// Tool-less thinking keeps the OpenAI reasoning lane; tools mode drops it
    /// (matching the non-streaming finalize).
    #[test]
    fn pipeline_reasoning_lane_gated_by_tools() {
        let (deltas, _) = pipeline_run(true, false, &["r</think>a"]);
        assert_eq!(deltas, vec![reasoning("r"), content("a")]);
        let (deltas, _) = pipeline_run(true, true, &["r</think>a"]);
        assert_eq!(deltas, vec![content("a")]);
    }
}
