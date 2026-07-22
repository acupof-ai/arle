use super::tool::ToolCall;
use super::tool::{
    DSML_TOOL_CALLS_BLOCK, HiddenBlock, THINK_BLOCK, TOOL_CALL_BLOCK, VISIBLE_TAGS, find_first_tag,
    hidden_block_for_open_tag, longest_tag_prefix_suffix, parse_dsml_tool_calls_block,
    parse_streaming_tool_call_block,
};

#[derive(Default)]
struct HiddenBlockStream {
    pending: String,
    hidden: Option<HiddenBlock>,
    tool_buf: String,
}

impl HiddenBlockStream {
    fn push(&mut self, chunk: &str, capture_tools: bool) -> (String, Vec<ToolCall>) {
        self.pending.push_str(chunk);
        self.drain(false, capture_tools)
    }

    fn finish(&mut self, capture_tools: bool) -> (String, Vec<ToolCall>) {
        self.drain(true, capture_tools)
    }

    fn drain(&mut self, flush: bool, capture_tools: bool) -> (String, Vec<ToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();

        loop {
            let Some(hidden) = self.hidden else {
                let Some((idx, tag)) = find_first_tag(&self.pending, &VISIBLE_TAGS) else {
                    if flush {
                        visible.push_str(&self.pending);
                        self.pending.clear();
                    } else {
                        let keep = longest_tag_prefix_suffix(&self.pending, &VISIBLE_TAGS);
                        let emit_len = self.pending.len().saturating_sub(keep);
                        visible.push_str(&self.pending[..emit_len]);
                        self.pending.drain(..emit_len);
                    }
                    break;
                };

                visible.push_str(&self.pending[..idx]);
                self.pending.drain(..idx + tag.len());
                self.hidden = hidden_block_for_open_tag(tag);
                continue;
            };

            let close = match hidden {
                HiddenBlock::ToolCall => TOOL_CALL_BLOCK.close,
                HiddenBlock::DsmlToolCalls => DSML_TOOL_CALLS_BLOCK.close,
                HiddenBlock::Think => THINK_BLOCK.close,
            };
            let close_idx = self.pending.find(close);
            let take_len = close_idx.unwrap_or_else(|| {
                if flush {
                    self.pending.len()
                } else {
                    self.pending
                        .len()
                        .saturating_sub(longest_tag_prefix_suffix(&self.pending, &[close]))
                }
            });

            if capture_tools && hidden != HiddenBlock::Think {
                self.tool_buf.push_str(&self.pending[..take_len]);
            }
            self.pending.drain(..take_len);

            if close_idx.is_some() {
                self.pending.drain(..close.len());
                self.parse_tool_buf(hidden, &mut calls);
                self.hidden = None;
                continue;
            }

            if flush {
                self.parse_tool_buf(hidden, &mut calls);
                self.pending.clear();
                self.hidden = None;
            }
            break;
        }

        (visible, calls)
    }

    fn parse_tool_buf(&mut self, hidden: HiddenBlock, calls: &mut Vec<ToolCall>) {
        match hidden {
            HiddenBlock::ToolCall => {
                if let Some(call) = parse_streaming_tool_call_block(self.tool_buf.trim()) {
                    calls.push(call);
                }
            }
            HiddenBlock::DsmlToolCalls => {
                calls.extend(parse_dsml_tool_calls_block(self.tool_buf.trim()));
            }
            HiddenBlock::Think => {}
        }
        self.tool_buf.clear();
    }
}

/// Incremental text filter for streamed assistant output.
///
/// This keeps user-visible text while stripping `<tool_call>...</tool_call>`
/// and `<think>...</think>` blocks across chunk boundaries.
#[derive(Default)]
pub struct VisibleTextStream {
    stream: HiddenBlockStream,
}

impl VisibleTextStream {
    pub fn push(&mut self, chunk: &str) -> String {
        self.stream.push(chunk, false).0
    }

    pub fn finish(&mut self) -> String {
        self.stream.finish(false).0
    }
}

/// Incremental filter that mirrors [`VisibleTextStream`]'s hiding of
/// `<think>...</think>` and `<tool_call>...</tool_call>` blocks, but also
/// captures the completed tool calls instead of discarding them.
///
/// Use this on the streaming path when the request carries tool definitions:
/// it emits user-visible text exactly as `VisibleTextStream` would while
/// surfacing each closed `<tool_call>` block as a parsed [`ToolCall`].
#[derive(Default)]
pub struct StreamingToolCalls {
    stream: HiddenBlockStream,
}

impl StreamingToolCalls {
    /// Feed a chunk; returns `(visible_text_to_emit, newly_completed_tool_calls)`.
    pub fn push(&mut self, chunk: &str) -> (String, Vec<ToolCall>) {
        self.stream.push(chunk, true)
    }

    /// Flush remaining buffered text and any complete unterminated tool call.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        self.stream.finish(true)
    }
}
