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
    tool_buf: Option<String>,
}

impl HiddenBlockStream {
    fn capture_tools() -> Self {
        Self {
            tool_buf: Some(String::new()),
            ..Self::default()
        }
    }

    fn push(&mut self, chunk: &str) -> (String, Vec<ToolCall>) {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    fn finish(&mut self) -> (String, Vec<ToolCall>) {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> (String, Vec<ToolCall>) {
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

            if hidden != HiddenBlock::Think
                && let Some(tool_buf) = &mut self.tool_buf
            {
                tool_buf.push_str(&self.pending[..take_len]);
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
        let Some(tool_buf) = &mut self.tool_buf else {
            return;
        };
        match hidden {
            HiddenBlock::ToolCall => {
                if let Some(call) = parse_streaming_tool_call_block(tool_buf.trim()) {
                    calls.push(call);
                }
            }
            HiddenBlock::DsmlToolCalls => {
                calls.extend(parse_dsml_tool_calls_block(tool_buf.trim()));
            }
            HiddenBlock::Think => {}
        }
        tool_buf.clear();
    }
}

/// Keeps user-visible text while stripping `<tool_call>...</tool_call>` and
/// `<think>...</think>` blocks across chunk boundaries.
#[derive(Default)]
pub struct VisibleTextStream {
    stream: HiddenBlockStream,
}

impl VisibleTextStream {
    pub fn push(&mut self, chunk: &str) -> String {
        self.stream.push(chunk).0
    }

    pub fn finish(&mut self) -> String {
        self.stream.finish().0
    }
}

/// Mirrors [`VisibleTextStream`]'s hiding of `<think>...</think>` and
/// `<tool_call>...</tool_call>` blocks, but captures completed tool calls
/// instead of discarding them. Use on the streaming path when the request
/// carries tool definitions.
pub struct StreamingToolCalls {
    stream: HiddenBlockStream,
}

impl Default for StreamingToolCalls {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingToolCalls {
    pub fn new() -> Self {
        Self {
            stream: HiddenBlockStream::capture_tools(),
        }
    }

    /// Feed a chunk; returns `(visible_text_to_emit, newly_completed_tool_calls)`.
    pub fn push(&mut self, chunk: &str) -> (String, Vec<ToolCall>) {
        self.stream.push(chunk)
    }

    /// Flush remaining buffered text and any complete unterminated tool call.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        self.stream.finish()
    }
}
