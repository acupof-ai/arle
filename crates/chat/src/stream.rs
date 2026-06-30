use super::tool::ToolCall;
use super::tool::{
    DSML_TOOL_CALLS_BLOCK, HiddenBlock, THINK_BLOCK, TOOL_CALL_BLOCK, VISIBLE_TAGS, find_first_tag,
    hidden_block_for_open_tag, longest_tag_prefix_suffix, parse_dsml_tool_calls_block,
    parse_streaming_tool_call_block,
};

/// Incremental text filter for streamed assistant output.
///
/// This keeps user-visible text while stripping `<tool_call>...</tool_call>`
/// and `<think>...</think>` blocks across chunk boundaries.
#[derive(Default)]
pub struct VisibleTextStream {
    pending: String,
    hidden: Option<HiddenBlock>,
}

impl VisibleTextStream {
    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> String {
        let mut visible = String::new();

        loop {
            match self.hidden {
                None => {
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
                }
                Some(HiddenBlock::ToolCall) => {
                    if let Some(idx) = self.pending.find(TOOL_CALL_BLOCK.close) {
                        self.pending.drain(..idx + TOOL_CALL_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep =
                            longest_tag_prefix_suffix(&self.pending, &[TOOL_CALL_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
                Some(HiddenBlock::DsmlToolCalls) => {
                    if let Some(idx) = self.pending.find(DSML_TOOL_CALLS_BLOCK.close) {
                        self.pending
                            .drain(..idx + DSML_TOOL_CALLS_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(
                            &self.pending,
                            &[DSML_TOOL_CALLS_BLOCK.close],
                        );
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
                Some(HiddenBlock::Think) => {
                    if let Some(idx) = self.pending.find(THINK_BLOCK.close) {
                        self.pending.drain(..idx + THINK_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(&self.pending, &[THINK_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
            }
        }

        visible
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
    pending: String,
    hidden: Option<HiddenBlock>,
    tool_buf: String,
}

impl StreamingToolCalls {
    /// Feed a chunk; returns `(visible_text_to_emit, newly_completed_tool_calls)`.
    pub fn push(&mut self, chunk: &str) -> (String, Vec<ToolCall>) {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    /// Flush remaining buffered text. An unterminated `<tool_call>` block is
    /// dropped (no partial tool call is emitted).
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> (String, Vec<ToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();

        loop {
            match self.hidden {
                None => {
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
                }
                Some(HiddenBlock::ToolCall) => {
                    if let Some(idx) = self.pending.find(TOOL_CALL_BLOCK.close) {
                        self.tool_buf.push_str(&self.pending[..idx]);
                        if let Some(call) = parse_streaming_tool_call_block(self.tool_buf.trim()) {
                            calls.push(call);
                        }
                        self.tool_buf.clear();
                        self.pending.drain(..idx + TOOL_CALL_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.tool_buf.push_str(&self.pending);
                        if let Some(call) = parse_streaming_tool_call_block(self.tool_buf.trim()) {
                            calls.push(call);
                        }
                        self.pending.clear();
                        self.tool_buf.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep =
                            longest_tag_prefix_suffix(&self.pending, &[TOOL_CALL_BLOCK.close]);
                        let take_len = self.pending.len().saturating_sub(keep);
                        self.tool_buf.push_str(&self.pending[..take_len]);
                        self.pending.drain(..take_len);
                        break;
                    }
                }
                Some(HiddenBlock::DsmlToolCalls) => {
                    if let Some(idx) = self.pending.find(DSML_TOOL_CALLS_BLOCK.close) {
                        self.tool_buf.push_str(&self.pending[..idx]);
                        calls.extend(parse_dsml_tool_calls_block(self.tool_buf.trim()));
                        self.tool_buf.clear();
                        self.pending
                            .drain(..idx + DSML_TOOL_CALLS_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.tool_buf.push_str(&self.pending);
                        calls.extend(parse_dsml_tool_calls_block(self.tool_buf.trim()));
                        self.pending.clear();
                        self.tool_buf.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(
                            &self.pending,
                            &[DSML_TOOL_CALLS_BLOCK.close],
                        );
                        let take_len = self.pending.len().saturating_sub(keep);
                        self.tool_buf.push_str(&self.pending[..take_len]);
                        self.pending.drain(..take_len);
                        break;
                    }
                }
                Some(HiddenBlock::Think) => {
                    if let Some(idx) = self.pending.find(THINK_BLOCK.close) {
                        self.pending.drain(..idx + THINK_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(&self.pending, &[THINK_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
            }
        }

        (visible, calls)
    }
}
