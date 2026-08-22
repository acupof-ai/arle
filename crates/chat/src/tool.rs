use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Copy)]
pub(crate) struct TaggedBlock {
    pub(crate) open: &'static str,
    pub(crate) close: &'static str,
}

impl TaggedBlock {
    pub(crate) fn strip_and_collect<T>(
        self,
        text: &str,
        mut parse_block: impl FnMut(&str) -> Option<T>,
    ) -> (String, Vec<T>) {
        let mut parsed = Vec::new();
        let mut remaining = text;
        let mut stripped = String::with_capacity(text.len());

        while let Some(start) = remaining.find(self.open) {
            stripped.push_str(&remaining[..start]);
            remaining = &remaining[start + self.open.len()..];

            if let Some(end) = remaining.find(self.close) {
                let block = remaining[..end].trim();
                if let Some(item) = parse_block(block) {
                    parsed.push(item);
                }
                remaining = &remaining[end + self.close.len()..];
            } else {
                stripped.push_str(remaining);
                remaining = "";
            }
        }

        stripped.push_str(remaining);
        (stripped, parsed)
    }

    pub(crate) fn strip_all(self, text: &str) -> String {
        let (stripped, _) = self.strip_and_collect::<()>(text, |_| None);
        stripped
    }
}

pub(crate) const TOOL_CALL_BLOCK: TaggedBlock = TaggedBlock {
    open: "<tool_call>",
    close: "</tool_call>",
};

pub(crate) const DSML_TOOL_CALLS_BLOCK: TaggedBlock = TaggedBlock {
    open: "<｜DSML｜tool_calls>",
    close: "</｜DSML｜tool_calls>",
};

const DSML_INVOKE_OPEN: &str = "<｜DSML｜invoke";
const DSML_INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const DSML_PARAMETER_OPEN: &str = "<｜DSML｜parameter";
const DSML_PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";

pub(crate) const THINK_BLOCK: TaggedBlock = TaggedBlock {
    open: "<think>",
    close: "</think>",
};

/// Open/close tags of every block the visible-text drain loops scan for. Shared
/// by `VisibleTextStream::drain` and `StreamingToolCalls::drain` so the set lives
/// in one place.
pub(crate) const VISIBLE_TAGS: [&str; 6] = [
    TOOL_CALL_BLOCK.open,
    TOOL_CALL_BLOCK.close,
    DSML_TOOL_CALLS_BLOCK.open,
    DSML_TOOL_CALLS_BLOCK.close,
    THINK_BLOCK.open,
    THINK_BLOCK.close,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HiddenBlock {
    ToolCall,
    DsmlToolCalls,
    Think,
}

/// Map a matched *open* tag to the [`HiddenBlock`] it starts. Close tags (and any
/// other slice) map to `None`, matching the drain loops' fallthrough. Shared by
/// both drain implementations.
pub(crate) fn hidden_block_for_open_tag(tag: &str) -> Option<HiddenBlock> {
    match tag {
        tag if tag == TOOL_CALL_BLOCK.open => Some(HiddenBlock::ToolCall),
        tag if tag == DSML_TOOL_CALLS_BLOCK.open => Some(HiddenBlock::DsmlToolCalls),
        tag if tag == THINK_BLOCK.open => Some(HiddenBlock::Think),
        _ => None,
    }
}

pub(crate) fn find_first_tag<'a>(text: &str, tags: &'a [&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|idx| (idx, *tag)))
        .min_by_key(|(idx, _)| *idx)
}

pub(crate) fn longest_tag_prefix_suffix(text: &str, tags: &[&str]) -> usize {
    let text = text.as_bytes();
    let max_len = tags
        .iter()
        .map(|tag| tag.len())
        .max()
        .unwrap_or(0)
        .min(text.len());

    (1..=max_len)
        .rev()
        .find(|&len| {
            let suffix = &text[text.len() - len..];
            tags.iter().any(|tag| tag.as_bytes().starts_with(suffix))
        })
        .unwrap_or(0)
}

pub(crate) fn json_object_len(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &c) in bytes.iter().enumerate() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Shares the exact decoding contract used by [`parse_tool_calls`].
pub(crate) fn parse_tool_call_block(json_str: &str) -> Option<ToolCall> {
    let value = serde_json::from_str::<Value>(json_str).ok()?;
    // A nameless payload is unroutable — prose *about* tool calls (e.g. an
    // example JSON in an explanation), not a call.
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())?
        .to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Map::default()));
    Some(ToolCall::new(name, arguments))
}

pub(crate) fn parse_streaming_tool_call_block(inner: &str) -> Option<ToolCall> {
    let inner = inner.trim();
    if inner.is_empty() {
        return None;
    }

    if let Some(function_pos) = inner.find("<function=")
        && inner[..function_pos].trim().is_empty()
    {
        return parse_native_function_block(&inner[function_pos..]);
    }

    let json_start = inner.find('{')?;
    if !inner[..json_start].trim().is_empty() {
        return None;
    }
    let json_len = json_object_len(&inner[json_start..])?;
    parse_tool_call_block(&inner[json_start..json_start + json_len])
}

pub(crate) fn parse_dsml_tool_calls_block(inner: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = inner;

    while let Some(invoke_start) = rest.find(DSML_INVOKE_OPEN) {
        let after_open = &rest[invoke_start + DSML_INVOKE_OPEN.len()..];
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let attrs = &after_open[..tag_end];
        let body_start = tag_end + 1;
        let body_and_tail = &after_open[body_start..];
        let Some(invoke_end) = body_and_tail.find(DSML_INVOKE_CLOSE) else {
            break;
        };

        if let Some(call) = parse_dsml_invoke_block(attrs, &body_and_tail[..invoke_end]) {
            calls.push(call);
        }
        rest = &body_and_tail[invoke_end + DSML_INVOKE_CLOSE.len()..];
    }

    calls
}

fn parse_dsml_invoke_block(attrs: &str, body: &str) -> Option<ToolCall> {
    let name = quoted_attr_value(attrs, "name")?.trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut args = Map::new();
    let mut rest = body;
    while let Some(param_start) = rest.find(DSML_PARAMETER_OPEN) {
        let after_open = &rest[param_start + DSML_PARAMETER_OPEN.len()..];
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let attrs = &after_open[..tag_end];
        let value_start = tag_end + 1;
        let value_and_tail = &after_open[value_start..];
        let Some(param_end) = value_and_tail.find(DSML_PARAMETER_CLOSE) else {
            break;
        };

        if let Some(key) = quoted_attr_value(attrs, "name").filter(|key| !key.trim().is_empty()) {
            let key = key.trim().to_string();
            let raw = value_and_tail[..param_end].trim();
            let is_string = quoted_attr_value(attrs, "string").as_deref() != Some("false");
            let value = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str::<Value>(raw)
                    .unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            args.insert(key, value);
        }

        rest = &value_and_tail[param_end + DSML_PARAMETER_CLOSE.len()..];
    }

    Some(ToolCall::new(name, Value::Object(args)))
}

fn quoted_attr_value(attrs: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = attrs.find(&needle)? + needle.len();
    let tail = &attrs[start..];
    let end = tail.find('"')?;
    Some(tail[..end].to_string())
}

pub(crate) fn parse_native_function_block(inner: &str) -> Option<ToolCall> {
    let fstart = inner.find("<function=")?;
    let after_fn = &inner[fstart + "<function=".len()..];
    let name_end = after_fn.find('>')?;
    let name = after_fn[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = Map::new();
    let mut rest = &after_fn[name_end + 1..];
    while let Some(ps) = rest.find("<parameter=") {
        let after_p = &rest[ps + "<parameter=".len()..];
        let Some(key_end) = after_p.find('>') else {
            break;
        };
        let key = after_p[..key_end].trim().to_string();
        let val_area = &after_p[key_end + 1..];
        let Some(pe) = val_area.find("</parameter>") else {
            break;
        };
        let value_text = strip_one_surrounding_newline(&val_area[..pe]);
        let value = serde_json::from_str::<Value>(value_text.trim())
            .ok()
            .filter(|v| !v.is_string())
            .unwrap_or_else(|| Value::String(value_text.to_string()));
        args.insert(key, value);
        rest = &val_area[pe + "</parameter>".len()..];
    }
    Some(ToolCall::new(name, Value::Object(args)))
}

fn strip_one_surrounding_newline(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.strip_suffix('\r').unwrap_or(s)
}

fn call_arguments_empty(call: &ToolCall) -> bool {
    match &call.arguments {
        Value::Object(map) => map.is_empty(),
        Value::Null => true,
        _ => false,
    }
}

fn split_arguments_object(s: &str) -> Option<(Value, usize)> {
    let after_ws = s.trim_start();
    let mut prefix = s.len() - after_ws.len();
    let body = if let Some(rest) = after_ws.strip_prefix(TOOL_CALL_BLOCK.close) {
        prefix += TOOL_CALL_BLOCK.close.len();
        let rest_trimmed = rest.trim_start();
        prefix += rest.len() - rest_trimmed.len();
        rest_trimmed
    } else {
        after_ws
    };

    if !body.starts_with('{') {
        return None;
    }
    let len = json_object_len(body)?;
    let value = serde_json::from_str::<Value>(&body[..len]).ok()?;
    if !value.is_object() {
        return None;
    }

    let mut consumed = prefix + len;
    let tail = &s[consumed..];
    if let Some(close_at) = tail.find(TOOL_CALL_BLOCK.close)
        && tail[..close_at].trim().is_empty()
    {
        consumed += close_at + TOOL_CALL_BLOCK.close.len();
    }

    Some((value, consumed))
}

fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    if !text.contains(TOOL_CALL_BLOCK.open) && !text.contains(DSML_TOOL_CALLS_BLOCK.open) {
        return (text.to_string(), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some((start, tag)) =
        find_first_tag(rest, &[TOOL_CALL_BLOCK.open, DSML_TOOL_CALLS_BLOCK.open])
    {
        out.push_str(&rest[..start]);
        let after = &rest[start + tag.len()..];

        if tag == DSML_TOOL_CALLS_BLOCK.open {
            if let Some(end) = after.find(DSML_TOOL_CALLS_BLOCK.close) {
                calls.extend(parse_dsml_tool_calls_block(after[..end].trim()));
                rest = &after[end + DSML_TOOL_CALLS_BLOCK.close.len()..];
            } else {
                calls.extend(parse_dsml_tool_calls_block(after.trim()));
                rest = "";
            }
            continue;
        }

        if let Some(fpos) = after.find("<function=")
            && after[..fpos].trim().is_empty()
        {
            let consumed = if let Some(fend) = after.find("</function>") {
                let end = fend + "</function>".len();
                if let Some(call) = parse_native_function_block(&after[..end]) {
                    calls.push(call);
                }
                match after[end..].find(TOOL_CALL_BLOCK.close) {
                    Some(c) if after[end..end + c].trim().is_empty() => {
                        end + c + TOOL_CALL_BLOCK.close.len()
                    }
                    _ => end,
                }
            } else {
                after.len()
            };
            rest = &after[consumed..];
            continue;
        }

        match after
            .find('{')
            .and_then(|b| json_object_len(&after[b..]).map(|len| (b, len)))
        {
            Some((b, len)) => {
                let mut consumed = b + len;
                let tail = &after[consumed..];
                if let Some(close_at) = tail.find(TOOL_CALL_BLOCK.close)
                    && tail[..close_at].trim().is_empty()
                {
                    consumed += close_at + TOOL_CALL_BLOCK.close.len();
                }

                if let Some(call) = parse_tool_call_block(after[b..b + len].trim()) {
                    if call_arguments_empty(&call)
                        && let Some((args, follow_consumed)) =
                            split_arguments_object(&after[consumed..])
                    {
                        calls.push(ToolCall::new(call.name, args));
                        consumed += follow_consumed;
                    } else {
                        calls.push(call);
                    }
                } else {
                    // Unparseable / nameless payload — a mention, not a call:
                    // keep the literal text instead of swallowing it.
                    out.push_str(tag);
                    out.push_str(&after[..consumed]);
                }
                rest = &after[consumed..];
            }
            None => {
                // No parseable JSON after the opener. A `{` adjacent to the
                // opener is a truncated real call — drop it, don't leak
                // half-JSON. Anything else is a literal tag mention: keep it.
                match after.find('{') {
                    Some(b) if after[..b].trim().is_empty() => rest = "",
                    _ => {
                        out.push_str(tag);
                        rest = after;
                    }
                }
            }
        }
    }
    out.push_str(rest);
    (out, calls)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    pub(crate) fn prompt_schema(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "arguments": compact_parameters(&self.parameters),
        })
    }
}

fn compact_parameters(parameters: &Value) -> Value {
    let Some(object) = parameters.as_object() else {
        return parameters.clone();
    };

    let Some(properties) = object.get("properties").and_then(Value::as_object) else {
        return parameters.clone();
    };

    let mut compact = Map::new();
    for (name, schema) in properties {
        let ty = schema
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("value")
            .to_string();
        compact.insert(name.clone(), Value::String(ty));
    }

    if let Some(required) = object.get("required").and_then(Value::as_array)
        && !required.is_empty()
    {
        compact.insert("required".to_string(), Value::Array(required.clone()));
    }

    Value::Object(compact)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// Render this call as the body that goes INSIDE a `<tool_call>...</tool_call>`
    /// wrapper, in the Qwen3.6-Coder native XML format the chat template trains on.
    pub(crate) fn prompt_payload(&self) -> String {
        let mut out = String::new();
        out.push_str("<function=");
        out.push_str(&self.name);
        out.push_str(">\n");
        match self.arguments.as_object() {
            Some(args) => {
                for (key, value) in args {
                    push_parameter(&mut out, key, value);
                }
            }
            None if !self.arguments.is_null() => {
                push_parameter(&mut out, "arguments", &self.arguments);
            }
            None => {}
        }
        out.push_str("</function>");
        out
    }
}

fn push_parameter(out: &mut String, key: &str, value: &Value) {
    out.push_str("<parameter=");
    out.push_str(key);
    out.push_str(">\n");
    match value {
        Value::String(s) => out.push_str(s),
        other => out.push_str(&serde_json::to_string(other).expect("tool arg serialization")),
    }
    out.push_str("\n</parameter>\n");
}

pub fn parse_tool_calls(text: &str) -> super::ParsedAssistantResponse {
    let (stripped, tool_calls) = extract_tool_calls(text);

    super::ParsedAssistantResponse {
        content: THINK_BLOCK.strip_all(&stripped).trim().to_string(),
        tool_calls,
    }
}
