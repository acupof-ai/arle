//! Shared chat/tool-call protocol helpers used by both the `infer` HTTP layer
//! and the root agent loop.

use serde::{Deserialize, Deserializer, Serialize};

#[path = "tool.rs"]
pub mod tool;
pub use tool::*;

#[path = "stream.rs"]
pub mod stream;
pub use stream::*;

#[path = "render.rs"]
pub mod render;
pub use render::*;

/// Role tags used by the shared ChatML formatter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl ChatRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role.as_str(),
        }
    }
}

impl From<ChatRole> for String {
    fn from(role: ChatRole) -> Self {
        role.as_str().to_string()
    }
}

impl From<&str> for ChatRole {
    fn from(role: &str) -> Self {
        match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<String> for ChatRole {
    fn from(role: String) -> Self {
        Self::from(role.as_str())
    }
}

/// Shared message shape used for prompt construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default, deserialize_with = "deserialize_string_or_empty")]
    pub content: String,
    #[serde(
        default,
        deserialize_with = "deserialize_tool_calls_or_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: ChatRole::System,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: ChatRole::User,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn assistant(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.to_string(),
            tool_calls,
        }
    }

    pub fn tool_result(_tool_name: &str, result: &str) -> Self {
        Self {
            role: ChatRole::Tool,
            content: result.to_string(),
            tool_calls: vec![],
        }
    }
}

/// Parsed assistant output with tool calls stripped from visible content.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAssistantResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

fn deserialize_string_or_empty<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_tool_calls_or_empty<'de, D>(deserializer: D) -> Result<Vec<ToolCall>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Vec<ToolCall>>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basic_user_message() {
        let prompt = messages_to_prompt(&[ChatMessage::user("hello")], &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn no_think_switch_pre_closes_think_block() {
        // With `/no_think` anywhere in the conversation, the generation prompt
        // gets a pre-closed think block so Qwen skips reasoning and acts.
        let prompt = messages_to_prompt(
            &[
                ChatMessage::system("be an agent /no_think"),
                ChatMessage::user("fix it"),
            ],
            &[],
        );
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        // Without the switch, thinking stays enabled (no pre-closed block).
        let normal = messages_to_prompt(&[ChatMessage::user("hello")], &[]);
        assert!(normal.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_chatml_single_message() {
        let prompt = render_chatml(
            &[ChatMlMessage {
                role: "user",
                content: "hello",
            }],
            true,
        );

        assert_eq!(
            prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_chatml_with_spans_tracks_body_range() {
        let rendered = render_chatml_with_spans(
            &[ChatMlMessage {
                role: "assistant",
                content: "\nhello",
            }],
            false,
        );

        assert_eq!(
            rendered.prompt,
            "<|im_start|>assistant\n\nhello<|im_end|>\n"
        );
        assert_eq!(rendered.spans.len(), 1);
        assert_eq!(rendered.spans[0].turn, 0..39);
        assert_eq!(rendered.spans[0].supervised, 22..38);
    }

    #[test]
    fn system_and_user_messages() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::system("You are helpful."),
                ChatMessage::user("hi"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn tool_definition_injected_into_system_prompt() {
        let prompt = messages_to_prompt(
            &[ChatMessage::user("list files")],
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
        );

        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("<tools>"));
    }

    #[test]
    fn tool_block_uses_compact_argument_shape() {
        let block = build_tool_block(&[ToolDefinition::new(
            "shell",
            "Run a shell command",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )]);

        assert!(block.contains(r#""arguments":{"command":"string","required":["command"]}"#));
        assert!(!block.contains(r#""type":"function""#));
        assert!(!block.contains(r#""properties""#));
        assert!(!block.contains("You may call one or more functions"));
    }

    #[test]
    fn assistant_tool_calls_render_as_xml_blocks() {
        let prompt = messages_to_prompt(
            &[ChatMessage::assistant(
                "Checking.",
                vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
            )],
            &[],
        );

        assert!(prompt.contains("Checking."));
        assert!(prompt.contains("Checking.\n<tool_call>\n"));
        assert!(prompt.contains(
            "<tool_call>\n<function=shell>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
        ));
    }

    #[test]
    fn empty_assistant_tool_call_matches_generation_prompt_continuation() {
        let user = ChatMessage::user("session w4 base prefix");
        let tool_call = ToolCall::new("retrieve_context", json!({ "query": "session-context" }));

        let warmup_prompt = messages_to_prompt(std::slice::from_ref(&user), &[]);
        let mut completed_warmup_prefix = warmup_prompt;
        completed_warmup_prefix.push_str("<tool_call>");
        completed_warmup_prefix.push('\n');
        completed_warmup_prefix.push_str(&tool_call.prompt_payload());
        completed_warmup_prefix.push('\n');
        completed_warmup_prefix.push_str("</tool_call>");
        completed_warmup_prefix.push_str("<|im_end|>\n");

        let resume_prompt = messages_to_prompt(
            &[
                user,
                ChatMessage::assistant("", vec![tool_call]),
                ChatMessage::tool_result("retrieve_context", "tool result payload"),
            ],
            &[],
        );

        assert!(resume_prompt.starts_with(&completed_warmup_prefix));
        assert!(!resume_prompt.contains("<|im_start|>assistant\n\n<tool_call>"));
    }

    #[test]
    fn chat_message_deserializes_tool_calls_and_null_content() {
        let message = serde_json::from_str::<ChatMessage>(
            r#"{"role":"assistant","content":null,"tool_calls":[{"name":"shell","arguments":{"command":"pwd"}}]}"#,
        )
        .expect("chat message should deserialize");

        assert_eq!(message.role, ChatRole::Assistant);
        assert_eq!(message.content, "");
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].name, "shell");
        assert_eq!(message.tool_calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn structured_render_only_labels_assistant_turns() {
        let rendered = render_structured_chatml_with_spans(
            &[
                ChatMessage::user("first"),
                ChatMessage::assistant(
                    "",
                    vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
                ),
                ChatMessage::tool_result("shell", "cwd"),
                ChatMessage::assistant("done", vec![]),
            ],
            false,
        );

        assert!(rendered.prompt.contains("<tool_call>"));
        assert!(rendered.prompt.contains("<tool_response>"));
        assert!(rendered.spans[0].supervised.is_empty());
        assert!(!rendered.spans[1].supervised.is_empty());
        assert!(rendered.spans[2].supervised.is_empty());
        assert!(!rendered.spans[3].supervised.is_empty());
    }

    #[test]
    fn parse_tool_call_basic() {
        let parsed = parse_tool_calls(
            "Sure.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}\n</tool_call>",
        );

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["cmd"], "ls");
        assert_eq!(parsed.content, "Sure.");
    }

    #[test]
    fn parse_tool_call_missing_close_tag() {
        // Qwen3.6 quirk: drops `</tool_call>`, sometimes puts `arguments` first.
        let parsed = parse_tool_calls(
            "<tool_call>\n\n{\"arguments\":{\"command\":\"ls -la\"},\"name\":\"shell\"}",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "ls -la");
        // The raw JSON must NOT leak into the visible content.
        assert_eq!(parsed.content, "");
        assert!(!parsed.content.contains("arguments"));
    }

    #[test]
    fn parse_tool_call_truncated_is_dropped_not_leaked() {
        // Generation cut off mid-JSON (unbalanced) — must hide, not leak.
        let parsed =
            parse_tool_calls("<tool_call>\n{\"arguments\":{\"command\":\"ls crates/cli/src/\"}");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "");
        assert!(!parsed.content.contains("arguments"));
    }

    #[test]
    fn parses_split_name_then_arguments_object() {
        // Untrained-Qwen3.6 split-emission quirk: the `<tool_call>` block holds only
        // `{"name":...}` and the arguments arrive as a bare object after the close
        // tag (with a stray trailing `</tool_call>`). Merge them into one call.
        let parsed = parse_tool_calls(
            "<tool_call>{\"name\":\"bash\"}</tool_call>\n{\"command\":\"ls lib\"}\n</tool_call>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({ "command": "ls lib" })
        );
        // The bare arguments JSON must NOT leak into the visible content.
        assert!(!parsed.content.contains("command"));
        assert!(!parsed.content.contains('{'));
    }

    #[test]
    fn name_only_tool_call_without_following_object_stays_empty() {
        // No bare arguments object follows — must NOT false-merge; arguments stay
        // empty and following prose is preserved as visible content.
        let parsed = parse_tool_calls("<tool_call>{\"name\":\"bash\"}</tool_call>\nokay then.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.tool_calls[0].arguments, json!({}));
        assert_eq!(parsed.content, "okay then.");
    }

    #[test]
    fn correct_single_object_call_unchanged_by_split_merge() {
        // Canonical single-object form with text after — the split-merge path must
        // not disturb it (arguments already present, so no following-object peek).
        let parsed = parse_tool_calls(
            "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}</tool_call>\n{\"command\":\"other\"}",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.tool_calls[0].arguments, json!({ "command": "ls" }));
        // The trailing bare object is NOT part of this call; it stays visible.
        assert!(parsed.content.contains("other"));
    }

    #[test]
    fn parse_tool_call_text_after_close_preserved() {
        let parsed = parse_tool_calls(
            "<tool_call>{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call> done now.",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.content, "done now.");
    }

    #[test]
    fn parse_tool_call_native_xml_function_format() {
        // Qwen3.6's native XML tool-call form.
        let parsed = parse_tool_calls(
            "<tool_call>\n<function=shell>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "ls -la");
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn parse_tool_call_native_xml_numeric_parameter() {
        let parsed = parse_tool_calls(
            "<tool_call><function=wait><parameter=seconds>30</parameter></function></tool_call>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments["seconds"], 30);
    }

    #[test]
    fn qwen3coder_roundtrip() {
        // THE case that broke under Hermes JSON: a command full of quotes and
        // pipes. Render to the native XML emit format, parse it back, and the
        // command string must survive verbatim.
        let call = ToolCall::new("bash", json!({ "command": "grep -rn \"foo|bar\" lib" }));
        let rendered = messages_to_prompt(&[ChatMessage::assistant("", vec![call.clone()])], &[]);
        let parsed = parse_tool_calls(&rendered);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.tool_calls[0].arguments, call.arguments);
        assert_eq!(
            parsed.tool_calls[0].arguments["command"],
            "grep -rn \"foo|bar\" lib"
        );
    }

    #[test]
    fn qwen3coder_parse_example() {
        // The literal chat-template example: a string arg and a numeric arg.
        let parsed = parse_tool_calls(
            "<tool_call>\n<function=read>\n<parameter=path>\nlib/ansible/x.py\n</parameter>\n<parameter=start>\n520\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments["path"], "lib/ansible/x.py");
        // `start` coerces to a JSON NUMBER, not a string.
        assert_eq!(parsed.tool_calls[0].arguments["start"], 520);
        assert!(parsed.tool_calls[0].arguments["start"].is_number());
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn qwen3coder_multiline_param() {
        // A multi-line `write` content must round-trip byte-for-byte, including
        // its inner blank lines, quotes, and braces.
        let content = "def main():\n    x = {\"a\": 1}\n\n    return x\n";
        let call = ToolCall::new("write", json!({ "path": "out.py", "content": content }));
        let rendered = messages_to_prompt(&[ChatMessage::assistant("", vec![call.clone()])], &[]);
        let parsed = parse_tool_calls(&rendered);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments["content"], content);
        assert_eq!(parsed.tool_calls[0].arguments["path"], "out.py");
    }

    #[test]
    fn malformed_xml_tool_call_not_leaked() {
        // Truncated emission: an open `<tool_call>` + `<function=` + a partial
        // parameter with NO closing tags. Must yield no tool call AND must not
        // leak the raw fragment into the visible content (a leaked fragment was
        // being treated as the agent's "final answer" and falsely terminating it).
        let parsed =
            parse_tool_calls("<tool_call>\n<function=bash>\n<parameter=command>\ngrep foo");
        assert!(parsed.tool_calls.is_empty());
        assert!(!parsed.content.contains("<tool_call>"));
        assert!(!parsed.content.contains("<function="));
        assert!(!parsed.content.contains("grep foo"));
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn parse_tool_call_deepseek_dsml_format() {
        let parsed = parse_tool_calls(
            "Before <｜DSML｜tool_calls>
<｜DSML｜invoke name=\"shell\">
<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>
<｜DSML｜parameter name=\"count\" string=\"false\">2</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls> after",
        );

        assert_eq!(parsed.content, "Before  after");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "pwd");
        assert_eq!(parsed.tool_calls[0].arguments["count"], 2);
    }

    #[test]
    fn parse_strips_think_blocks() {
        let parsed = parse_tool_calls("<think>\nI should check.\n</think>\nHere is the answer.");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "Here is the answer.");
    }

    #[test]
    fn visible_text_stream_strips_hidden_blocks_across_chunk_boundaries() {
        let mut stream = VisibleTextStream::default();
        let mut visible = String::new();

        for chunk in [
            "Hello<th",
            "ink>secret</th",
            "ink> world<tool",
            "_call>{\"name\":\"shell\"}</tool_call>!",
        ] {
            visible.push_str(&stream.push(chunk));
        }
        visible.push_str(&stream.finish());

        assert_eq!(visible, "Hello world!");
    }

    #[test]
    fn visible_text_stream_keeps_partial_tag_prefix_hidden_until_resolved() {
        let mut stream = VisibleTextStream::default();

        assert_eq!(stream.push("abc<th"), "abc");
        assert_eq!(stream.push("ink>secret</think>def"), "def");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn visible_text_stream_handles_multibyte_text_before_partial_tag() {
        let mut stream = VisibleTextStream::default();

        assert_eq!(stream.push("User:** \"你<th"), "User:** \"你");
        assert_eq!(stream.push("ink>secret</think>好"), "好");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn multi_turn_conversation() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::user("first"),
                ChatMessage::assistant("response", vec![]),
                ChatMessage::user("second"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>assistant\nresponse<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    fn drive_streaming(
        stream: &mut StreamingToolCalls,
        chunks: &[&str],
    ) -> (String, Vec<ToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            let (text, new_calls) = stream.push(chunk);
            visible.push_str(&text);
            calls.extend(new_calls);
        }
        let (text, new_calls) = stream.finish();
        visible.push_str(&text);
        calls.extend(new_calls);
        (visible, calls)
    }

    #[test]
    fn streaming_tool_calls_single_call_split_across_chunks() {
        let mut stream = StreamingToolCalls::default();
        // The `<tool_call>` open tag is split mid-tag across chunk boundaries.
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Looking<tool",
                "_call>{\"name\":\"shell\",\"argum",
                "ents\":{\"command\":\"pwd\"}}</tool_call> done",
            ],
        );

        assert_eq!(visible, "Looking done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn streaming_tool_calls_two_sequential_calls() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call>",
                "<tool_call>{\"name\":\"b\",\"arguments\":{\"y\":2}}</tool_call>",
            ],
        );

        assert_eq!(visible, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments["x"], 1);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments["y"], 2);
    }

    #[test]
    fn streaming_tool_calls_native_xml_split_across_chunks() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Checking <tool_call><function=shell><parameter=command>",
                "ls -la</parameter></function>",
                "</tool_call> done",
            ],
        );

        assert_eq!(visible, "Checking  done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls -la");
    }

    #[test]
    fn streaming_tool_calls_missing_close_json_is_parsed_on_finish() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &["<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}"],
        );

        assert_eq!(visible, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn streaming_tool_calls_missing_close_native_xml_is_parsed_on_finish() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &["<tool_call><function=shell><parameter=command>pwd</parameter></function>"],
        );

        assert_eq!(visible, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn streaming_tool_calls_deepseek_dsml_split_across_chunks() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Checking <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"shell\">\n",
                "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n",
                "<｜DSML｜parameter name=\"count\" string=\"false\">2</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n</｜DSML｜tool_calls> done",
            ],
        );

        assert_eq!(visible, "Checking  done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
        assert_eq!(calls[0].arguments["count"], 2);
    }

    #[test]
    fn streaming_tool_calls_hides_think_and_keeps_visible_text() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Hello <think>private</think>world ",
                "<tool_call>{\"name\":\"shell\",\"arguments\":{}}</tool_call>",
                "!",
            ],
        );

        assert_eq!(visible, "Hello world !");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn streaming_tool_calls_plain_text_passes_through() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(&mut stream, &["just ", "plain ", "text answer"]);

        assert_eq!(visible, "just plain text answer");
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_tool_calls_drops_unterminated_block() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) =
            drive_streaming(&mut stream, &["text <tool_call>{\"name\":\"shell\",\"argu"]);

        assert_eq!(visible, "text ");
        assert!(calls.is_empty());
    }

    #[test]
    fn build_tool_block_with_choice_none_is_empty() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::None,
        );
        assert!(block.is_empty());
    }

    #[test]
    fn build_tool_block_with_choice_required_appends_directive() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::Required,
        );
        assert!(block.contains("<tools>"));
        assert!(block.contains("You MUST respond by emitting a <tool_call>"));
    }

    #[test]
    fn build_tool_block_with_choice_function_names_the_tool() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::Function("shell".to_string()),
        );
        assert!(block.contains("You MUST call the `shell` tool via <tool_call>"));
    }

    #[test]
    fn build_tool_block_required_with_no_tools_is_empty() {
        let block = build_tool_block_with_choice(&[], &ToolChoiceMode::Required);
        assert!(block.is_empty());
    }

    #[test]
    fn build_tool_block_renders_qwen3coder_native_header() {
        let block = build_tool_block(&[ToolDefinition::new(
            "shell",
            "Run a shell command",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )]);
        // Qwen3.6-Coder native tools-block contract from the chat template.
        assert!(block.contains("# Tools"));
        assert!(block.contains("You have access to the following functions:"));
        assert!(block.contains("<tools>"));
        assert!(block.contains("</tools>"));
        assert!(block.contains("<function=example_function_name>"));
        assert!(block.contains("<parameter=example_parameter_1>"));
        assert!(block.contains("<IMPORTANT>"));
        assert!(block.contains("</IMPORTANT>"));
        // The tool itself still serializes as compact OpenAI-function JSON.
        assert!(block.contains(r#""name":"shell""#));
        // The old Hermes JSON-call instruction is gone.
        assert!(!block.contains(r#"Use <tool_call>{"name""#));
    }
}
