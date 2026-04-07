pub type Message = infer::chat_protocol::ChatMessage;
#[cfg_attr(not(test), allow(dead_code))]
pub type ToolCall = infer::chat_protocol::ToolCall;

use infer::chat_protocol::{ParsedAssistantResponse, ToolDefinition as ProtocolToolDefinition};

use crate::tools::Tool;

pub fn format_prompt(messages: &[Message], tools: &[Tool]) -> String {
    let protocol_tools: Vec<ProtocolToolDefinition> =
        tools.iter().map(Tool::to_definition).collect();
    infer::chat_protocol::messages_to_prompt(messages, &protocol_tools)
}

pub fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    infer::chat_protocol::parse_tool_calls(text)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_calls_uses_shared_protocol_parser() {
        let parsed = parse_tool_calls(
            "Let me check.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>",
        );

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "pwd");
        assert_eq!(parsed.content, "Let me check.");
    }

    #[test]
    fn format_prompt_uses_shared_protocol_formatter() {
        let prompt = format_prompt(
            &[
                Message::system("You are helpful."),
                Message::assistant(
                    "Checking.",
                    vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
                ),
            ],
            &[Tool {
                name: "shell".into(),
                description: "Run a shell command".into(),
                parameters: json!({}),
            }],
        );

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains(r#""name":"shell""#));
        assert!(prompt.contains(r#""command":"pwd""#));
    }
}
