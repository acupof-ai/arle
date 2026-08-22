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
