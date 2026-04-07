use anyhow::Result;
use log::info;

use infer::sampler::SamplingParams;
use infer::server_engine::CompleteRequest;

use crate::chat::{Message, format_prompt, parse_tool_calls};
use crate::engine::AgentEngine;
use crate::tools::{Tool, execute_tool_call};

const SYSTEM_PROMPT: &str = r#"You are a helpful AI assistant with access to tools. You can use tools to help answer questions, perform calculations, run commands, and more.

When you need to use a tool, output a tool call in the specified format. You may call multiple tools in a single response. After receiving tool results, continue your reasoning and provide a final answer.

Think step by step. If a task requires multiple steps, use tools iteratively. Always verify your work when possible."#;

#[derive(Clone, Copy, Debug)]
pub struct AgentSettings {
    pub max_turns: usize,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTurnResult {
    pub text: String,
    pub tool_calls_executed: usize,
    pub max_turns_reached: bool,
}

#[derive(Debug, Clone)]
pub struct AgentSession {
    messages: Vec<Message>,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    pub fn new() -> Self {
        Self {
            messages: vec![Message::system(SYSTEM_PROMPT)],
        }
    }

    pub fn reset(&mut self) {
        self.messages.truncate(1);
    }

    #[cfg(test)]
    fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn run_turn<E: AgentEngine + ?Sized>(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[Tool],
        settings: AgentSettings,
    ) -> Result<AgentTurnResult> {
        self.messages.push(Message::user(user_input));

        let mut tool_calls_executed = 0usize;

        for turn in 0..settings.max_turns {
            let prompt = format_prompt(&self.messages, tools);
            info!(
                "Agent turn {}/{}: prompt length = {} chars",
                turn + 1,
                settings.max_turns,
                prompt.len()
            );

            let output = engine.complete(CompleteRequest {
                prompt,
                max_tokens: settings.max_tokens,
                sampling: SamplingParams {
                    temperature: settings.temperature,
                    ..SamplingParams::default()
                },
                stop: Some(vec!["<|im_end|>".to_string()]),
            })?;

            info!(
                "Generated {} chars, finish_reason={:?}",
                output.text.len(),
                output.finish_reason
            );

            let parsed = parse_tool_calls(&output.text);
            let content = parsed.content;
            let tool_calls = parsed.tool_calls;

            self.messages
                .push(Message::assistant(&content, tool_calls.clone()));

            if tool_calls.is_empty() {
                return Ok(AgentTurnResult {
                    text: content,
                    tool_calls_executed,
                    max_turns_reached: false,
                });
            }

            if !content.is_empty() {
                println!("\x1b[2m{}\x1b[0m", content);
            }

            println!();
            for tool_call in &tool_calls {
                tool_calls_executed += 1;
                println!(
                    "\x1b[33m[tool: {}]\x1b[0m {}",
                    tool_call.name,
                    serde_json::to_string(&tool_call.arguments).unwrap_or_default()
                );

                let result = execute_tool_call(tool_call);
                let display_result = if result.len() > 500 {
                    format!("{}... ({} chars total)", &result[..500], result.len())
                } else {
                    result.clone()
                };

                println!("\x1b[36m{}\x1b[0m", display_result);
                println!();

                self.messages
                    .push(Message::tool_result(&tool_call.name, &result));
            }
        }

        Ok(AgentTurnResult {
            text: "(max turns reached - agent stopped)".to_string(),
            tool_calls_executed,
            max_turns_reached: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use anyhow::{Result, anyhow};
    use serde_json::json;

    use infer::server_engine::{CompleteOutput, CompleteRequest, FinishReason, Usage};

    use crate::tools::Tool;

    use super::{AgentEngine, AgentSession, AgentSettings};

    struct FakeEngine {
        outputs: VecDeque<String>,
        prompts: Vec<String>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<&str>) -> Self {
            Self {
                outputs: outputs.into_iter().map(str::to_string).collect(),
                prompts: Vec::new(),
            }
        }
    }

    impl AgentEngine for FakeEngine {
        fn model_id(&self) -> &str {
            "fake"
        }

        fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
            self.prompts.push(req.prompt);
            let text = self
                .outputs
                .pop_front()
                .ok_or_else(|| anyhow!("fake engine exhausted"))?;
            Ok(CompleteOutput {
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                text,
                finish_reason: FinishReason::Stop,
            })
        }
    }

    fn settings() -> AgentSettings {
        AgentSettings {
            max_turns: 4,
            max_tokens: 128,
            temperature: 0.0,
        }
    }

    #[test]
    fn session_persists_conversation_history_across_turns() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["alpha", "beta"]);

        let first = session
            .run_turn(&mut engine, "remember alpha", &[], settings())
            .expect("first turn");
        assert_eq!(first.text, "alpha");

        let second = session
            .run_turn(&mut engine, "what did I say before?", &[], settings())
            .expect("second turn");
        assert_eq!(second.text, "beta");

        let second_prompt = &engine.prompts[1];
        assert!(second_prompt.contains("remember alpha"));
        assert!(second_prompt.contains("alpha"));
    }

    #[test]
    fn tool_call_messages_are_not_duplicated_in_followup_prompt() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "Checking.\n<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(2 + 2)\"}}\n</tool_call>",
            "The answer is 4.",
        ]);
        let tools = vec![Tool {
            name: "python".into(),
            description: "Run Python".into(),
            parameters: json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            }),
        }];

        let result = session
            .run_turn(&mut engine, "compute 2+2", &tools, settings())
            .expect("tool turn");

        assert_eq!(result.text, "The answer is 4.");
        assert_eq!(result.tool_calls_executed, 1);

        let followup_prompt = &engine.prompts[1];
        assert_eq!(followup_prompt.matches("print(2 + 2)").count(), 1);
        assert!(followup_prompt.contains("Checking."));
        assert!(followup_prompt.contains("4"));
    }

    #[test]
    fn reset_clears_messages_but_keeps_system_prompt() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["hello"]);

        session
            .run_turn(&mut engine, "hi", &[], settings())
            .expect("turn");
        session.reset();

        assert_eq!(session.messages().len(), 1);
        assert_eq!(
            session.messages()[0].role,
            infer::chat_protocol::ChatRole::System
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn live_model_path() -> Option<String> {
        std::env::var("AGENT_INFER_TEST_MODEL_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                std::env::var("PEGAINFER_TEST_MODEL_PATH")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    #[ignore = "requires AGENT_INFER_TEST_MODEL_PATH or PEGAINFER_TEST_MODEL_PATH"]
    fn live_local_model_session_persists_context() {
        use crate::engine::LoadedAgentEngine;

        let Some(model_path) = live_model_path() else {
            eprintln!("Skipping live test: no local model path provided");
            return;
        };

        infer::logging::init_default();

        let mut engine = LoadedAgentEngine::load(&model_path, true).expect("load local engine");
        let mut session = AgentSession::new();
        let tools: Vec<Tool> = Vec::new();

        let first = session
            .run_turn(
                &mut engine,
                "Reply with exactly the word RIVER and nothing else.",
                &tools,
                AgentSettings {
                    max_turns: 2,
                    max_tokens: 32,
                    temperature: 0.0,
                },
            )
            .expect("first live turn");
        assert!(
            first.text.to_ascii_lowercase().contains("river"),
            "unexpected first response: {:?}",
            first.text
        );

        let second = session
            .run_turn(
                &mut engine,
                "What exact word did you reply with in your previous answer? Reply with that word only.",
                &tools,
                AgentSettings {
                    max_turns: 2,
                    max_tokens: 32,
                    temperature: 0.0,
                },
            )
            .expect("second live turn");
        assert!(
            second.text.to_ascii_lowercase().contains("river"),
            "model did not preserve conversation context: {:?}",
            second.text
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    #[ignore = "requires AGENT_INFER_TEST_MODEL_PATH or PEGAINFER_TEST_MODEL_PATH"]
    fn live_local_model_executes_python_tool() {
        use crate::engine::LoadedAgentEngine;

        let Some(model_path) = live_model_path() else {
            eprintln!("Skipping live test: no local model path provided");
            return;
        };

        infer::logging::init_default();

        let mut engine = LoadedAgentEngine::load(&model_path, true).expect("load local engine");
        let mut session = AgentSession::new();
        let tools = crate::tools::builtin_tools();

        let result = session
            .run_turn(
                &mut engine,
                "Use the python tool to compute 123 * 456. Do not do the math mentally. After the tool returns, answer with just the integer.",
                &tools,
                AgentSettings {
                    max_turns: 4,
                    max_tokens: 128,
                    temperature: 0.0,
                },
            )
            .expect("live tool turn");

        assert!(
            result.tool_calls_executed > 0,
            "model did not execute any tool calls; final response: {:?}",
            result.text
        );
        assert!(
            result.text.contains("56088"),
            "tool result not reflected in final answer: {:?}",
            result.text
        );
    }
}
