use super::*;

#[test]
fn completion_request_accepts_session_id() {
    let raw = r#"{"prompt":"hi","session_id":"agent-42"}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.session_id_parsed().unwrap().as_str(), "agent-42");
}

#[test]
fn completion_request_missing_session_id_is_none() {
    let raw = r#"{"prompt":"hi"}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    assert!(req.session_id_parsed().is_none());
    assert!(req.speculative.is_none());
}

#[test]
fn completion_request_accepts_speculative_override() {
    let raw = r#"{
        "prompt":"hi",
        "speculative":{
            "enabled":true,
            "draft_k":5,
            "acceptance_threshold":0.6,
            "draft_model":"self"
        }
    }"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    req.validate().unwrap();
    let spec = req.speculative.as_ref().unwrap();
    assert_eq!(spec.enabled, Some(true));
    assert_eq!(spec.draft_k, Some(5));
    assert_eq!(spec.acceptance_threshold, Some(0.6));
    assert_eq!(spec.draft_model.as_deref(), Some("self"));
}

#[test]
fn completion_request_empty_and_whitespace_session_id_is_none() {
    let empty: CompletionRequest =
        serde_json::from_str(r#"{"prompt":"hi","session_id":""}"#).unwrap();
    assert!(empty.session_id_parsed().is_none());

    let whitespace: CompletionRequest =
        serde_json::from_str(r#"{"prompt":"hi","session_id":"   "}"#).unwrap();
    assert!(whitespace.session_id_parsed().is_none());
}

#[test]
fn completion_request_accepts_user_alias() {
    // OpenAI's canonical "user" field is the standard per-user identifier;
    // we accept it as an alias so existing clients opt into sticky routing
    // without changing their payloads.
    let raw = r#"{"prompt":"hi","user":"client-9"}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.session_id_parsed().unwrap().as_str(), "client-9");
}

#[test]
fn completion_request_accepts_return_token_ids_extension() {
    let raw = r#"{"prompt":"hi","return_token_ids":true}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    assert!(req.return_token_ids_or_default());
}

#[test]
fn completion_request_rejects_return_token_ids_with_streaming() {
    let raw = r#"{"prompt":"hi","stream":true,"return_token_ids":true}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    let err = req
        .validate()
        .expect_err("streaming token id trajectories are not exposed");
    assert_eq!(err.body.code, "invalid_parameter");
    assert_eq!(err.body.param.as_deref(), Some("return_token_ids"));
}

#[test]
fn completion_request_accepts_model_alias_by_final_segment() {
    let raw = r#"{"model":"Qwen/Qwen3-4B","prompt":"hi"}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    req.validate_for_model("qwen3-4b").unwrap();
}

#[test]
fn completion_request_rejects_unavailable_model() {
    let raw = r#"{"model":"qwen3-8b","prompt":"hi"}"#;
    let req: CompletionRequest = serde_json::from_str(raw).unwrap();
    let err = req
        .validate_for_model("Qwen3-4B")
        .expect_err("unexpected model should fail");
    assert_eq!(err.body.code, "model_not_found");
    assert!(err.body.message.contains("qwen3-8b"));
    assert!(err.body.message.contains("Qwen3-4B"));
}

#[test]
fn chat_completion_request_accepts_session_id_and_trims() {
    let raw = r#"{"messages":[{"role":"user","content":"hi"}],"session_id":"  sess-1  "}"#;
    let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.session_id_parsed().unwrap().as_str(), "sess-1");
}

#[test]
fn chat_completion_request_accepts_speculative_override() {
    let raw = r#"{
        "messages":[{"role":"user","content":"hi"}],
        "speculative":{"enabled":true,"draft_k":4,"acceptance_threshold":0.7,"draft_model":"self"}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
    req.validate().unwrap();
    let spec = req.speculative.as_ref().unwrap();
    assert_eq!(spec.enabled, Some(true));
    assert_eq!(spec.draft_k, Some(4));
    assert_eq!(spec.acceptance_threshold, Some(0.7));
    assert_eq!(spec.draft_model.as_deref(), Some("self"));
}

#[test]
fn responses_request_accepts_string_input_and_user_alias() {
    let raw = r#"{"input":"hi","user":"agent-7"}"#;
    let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.session_id_parsed().unwrap().as_str(), "agent-7");
    assert!(matches!(req.input, ResponsesInput::Text(_)));
}

#[test]
fn responses_request_accepts_speculative_override() {
    let raw = r#"{
        "input":"hi",
        "speculative":{"enabled":false,"draft_k":3,"acceptance_threshold":0.5,"draft_model":"external:/models/draft"}
    }"#;
    let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
    req.validate().unwrap();
    let spec = req.speculative.as_ref().unwrap();
    assert_eq!(spec.enabled, Some(false));
    assert_eq!(spec.draft_k, Some(3));
    assert_eq!(spec.acceptance_threshold, Some(0.5));
    assert_eq!(spec.draft_model.as_deref(), Some("external:/models/draft"));
}

#[test]
fn responses_request_accepts_max_tokens_alias() {
    let raw = r#"{"input":"hi","max_tokens":23}"#;
    let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.max_output_tokens_or_default(), 23);
}

#[test]
fn responses_response_exposes_output_text_and_function_calls() {
    let response = ResponsesResponse::from_output(
        "Qwen3-4B".to_string(),
        1,
        CompletionOutput {
            text: "Let me check.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>".to_string(),
            finish_reason: crate::server_engine::FinishReason::Stop,
            usage: crate::server_engine::TokenUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
            token_logprobs: Vec::new(),
            prompt_token_ids: Vec::new(),
            response_token_ids: Vec::new(),
        },
    );

    let payload = serde_json::to_value(response).unwrap();
    assert_eq!(payload["object"], "response");
    assert_eq!(payload["usage"]["input_tokens"], 2);
    assert_eq!(payload["usage"]["output_tokens"], 3);
    assert_eq!(
        payload["output"][1]["type"],
        serde_json::Value::String("function_call".to_string())
    );
}

#[test]
fn completion_request_rejects_zero_max_tokens() {
    let req: CompletionRequest = serde_json::from_str(r#"{"prompt":"hi","max_tokens":0}"#).unwrap();
    let err = req.validate().expect_err("zero max_tokens should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("max_tokens"));
}

#[test]
fn completion_request_rejects_empty_prompt() {
    let req: CompletionRequest = serde_json::from_str(r#"{"prompt":"   "}"#).unwrap();
    let err = req.validate().expect_err("empty prompt should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert_eq!(err.body.param.as_deref(), Some("prompt"));
    assert!(err.body.message.contains("prompt"));
}

#[test]
fn completion_request_rejects_multi_choice_n() {
    let req: CompletionRequest = serde_json::from_str(r#"{"prompt":"hi","n":2}"#).unwrap();
    let err = req.validate().expect_err("n > 1 should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("only `1`"));
}

#[test]
fn completion_request_rejects_logprobs_request() {
    let req: CompletionRequest = serde_json::from_str(r#"{"prompt":"hi","logprobs":1}"#).unwrap();
    let err = req.validate().expect_err("logprobs should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("logprobs"));
}

#[test]
fn chat_request_rejects_stream_options_without_stream() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{"role":"user","content":"hi"}],
            "stream_options":{"include_usage":true}
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("stream_options without stream should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("stream_options"));
}

#[test]
fn completion_request_accepts_continuous_usage_stats_with_include_usage() {
    let req: CompletionRequest = serde_json::from_str(
        r#"{
            "prompt":"hi",
            "stream":true,
            "stream_options":{"include_usage":true,"continuous_usage_stats":true}
        }"#,
    )
    .unwrap();
    req.validate()
        .expect("continuous_usage_stats should be accepted");
    assert!(req.include_usage_or_default());
    assert!(req.continuous_usage_stats_or_default());
}

#[test]
fn completion_request_rejects_continuous_usage_stats_without_include_usage() {
    let req: CompletionRequest = serde_json::from_str(
        r#"{
            "prompt":"hi",
            "stream":true,
            "stream_options":{"continuous_usage_stats":true}
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("continuous_usage_stats without include_usage should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("continuous_usage_stats"));
}

#[test]
fn chat_request_rejects_non_text_content_parts() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("non-text message parts should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("image_url"));
}

#[test]
fn chat_request_rejects_unknown_message_role() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{"role":"developer","content":"hi"}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("unsupported message roles should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("messages[0].role"));
    assert!(err.body.message.contains("developer"));
}

#[test]
fn chat_request_rejects_empty_messages() {
    let req: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[]}"#).unwrap();
    let err = req.validate().expect_err("empty messages should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("messages"));
}

#[test]
fn chat_request_rejects_non_function_tools() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"web_search","function":{"name":"search"}}]
        }"#,
    )
    .unwrap();
    let err = req.validate().expect_err("non-function tools should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("tools[0].type"));
    assert!(err.body.message.contains("web_search"));
}

#[test]
fn chat_request_rejects_streaming_tools() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{"role":"user","content":"hi"}],
            "stream":true,
            "tools":[{"type":"function","function":{"name":"shell"}}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("streaming chat tools should fail until delta.tool_calls exist");
    assert_eq!(err.body.code, "invalid_parameter");
    assert_eq!(err.body.param.as_deref(), Some("stream"));
    assert!(err.body.message.contains("stream=true"));
    assert!(err.body.message.contains("tool calls"));
}

#[test]
fn chat_request_rejects_invalid_assistant_tool_call_arguments() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{
            "messages":[{
                "role":"assistant",
                "tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"shell","arguments":"not-json"}
                }]
            }]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("invalid assistant tool_call arguments should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(
        err.body
            .message
            .contains("messages[0].tool_calls[0].function.arguments")
    );
}

#[test]
fn responses_request_rejects_tool_message_without_tool_call_id() {
    let req: ResponsesRequest = serde_json::from_str(
        r#"{
            "input":[{"role":"tool","content":"done"}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("tool messages without tool_call_id should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("input[0].tool_call_id"));
}

#[test]
fn responses_request_rejects_streaming_tools() {
    let req: ResponsesRequest = serde_json::from_str(
        r#"{
            "input":"hello",
            "stream":true,
            "tools":[{"type":"function","function":{"name":"shell"}}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("streaming responses tools should fail until function-call deltas exist");
    assert_eq!(err.body.code, "invalid_parameter");
    assert_eq!(err.body.param.as_deref(), Some("stream"));
    assert!(err.body.message.contains("stream=true"));
    assert!(err.body.message.contains("tool calls"));
}

#[test]
fn responses_request_rejects_empty_input() {
    let req: ResponsesRequest = serde_json::from_str(r#"{"input":"   "}"#).unwrap();
    let err = req.validate().expect_err("empty input should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("input"));
}

#[test]
fn responses_request_rejects_invalid_top_p() {
    let req: ResponsesRequest = serde_json::from_str(r#"{"input":"hi","top_p":1.5}"#).unwrap();
    let err = req.validate().expect_err("top_p > 1 should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("top_p"));
}

#[test]
fn responses_request_rejects_non_text_message_parts() {
    let req: ResponsesRequest = serde_json::from_str(
        r#"{
            "input":[{"role":"user","content":[{"type":"input_audio","audio":{"data":"...","format":"wav"}}]}]
        }"#,
    )
    .unwrap();
    let err = req
        .validate()
        .expect_err("non-text responses input parts should fail");
    assert_eq!(err.body.code, "invalid_parameter");
    assert!(err.body.message.contains("input_audio"));
}

#[test]
fn completion_response_drops_non_finite_logprobs() {
    let response = CompletionResponse::from_output(
        "Qwen3-4B".to_string(),
        1,
        CompletionOutput {
            text: "hello".to_string(),
            finish_reason: crate::server_engine::FinishReason::Stop,
            usage: crate::server_engine::TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            token_logprobs: vec![f32::NAN, f32::NEG_INFINITY],
            prompt_token_ids: Vec::new(),
            response_token_ids: Vec::new(),
        },
        false,
    );

    let payload = serde_json::to_value(response).unwrap();
    assert!(payload["choices"][0]["logprobs"].is_null());
    assert!(payload["choices"][0].get("token_ids").is_none());
}

#[test]
fn completion_response_exposes_token_ids_when_requested() {
    let response = CompletionResponse::from_output(
        "Qwen3-4B".to_string(),
        1,
        CompletionOutput {
            text: "hello".to_string(),
            finish_reason: crate::server_engine::FinishReason::Stop,
            usage: crate::server_engine::TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
            token_logprobs: Vec::new(),
            prompt_token_ids: Vec::new(),
            response_token_ids: vec![11, 22],
        },
        true,
    );

    let payload = serde_json::to_value(response).unwrap();
    assert_eq!(
        payload["choices"][0]["token_ids"],
        serde_json::json!([11, 22])
    );
}
