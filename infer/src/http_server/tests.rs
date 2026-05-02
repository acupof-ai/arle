//! HTTP server end-to-end tests.
//!
//! Split out of `http_server.rs` (pure structural refactor — no behavior change).

#[cfg(test)]
mod tests {
    // Pull in the public surface and the internal items the test bodies
    // reference.
    use super::super::router::{build_app, build_app_with_config, build_app_with_metrics};
    use super::super::types::{
        HTTP_REQUEST_ID_HEADER, HealthResponse, HttpServerConfig, TrainControlTarget,
    };
    use crate::metrics::ServerMetrics;
    use crate::scheduler::{IncomingRequest, SchedulerHandle};
    use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
    use std::sync::Arc;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use tower::util::ServiceExt;

    fn mock_scheduler(model_id: &str) -> SchedulerHandle {
        mock_scheduler_with_deltas(
            model_id,
            vec![
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                    token_ids: Vec::new(),
                },
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                    token_ids: Vec::new(),
                },
            ],
            true,
        )
    }

    fn mock_scheduler_with_deltas(
        model_id: &str,
        deltas: Vec<CompletionStreamDelta>,
        prefix_prompt_on_first_delta: bool,
    ) -> SchedulerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IncomingRequest>();
        let model_id = model_id.to_string();

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                for (index, delta) in deltas.iter().enumerate() {
                    let text_delta = if prefix_prompt_on_first_delta && index == 0 {
                        format!("ok:{}{}", req.prompt, delta.text_delta)
                    } else {
                        delta.text_delta.clone()
                    };
                    let _ = req.delta_tx.send(CompletionStreamDelta {
                        text_delta,
                        finish_reason: delta.finish_reason,
                        usage: delta.usage,
                        logprob: delta.logprob,
                        token_ids: delta.token_ids.clone(),
                    });
                }
            }
        });

        SchedulerHandle::from_parts(tx, &model_id)
    }

    fn spawn_train_control_stub_once(
        expected_method: &'static str,
        expected_target: &'static str,
        status: u16,
        body: &'static str,
    ) -> (TrainControlTarget, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind train control stub");
        let addr = listener.local_addr().expect("stub addr");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept train control stub");
            let mut reader = BufReader::new(stream.try_clone().expect("clone stub stream"));
            let mut request_line = String::new();
            reader
                .read_line(&mut request_line)
                .expect("read request line");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("");
            let target = parts.next().unwrap_or("");
            assert_eq!(method, expected_method);
            assert_eq!(target, expected_target);

            loop {
                let mut line = String::new();
                let bytes = reader.read_line(&mut line).expect("read header");
                if bytes == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
            }

            write!(
                stream,
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .expect("write stub response");
        });
        let target = TrainControlTarget::parse(&format!("http://{addr}"))
            .expect("parse train control target");
        (target, handle)
    }

    #[test]
    fn train_control_target_parses_and_normalizes_base_path() {
        let target =
            TrainControlTarget::parse("http://127.0.0.1:9123/base/child/").expect("parse target");
        assert_eq!(target.authority(), "127.0.0.1:9123");
        assert_eq!(
            target.request_path("/v1/train/status", None),
            "/base/child/v1/train/status"
        );
        assert_eq!(
            target.request_path("/v1/train/events", Some("after_seq=7")),
            "/base/child/v1/train/events?after_seq=7"
        );
    }

    #[tokio::test]
    async fn completion_rejects_unavailable_model_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["error"]["code"], "model_not_found");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("qwen3-8b")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_response_includes_generated_request_id_header() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let request_id = response
            .headers()
            .get(HTTP_REQUEST_ID_HEADER)
            .expect("generated x-request-id header")
            .to_str()
            .expect("request id header ascii");
        assert!(uuid::Uuid::parse_str(request_id).is_ok());
    }

    #[tokio::test]
    async fn response_preserves_client_supplied_request_id_header() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .header(HTTP_REQUEST_ID_HEADER, "req-client-42")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[HTTP_REQUEST_ID_HEADER], "req-client-42");
    }

    #[tokio::test]
    async fn completion_response_includes_token_ids_when_requested() {
        let app = build_app(mock_scheduler_with_deltas(
            "Qwen3-4B",
            vec![
                CompletionStreamDelta {
                    text_delta: "A".to_string(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                    token_ids: vec![11],
                },
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 2,
                        total_tokens: 3,
                    }),
                    logprob: None,
                    token_ids: vec![22],
                },
            ],
            true,
        ));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":2,"return_token_ids":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            payload["choices"][0]["token_ids"],
            serde_json::json!([11, 22])
        );
    }

    #[tokio::test]
    async fn streaming_response_uses_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","prompt":"hello","max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""model":"Qwen3-8B""#),
            "payload={payload}"
        );
        assert!(
            !payload.contains(r#""model":"qwen3-4b""#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"));
    }

    #[tokio::test]
    async fn streaming_response_includes_usage_when_requested() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload
                .contains(r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn streaming_response_accepts_continuous_usage_stats_probe_shape() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""text":"ok:hello""#),
            "payload={payload}"
        );
        assert!(
            payload
                .contains(r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"), "payload={payload}");
    }

    #[tokio::test]
    async fn completion_rejects_empty_prompt() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"   ","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert_eq!(payload["error"]["param"], "prompt");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("prompt")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_rejects_malformed_json_with_openai_error_body() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":"oops"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "invalid_json");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Invalid JSON request body")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_requires_json_content_type() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":1}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_json");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Content-Type")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_rejects_payload_too_large_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let oversized_body = format!(
            r#"{{"prompt":"{}","max_tokens":1}}"#,
            "x".repeat(17 * 1024 * 1024)
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(oversized_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "payload_too_large");
    }

    #[tokio::test]
    async fn completion_accepts_large_body_within_explicit_limit() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let large_body = format!(
            r#"{{"model":"qwen3-4b","prompt":"{}","max_tokens":1}}"#,
            "x".repeat(3 * 1024 * 1024)
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(large_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_route_returns_structured_not_found_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/unknown")
            .header(HTTP_REQUEST_ID_HEADER, "req-404")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );
        assert_eq!(response.headers()[HTTP_REQUEST_ID_HEADER], "req-404");

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "route_not_found");
        assert_eq!(
            payload["error"]["message"],
            "Route `/v1/unknown` was not found"
        );
    }

    #[tokio::test]
    async fn wrong_method_returns_structured_method_not_allowed_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/completions")
            .header(HTTP_REQUEST_ID_HEADER, "req-405")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );
        assert_eq!(response.headers()["allow"], "POST");
        assert_eq!(response.headers()[HTTP_REQUEST_ID_HEADER], "req-405");

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "method_not_allowed");
        assert_eq!(
            payload["error"]["message"],
            "Method `GET` is not allowed for `/v1/completions`"
        );
    }

    #[tokio::test]
    async fn get_route_method_errors_include_allow_header() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(response.headers()["allow"], "GET, HEAD");
    }

    #[tokio::test]
    async fn completion_rejects_zero_max_tokens_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":0}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("max_tokens")
        );
    }

    #[tokio::test]
    async fn completion_rejects_unsupported_parameter_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"prompt":"hello","max_tokens":1,"response_format":{"type":"json_object"}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert_eq!(payload["error"]["param"], "response_format");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("response_format")),
            "payload={payload}"
        );
    }

    // -----------------------------------------------------------------------
    // /v1/chat/completions tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn chat_completion_basic() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Model id comes from the loaded model, not the request.
        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        // Content should contain something (mock returns "ok:<prompt>").
        assert!(
            payload["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "expected non-empty content, got: {payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_empty_messages() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("messages")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_stream_options_without_stream() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stream_options")
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_unsupported_parameter_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("tool_choice")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_unsupported_message_role_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"developer","content":"hi"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("messages[0].role")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_unavailable_model_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "model_not_found");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("qwen3-8b")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_streaming() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("chat.completion.chunk"),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"), "payload={payload}");
        // First chunk must carry role.
        assert!(
            payload.contains(r#""role":"assistant""#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("infer_requests_total"),
            "payload={payload}"
        );
        assert!(
            payload.contains("infer_requests_active"),
            "payload={payload}"
        );
        assert!(
            payload.contains("infer_scheduler_plan_total"),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn healthz_endpoint_returns_json_status() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<HealthResponse>(&body).unwrap(),
            HealthResponse::live()
        );
    }

    #[tokio::test]
    async fn readyz_endpoint_returns_model_identity() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/readyz")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<HealthResponse>(&body).unwrap(),
            HealthResponse::ready("Qwen3-4B")
        );
    }

    #[tokio::test]
    async fn stats_endpoint_returns_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();
        assert!(!payload.is_empty(), "stats body should not be empty");
        assert!(payload.contains("step_phase_us="));
        assert!(payload.contains("plan_label="));
        assert!(payload.contains("matched_prefix_tokens="));
        assert!(payload.contains("resume_prefill_tokens="));
    }

    #[tokio::test]
    async fn stats_endpoint_returns_json_when_requested() {
        let metrics = ServerMetrics::new("Qwen3-4B");
        metrics.record_request_cache(
            Some(&crate::types::SessionId::from("w3-warm-000")),
            48,
            120,
            72,
        );
        let app = build_app_with_metrics(mock_scheduler("Qwen3-4B"), metrics);
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats?format=json")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["prefix_hit_rate"], serde_json::json!(1.0));
        assert_eq!(payload["prefix_skip_rate"], serde_json::json!(0.4));
        assert_eq!(payload["session_affinity_hit"], serde_json::json!(1));
        assert_eq!(payload["session_affinity_miss"], serde_json::json!(0));
        assert_eq!(payload["matched_prefix_tokens"], serde_json::json!(48));
        assert_eq!(payload["resume_prefill_tokens"], serde_json::json!(72));
        assert_eq!(
            payload["last_request"]["session_id"],
            serde_json::json!("w3-warm-000")
        );
        assert_eq!(
            payload["sessions"]["w3-warm-000"]["matched_prefix_tokens"],
            serde_json::json!(48)
        );
    }

    #[tokio::test]
    async fn train_status_returns_not_found_when_bridge_is_unconfigured() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn train_status_proxies_json_from_control_plane() {
        let (target, handle) = spawn_train_control_stub_once(
            "GET",
            "/v1/train/status",
            200,
            r#"{"iter":3,"finished":false}"#,
        );
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                train_control_target: Some(target),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"iter":3,"finished":false})
        );
        handle.join().expect("join train control stub");
    }

    #[tokio::test]
    async fn train_events_proxy_forwards_after_seq_query() {
        let (target, handle) = spawn_train_control_stub_once(
            "GET",
            "/v1/train/events?after_seq=7",
            200,
            r#"{"events":[],"latest_seq":7}"#,
        );
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                train_control_target: Some(target),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/events?after_seq=7")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"events":[],"latest_seq":7})
        );
        handle.join().expect("join train control stub");
    }

    #[tokio::test]
    async fn completions_reject_missing_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()["www-authenticate"],
            r#"Bearer realm="agent-infer""#
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "unauthorized");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Missing Authorization")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completions_accept_valid_bearer_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret-token")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn stats_reject_missing_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()["www-authenticate"],
            r#"Bearer realm="agent-infer""#
        );
    }

    #[tokio::test]
    async fn completions_reject_invalid_api_key_with_bearer_challenge() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wrong-token")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()["www-authenticate"],
            r#"Bearer realm="agent-infer""#
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "unauthorized");
        assert_eq!(payload["error"]["message"], "Invalid API key");
    }

    #[tokio::test]
    async fn train_status_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                train_control_target: Some(
                    TrainControlTarget::parse("http://127.0.0.1:9123").expect("parse target"),
                ),
                pool_models: Vec::new(),
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn chat_completion_streaming_with_usage() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""prompt_tokens":1"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_streaming_rejects_tools() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "messages":[{"role":"user","content":"hi"}],
                    "stream":true,
                    "tools":[{"type":"function","function":{"name":"shell"}}]
                }"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert_eq!(payload["error"]["param"], "stream");
        assert!(
            payload["error"]["message"].as_str().is_some_and(|message| {
                message.contains("stream=true") && message.contains("tool calls")
            }),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_returns_structured_tool_calls() {
        let app = build_app(mock_scheduler_with_deltas(
            "Qwen3-4B",
            vec![
                CompletionStreamDelta {
                    text_delta:
                        "\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>"
                            .to_string(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                    token_ids: Vec::new(),
                },
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                    token_ids: Vec::new(),
                },
            ],
            false,
        ));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
        assert!(payload["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "shell"
        );
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!({"command":"pwd"}).to_string()
        );
    }

    #[tokio::test]
    async fn models_endpoint_returns_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["object"], "list");
        assert_eq!(payload["data"][0]["id"], "Qwen3-8B");
    }

    #[tokio::test]
    async fn models_endpoint_lists_configured_pool_models_as_unloaded_stubs() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-8B"),
            crate::metrics::ServerMetrics::new(""),
            HttpServerConfig {
                pool_models: vec![
                    crate::server_engine::EnginePoolModelSpec::parse_cli(
                        "embed=/models/embed,type=embedding,aliases=vision-embed",
                    )
                    .expect("pool spec"),
                ],
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["data"][0]["id"], "Qwen3-8B");
        assert_eq!(payload["data"][1]["id"], "embed");
        assert_eq!(payload["data"][1]["model_type"], "embedding");
        assert_eq!(payload["data"][1]["loaded"], false);
        assert_eq!(payload["data"][1]["aliases"][0], "vision-embed");
    }

    #[tokio::test]
    async fn serving_identity_is_snapshotted_once_and_reused_by_http_handlers() {
        use crate::request_handle::{DflashStatus, RequestHandle, SubmitError};
        use crate::scheduler::IncomingRequest;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SnapshotHandle {
            submit_tx: tokio::sync::mpsc::UnboundedSender<IncomingRequest>,
            model_id: String,
            dflash_status: Option<DflashStatus>,
            model_id_calls: Arc<AtomicUsize>,
            dflash_calls: Arc<AtomicUsize>,
        }

        impl RequestHandle for SnapshotHandle {
            fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
                self.submit_tx.send(req).map_err(|_| SubmitError)
            }

            fn model_id(&self) -> &str {
                let calls = self.model_id_calls.fetch_add(1, Ordering::SeqCst) + 1;
                assert_eq!(calls, 1, "model_id() should only be called at build time");
                &self.model_id
            }

            fn dflash_status(&self) -> Option<DflashStatus> {
                let calls = self.dflash_calls.fetch_add(1, Ordering::SeqCst) + 1;
                assert_eq!(
                    calls, 1,
                    "dflash_status() should only be called at build time"
                );
                self.dflash_status.clone()
            }
        }

        let model_id_calls = Arc::new(AtomicUsize::new(0));
        let dflash_calls = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IncomingRequest>();
        let model_id = "BootModel".to_string();
        let dflash_status = Some(DflashStatus {
            draft_model: "draft/boot-model".to_string(),
            speculative_tokens: 4,
        });
        let handle = SnapshotHandle {
            submit_tx: tx,
            model_id: model_id.clone(),
            dflash_status: dflash_status.clone(),
            model_id_calls: model_id_calls.clone(),
            dflash_calls: dflash_calls.clone(),
        };

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: format!("ok:{}", req.prompt),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                    token_ids: Vec::new(),
                });
            }
        });

        let app = build_app(handle);
        assert_eq!(model_id_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dflash_calls.load(Ordering::SeqCst), 1);

        for (method, uri, body) in [
            (
                "POST",
                "/v1/completions",
                r#"{"prompt":"hello","max_tokens":1}"#,
            ),
            (
                "POST",
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#,
            ),
            (
                "POST",
                "/v1/responses",
                r#"{"input":"hello","max_output_tokens":1}"#,
            ),
        ] {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();

            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(payload["model"], model_id, "uri={uri}");
        }

        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["data"][0]["id"], model_id);
        let dflash = &payload["data"][0]["dflash"];
        assert_eq!(dflash["enabled"], true);
        assert_eq!(dflash["draft"], "draft/boot-model");
        assert_eq!(dflash["speculative_tokens"], 4);
        assert!(dflash["acceptance_rate"].is_null());

        assert_eq!(model_id_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dflash_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn models_endpoint_omits_dflash_when_handle_reports_none() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Baseline (non-DFlash) runtimes must keep the original JSON shape —
        // the `dflash` key is skipped entirely, not emitted as `null`.
        assert!(
            !payload["data"][0]
                .as_object()
                .unwrap()
                .contains_key("dflash"),
            "dflash key must be omitted when RequestHandle reports None, got {payload}"
        );
    }

    #[tokio::test]
    async fn models_endpoint_surfaces_dflash_status_when_reported() {
        use crate::request_handle::{DflashStatus, RequestHandle, SubmitError};
        use crate::scheduler::IncomingRequest;

        struct DflashHandle;
        impl RequestHandle for DflashHandle {
            fn submit(&self, _req: IncomingRequest) -> Result<(), SubmitError> {
                Ok(())
            }
            fn model_id(&self) -> &str {
                "Qwen3.5-4B-MLX-4bit"
            }
            fn dflash_status(&self) -> Option<DflashStatus> {
                Some(DflashStatus {
                    draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
                    speculative_tokens: 5,
                })
            }
        }

        let app = build_app(DflashHandle);
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let dflash = &payload["data"][0]["dflash"];
        assert_eq!(dflash["enabled"], true);
        assert_eq!(dflash["draft"], "z-lab/Qwen3.5-4B-DFlash");
        assert_eq!(dflash["speculative_tokens"], 5);
        // No speculative blocks have run in a test build → `acceptance_rate`
        // must serialise as JSON `null`, not 0.0, so dashboards can tell
        // "no data yet" apart from "everything rejected".
        assert!(
            dflash["acceptance_rate"].is_null(),
            "acceptance_rate must be null before any blocks run, got {dflash}"
        );
    }

    #[tokio::test]
    async fn models_endpoint_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn responses_endpoint_returns_openai_style_response_object() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","input":"hello","max_output_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["object"], "response");
        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["usage"]["input_tokens"], 1);
        assert!(
            payload["output_text"]
                .as_str()
                .is_some_and(|text| text.contains("hello")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_invalid_sampling_knob() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"hello","top_p":1.5}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("top_p")
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_empty_input_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"   "}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("input")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_unavailable_model_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","input":"hello","max_output_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "model_not_found");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("qwen3-8b")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_unsupported_parameter_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"input":"hello","max_output_tokens":1,"parallel_tool_calls":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("parallel_tool_calls")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_non_function_tools_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "input":"hello",
                    "max_output_tokens":1,
                    "tools":[{"type":"web_search","function":{"name":"search"}}]
                }"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("tools[0].type")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_streaming_tools_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "input":"hello",
                    "stream":true,
                    "tools":[{"type":"function","function":{"name":"shell"}}]
                }"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert_eq!(payload["error"]["param"], "stream");
        assert!(
            payload["error"]["message"].as_str().is_some_and(|message| {
                message.contains("stream=true") && message.contains("tool calls")
            }),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_streams_deltas_and_final_event_before_done() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"input":"hello","max_output_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        let created_pos = payload
            .find("event: response.created")
            .unwrap_or_else(|| panic!("missing created event: {payload}"));
        let delta_pos = payload
            .find("event: response.output_text.delta")
            .unwrap_or_else(|| panic!("missing delta event: {payload}"));
        let completed_pos = payload
            .find("event: response.completed")
            .unwrap_or_else(|| panic!("missing completed event: {payload}"));
        let done_pos = payload
            .find("[DONE]")
            .unwrap_or_else(|| panic!("missing terminal done event: {payload}"));

        assert!(
            created_pos < delta_pos && delta_pos < completed_pos && completed_pos < done_pos,
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""delta":"ok:<|im_start|>user"#),
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""status":"completed""#),
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"hello","max_output_tokens":1}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
