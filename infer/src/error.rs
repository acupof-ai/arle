//! Structured error types for the inference engine.
//!
//! `ApiError` provides OpenAI-compatible JSON error responses for the HTTP API.
//! Internal error categories use `thiserror` for structured error variants.

use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Error response body following the OpenAI API error format.
///
/// ```json
/// { "error": { "message": "...", "type": "...", "code": "..." } }
/// ```
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: &'static str,
}

/// HTTP error response with status code and OpenAI-compatible JSON body.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ApiErrorBody,
    headers: Vec<(HeaderName, HeaderValue)>,
}

impl ApiError {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        error_type: &'static str,
        code: &'static str,
    ) -> Self {
        Self {
            status,
            body: ApiErrorBody {
                message: message.into(),
                error_type,
                code,
            },
            headers: Vec::new(),
        }
    }

    /// 400 Bad Request — invalid client input.
    pub fn bad_request(message: impl Into<String>, code: &'static str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            code,
        )
    }

    /// 503 Service Unavailable — scheduler overloaded or unavailable.
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            message,
            "server_error",
            "service_unavailable",
        )
    }

    /// 401 Unauthorized — missing or invalid authentication credentials.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        let mut error = Self::new(
            StatusCode::UNAUTHORIZED,
            message,
            "invalid_request_error",
            "unauthorized",
        );
        error.headers.push((
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static(r#"Bearer realm="agent-infer""#),
        ));
        error
    }

    /// 404 Not Found — route or optional subsystem is not available.
    pub fn not_found(message: impl Into<String>, code: &'static str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            message,
            "invalid_request_error",
            code,
        )
    }

    /// 405 Method Not Allowed — route exists but doesn't accept this method.
    pub fn method_not_allowed(message: impl Into<String>, code: &'static str) -> Self {
        Self::new(
            StatusCode::METHOD_NOT_ALLOWED,
            message,
            "invalid_request_error",
            code,
        )
    }

    /// 413 Payload Too Large — request body exceeded the accepted limit.
    pub fn payload_too_large(message: impl Into<String>, code: &'static str) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            message,
            "invalid_request_error",
            code,
        )
    }

    /// 504 Gateway Timeout — request took too long.
    pub fn timeout(elapsed_secs: u64) -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            format!("Request timed out after {elapsed_secs}s"),
            "server_error",
            "timeout",
        )
    }
}

#[derive(Serialize)]
struct ErrorWrapper<'a> {
    error: &'a ApiErrorBody,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        for (name, value) in self.headers {
            headers.insert(name, value);
        }
        let wrapper = ErrorWrapper { error: &self.body };
        let body = serde_json::to_string(&wrapper).unwrap_or_else(|_| {
            r#"{"error":{"message":"Internal error","type":"server_error","code":"serialization_failed"}}"#.to_string()
        });
        (self.status, headers, body).into_response()
    }
}
