//! Structured error types for the inference engine.
//!
//! `ApiError` provides OpenAI-compatible JSON error responses for the HTTP API.
//! Internal error categories use `thiserror` for structured error variants.

use axum::http::StatusCode;
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
}

impl ApiError {
    /// 400 Bad Request — invalid client input.
    pub fn bad_request(message: impl Into<String>, code: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ApiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error",
                code,
            },
        }
    }

    /// 503 Service Unavailable — scheduler overloaded or unavailable.
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ApiErrorBody {
                message: message.into(),
                error_type: "server_error",
                code: "service_unavailable",
            },
        }
    }

    /// 401 Unauthorized — missing or invalid authentication credentials.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: ApiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error",
                code: "unauthorized",
            },
        }
    }

    /// 504 Gateway Timeout — request took too long.
    pub fn timeout(elapsed_secs: u64) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            body: ApiErrorBody {
                message: format!("Request timed out after {elapsed_secs}s"),
                error_type: "server_error",
                code: "timeout",
            },
        }
    }
}

#[derive(Serialize)]
struct ErrorWrapper<'a> {
    error: &'a ApiErrorBody,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let wrapper = ErrorWrapper { error: &self.body };
        let body = serde_json::to_string(&wrapper).unwrap_or_else(|_| {
            r#"{"error":{"message":"Internal error","type":"server_error","code":"serialization_failed"}}"#.to_string()
        });
        (
            self.status,
            [(
                axum::http::header::CONTENT_TYPE,
                "application/json; charset=utf-8",
            )],
            body,
        )
            .into_response()
    }
}
