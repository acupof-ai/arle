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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Box<ApiErrorDetails>>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetails {
    pub kind: String,
    pub chain: Vec<String>,
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
                param: None,
                details: None,
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

    /// 503 Service Unavailable with structured cause details. Use this for
    /// real admission/capacity failures where retrying may be appropriate,
    /// but the operator still needs the upstream error chain in logs/body.
    pub fn service_unavailable_with_details(
        message: impl Into<String>,
        kind: impl Into<String>,
        chain: Vec<String>,
    ) -> Self {
        let mut error = Self::service_unavailable(message);
        error.body.details = Some(Box::new(ApiErrorDetails {
            kind: kind.into(),
            chain,
        }));
        error
    }

    /// 500/501 inference failure — model/scheduler/kernel execution failed
    /// after request admission. The chain carries operator/kernel context.
    pub fn inference_failed(kind: impl Into<String>, chain: Vec<String>) -> Self {
        let kind = kind.into();
        let architectural = kind == "architectural_deferral"
            || chain.iter().any(|cause| {
                cause.contains("CUDA_ERROR_NOT_SUPPORTED")
                    || cause.contains("cudaErrorNotSupported")
                    || cause.contains("operation not supported")
            });
        let mut error = Self::new(
            if architectural {
                StatusCode::NOT_IMPLEMENTED
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            },
            chain
                .first()
                .cloned()
                .unwrap_or_else(|| "Inference request failed".to_string()),
            "server_error",
            if architectural {
                "architectural_deferral"
            } else {
                "inference_failed"
            },
        );
        error.body.details = Some(Box::new(ApiErrorDetails { kind, chain }));
        error
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

    #[must_use]
    pub fn with_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.push((name, value));
        self
    }

    #[must_use]
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.body.param = Some(param.into());
        self
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
