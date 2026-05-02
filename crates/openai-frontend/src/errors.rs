use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct OpenAiError {
    status: StatusCode,
    message: String,
    error_type: String,
    code: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiErrorKind {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    PayloadTooLarge,
    Timeout,
    Internal,
    ServiceUnavailable,
    ContextLengthExceeded,
    UnsupportedFeature,
}

impl OpenAiError {
    pub fn from_kind(
        status: StatusCode,
        kind: OpenAiErrorKind,
        message: impl Into<String>,
    ) -> Self {
        let (error_type, code) = kind_to_openai_fields(kind);
        Self {
            status,
            message: message.into(),
            error_type: error_type.to_string(),
            code: Some(code.to_string()),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            OpenAiErrorKind::InvalidRequest,
            message,
        )
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::from_kind(
            StatusCode::NOT_FOUND,
            OpenAiErrorKind::NotFound,
            format!("model not found: {model}"),
        )
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_GATEWAY,
            OpenAiErrorKind::ServiceUnavailable,
            message,
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::INTERNAL_SERVER_ERROR,
            OpenAiErrorKind::Internal,
            message,
        )
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            OpenAiErrorKind::UnsupportedFeature,
            message,
        )
    }

    pub fn route_not_found(path: impl std::fmt::Display) -> Self {
        Self::from_kind(
            StatusCode::NOT_FOUND,
            OpenAiErrorKind::InvalidRequest,
            format!("route not found: {path}"),
        )
        .with_code("not_found")
    }

    pub fn method_not_allowed(method: impl std::fmt::Display) -> Self {
        Self::from_kind(
            StatusCode::METHOD_NOT_ALLOWED,
            OpenAiErrorKind::InvalidRequest,
            format!("method not allowed: {method}"),
        )
        .with_code("method_not_allowed")
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::PAYLOAD_TOO_LARGE,
            OpenAiErrorKind::PayloadTooLarge,
            message,
        )
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            OpenAiErrorKind::ContextLengthExceeded,
            message,
        )
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::GATEWAY_TIMEOUT,
            OpenAiErrorKind::Timeout,
            message,
        )
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn body(&self) -> ErrorResponse {
        ErrorResponse {
            error: ErrorBody {
                message: self.message.clone(),
                error_type: self.error_type.clone(),
                param: None,
                code: self.code.clone(),
            },
        }
    }
}

impl std::fmt::Display for OpenAiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for OpenAiError {}

fn map_upstream_kind(status_code: u16, upstream_type: &str) -> OpenAiErrorKind {
    match (status_code, upstream_type) {
        (400, "invalid_request_error") => OpenAiErrorKind::InvalidRequest,
        (401, "authentication_error") => OpenAiErrorKind::Authentication,
        (404, "not_found_error") => OpenAiErrorKind::NotFound,
        (500, "server_error") => OpenAiErrorKind::Internal,
        (403, "permission_error") => OpenAiErrorKind::Permission,
        (501, "not_supported_error") => OpenAiErrorKind::UnsupportedFeature,
        (503, "unavailable_error") => OpenAiErrorKind::ServiceUnavailable,
        (400, "exceed_context_size_error") => OpenAiErrorKind::ContextLengthExceeded,
        (400, _) => OpenAiErrorKind::InvalidRequest,
        (401, _) => OpenAiErrorKind::Authentication,
        (403, _) => OpenAiErrorKind::Permission,
        (404, _) => OpenAiErrorKind::NotFound,
        (429, _) => OpenAiErrorKind::RateLimit,
        (503, _) => OpenAiErrorKind::ServiceUnavailable,
        _ => OpenAiErrorKind::Internal,
    }
}

fn kind_to_openai_fields(kind: OpenAiErrorKind) -> (&'static str, &'static str) {
    match kind {
        OpenAiErrorKind::InvalidRequest => ("invalid_request_error", "invalid_value"),
        OpenAiErrorKind::Authentication => ("authentication_error", "invalid_api_key"),
        OpenAiErrorKind::Permission => ("permission_error", "insufficient_quota"),
        OpenAiErrorKind::NotFound => ("invalid_request_error", "model_not_found"),
        OpenAiErrorKind::RateLimit => ("rate_limit_error", "rate_limit_exceeded"),
        OpenAiErrorKind::PayloadTooLarge => ("invalid_request_error", "payload_too_large"),
        OpenAiErrorKind::Timeout => ("server_error", "timeout"),
        OpenAiErrorKind::Internal => ("server_error", "internal_server_error"),
        OpenAiErrorKind::ServiceUnavailable => ("server_error", "service_unavailable"),
        OpenAiErrorKind::ContextLengthExceeded => {
            ("invalid_request_error", "context_length_exceeded")
        }
        OpenAiErrorKind::UnsupportedFeature => {
            ("invalid_request_error", "unsupported_model_feature")
        }
    }
}

fn extract_message(value: &Value) -> Option<String> {
    value
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

fn extract_upstream_type(value: &Value) -> Option<String> {
    value
        .get("type")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

pub fn already_openai_error(value: &Value) -> bool {
    value
        .get("error")
        .and_then(Value::as_object)
        .map(|error| {
            error.get("message").and_then(Value::as_str).is_some()
                && error.get("type").and_then(Value::as_str).is_some()
        })
        .unwrap_or(false)
}

pub fn map_upstream_error_body(status_code: u16, body: &[u8]) -> Option<Vec<u8>> {
    if status_code < 400 {
        return None;
    }

    let parsed = serde_json::from_slice::<Value>(body).ok();
    if let Some(value) = parsed.as_ref() {
        if already_openai_error(value) {
            return None;
        }
    }

    let message = parsed
        .as_ref()
        .and_then(extract_message)
        .or_else(|| {
            let text = String::from_utf8_lossy(body).trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        })
        .unwrap_or_else(|| "Unknown error".to_string());

    let upstream_type = parsed
        .as_ref()
        .and_then(extract_upstream_type)
        .unwrap_or_default();
    let kind = map_upstream_kind(status_code, &upstream_type);
    let status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let error = OpenAiError::from_kind(status, kind, message);
    Some(serde_json::to_vec(&error.body()).expect("serializing OpenAI error body should not fail"))
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}
