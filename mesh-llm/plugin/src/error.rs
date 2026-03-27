use rmcp::model::ErrorCode;

use crate::proto;

pub const STARTUP_DISABLED_ERROR_CODE: i32 = -32_010;

#[derive(Debug, Clone)]
pub struct PluginError {
    pub code: i32,
    pub message: String,
    pub data_json: String,
}

impl PluginError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INVALID_REQUEST.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn method_not_found(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::METHOD_NOT_FOUND.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INVALID_PARAMS.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn startup_disabled(message: impl Into<String>) -> Self {
        Self {
            code: STARTUP_DISABLED_ERROR_CODE,
            message: message.into(),
            data_json: serde_json::json!({ "status": "disabled" }).to_string(),
        }
    }

    pub(crate) fn into_error_response(self) -> proto::ErrorResponse {
        proto::ErrorResponse {
            code: self.code,
            message: self.message,
            data_json: self.data_json,
        }
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PluginError {}

impl From<anyhow::Error> for PluginError {
    fn from(value: anyhow::Error) -> Self {
        Self::internal(value.to_string())
    }
}

pub type PluginResult<T> = std::result::Result<T, PluginError>;
pub type PluginRpcResult = PluginResult<proto::envelope::Payload>;
