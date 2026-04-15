use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    Internal,
    ServiceUnavailable,
    ContextLengthExceeded,
    UnsupportedFeature,
}

fn map_kind(status_code: u16, upstream_type: &str) -> ErrorKind {
    match (status_code, upstream_type) {
        (400, "invalid_request_error") => ErrorKind::InvalidRequest,
        (401, "authentication_error") => ErrorKind::Authentication,
        (404, "not_found_error") => ErrorKind::NotFound,
        (500, "server_error") => ErrorKind::Internal,
        (403, "permission_error") => ErrorKind::Permission,
        (501, "not_supported_error") => ErrorKind::UnsupportedFeature,
        (503, "unavailable_error") => ErrorKind::ServiceUnavailable,
        (400, "exceed_context_size_error") => ErrorKind::ContextLengthExceeded,
        // Fallback by status code if upstream type does not match.
        (400, _) => ErrorKind::InvalidRequest,
        (401, _) => ErrorKind::Authentication,
        (403, _) => ErrorKind::Permission,
        (404, _) => ErrorKind::NotFound,
        (429, _) => ErrorKind::RateLimit,
        (503, _) => ErrorKind::ServiceUnavailable,
        _ => ErrorKind::Internal,
    }
}

fn kind_to_openai_fields(kind: ErrorKind) -> (&'static str, &'static str) {
    match kind {
        ErrorKind::InvalidRequest => ("invalid_request_error", "invalid_value"),
        ErrorKind::Authentication => ("authentication_error", "invalid_api_key"),
        ErrorKind::Permission => ("permission_error", "insufficient_quota"),
        ErrorKind::NotFound => ("invalid_request_error", "model_not_found"),
        ErrorKind::RateLimit => ("rate_limit_error", "rate_limit_exceeded"),
        ErrorKind::Internal => ("server_error", "internal_server_error"),
        ErrorKind::ServiceUnavailable => ("server_error", "service_unavailable"),
        ErrorKind::ContextLengthExceeded => ("invalid_request_error", "context_length_exceeded"),
        ErrorKind::UnsupportedFeature => ("invalid_request_error", "unsupported_model_feature"),
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

pub(crate) fn already_openai_error(value: &Value) -> bool {
    value
        .get("error")
        .and_then(Value::as_object)
        .map(|error| {
            error.get("message").and_then(Value::as_str).is_some()
                && error.get("type").and_then(Value::as_str).is_some()
        })
        .unwrap_or(false)
}

pub(crate) fn map_upstream_error_body(status_code: u16, body: &[u8]) -> Option<Vec<u8>> {
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
    let kind = map_kind(status_code, &upstream_type);
    let (error_type, code) = kind_to_openai_fields(kind);

    Some(
        serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": Value::Null,
                "code": code,
            }
        })
        .to_string()
        .into_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_error_mapping_from_llama_types() {
        let mappings = vec![
            (
                400,
                json!({"type": "invalid_request_error", "message": "error"}),
                "invalid_request_error",
                "invalid_value",
            ),
            (
                401,
                json!({"type": "authentication_error", "message": "error"}),
                "authentication_error",
                "invalid_api_key",
            ),
            (
                404,
                json!({"type": "not_found_error", "message": "error"}),
                "invalid_request_error",
                "model_not_found",
            ),
            (
                500,
                json!({"type": "server_error", "message": "error"}),
                "server_error",
                "internal_server_error",
            ),
            (
                403,
                json!({"type": "permission_error", "message": "error"}),
                "permission_error",
                "insufficient_quota",
            ),
            (
                501,
                json!({"type": "not_supported_error", "message": "error"}),
                "invalid_request_error",
                "unsupported_model_feature",
            ),
            (
                503,
                json!({"type": "unavailable_error", "message": "error"}),
                "server_error",
                "service_unavailable",
            ),
            (
                400,
                json!({"type": "exceed_context_size_error", "message": "error"}),
                "invalid_request_error",
                "context_length_exceeded",
            ),
        ];

        for (status, body, expected_type, expected_code) in mappings {
            let mapped = map_upstream_error_body(status, body.to_string().as_bytes())
                .expect("mapping should produce a body");
            let value: Value = serde_json::from_slice(&mapped).expect("mapped body must be json");
            assert_eq!(
                value["error"]["type"].as_str().unwrap_or_default(),
                expected_type
            );
            assert_eq!(
                value["error"]["code"].as_str().unwrap_or_default(),
                expected_code
            );
        }
    }

    #[test]
    fn test_openai_error_passthrough_not_remapped() {
        let openai = json!({
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_value"
            }
        });
        assert!(map_upstream_error_body(400, openai.to_string().as_bytes()).is_none());
    }

    #[test]
    fn test_plain_text_error_maps_message() {
        let mapped = map_upstream_error_body(503, b"backend down").expect("must map text");
        let value: Value = serde_json::from_slice(&mapped).expect("mapped body must be json");
        assert_eq!(
            value["error"]["message"].as_str().unwrap_or_default(),
            "backend down"
        );
        assert_eq!(
            value["error"]["type"].as_str().unwrap_or_default(),
            "server_error"
        );
        assert_eq!(
            value["error"]["code"].as_str().unwrap_or_default(),
            "service_unavailable"
        );
    }
}
