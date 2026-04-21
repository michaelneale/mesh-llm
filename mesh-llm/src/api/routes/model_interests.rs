use super::super::{
    http::{respond_error, respond_json},
    LocalModelInterest, MeshApi,
};
use crate::models::canonicalize_interest_model_ref;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

#[derive(Debug, Deserialize)]
struct UpsertModelInterestRequest {
    model_ref: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Debug)]
struct ParsedUpsertModelInterestRequest {
    model_ref: String,
    source: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelInterestListResponse {
    model_interests: Vec<LocalModelInterest>,
}

#[derive(Debug, Serialize)]
struct UpsertModelInterestResponse {
    created: bool,
    interest: LocalModelInterest,
    model_interests: Vec<LocalModelInterest>,
}

#[derive(Debug, Serialize)]
struct DeleteModelInterestResponse {
    removed: bool,
    model_ref: String,
    model_interests: Vec<LocalModelInterest>,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path) {
        ("GET", "/api/model-interests") => handle_list(stream, state).await,
        ("POST", "/api/model-interests") => handle_upsert(stream, state, body).await,
        ("DELETE", path) if path.starts_with("/api/model-interests/") => {
            handle_delete(stream, state, path).await
        }
        _ => Ok(()),
    }
}

async fn handle_list(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    respond_json(
        stream,
        200,
        &ModelInterestListResponse {
            model_interests: state.model_interests().await,
        },
    )
    .await
}

async fn handle_upsert(stream: &mut TcpStream, state: &MeshApi, body: &str) -> anyhow::Result<()> {
    let request = match parse_upsert_request(body) {
        Ok(request) => request,
        Err(message) => return respond_error(stream, 400, &message).await,
    };

    let canonical_ref = match canonicalize_interest_model_ref(&request.model_ref) {
        Ok(model_ref) => model_ref,
        Err(err) => return respond_error(stream, 400, &err.to_string()).await,
    };

    let (interest, created) = state
        .upsert_model_interest(canonical_ref, normalize_submission_source(request.source))
        .await;
    let model_interests = state.model_interests().await;
    respond_json(
        stream,
        if created { 201 } else { 200 },
        &UpsertModelInterestResponse {
            created,
            interest,
            model_interests,
        },
    )
    .await
}

async fn handle_delete(stream: &mut TcpStream, state: &MeshApi, path: &str) -> anyhow::Result<()> {
    let Some(decoded_ref) = decode_model_interest_path(path) else {
        return respond_error(stream, 400, "Missing model interest path").await;
    };
    let canonical_ref = match canonicalize_interest_model_ref(&decoded_ref) {
        Ok(model_ref) => model_ref,
        Err(err) => return respond_error(stream, 400, &err.to_string()).await,
    };

    let removed = state.remove_model_interest(&canonical_ref).await;
    let model_interests = state.model_interests().await;
    respond_json(
        stream,
        200,
        &DeleteModelInterestResponse {
            removed,
            model_ref: canonical_ref,
            model_interests,
        },
    )
    .await
}

fn parse_upsert_request(body: &str) -> Result<ParsedUpsertModelInterestRequest, String> {
    let request: UpsertModelInterestRequest =
        serde_json::from_str(body).map_err(|err| format!("Invalid JSON body: {err}"))?;
    let model_ref = request.model_ref.unwrap_or_default().trim().to_string();
    if model_ref.is_empty() {
        return Err("Missing 'model_ref' field".to_string());
    }
    Ok(ParsedUpsertModelInterestRequest {
        model_ref,
        source: request.source,
    })
}

fn normalize_submission_source(source: Option<String>) -> Option<String> {
    source
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn decode_model_interest_path(path: &str) -> Option<String> {
    decode_path_suffix(path, "/api/model-interests/")
}

fn decode_path_suffix(path: &str, prefix: &str) -> Option<String> {
    let raw = path.strip_prefix(prefix)?;
    if raw.is_empty() {
        return None;
    }

    let bytes = raw.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = bytes[i + 1] as char;
                let lo = bytes[i + 2] as char;
                let hex = [hi, lo].iter().collect::<String>();
                if let Ok(value) = u8::from_str_radix(&hex, 16) {
                    decoded.push(value);
                    i += 3;
                    continue;
                }
                return None;
            }
            b'+' => decoded.push(b'+'),
            byte => decoded.push(byte),
        }
        i += 1;
    }

    String::from_utf8(decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_upsert_request_requires_non_empty_model_ref() {
        let err = parse_upsert_request(r#"{"source":"ui"}"#).unwrap_err();
        assert_eq!(err, "Missing 'model_ref' field");

        let err = parse_upsert_request(r#"{"model_ref":"   ","source":"ui"}"#).unwrap_err();
        assert_eq!(err, "Missing 'model_ref' field");
    }

    #[test]
    fn parse_upsert_request_preserves_optional_source() {
        let request =
            parse_upsert_request("{\"model_ref\":\"Qwen3-8B-Q4_K_M\",\"source\":\" ui \"}")
                .unwrap();
        assert_eq!(request.model_ref, "Qwen3-8B-Q4_K_M");
        assert_eq!(request.source.as_deref(), Some(" ui "));
    }

    #[test]
    fn normalize_submission_source_trims_optional_values() {
        assert_eq!(
            normalize_submission_source(Some(" ui ".to_string())),
            Some("ui".to_string())
        );
        assert_eq!(normalize_submission_source(Some("   ".to_string())), None);
    }

    #[test]
    fn decode_model_interest_path_decodes_percent_encoded_model_refs() {
        let decoded = decode_model_interest_path(
            "/api/model-interests/Qwen%2FQwen3-Coder-Next-GGUF%3AQ4_K_M",
        )
        .unwrap();
        assert_eq!(decoded, "Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M");
    }

    #[test]
    fn decode_model_interest_path_rejects_invalid_utf8_bytes() {
        assert_eq!(decode_model_interest_path("/api/model-interests/%80"), None);
    }
}
