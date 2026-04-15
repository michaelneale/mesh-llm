use anyhow::{anyhow, bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseAdapterMode {
    None,
    OpenAiResponsesJson,
    OpenAiResponsesStream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NormalizationOutcome {
    pub changed: bool,
    pub rewritten_path: Option<String>,
    pub response_adapter: ResponseAdapterMode,
}

fn path_only(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn rewrite_path_preserving_query(path: &str, new_path: &str) -> String {
    match path.split_once('?') {
        Some((_, query)) => format!("{new_path}?{query}"),
        None => new_path.to_string(),
    }
}

fn alias_max_tokens(object: &mut serde_json::Map<String, serde_json::Value>) -> bool {
    let mut changed = false;
    for alias in ["max_completion_tokens", "max_output_tokens"] {
        let Some(value) = object.remove(alias) else {
            continue;
        };
        changed = true;
        object.entry("max_tokens".to_string()).or_insert(value);
    }
    changed
}

fn map_response_role(role: &str) -> String {
    match role {
        "developer" => "system".to_string(),
        other => other.to_string(),
    }
}

fn object_or_url_container(
    value: Option<&serde_json::Value>,
    fallback_url: Option<&str>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(serde_json::Value::Object(map)) => Some(map.clone()),
        Some(serde_json::Value::String(url)) => Some(serde_json::Map::from_iter([(
            "url".to_string(),
            serde_json::Value::String(url.clone()),
        )])),
        _ => fallback_url.map(|url| {
            serde_json::Map::from_iter([(
                "url".to_string(),
                serde_json::Value::String(url.to_string()),
            )])
        }),
    }
}

fn translate_responses_content_item(item: &serde_json::Value) -> Result<serde_json::Value> {
    let Some(object) = item.as_object() else {
        return Ok(serde_json::json!({
            "type": "text",
            "text": item.as_str().unwrap_or_default(),
        }));
    };
    let item_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("text");

    match item_type {
        "input_text" | "text" => {
            let text = object
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            Ok(serde_json::json!({"type": "text", "text": text}))
        }
        "input_image" | "image_url" | "image" => {
            let container = object_or_url_container(
                object.get("image_url").or_else(|| object.get("image")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .ok_or_else(|| anyhow!("responses input_image block is missing image_url/url"))?;
            Ok(serde_json::json!({"type": "image_url", "image_url": container}))
        }
        "input_audio" | "audio" | "audio_url" => {
            let mut container = object_or_url_container(
                object
                    .get("input_audio")
                    .or_else(|| object.get("audio_url")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .unwrap_or_default();
            for key in [
                "data",
                "format",
                "mime_type",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            if container.is_empty() {
                bail!("responses input_audio block is missing input_audio/audio_url/url");
            }
            Ok(serde_json::json!({"type": "input_audio", "input_audio": container}))
        }
        "input_file" | "file" => {
            let mut container = object_or_url_container(
                object.get("input_file").or_else(|| object.get("file")),
                object.get("url").and_then(|value| value.as_str()),
            )
            .ok_or_else(|| anyhow!("responses input_file block is missing input_file/file/url"))?;
            for key in [
                "mime_type",
                "file_name",
                "filename",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            Ok(serde_json::json!({"type": "input_file", "input_file": container}))
        }
        other => bail!("unsupported /v1/responses content block type '{other}'"),
    }
}

fn collapse_blocks_if_text_only(blocks: Vec<serde_json::Value>) -> serde_json::Value {
    if blocks.len() == 1 {
        if let Some(text) = blocks[0].get("text").and_then(|value| value.as_str()) {
            return serde_json::Value::String(text.to_string());
        }
    }
    serde_json::Value::Array(blocks)
}

fn translate_responses_message_content(content: &serde_json::Value) -> Result<serde_json::Value> {
    match content {
        serde_json::Value::String(text) => Ok(serde_json::Value::String(text.clone())),
        serde_json::Value::Array(items) => {
            let blocks = items
                .iter()
                .map(translate_responses_content_item)
                .collect::<Result<Vec<_>>>()?;
            Ok(collapse_blocks_if_text_only(blocks))
        }
        serde_json::Value::Object(_) => Ok(collapse_blocks_if_text_only(vec![
            translate_responses_content_item(content)?,
        ])),
        _ => bail!("unsupported /v1/responses input content shape"),
    }
}

fn translate_responses_input_message(
    message: &serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let Some(object) = message.as_object() else {
        bail!("unsupported /v1/responses message shape");
    };

    let role = map_response_role(
        object
            .get("role")
            .and_then(|value| value.as_str())
            .unwrap_or("user"),
    );
    let content_value = object
        .get("content")
        .map(translate_responses_message_content)
        .transpose()?
        .unwrap_or_else(|| serde_json::Value::String(String::new()));

    Ok(serde_json::Map::from_iter([
        ("role".to_string(), serde_json::Value::String(role)),
        ("content".to_string(), content_value),
    ]))
}

fn translate_responses_input_to_messages(
    input: &serde_json::Value,
) -> Result<Vec<serde_json::Value>> {
    match input {
        serde_json::Value::String(text) => Ok(vec![serde_json::json!({
            "role": "user",
            "content": text,
        })]),
        serde_json::Value::Array(items) => {
            let looks_like_messages = items.iter().all(|item| {
                item.as_object()
                    .map(|object| object.contains_key("role") || object.contains_key("content"))
                    .unwrap_or(false)
            });
            if looks_like_messages {
                items
                    .iter()
                    .map(translate_responses_input_message)
                    .map(|result| result.map(serde_json::Value::Object))
                    .collect()
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        serde_json::Value::Object(object) => {
            if object.contains_key("role") || object.contains_key("content") {
                Ok(vec![serde_json::Value::Object(
                    translate_responses_input_message(input)?,
                )])
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        _ => bail!("unsupported /v1/responses input shape"),
    }
}

fn translate_openai_responses_input(
    object: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<bool> {
    let mut changed = false;
    let mut messages = Vec::new();

    if let Some(instructions_value) = object.remove("instructions") {
        if let Some(instructions) = instructions_value.as_str().map(str::trim) {
            if !instructions.is_empty() {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": instructions,
                }));
            }
        }
        changed = true;
    }

    if let Some(input) = object.remove("input") {
        messages.extend(translate_responses_input_to_messages(&input)?);
        changed = true;
    } else if let Some(existing_messages) = object.remove("messages") {
        messages.extend(translate_responses_input_to_messages(&existing_messages)?);
        changed = true;
    }

    if !messages.is_empty() {
        object.insert("messages".to_string(), serde_json::Value::Array(messages));
    }

    for key in [
        "include",
        "output",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "reasoning",
        "store",
        "text",
        "truncation",
    ] {
        if object.remove(key).is_some() {
            changed = true;
        }
    }

    Ok(changed)
}

pub(crate) fn normalize_openai_compat_request(
    path: &str,
    body: &mut serde_json::Value,
) -> Result<NormalizationOutcome> {
    let Some(object) = body.as_object_mut() else {
        return Ok(NormalizationOutcome {
            changed: false,
            rewritten_path: None,
            response_adapter: ResponseAdapterMode::None,
        });
    };

    let mut changed = alias_max_tokens(object);
    let mut rewritten_path = None;
    let mut response_adapter = ResponseAdapterMode::None;

    if path_only(path) == "/v1/responses" {
        let is_stream = object
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        changed |= translate_openai_responses_input(object)?;
        rewritten_path = Some(rewrite_path_preserving_query(path, "/v1/chat/completions"));
        response_adapter = if is_stream {
            ResponseAdapterMode::OpenAiResponsesStream
        } else {
            ResponseAdapterMode::OpenAiResponsesJson
        };
    }

    Ok(NormalizationOutcome {
        changed,
        rewritten_path,
        response_adapter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_chat_aliases_max_tokens() {
        let mut body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 128
        });
        let normalized =
            normalize_openai_compat_request("/v1/chat/completions", &mut body).unwrap();

        assert!(normalized.changed);
        assert_eq!(normalized.rewritten_path, None);
        assert_eq!(normalized.response_adapter, ResponseAdapterMode::None);
        assert_eq!(body["max_tokens"], 128);
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn normalize_responses_rewrites_path_and_messages() {
        let mut body = serde_json::json!({
            "model": "qwen",
            "stream": true,
            "instructions": "be concise",
            "input": "hello"
        });
        let normalized = normalize_openai_compat_request("/v1/responses?foo=1", &mut body).unwrap();

        assert!(normalized.changed);
        assert_eq!(
            normalized.rewritten_path.as_deref(),
            Some("/v1/chat/completions?foo=1")
        );
        assert_eq!(
            normalized.response_adapter,
            ResponseAdapterMode::OpenAiResponsesStream
        );
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hello");
    }
}
