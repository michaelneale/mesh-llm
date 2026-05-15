use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    chat::{ChatCompletionChunk, ChatCompletionResponse},
    common::Usage,
    errors::OpenAiError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseAdapterMode {
    None,
    OpenAiResponsesJson,
    OpenAiResponsesStream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationOutcome {
    pub changed: bool,
    pub rewritten_path: Option<String>,
    pub response_adapter: ResponseAdapterMode,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamChunk {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatCompletionStreamChoice>,
    #[serde(default)]
    pub usage: Option<StreamUsage>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamChoice {
    #[serde(default)]
    pub delta: Option<ChatCompletionStreamDelta>,
    #[serde(default)]
    pub logprobs: Option<Value>,
    #[serde(rename = "finish_reason", default)]
    _finish_reason: Option<String>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
    #[serde(rename = "role", default)]
    _role: Option<String>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<StreamPromptTokensDetails>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
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

fn alias_max_tokens(object: &mut Map<String, Value>) -> bool {
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
    value: Option<&Value>,
    fallback_url: Option<&str>,
) -> Option<Map<String, Value>> {
    match value {
        Some(Value::Object(map)) => Some(map.clone()),
        Some(Value::String(url)) => Some(Map::from_iter([(
            "url".to_string(),
            Value::String(url.clone()),
        )])),
        _ => fallback_url
            .map(|url| Map::from_iter([("url".to_string(), Value::String(url.to_string()))])),
    }
}

fn translate_responses_content_item(item: &Value) -> Result<Value, OpenAiError> {
    let Some(object) = item.as_object() else {
        return Ok(serde_json::json!({
            "type": "text",
            "text": item.as_str().unwrap_or_default(),
        }));
    };
    let item_type = object.get("type").and_then(Value::as_str).unwrap_or("text");

    match item_type {
        "input_text" | "text" => {
            let text = object
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(serde_json::json!({"type": "text", "text": text}))
        }
        "input_image" | "image_url" | "image" => {
            let container = object_or_url_container(
                object.get("image_url").or_else(|| object.get("image")),
                object.get("url").and_then(Value::as_str),
            )
            .ok_or_else(|| {
                OpenAiError::invalid_request("responses input_image block is missing image_url/url")
            })?;
            Ok(serde_json::json!({"type": "image_url", "image_url": container}))
        }
        "input_audio" | "audio" | "audio_url" => {
            let mut container = object_or_url_container(
                object
                    .get("input_audio")
                    .or_else(|| object.get("audio_url")),
                object.get("url").and_then(Value::as_str),
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
                return Err(OpenAiError::invalid_request(
                    "responses input_audio block is missing input_audio/audio_url/url",
                ));
            }
            Ok(serde_json::json!({"type": "input_audio", "input_audio": container}))
        }
        "input_file" | "file" => {
            let mut container = object_or_url_container(
                object.get("input_file").or_else(|| object.get("file")),
                object.get("url").and_then(Value::as_str),
            )
            .ok_or_else(|| {
                OpenAiError::invalid_request(
                    "responses input_file block is missing input_file/file/url",
                )
            })?;
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
        other => Err(OpenAiError::unsupported(format!(
            "unsupported /v1/responses content block type '{other}'"
        ))),
    }
}

fn collapse_blocks_if_text_only(blocks: Vec<Value>) -> Value {
    if blocks.len() == 1 {
        if let Some(text) = blocks[0].get("text").and_then(Value::as_str) {
            return Value::String(text.to_string());
        }
    }
    Value::Array(blocks)
}

fn translate_responses_message_content(content: &Value) -> Result<Value, OpenAiError> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(items) => {
            let blocks = items
                .iter()
                .map(translate_responses_content_item)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(collapse_blocks_if_text_only(blocks))
        }
        Value::Object(_) => Ok(collapse_blocks_if_text_only(vec![
            translate_responses_content_item(content)?,
        ])),
        _ => Err(OpenAiError::unsupported(
            "unsupported /v1/responses input content shape",
        )),
    }
}

fn translate_responses_input_message(message: &Value) -> Result<Map<String, Value>, OpenAiError> {
    let Some(object) = message.as_object() else {
        return Err(OpenAiError::unsupported(
            "unsupported /v1/responses message shape",
        ));
    };

    let role = map_response_role(object.get("role").and_then(Value::as_str).unwrap_or("user"));
    let content_value = object
        .get("content")
        .map(translate_responses_message_content)
        .transpose()?
        .unwrap_or_else(|| Value::String(String::new()));

    Ok(Map::from_iter([
        ("role".to_string(), Value::String(role)),
        ("content".to_string(), content_value),
    ]))
}

fn translate_responses_input_to_messages(input: &Value) -> Result<Vec<Value>, OpenAiError> {
    match input {
        Value::String(text) => Ok(vec![serde_json::json!({
            "role": "user",
            "content": text,
        })]),
        Value::Array(items) => {
            let looks_like_messages = items.iter().all(|item| {
                item.as_object()
                    .map(|object| object.contains_key("role") || object.contains_key("content"))
                    .unwrap_or(false)
            });
            if looks_like_messages {
                items
                    .iter()
                    .map(translate_responses_input_message)
                    .map(|result| result.map(Value::Object))
                    .collect()
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        Value::Object(object) => {
            if object.contains_key("role") || object.contains_key("content") {
                Ok(vec![Value::Object(translate_responses_input_message(
                    input,
                )?)])
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        _ => Err(OpenAiError::unsupported(
            "unsupported /v1/responses input shape",
        )),
    }
}

fn translate_openai_responses_input(object: &mut Map<String, Value>) -> Result<bool, OpenAiError> {
    let mut changed = false;
    let mut messages = Vec::new();
    let mut state_cache_key = None;

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
        object.insert("messages".to_string(), Value::Array(messages));
    }

    if let Some(value) = object.get("previous_response_id") {
        state_cache_key = value.as_str().map(ToString::to_string);
    }
    if state_cache_key.is_none() {
        if let Some(value) = object.get("conversation") {
            state_cache_key = responses_conversation_cache_key(value);
        }
    }
    if let Some(cache_key) = state_cache_key {
        object
            .entry("prompt_cache_key".to_string())
            .or_insert(Value::String(cache_key));
    }

    for key in [
        "conversation",
        "include",
        "output",
        "output_text",
        "previous_response_id",
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

fn responses_conversation_cache_key(value: &Value) -> Option<String> {
    if let Some(id) = value.as_str() {
        return Some(id.to_string());
    }
    let object = value.as_object()?;
    for key in ["id", "conversation_id"] {
        if let Some(id) = object.get(key).and_then(Value::as_str) {
            return Some(id.to_string());
        }
    }
    None
}

pub fn normalize_openai_compat_request(
    path: &str,
    body: &mut Value,
) -> Result<NormalizationOutcome, OpenAiError> {
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
            .and_then(Value::as_bool)
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

fn chat_completion_message_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn chat_completion_first_choice(value: &Value) -> Option<&Value> {
    value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
}

fn response_output_text_content(text: &str, logprobs: Option<Value>) -> Value {
    let mut content = serde_json::json!({
        "type": "output_text",
        "text": text,
        "annotations": [],
    });
    if let Some(logprobs) = logprobs {
        if let Some(object) = content.as_object_mut() {
            object.insert("logprobs".to_string(), logprobs);
        }
    }
    content
}

fn response_function_call_items(message: &Value, created_at: i64) -> Vec<Value> {
    let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };

    tool_calls
        .iter()
        .enumerate()
        .filter_map(|(index, tool_call)| {
            let object = tool_call.as_object()?;
            let call_id = object
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("call_{created_at}_{index}"));
            let function = object.get("function").and_then(Value::as_object);
            let name = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .or_else(|| object.get("name").and_then(Value::as_str))
                .unwrap_or("tool");
            let arguments = function
                .and_then(|function| function.get("arguments"))
                .or_else(|| object.get("arguments"))
                .map(|arguments| match arguments {
                    Value::String(arguments) => arguments.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();

            Some(serde_json::json!({
                "id": format!("fc_{created_at}_{index}"),
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }))
        })
        .collect()
}

fn insert_absent(object: &mut Map<String, Value>, key: &str, value: Value) {
    object.entry(key.to_string()).or_insert(value);
}

fn responses_text_defaults() -> Value {
    serde_json::json!({
        "format": {
            "type": "text",
        },
    })
}

fn responses_reasoning_defaults() -> Value {
    serde_json::json!({
        "effort": Value::Null,
        "summary": Value::Null,
    })
}

fn response_completed_at_value(status: &str, created_at: i64) -> Value {
    if status == "completed" {
        Value::from(created_at)
    } else {
        Value::Null
    }
}

fn apply_agent_compat_response_defaults(response: &mut Map<String, Value>, created_at: i64) {
    let status = response
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed")
        .to_string();

    insert_absent(response, "background", Value::Bool(false));
    insert_absent(
        response,
        "completed_at",
        response_completed_at_value(&status, created_at),
    );
    insert_absent(response, "conversation", Value::Null);
    insert_absent(response, "error", Value::Null);
    insert_absent(response, "incomplete_details", Value::Null);
    insert_absent(response, "instructions", Value::Null);
    insert_absent(response, "max_output_tokens", Value::Null);
    insert_absent(response, "max_tool_calls", Value::Null);
    insert_absent(response, "metadata", Value::Object(Map::new()));
    insert_absent(response, "parallel_tool_calls", Value::Bool(true));
    insert_absent(response, "previous_response_id", Value::Null);
    insert_absent(response, "prompt", Value::Null);
    insert_absent(response, "prompt_cache_key", Value::Null);
    insert_absent(response, "prompt_cache_retention", Value::Null);
    insert_absent(response, "reasoning", responses_reasoning_defaults());
    insert_absent(response, "safety_identifier", Value::Null);
    insert_absent(response, "service_tier", Value::Null);
    insert_absent(response, "temperature", Value::Null);
    insert_absent(response, "text", responses_text_defaults());
    insert_absent(response, "tool_choice", Value::String("auto".to_string()));
    insert_absent(response, "tools", Value::Array(Vec::new()));
    insert_absent(response, "top_logprobs", Value::Null);
    insert_absent(response, "top_p", Value::Null);
    insert_absent(
        response,
        "truncation",
        Value::String("disabled".to_string()),
    );
    insert_absent(response, "usage", Value::Null);
    insert_absent(response, "user", Value::Null);
}

pub fn translate_chat_completion_to_responses(body: &[u8]) -> Result<Vec<u8>, OpenAiError> {
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        OpenAiError::invalid_request(format!("parse chat completion response body: {error}"))
    })?;
    translate_chat_completion_value_to_responses(&value)
}

pub fn translate_chat_completion_response_to_responses(
    response: &ChatCompletionResponse,
) -> Result<Value, OpenAiError> {
    let value = serde_json::to_value(response).map_err(|error| {
        OpenAiError::internal(format!("serialize chat completion response: {error}"))
    })?;
    let bytes = translate_chat_completion_value_to_responses(&value)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        OpenAiError::internal(format!("parse translated responses response: {error}"))
    })
}

fn translate_chat_completion_value_to_responses(value: &Value) -> Result<Vec<u8>, OpenAiError> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("resp_mesh_llm")
        .to_string();
    let created_at = value
        .get("created")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_unix_secs_i64);
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let first_choice = chat_completion_first_choice(value);
    let assistant_message = first_choice
        .and_then(|choice| choice.get("message"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"role": "assistant", "content": ""}));
    let output_text = chat_completion_message_text(&assistant_message);
    let finish_reason = first_choice
        .and_then(|choice| choice.get("finish_reason"))
        .cloned()
        .unwrap_or(Value::Null);
    let logprobs = first_choice
        .and_then(|choice| choice.get("logprobs"))
        .filter(|logprobs| !logprobs.is_null())
        .cloned();

    let usage = value.get("usage").map(chat_usage_to_responses_usage);
    let mut output = Vec::new();
    let tool_call_items = response_function_call_items(&assistant_message, created_at);
    if !output_text.is_empty() || tool_call_items.is_empty() {
        output.push(serde_json::json!({
            "id": format!("msg_{created_at}"),
            "type": "message",
            "status": "completed",
            "role": assistant_message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant"),
            "content": [response_output_text_content(&output_text, logprobs.clone())],
        }));
    }
    output.extend(tool_call_items);

    let mut response = serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "model": model,
        "output": output,
        "output_text": output_text,
        "finish_reason": finish_reason,
        "usage": usage.unwrap_or(Value::Null),
    });
    if let Some(object) = response.as_object_mut() {
        apply_agent_compat_response_defaults(object, created_at);
    }
    serde_json::to_vec(&response)
        .map_err(|error| OpenAiError::internal(format!("serialize /v1/responses body: {error}")))
}

pub fn parse_chat_stream_chunk(data: &str) -> Result<ChatCompletionStreamChunk, OpenAiError> {
    serde_json::from_str(data)
        .map_err(|error| OpenAiError::invalid_request(format!("parse chat stream chunk: {error}")))
}

pub fn responses_stream_created_event(model: &str, created_at: i64) -> Value {
    responses_stream_created_event_with_sequence(model, created_at, 0)
}

pub fn responses_stream_created_event_with_sequence(
    model: &str,
    created_at: i64,
    sequence_number: i32,
) -> Value {
    let mut event = serde_json::json!({
        "type": "response.created",
        "sequence_number": sequence_number,
        "response": {
            "id": format!("resp_{created_at}"),
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
            "output_text": "",
        }
    });
    if let Some(response) = event.get_mut("response").and_then(Value::as_object_mut) {
        apply_agent_compat_response_defaults(response, created_at);
    }
    event
}

pub fn responses_stream_output_item_added_event(item_id: &str, sequence_number: i32) -> Value {
    serde_json::json!({
        "type": "response.output_item.added",
        "sequence_number": sequence_number,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    })
}

pub fn responses_stream_content_part_added_event(item_id: &str, sequence_number: i32) -> Value {
    serde_json::json!({
        "type": "response.content_part.added",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": [],
        },
    })
}

pub fn responses_stream_delta_event(item_id: &str, delta: &str) -> Value {
    responses_stream_delta_event_with_logprobs(item_id, delta, None)
}

pub fn responses_stream_delta_event_with_logprobs(
    item_id: &str,
    delta: &str,
    logprobs: Option<Value>,
) -> Value {
    responses_stream_delta_event_with_logprobs_and_sequence(item_id, delta, logprobs, 0)
}

pub fn responses_stream_delta_event_with_logprobs_and_sequence(
    item_id: &str,
    delta: &str,
    logprobs: Option<Value>,
    sequence_number: i32,
) -> Value {
    let mut event = serde_json::json!({
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    });
    if let Some(logprobs) = logprobs {
        if let Some(object) = event.as_object_mut() {
            object.insert("logprobs".to_string(), logprobs);
        }
    }
    event
}

pub fn responses_stream_reasoning_delta_event_with_sequence(
    item_id: &str,
    delta: &str,
    sequence_number: i32,
) -> Value {
    serde_json::json!({
        "type": "response.reasoning_text.delta",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    })
}

pub fn responses_stream_text_done_event(item_id: &str, text: &str) -> Value {
    serde_json::json!({
        "type": "response.output_text.done",
        "sequence_number": 0,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
}

pub fn responses_stream_text_done_event_with_sequence(
    item_id: &str,
    text: &str,
    sequence_number: i32,
) -> Value {
    serde_json::json!({
        "type": "response.output_text.done",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
}

pub fn responses_stream_content_part_done_event(
    item_id: &str,
    text: &str,
    sequence_number: i32,
) -> Value {
    serde_json::json!({
        "type": "response.content_part.done",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": text,
            "annotations": [],
        },
    })
}

pub fn responses_stream_output_item_done_event(
    item_id: &str,
    text: &str,
    sequence_number: i32,
) -> Value {
    serde_json::json!({
        "type": "response.output_item.done",
        "sequence_number": sequence_number,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        },
    })
}

pub fn responses_stream_completed_event(
    response_id: &str,
    created_at: i64,
    model: &str,
    item_id: &str,
    text: &str,
    usage: Option<Value>,
) -> Value {
    responses_stream_completed_event_with_sequence(
        response_id,
        created_at,
        model,
        item_id,
        text,
        usage,
        0,
    )
}

pub fn responses_stream_completed_event_with_sequence(
    response_id: &str,
    created_at: i64,
    model: &str,
    item_id: &str,
    text: &str,
    usage: Option<Value>,
    sequence_number: i32,
) -> Value {
    let mut event = serde_json::json!({
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "error": Value::Null,
            "incomplete_details": Value::Null,
            "model": model,
            "output": [{
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }],
            }],
            "output_text": text,
            "usage": usage.unwrap_or(Value::Null),
        }
    });
    if let Some(response) = event.get_mut("response").and_then(Value::as_object_mut) {
        apply_agent_compat_response_defaults(response, created_at);
    }
    event
}

pub fn chat_usage_to_responses_usage(usage: &Value) -> Value {
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .cloned()
        .unwrap_or(Value::Null);
    serde_json::json!({
        "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(Value::Null),
        "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(Value::Null),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or(Value::Null),
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
    })
}

pub fn stream_usage_to_responses_usage(usage: &StreamUsage) -> Value {
    let cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .map(Value::from)
        .unwrap_or(Value::Null);
    serde_json::json!({
        "input_tokens": usage.prompt_tokens.map(Value::from).unwrap_or(Value::Null),
        "output_tokens": usage.completion_tokens.map(Value::from).unwrap_or(Value::Null),
        "total_tokens": usage.total_tokens.map(Value::from).unwrap_or(Value::Null),
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
    })
}

pub fn usage_to_responses_usage(usage: &Usage) -> Value {
    let cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .map(|details| Value::from(details.cached_tokens))
        .unwrap_or(Value::Null);
    serde_json::json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
    })
}

pub fn now_unix_secs_i64() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

pub fn chunk_delta_text(chunk: &ChatCompletionChunk) -> Option<String> {
    chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.content.clone())
}

pub fn chunk_model<'a>(chunk: &'a ChatCompletionChunk, fallback: &'a str) -> &'a str {
    if chunk.model.is_empty() {
        fallback
    } else {
        &chunk.model
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResponseSseState {
    pub response_id: String,
    pub item_id: String,
    pub created_at: i64,
    pub sequence_number: i32,
    pub model: String,
    pub output_text: String,
    pub usage: Option<Value>,
    pub created_emitted: bool,
    pub output_item_emitted: bool,
}

impl ResponseSseState {
    pub fn new(model: impl Into<String>) -> Self {
        let created_at = now_unix_secs_i64();
        Self {
            response_id: format!("resp_{created_at}"),
            item_id: format!("msg_{created_at}"),
            created_at,
            sequence_number: 0,
            model: model.into(),
            output_text: String::new(),
            usage: None,
            created_emitted: false,
            output_item_emitted: false,
        }
    }

    pub fn next_sequence_number(&mut self) -> i32 {
        self.sequence_number = self.sequence_number.saturating_add(1);
        self.sequence_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_responses_rewrites_path_and_messages() {
        let mut body = json!({
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

    #[test]
    fn normalize_responses_preserves_tool_and_structured_fields() {
        let mut body = json!({
            "model": "qwen",
            "input": "call a tool",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
            "logprobs": true,
            "top_logprobs": 3
        });
        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();

        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["logprobs"], true);
        assert_eq!(body["top_logprobs"], 3);
    }

    #[test]
    fn normalize_responses_maps_state_to_prompt_cache_key() {
        let mut body = json!({
            "model": "qwen",
            "input": "continue",
            "previous_response_id": "resp_abc"
        });
        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();

        assert_eq!(body["prompt_cache_key"], "resp_abc");
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn normalize_responses_keeps_explicit_prompt_cache_key() {
        let mut body = json!({
            "model": "qwen",
            "input": "continue",
            "conversation": {"id": "conv_abc"},
            "prompt_cache_key": "caller-key"
        });
        normalize_openai_compat_request("/v1/responses", &mut body).unwrap();

        assert_eq!(body["prompt_cache_key"], "caller-key");
        assert!(body.get("conversation").is_none());
    }

    #[test]
    fn translate_chat_completion_to_responses_maps_core_fields() {
        let translated = translate_chat_completion_to_responses(
            json!({
                "id": "chatcmpl_123",
                "created": 123,
                "model": "qwen",
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"}
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "prompt_tokens_details": {
                        "cached_tokens": 1
                    }
                }
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let parsed: Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(parsed["object"], "response");
        assert_eq!(parsed["output_text"], "hello");
        assert_eq!(parsed["usage"]["input_tokens"], 1);
        assert_eq!(parsed["usage"]["output_tokens"], 2);
        assert_eq!(parsed["usage"]["total_tokens"], 3);
        assert_eq!(parsed["usage"]["input_tokens_details"]["cached_tokens"], 1);
    }

    #[test]
    fn translate_chat_completion_to_responses_emits_agent_compat_fields() {
        let translated = translate_chat_completion_to_responses(
            json!({
                "id": "chatcmpl_123",
                "created": 123,
                "model": "qwen",
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"}
                }]
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let parsed: Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(parsed["status"], "completed");
        assert_eq!(parsed["completed_at"], 123);
        assert_eq!(parsed["background"], false);
        assert_eq!(parsed["metadata"], json!({}));
        assert_eq!(parsed["parallel_tool_calls"], true);
        assert_eq!(parsed["text"]["format"]["type"], "text");
        assert_eq!(parsed["tool_choice"], "auto");
        assert_eq!(parsed["tools"], json!([]));
        assert_eq!(parsed["truncation"], "disabled");
        for key in [
            "conversation",
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "temperature",
            "top_logprobs",
            "top_p",
            "user",
        ] {
            assert!(parsed[key].is_null(), "{key} should default to null");
        }
    }

    #[test]
    fn translate_chat_completion_to_responses_preserves_tool_calls_and_logprobs() {
        let translated = translate_chat_completion_to_responses(
            json!({
                "id": "chatcmpl_123",
                "created": 123,
                "model": "qwen",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "calling lookup",
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"city\":\"Sydney\"}"
                            }
                        }]
                    },
                    "logprobs": {
                        "content": [{
                            "token": "{",
                            "logprob": -0.1
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let parsed: Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(parsed["output_text"], "calling lookup");
        assert_eq!(parsed["finish_reason"], "tool_calls");
        assert_eq!(parsed["output"][0]["type"], "message");
        assert_eq!(
            parsed["output"][0]["content"][0]["logprobs"]["content"][0]["token"],
            "{"
        );
        assert_eq!(parsed["output"][1]["type"], "function_call");
        assert_eq!(parsed["output"][1]["call_id"], "call_123");
        assert_eq!(parsed["output"][1]["name"], "lookup");
        assert_eq!(parsed["output"][1]["arguments"], "{\"city\":\"Sydney\"}");
    }

    #[test]
    fn stream_usage_to_responses_usage_maps_missing_fields_to_null() {
        let usage: StreamUsage = serde_json::from_value(json!({
            "prompt_tokens": 11,
            "total_tokens": 14
        }))
        .unwrap();
        let mapped = stream_usage_to_responses_usage(&usage);

        assert_eq!(mapped["input_tokens"], 11);
        assert!(mapped["output_tokens"].is_null());
        assert_eq!(mapped["total_tokens"], 14);
        assert!(mapped["input_tokens_details"]["cached_tokens"].is_null());
    }

    #[test]
    fn parse_chat_stream_chunk_reads_model_delta_and_usage() {
        let payload = json!({
            "model": "qwen",
            "choices": [{
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": null
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}
        })
        .to_string();

        let parsed = parse_chat_stream_chunk(&payload).expect("stream chunk parse should succeed");
        assert_eq!(parsed.model.as_deref(), Some("qwen"));
        let delta = parsed
            .choices
            .first()
            .and_then(|choice| choice.delta.as_ref())
            .and_then(|delta| delta.content.as_deref());
        assert_eq!(delta, Some("hello"));
        assert_eq!(
            parsed.usage.as_ref().and_then(|usage| usage.total_tokens),
            Some(13)
        );
    }

    #[test]
    fn parse_chat_stream_chunk_reads_reasoning_delta() {
        let payload = json!({
            "model": "qwen",
            "choices": [{
                "delta": {"reasoning_content": "Checking facts."},
                "finish_reason": null
            }]
        })
        .to_string();

        let parsed = parse_chat_stream_chunk(&payload).expect("stream chunk parse should succeed");
        let delta = parsed
            .choices
            .first()
            .and_then(|choice| choice.delta.as_ref())
            .and_then(|delta| delta.reasoning_content.as_deref());
        assert_eq!(delta, Some("Checking facts."));
    }

    #[test]
    fn usage_to_responses_usage_maps_cached_tokens() {
        let usage = Usage::new(128, 8).with_cached_tokens(96);
        let mapped = usage_to_responses_usage(&usage);

        assert_eq!(mapped["input_tokens"], 128);
        assert_eq!(mapped["input_tokens_details"]["cached_tokens"], 96);
    }

    #[test]
    fn responses_stream_events_emit_agent_compat_response_fields() {
        let created = responses_stream_created_event("qwen", 123);
        assert_eq!(created["sequence_number"], 0);
        let created_response = &created["response"];
        assert_eq!(created_response["status"], "in_progress");
        assert!(created_response["completed_at"].is_null());
        assert_eq!(created_response["output_text"], "");
        assert_eq!(created_response["parallel_tool_calls"], true);
        assert_eq!(created_response["text"]["format"]["type"], "text");

        let completed =
            responses_stream_completed_event("resp_123", 123, "qwen", "msg_123", "hello", None);
        let completed_response = &completed["response"];
        assert_eq!(completed_response["status"], "completed");
        assert_eq!(completed_response["completed_at"], 123);
        assert_eq!(completed_response["output_text"], "hello");
        assert_eq!(completed_response["tool_choice"], "auto");
        assert_eq!(completed_response["tools"], json!([]));
    }

    #[test]
    fn responses_stream_events_emit_openai_responses_scaffold() {
        let item_added = responses_stream_output_item_added_event("msg_123", 2);
        assert_eq!(item_added["type"], "response.output_item.added");
        assert_eq!(item_added["sequence_number"], 2);
        assert_eq!(item_added["item"]["id"], "msg_123");
        assert_eq!(item_added["item"]["type"], "message");
        assert_eq!(item_added["item"]["status"], "in_progress");

        let part_added = responses_stream_content_part_added_event("msg_123", 3);
        assert_eq!(part_added["type"], "response.content_part.added");
        assert_eq!(part_added["sequence_number"], 3);
        assert_eq!(part_added["part"]["type"], "output_text");

        let delta =
            responses_stream_delta_event_with_logprobs_and_sequence("msg_123", "hello", None, 4);
        assert_eq!(delta["type"], "response.output_text.delta");
        assert_eq!(delta["sequence_number"], 4);
        assert_eq!(delta["delta"], "hello");

        let reasoning_delta =
            responses_stream_reasoning_delta_event_with_sequence("msg_123", "Checking facts.", 5);
        assert_eq!(reasoning_delta["type"], "response.reasoning_text.delta");
        assert_eq!(reasoning_delta["sequence_number"], 5);
        assert_eq!(reasoning_delta["delta"], "Checking facts.");

        let part_done = responses_stream_content_part_done_event("msg_123", "hello", 6);
        assert_eq!(part_done["type"], "response.content_part.done");
        assert_eq!(part_done["part"]["text"], "hello");

        let item_done = responses_stream_output_item_done_event("msg_123", "hello", 7);
        assert_eq!(item_done["type"], "response.output_item.done");
        assert_eq!(item_done["item"]["content"][0]["text"], "hello");
    }
}
