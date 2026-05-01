use anyhow::Context;
use anyhow::Result;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::network::openai::schema;

fn chat_completion_message_text(message: &serde_json::Value) -> String {
    match message.get("content") {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

pub(crate) fn translate_chat_completion_to_responses(body: &[u8]) -> Result<Vec<u8>> {
    let value: serde_json::Value =
        serde_json::from_slice(body).context("parse chat completion response body")?;
    let id = value
        .get("id")
        .and_then(|field| field.as_str())
        .unwrap_or("resp_mesh_llm")
        .to_string();
    let created_at = value
        .get("created")
        .and_then(|field| field.as_i64())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs() as i64)
                .unwrap_or(0)
        });
    let model = value
        .get("model")
        .and_then(|field| field.as_str())
        .unwrap_or("unknown")
        .to_string();
    let assistant_message = value
        .get("choices")
        .and_then(|field| field.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"role": "assistant", "content": ""}));
    let output_text = chat_completion_message_text(&assistant_message);

    let usage = value.get("usage").map(|usage| {
        serde_json::json!({
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(serde_json::Value::Null),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(serde_json::Value::Null),
            "total_tokens": usage.get("total_tokens").cloned().unwrap_or(serde_json::Value::Null),
        })
    });

    let response = serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": serde_json::Value::Null,
        "incomplete_details": serde_json::Value::Null,
        "model": model,
        "output": [{
            "id": format!("msg_{created_at}"),
            "type": "message",
            "status": "completed",
            "role": assistant_message
                .get("role")
                .and_then(|field| field.as_str())
                .unwrap_or("assistant"),
            "content": [{
                "type": "output_text",
                "text": output_text.clone(),
                "annotations": [],
            }],
        }],
        "output_text": output_text,
        "usage": usage.unwrap_or(serde_json::Value::Null),
    });
    serde_json::to_vec(&response).context("serialize translated /v1/responses body")
}

fn sse_frame(event: Option<&str>, data: &str) -> Vec<u8> {
    let mut frame = Vec::new();
    if let Some(event_name) = event {
        frame.extend_from_slice(format!("event: {event_name}\n").as_bytes());
    }
    for line in data.lines() {
        frame.extend_from_slice(b"data: ");
        frame.extend_from_slice(line.as_bytes());
        frame.extend_from_slice(b"\n");
    }
    if data.is_empty() {
        frame.extend_from_slice(b"data: \n");
    }
    frame.extend_from_slice(b"\n");
    frame
}

async fn write_chunked_bytes(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    let header = format!("{:x}\r\n", bytes.len());
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.write_all(b"\r\n").await
}

pub(crate) async fn write_chunked_sse_event(
    stream: &mut TcpStream,
    event: Option<&str>,
    data: &str,
) -> std::io::Result<()> {
    let frame = sse_frame(event, data);
    write_chunked_bytes(stream, &frame).await
}

pub(crate) fn responses_stream_created_event(model: &str, created_at: i64) -> serde_json::Value {
    serde_json::json!({
        "type": "response.created",
        "response": {
            "id": format!("resp_{created_at}"),
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
        }
    })
}

pub(crate) fn responses_stream_delta_event(item_id: &str, delta: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    })
}

pub(crate) fn responses_stream_text_done_event(item_id: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
}

pub(crate) fn responses_stream_completed_event(
    response_id: &str,
    created_at: i64,
    model: &str,
    item_id: &str,
    text: &str,
    usage: Option<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "error": serde_json::Value::Null,
            "incomplete_details": serde_json::Value::Null,
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
            "usage": usage.unwrap_or(serde_json::Value::Null),
        }
    })
}

pub(crate) fn stream_usage_to_responses_usage(usage: &schema::StreamUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.prompt_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "output_tokens": usage.completion_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "total_tokens": usage.total_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_chat_completion_to_responses_maps_core_fields() {
        let translated = translate_chat_completion_to_responses(
            serde_json::json!({
                "id": "chatcmpl_123",
                "created": 123,
                "model": "qwen",
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"}
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3
                }
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(parsed["object"], "response");
        assert_eq!(parsed["output_text"], "hello");
        assert_eq!(parsed["usage"]["input_tokens"], 1);
        assert_eq!(parsed["usage"]["output_tokens"], 2);
        assert_eq!(parsed["usage"]["total_tokens"], 3);
    }

    #[test]
    fn stream_usage_to_responses_usage_maps_missing_fields_to_null() {
        let usage: schema::StreamUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 11,
            "total_tokens": 14
        }))
        .unwrap();
        let mapped = stream_usage_to_responses_usage(&usage);

        assert_eq!(mapped["input_tokens"], 11);
        assert!(mapped["output_tokens"].is_null());
        assert_eq!(mapped["total_tokens"], 14);
    }
}
