use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::network::openai::schema;

pub(crate) use openai_frontend::{
    responses_stream_completed_event, responses_stream_created_event,
    responses_stream_delta_event_with_logprobs, responses_stream_text_done_event,
    translate_chat_completion_to_responses,
};

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

pub(crate) fn stream_usage_to_responses_usage(usage: &schema::StreamUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.prompt_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "output_tokens": usage.completion_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "total_tokens": usage.total_tokens.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
    })
}
