use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(crate) use openai_frontend::{
    responses_stream_completed_event_with_sequence, responses_stream_content_part_added_event,
    responses_stream_content_part_done_event, responses_stream_created_event_with_sequence,
    responses_stream_delta_event_with_logprobs_and_sequence,
    responses_stream_output_item_added_event, responses_stream_output_item_done_event,
    responses_stream_text_done_event_with_sequence, stream_usage_to_responses_usage,
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
