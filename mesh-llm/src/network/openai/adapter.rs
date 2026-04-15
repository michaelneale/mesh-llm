use crate::network::openai::schema;
use anyhow::{Context, Result};

pub(crate) fn parse_chat_stream_chunk(payload: &str) -> Result<schema::ChatCompletionStreamChunk> {
    serde_json::from_str(payload).context("parse typed upstream chat stream chunk")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_chat_stream_chunk_typed() {
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
}
