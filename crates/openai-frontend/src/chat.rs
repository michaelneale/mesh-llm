use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    common::{
        completion_id, now_unix_secs, FinishReason, PromptCacheRetention, ReasoningConfig,
        ReasoningEffort, StopSequence, StreamOptions, Usage,
    },
    errors::OpenAiError,
};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub response_format: Option<Value>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub user: Option<String>,
    pub stop: Option<StopSequence>,
    pub seed: Option<u64>,
    pub reasoning: Option<ReasoningConfig>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ChatCompletionRequest {
    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(StreamOptions::include_usage)
            .unwrap_or(false)
    }

    pub fn validate(&self) -> Result<(), OpenAiError> {
        if self.model.trim().is_empty() {
            return Err(OpenAiError::invalid_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(OpenAiError::invalid_request("messages is required"));
        }
        if matches!(self.max_tokens, Some(0)) || matches!(self.max_completion_tokens, Some(0)) {
            return Err(OpenAiError::invalid_request(
                "max_tokens must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n == 0) {
            return Err(OpenAiError::invalid_request("n must be greater than zero"));
        }
        if self.n.is_some_and(|n| n > 1) {
            return Err(OpenAiError::unsupported(
                "n > 1 is parsed but multiple choices are not yet implemented",
            ));
        }
        if self.top_logprobs.is_some() && !self.logprobs.unwrap_or(false) {
            return Err(OpenAiError::invalid_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if self.tools.as_ref().is_some_and(invalid_tools_value) {
            return Err(OpenAiError::invalid_request("tools must be an array"));
        }
        if self
            .response_format
            .as_ref()
            .is_some_and(invalid_response_format_value)
        {
            return Err(OpenAiError::invalid_request(
                "response_format must be an object with a type field",
            ));
        }
        let prompt_is_empty = messages_to_plain_prompt(&self.messages).trim().is_empty();
        let media_can_supply_prompt = crate::hooks::first_chat_media(&self.messages).is_some();
        if prompt_is_empty && !media_can_supply_prompt {
            return Err(OpenAiError::invalid_request(
                "messages produced an empty prompt",
            ));
        }
        Ok(())
    }
}

fn invalid_response_format_value(value: &Value) -> bool {
    !value
        .as_object()
        .and_then(|object| object.get("type"))
        .is_some_and(Value::is_string)
}

fn invalid_tools_value(value: &Value) -> bool {
    !matches!(value, Value::Array(_))
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
    Other(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MessageContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

pub fn messages_to_plain_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .filter_map(message_content_to_text)
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn message_content_to_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter(|part| part.content_type == "text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        MessageContent::Other(_) => None,
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(model: impl Into<String>, content: impl Into<String>, usage: Usage) -> Self {
        Self::new_with_reason(model, content, usage, FinishReason::Stop)
    }

    pub fn new_with_reason(
        model: impl Into<String>,
        content: impl Into<String>,
        usage: Usage,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            object: "chat.completion",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some(content.into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    pub fn role(model: impl Into<String>) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some("assistant"),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    pub fn delta(model: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: Some(content.into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    pub fn done(model: impl Into<String>) -> Self {
        Self::done_with_reason(model, FinishReason::Stop)
    }

    pub fn done_with_reason(model: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: None,
        }
    }

    pub fn usage(model: impl Into<String>, usage: Usage) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: now_unix_secs(),
            model: model.into(),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn assistant_message_serializes_reasoning_content_when_present() {
        let message = AssistantMessage {
            role: "assistant",
            content: Some("Final answer.".to_string()),
            reasoning_content: Some("Checked the facts first.".to_string()),
            tool_calls: None,
        };

        let value = serde_json::to_value(message).unwrap();

        assert_eq!(
            value,
            json!({
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "Checked the facts first."
            })
        );
    }

    #[test]
    fn chat_delta_serializes_reasoning_content_without_text_content() {
        let delta = ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: Some("Still thinking.".to_string()),
            tool_calls: None,
        };

        let value = serde_json::to_value(delta).unwrap();

        assert_eq!(value, json!({ "reasoning_content": "Still thinking." }));
    }
}
