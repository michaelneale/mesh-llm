use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatCompletionStreamChunk {
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
pub(crate) struct ChatCompletionStreamChoice {
    #[serde(default)]
    pub delta: Option<ChatCompletionDelta>,
    #[serde(rename = "finish_reason", default)]
    _finish_reason: Option<String>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatCompletionDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(rename = "role", default)]
    _role: Option<String>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct StreamUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(flatten)]
    _extra: Map<String, Value>,
}
