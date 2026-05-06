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
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub best_of: Option<u32>,
    pub suffix: Option<String>,
    pub echo: Option<bool>,
    pub logprobs: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
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

impl CompletionRequest {
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
        if matches!(self.max_tokens, Some(0)) {
            return Err(OpenAiError::invalid_request(
                "max_tokens must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n == 0) {
            return Err(OpenAiError::invalid_request("n must be greater than zero"));
        }
        if self.n.is_some_and(|n| n > 1) || self.best_of.is_some_and(|best_of| best_of > 1) {
            return Err(OpenAiError::unsupported(
                "multiple choices are parsed but not yet implemented",
            ));
        }
        if self
            .suffix
            .as_ref()
            .is_some_and(|suffix| !suffix.is_empty())
        {
            return Err(OpenAiError::unsupported(
                "suffix is parsed but not yet implemented",
            ));
        }
        if self.echo.unwrap_or(false) {
            return Err(OpenAiError::unsupported(
                "echo is parsed but not yet implemented",
            ));
        }
        if self.prompt.is_empty() {
            return Err(OpenAiError::invalid_request("prompt is required"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    ManyText(Vec<String>),
    Tokens(Vec<i32>),
    ManyTokens(Vec<Vec<i32>>),
    Other(Value),
}

impl CompletionPrompt {
    pub fn text_lossy(&self) -> String {
        match self {
            CompletionPrompt::Text(text) => text.clone(),
            CompletionPrompt::ManyText(values) => values.join("\n"),
            CompletionPrompt::Tokens(values) => values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" "),
            CompletionPrompt::ManyTokens(values) => values
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join("\n"),
            CompletionPrompt::Other(value) => value.to_string(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            CompletionPrompt::Text(text) => text.trim().is_empty(),
            CompletionPrompt::ManyText(values) => values.iter().all(|text| text.trim().is_empty()),
            CompletionPrompt::Tokens(values) => values.is_empty(),
            CompletionPrompt::ManyTokens(values) => values.iter().all(Vec::is_empty),
            CompletionPrompt::Other(_) => true,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(model: impl Into<String>, text: impl Into<String>, usage: Usage) -> Self {
        Self::new_with_reason(model, text, usage, FinishReason::Stop)
    }

    pub fn new_with_reason(
        model: impl Into<String>,
        text: impl Into<String>,
        usage: Usage,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            id: completion_id("cmpl"),
            object: "text_completion",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![CompletionChoice {
                text: text.into(),
                index: 0,
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl CompletionChunk {
    pub fn delta(model: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: completion_id("cmpl"),
            object: "text_completion",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![CompletionChunkChoice {
                text: text.into(),
                index: 0,
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
            id: completion_id("cmpl"),
            object: "text_completion",
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![CompletionChunkChoice {
                text: String::new(),
                index: 0,
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: None,
        }
    }

    pub fn usage(model: impl Into<String>, usage: Usage) -> Self {
        Self {
            id: completion_id("cmpl"),
            object: "text_completion",
            created: now_unix_secs(),
            model: model.into(),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}
