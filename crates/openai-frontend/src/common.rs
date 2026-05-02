use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::OpenAiError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>),
}

impl StopSequence {
    pub fn values(&self) -> Vec<&str> {
        match self {
            StopSequence::One(value) => vec![value.as_str()],
            StopSequence::Many(values) => values.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

impl StreamOptions {
    pub fn include_usage(&self) -> bool {
        self.include_usage.unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReasoningConfig {
    pub enabled: Option<bool>,
    pub effort: Option<ReasoningEffort>,
    pub max_tokens: Option<u32>,
    pub exclude: Option<bool>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReasoningTemplateOptions {
    pub enable_thinking: Option<bool>,
}

pub const THINKING_BOOLEAN_ALIASES: &[&str] = &[
    "enable_thinking",
    "enableThinking",
    "enable_reasoning",
    "use_reasoning",
    "reasoning_enabled",
    "use_thinking",
    "thinking_enabled",
    "enable_think",
    "think_enabled",
];

pub fn normalize_reasoning_template_options(
    reasoning: Option<&ReasoningConfig>,
    reasoning_effort: Option<ReasoningEffort>,
    extra: &BTreeMap<String, Value>,
) -> Result<ReasoningTemplateOptions, OpenAiError> {
    let mut options = ReasoningTemplateOptions::default();

    if let Some(reasoning) = reasoning {
        if reasoning.enabled == Some(false)
            || matches!(reasoning.effort, Some(ReasoningEffort::None))
            || reasoning.max_tokens == Some(0)
        {
            options.enable_thinking = Some(false);
        } else if reasoning.enabled == Some(true)
            || reasoning.effort.is_some()
            || reasoning.max_tokens.is_some()
        {
            options.enable_thinking = Some(true);
        }
    }

    if let Some(effort) = reasoning_effort {
        options.enable_thinking = Some(!matches!(effort, ReasoningEffort::None));
    }

    for field in THINKING_BOOLEAN_ALIASES {
        if let Some(enable_thinking) = optional_bool_extra(extra, field)? {
            options.enable_thinking = Some(enable_thinking);
        }
    }
    if optional_u32_extra(extra, "thinking_budget")? == Some(0) {
        options.enable_thinking = Some(false);
    }

    if let Some(value) = extra.get("chat_template_kwargs") {
        let object = value.as_object().ok_or_else(|| {
            OpenAiError::invalid_request("chat_template_kwargs must be an object")
        })?;
        for field in THINKING_BOOLEAN_ALIASES {
            if let Some(value) = object.get(*field) {
                let enabled = value.as_bool().ok_or_else(|| {
                    OpenAiError::invalid_request(format!(
                        "chat_template_kwargs.{field} must be a boolean"
                    ))
                })?;
                options.enable_thinking = Some(enabled);
            }
        }
    }

    Ok(options)
}

fn optional_u32_extra(
    extra: &BTreeMap<String, Value>,
    field: &str,
) -> Result<Option<u32>, OpenAiError> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}

fn optional_bool_extra(
    extra: &BTreeMap<String, Value>,
    field: &str,
) -> Result<Option<bool>, OpenAiError> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| OpenAiError::invalid_request(format!("{field} must be a boolean")))
        })
        .transpose()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
        }
    }
}

pub fn completion_id(prefix: &str) -> String {
    format!("{prefix}-{}", now_unix_millis())
}

pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_reasoning_disables_thinking() {
        let reasoning = ReasoningConfig {
            enabled: Some(false),
            ..ReasoningConfig::default()
        };

        let options =
            normalize_reasoning_template_options(Some(&reasoning), None, &BTreeMap::new()).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_reasoning_effort_none_disables_thinking() {
        let options = normalize_reasoning_template_options(
            None,
            Some(ReasoningEffort::None),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_provider_aliases_override_canonical_reasoning() {
        let reasoning = ReasoningConfig {
            enabled: Some(false),
            ..ReasoningConfig::default()
        };
        let mut extra = BTreeMap::new();
        extra.insert("enable_thinking".to_string(), json!(true));

        let options = normalize_reasoning_template_options(Some(&reasoning), None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(true));
    }

    #[test]
    fn normalize_chat_template_kwargs_aliases() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "chat_template_kwargs".to_string(),
            json!({"enable_thinking": false}),
        );

        let options = normalize_reasoning_template_options(None, None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_thinking_budget_zero_disables_thinking() {
        let reasoning = ReasoningConfig {
            enabled: Some(true),
            ..ReasoningConfig::default()
        };
        let mut extra = BTreeMap::new();
        extra.insert("thinking_budget".to_string(), json!(0));

        let options = normalize_reasoning_template_options(Some(&reasoning), None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }
}
