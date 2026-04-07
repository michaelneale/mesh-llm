use super::build_hf_tokio_api;
use super::catalog;
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    #[default]
    None,
    Likely,
    Supported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub multimodal: bool,
    pub vision: CapabilityLevel,
    pub audio: CapabilityLevel,
    pub reasoning: CapabilityLevel,
    pub tool_use: CapabilityLevel,
    pub moe: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            multimodal: false,
            vision: CapabilityLevel::None,
            audio: CapabilityLevel::None,
            reasoning: CapabilityLevel::None,
            tool_use: CapabilityLevel::None,
            moe: false,
        }
    }
}

impl ModelCapabilities {
    pub fn supports_multimodal_runtime(self) -> bool {
        self.multimodal || self.supports_vision_runtime() || self.supports_audio_runtime()
    }

    pub fn supports_vision_runtime(self) -> bool {
        matches!(self.vision, CapabilityLevel::Supported)
    }

    pub fn supports_audio_runtime(self) -> bool {
        matches!(self.audio, CapabilityLevel::Supported)
    }

    pub fn multimodal_status(self) -> &'static str {
        if self.supports_multimodal_runtime() {
            "supported"
        } else {
            "none"
        }
    }

    pub fn multimodal_label(self) -> Option<&'static str> {
        if self.supports_multimodal_runtime() {
            Some("yes")
        } else {
            None
        }
    }

    pub fn vision_status(self) -> &'static str {
        match self.vision {
            CapabilityLevel::Supported => "supported",
            CapabilityLevel::Likely => "likely",
            CapabilityLevel::None => "none",
        }
    }

    pub fn vision_label(self) -> Option<&'static str> {
        match self.vision {
            CapabilityLevel::Supported => Some("yes"),
            CapabilityLevel::Likely => Some("likely"),
            CapabilityLevel::None => None,
        }
    }

    pub fn audio_status(self) -> &'static str {
        match self.audio {
            CapabilityLevel::Supported => "supported",
            CapabilityLevel::Likely => "likely",
            CapabilityLevel::None => "none",
        }
    }

    pub fn audio_label(self) -> Option<&'static str> {
        match self.audio {
            CapabilityLevel::Supported => Some("yes"),
            CapabilityLevel::Likely => Some("likely"),
            CapabilityLevel::None => None,
        }
    }

    pub fn reasoning_status(self) -> &'static str {
        match self.reasoning {
            CapabilityLevel::Supported => "supported",
            CapabilityLevel::Likely => "likely",
            CapabilityLevel::None => "none",
        }
    }

    pub fn reasoning_label(self) -> Option<&'static str> {
        match self.reasoning {
            CapabilityLevel::Supported => Some("yes"),
            CapabilityLevel::Likely => Some("likely"),
            CapabilityLevel::None => None,
        }
    }

    pub fn tool_use_status(self) -> &'static str {
        match self.tool_use {
            CapabilityLevel::Supported => "supported",
            CapabilityLevel::Likely => "likely",
            CapabilityLevel::None => "none",
        }
    }

    pub fn tool_use_label(self) -> Option<&'static str> {
        match self.tool_use {
            CapabilityLevel::Supported => Some("yes"),
            CapabilityLevel::Likely => Some("likely"),
            CapabilityLevel::None => None,
        }
    }

    fn upgrade_vision(&mut self, level: CapabilityLevel) {
        self.vision = self.vision.max(level);
        if self.vision != CapabilityLevel::None {
            self.multimodal = true;
        }
    }

    fn upgrade_audio(&mut self, level: CapabilityLevel) {
        self.audio = self.audio.max(level);
        if self.audio != CapabilityLevel::None {
            self.multimodal = true;
        }
    }

    fn upgrade_reasoning(&mut self, level: CapabilityLevel) {
        self.reasoning = self.reasoning.max(level);
    }

    fn upgrade_tool_use(&mut self, level: CapabilityLevel) {
        self.tool_use = self.tool_use.max(level);
    }

    fn normalize(mut self) -> Self {
        if self.vision != CapabilityLevel::None || self.audio != CapabilityLevel::None {
            self.multimodal = true;
        }
        self
    }
}

pub fn infer_catalog_capabilities(model: &catalog::CatalogModel) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    if model.mmproj.is_some() {
        caps.upgrade_vision(CapabilityLevel::Supported);
    }
    caps.moe = model.moe.is_some();
    caps = merge_name_signals(
        caps,
        &[
            model.name.as_str(),
            model.file.as_str(),
            model.description.as_str(),
        ],
    );
    caps.normalize()
}

pub fn infer_local_model_capabilities(
    model_name: &str,
    path: &Path,
    catalog: Option<&catalog::CatalogModel>,
) -> ModelCapabilities {
    let mut caps = catalog.map(infer_catalog_capabilities).unwrap_or_default();
    caps = merge_name_signals(
        caps,
        &[
            model_name,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        ],
    );
    for config in read_local_metadata_jsons(path) {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

pub async fn infer_remote_hf_capabilities(
    repo: &str,
    revision: Option<&str>,
    file: &str,
    siblings: Option<&[String]>,
) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    caps = merge_name_signals(caps, &[repo, file]);
    if let Some(files) = siblings {
        caps = merge_sibling_signals(caps, files.iter().map(String::as_str));
    }
    for config in fetch_remote_metadata_jsons(repo, revision).await {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

pub fn merge_name_signals(mut caps: ModelCapabilities, values: &[&str]) -> ModelCapabilities {
    if values.iter().any(|value| strong_vision_name_signal(value)) {
        caps.upgrade_vision(CapabilityLevel::Supported);
    } else if values.iter().any(|value| likely_vision_name_signal(value)) {
        caps.upgrade_vision(CapabilityLevel::Likely);
    }

    if values.iter().any(|value| strong_audio_name_signal(value)) {
        caps.upgrade_audio(CapabilityLevel::Supported);
    } else if values.iter().any(|value| likely_audio_name_signal(value)) {
        caps.upgrade_audio(CapabilityLevel::Likely);
    }

    if values
        .iter()
        .any(|value| strong_reasoning_name_signal(value))
    {
        caps.upgrade_reasoning(CapabilityLevel::Supported);
    } else if values
        .iter()
        .any(|value| likely_reasoning_name_signal(value))
    {
        caps.upgrade_reasoning(CapabilityLevel::Likely);
    }

    if values
        .iter()
        .any(|value| strong_tool_use_name_signal(value))
    {
        caps.upgrade_tool_use(CapabilityLevel::Supported);
    } else if values
        .iter()
        .any(|value| likely_tool_use_name_signal(value))
    {
        caps.upgrade_tool_use(CapabilityLevel::Likely);
    }

    caps.normalize()
}

pub fn merge_sibling_signals<I, S>(mut caps: ModelCapabilities, siblings: I) -> ModelCapabilities
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut saw_processor = false;
    let mut saw_reasoning_template = false;
    let mut saw_tool_template = false;
    for sibling in siblings {
        let name = sibling.as_ref().to_lowercase();
        if name.contains("mmproj") {
            caps.upgrade_vision(CapabilityLevel::Supported);
        }
        if name.contains("audio") || name.contains("whisper") || name.contains("ultravox") {
            caps.upgrade_audio(CapabilityLevel::Likely);
        }
        if name.ends_with("preprocessor_config.json")
            || name.ends_with("processor_config.json")
            || name.ends_with("image_processor_config.json")
        {
            saw_processor = true;
        }
        if name.ends_with("tokenizer_config.json")
            || name.ends_with("chat_template.jinja")
            || name.ends_with("chat_template.json")
            || name.contains("reasoning")
            || name.contains("thinking")
        {
            saw_reasoning_template = true;
        }
        if name.contains("tool") || name.contains("function") {
            saw_tool_template = true;
        }
    }
    if saw_processor {
        caps.upgrade_vision(CapabilityLevel::Likely);
    }
    if saw_reasoning_template {
        caps.upgrade_reasoning(CapabilityLevel::Likely);
    }
    if saw_tool_template {
        caps.upgrade_tool_use(CapabilityLevel::Likely);
    }
    caps.normalize()
}

pub fn merge_config_signals(mut caps: ModelCapabilities, config: &Value) -> ModelCapabilities {
    if config.get("vision_config").is_some() {
        caps.upgrade_vision(CapabilityLevel::Supported);
    }

    if config.get("audio_config").is_some() {
        caps.upgrade_audio(CapabilityLevel::Supported);
    }

    for key in [
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_token_id",
    ] {
        if config.get(key).is_some() {
            caps.upgrade_vision(CapabilityLevel::Supported);
        }
    }

    for key in [
        "audio_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "audio_bos_token_id",
        "audio_eos_token_id",
        "audio_chunk_size",
    ] {
        if config.get(key).is_some() {
            caps.upgrade_audio(CapabilityLevel::Supported);
        }
    }

    if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(strong_vision_name_signal)
    {
        caps.upgrade_vision(CapabilityLevel::Supported);
    }

    if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(strong_audio_name_signal)
    {
        caps.upgrade_audio(CapabilityLevel::Supported);
    } else if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(likely_audio_name_signal)
    {
        caps.upgrade_audio(CapabilityLevel::Likely);
    }

    if config
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(strong_vision_name_signal)
        .unwrap_or(false)
    {
        caps.upgrade_vision(CapabilityLevel::Supported);
    }

    if let Some(model_type) = config.get("model_type").and_then(|value| value.as_str()) {
        if strong_audio_name_signal(model_type) {
            caps.upgrade_audio(CapabilityLevel::Supported);
        } else if likely_audio_name_signal(model_type) {
            caps.upgrade_audio(CapabilityLevel::Likely);
        }
    }

    if json_contains_reasoning_tokens(config) {
        caps.upgrade_reasoning(CapabilityLevel::Supported);
    }

    if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(strong_reasoning_name_signal)
    {
        caps.upgrade_reasoning(CapabilityLevel::Supported);
    } else if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(likely_reasoning_name_signal)
    {
        caps.upgrade_reasoning(CapabilityLevel::Likely);
    }

    if let Some(model_type) = config.get("model_type").and_then(|value| value.as_str()) {
        if strong_reasoning_name_signal(model_type) {
            caps.upgrade_reasoning(CapabilityLevel::Supported);
        } else if likely_reasoning_name_signal(model_type) {
            caps.upgrade_reasoning(CapabilityLevel::Likely);
        }
    }

    if json_contains_tool_use_tokens(config) {
        caps.upgrade_tool_use(CapabilityLevel::Supported);
    }

    if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(strong_tool_use_name_signal)
    {
        caps.upgrade_tool_use(CapabilityLevel::Supported);
    } else if config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .any(likely_tool_use_name_signal)
    {
        caps.upgrade_tool_use(CapabilityLevel::Likely);
    }

    if let Some(model_type) = config.get("model_type").and_then(|value| value.as_str()) {
        if strong_tool_use_name_signal(model_type) {
            caps.upgrade_tool_use(CapabilityLevel::Supported);
        } else if likely_tool_use_name_signal(model_type) {
            caps.upgrade_tool_use(CapabilityLevel::Likely);
        }
    }

    if config
        .get("num_experts")
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
        > 1
        || config
            .get("num_experts_per_tok")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            > 0
    {
        caps.moe = true;
    }

    caps.normalize()
}

fn strong_vision_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    [
        "vision",
        "qwen3-vl",
        "qwen3_vl",
        "qwen3vl",
        "qwen2-vl",
        "qwen2_vl",
        "qwen2.5-vl",
        "qwen2_5_vl",
        "llava",
        "mllama",
        "paligemma",
        "idefics",
        "molmo",
        "internvl",
        "glm-4v",
        "glm4v",
        "ovis",
        "florence",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn likely_vision_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    value.contains("-vl")
        || value.contains("vl-")
        || value.contains("_vl")
        || value.contains("video")
        || value.contains("multimodal")
        || value.contains("image")
}

fn strong_audio_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    [
        "audio",
        "qwen2-audio",
        "qwen2_audio",
        "seallm-audio",
        "seallm_audio",
        "ultravox",
        "omni",
        "speech",
        "whisper",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn likely_audio_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    value.contains("audio")
        || value.contains("speech")
        || value.contains("voice")
        || value.contains("omni")
}

fn strong_reasoning_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    [
        "reasoning",
        "reasoner",
        "reason",
        "thinking",
        "deepthink",
        "deep_think",
        "<think>",
        "</think>",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn likely_reasoning_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    [
        "-r1",
        "_r1",
        " r1",
        "think",
        "thought",
        "chain-of-thought",
        "cot",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn strong_tool_use_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    [
        "tool calling",
        "tool-calling",
        "tool use",
        "function calling",
        "function-calling",
        "function call",
        "tool_use",
        "tool_calls",
        "function_call",
        "function_calls",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn likely_tool_use_name_signal(value: &str) -> bool {
    let value = value.to_lowercase();
    ["tool", "agentic", "function", "coding"]
        .iter()
        .any(|needle| value.contains(needle))
}

fn json_contains_reasoning_tokens(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
        Value::String(text) => {
            let lower = text.to_lowercase();
            lower.contains("<think>")
                || lower.contains("</think>")
                || lower.contains("reasoning")
                || lower.contains("thinking")
        }
        Value::Array(items) => items.iter().any(json_contains_reasoning_tokens),
        Value::Object(map) => map.iter().any(|(key, value)| {
            let key_lower = key.to_lowercase();
            key_lower.contains("reason")
                || key_lower.contains("think")
                || json_contains_reasoning_tokens(value)
        }),
    }
}

fn json_contains_tool_use_tokens(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
        Value::String(text) => {
            let lower = text.to_lowercase();
            lower.contains("tool_call")
                || lower.contains("tool_calls")
                || lower.contains("tool_use")
                || lower.contains("tool_result")
                || lower.contains("function_call")
                || lower.contains("function_calls")
                || lower.contains("parallel_tool_calls")
                || lower.contains("\"tool\"")
        }
        Value::Array(items) => items.iter().any(json_contains_tool_use_tokens),
        Value::Object(map) => map.iter().any(|(key, value)| {
            let key_lower = key.to_lowercase();
            key_lower == "tool_calls"
                || key_lower == "tool_call"
                || key_lower == "tool_use"
                || key_lower == "tool_result"
                || key_lower == "parallel_tool_calls"
                || key_lower == "function_call"
                || key_lower == "function_calls"
                || json_contains_tool_use_tokens(value)
        }),
    }
}

fn read_local_metadata_jsons(path: &Path) -> Vec<Value> {
    let mut values = Vec::new();
    for dir in path.ancestors().skip(1).take(6) {
        for name in [
            "config.json",
            "tokenizer_config.json",
            "chat_template.json",
            "chat_template.jinja",
        ] {
            let candidate = dir.join(name);
            if !candidate.is_file() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            if name.ends_with(".jinja") {
                values.push(Value::String(text));
            } else if let Ok(value) = serde_json::from_str(&text) {
                values.push(value);
            }
        }
    }
    values
}

async fn fetch_remote_metadata_jsons(repo: &str, revision: Option<&str>) -> Vec<Value> {
    let mut values = Vec::new();
    for filename in [
        "config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ] {
        if let Some(value) = fetch_remote_json(repo, revision, filename).await {
            values.push(value);
        }
    }
    values
}

async fn fetch_remote_json(repo: &str, revision: Option<&str>, file: &str) -> Option<Value> {
    let api = build_hf_tokio_api(false).ok()?;
    let repo = match revision {
        Some(revision) => {
            Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string())
        }
        None => Repo::new(repo.to_string(), RepoType::Model),
    };
    let path = api.repo(repo).get(file).await.ok()?;
    let text = tokio::fs::read_to_string(path).await.ok()?;
    if file.ends_with(".jinja") {
        Some(Value::String(text))
    } else {
        serde_json::from_str(&text).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{merge_name_signals, CapabilityLevel};

    #[test]
    fn qwen3vl_name_signal_is_supported_vision() {
        let caps = merge_name_signals(
            Default::default(),
            &[
                "Qwen3VL-2B-Instruct-Q4_K_M",
                "Qwen/Qwen3-VL-2B-Instruct-GGUF",
            ],
        );
        assert_eq!(caps.vision, CapabilityLevel::Supported);
        assert!(caps.multimodal);
    }
}
