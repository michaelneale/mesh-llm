use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::LazyLock;

static CHAT_TEMPLATE_RE: LazyLock<regex_lite::Regex> = LazyLock::new(|| {
    regex_lite::Regex::new(r#""chat_template"\s*:\s*"((?:\\.|[^"\\])*)""#)
        .expect("Failed to compile CHAT_TEMPLATE_RE regex pattern")
});

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPromptBehavior {
    pub prompt_template: Option<String>,
    pub default_system_prompt: Option<String>,
    pub template_source: Option<String>,
}

pub fn infer_prompt_behavior_for_dir(dir: &Path) -> Option<ModelPromptBehavior> {
    let config = read_config_json(dir);
    if let Some(template) = read_template_text(dir) {
        let mut behavior = classify_template_behavior(&template, config.as_ref());
        behavior.template_source = Some("huggingface".to_string());
        return Some(behavior);
    }
    config
        .as_ref()
        .and_then(heuristic_prompt_behavior)
        .map(|mut behavior| {
            behavior.template_source = Some("fallback".to_string());
            behavior
        })
}

fn read_config_json(dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// Scans a model directory for a chat template and returns `(source_filename, template_text)`.
///
/// Checks files in priority order:
/// 1. `chat_template.jinja`
/// 2. `chat_template.json`
/// 3. `tokenizer_config.json`
///
/// This shared helper is the single source of truth used by both
/// `infer_prompt_behavior_for_dir` and the MLX template loader.
pub fn find_template_with_source(dir: &Path) -> Option<(String, String)> {
    for filename in [
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer_config.json",
    ] {
        let Ok(text) = std::fs::read_to_string(dir.join(filename)) else {
            continue;
        };
        if filename.ends_with(".jinja") {
            return Some((filename.to_string(), text));
        }
        if let Some(template) = extract_template_text_from_json_text(&text) {
            return Some((filename.to_string(), template));
        }
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(template) = extract_template_text(&value) {
            return Some((filename.to_string(), template));
        }
    }
    None
}

fn read_template_text(dir: &Path) -> Option<String> {
    find_template_with_source(dir).map(|(_source, text)| text)
}

fn extract_template_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => map
            .get("chat_template")
            .and_then(|template| template.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    }
}

fn extract_template_text_from_json_text(text: &str) -> Option<String> {
    let captures = CHAT_TEMPLATE_RE.captures(text)?;
    serde_json::from_str::<String>(&format!("\"{}\"", &captures[1])).ok()
}

fn classify_template_behavior(template: &str, config: Option<&Value>) -> ModelPromptBehavior {
    let prompt_template = if template.contains("<|im_start|>") {
        Some("chatml".to_string())
    } else if template.contains("<start_of_turn>") && template.contains("<end_of_turn>") {
        Some("gemma3".to_string())
    } else if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
        Some("llama3".to_string())
    } else {
        Some("hf_template".to_string())
    };
    ModelPromptBehavior {
        prompt_template,
        default_system_prompt: inferred_default_system_prompt(config, template),
        template_source: None,
    }
}

fn heuristic_prompt_behavior(config: &Value) -> Option<ModelPromptBehavior> {
    let family = model_family(config)?;
    let prompt_template = match family.as_str() {
        "qwen" => "chatml",
        "gemma" => "gemma3",
        "llama" => "llama3",
        _ => return None,
    };
    Some(ModelPromptBehavior {
        prompt_template: Some(prompt_template.to_string()),
        default_system_prompt: if family == "qwen" {
            Some("You are a helpful assistant.".to_string())
        } else {
            None
        },
        template_source: None,
    })
}

fn inferred_default_system_prompt(config: Option<&Value>, template: &str) -> Option<String> {
    if template.contains("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") {
        return Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".into());
    }
    if template.contains("You are a helpful assistant.") {
        return Some("You are a helpful assistant.".to_string());
    }
    match config.and_then(model_family).as_deref() {
        Some("qwen") if template.contains("<|im_start|>system") => {
            Some("You are a helpful assistant.".to_string())
        }
        _ => None,
    }
}

fn model_family(config: &Value) -> Option<String> {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();

    if model_type.starts_with("qwen") || architectures.iter().any(|value| value.contains("qwen")) {
        return Some("qwen".to_string());
    }
    if model_type.starts_with("gemma") || architectures.iter().any(|value| value.contains("gemma"))
    {
        return Some("gemma".to_string());
    }
    if model_type == "llama" || architectures.iter().any(|value| value.contains("llama")) {
        return Some("llama".to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_huggingface_template_behavior_for_qwen() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-prompt-qwen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({"model_type":"qwen2"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            })
            .to_string(),
        )
        .unwrap();

        let behavior = infer_prompt_behavior_for_dir(&root).unwrap();
        assert_eq!(behavior.prompt_template.as_deref(), Some("chatml"));
        assert_eq!(
            behavior.default_system_prompt.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert_eq!(behavior.template_source.as_deref(), Some("huggingface"));
    }

    #[test]
    fn infers_fallback_behavior_for_llama() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-prompt-llama-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "model_type":"llama",
                "architectures":["LlamaForCausalLM"]
            })
            .to_string(),
        )
        .unwrap();

        let behavior = infer_prompt_behavior_for_dir(&root).unwrap();
        assert_eq!(behavior.prompt_template.as_deref(), Some("llama3"));
        assert_eq!(behavior.template_source.as_deref(), Some("fallback"));
    }

    #[test]
    fn infers_fallback_behavior_for_gemma() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-prompt-gemma-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "model_type":"gemma3",
                "architectures":["Gemma3ForConditionalGeneration"]
            })
            .to_string(),
        )
        .unwrap();

        let behavior = infer_prompt_behavior_for_dir(&root).unwrap();
        assert_eq!(behavior.prompt_template.as_deref(), Some("gemma3"));
        assert_eq!(behavior.template_source.as_deref(), Some("fallback"));
    }

    #[test]
    fn infers_huggingface_template_behavior_for_gemma() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-prompt-gemma-hf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "model_type":"gemma3",
                "architectures":["Gemma3ForConditionalGeneration"]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{{ bos_token }}<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
            })
            .to_string(),
        )
        .unwrap();

        let behavior = infer_prompt_behavior_for_dir(&root).unwrap();
        assert_eq!(behavior.prompt_template.as_deref(), Some("gemma3"));
        assert_eq!(behavior.template_source.as_deref(), Some("huggingface"));
    }
}
