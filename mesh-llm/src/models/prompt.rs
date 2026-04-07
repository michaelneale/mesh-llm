use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPromptBehavior {
    pub prompt_template: Option<String>,
    pub default_system_prompt: Option<String>,
    pub template_source: Option<String>,
}

pub fn infer_prompt_behavior_for_dir(_dir: &Path) -> Option<ModelPromptBehavior> {
    None
}

pub fn find_template_with_source(dir: &Path) -> Option<(String, String)> {
    let jinja = dir.join("chat_template.jinja");
    if let Ok(text) = std::fs::read_to_string(&jinja) {
        return Some(("chat_template.jinja".to_string(), text));
    }

    let json_path = dir.join("chat_template.json");
    if let Ok(text) = std::fs::read_to_string(&json_path) {
        if let Ok(template) = serde_json::from_str::<String>(&text) {
            return Some(("chat_template.json".to_string(), template));
        }
    }

    let tokenizer_config = dir.join("tokenizer_config.json");
    if let Ok(text) = std::fs::read_to_string(&tokenizer_config) {
        if let Ok(value) = serde_json::from_str::<Value>(&text) {
            if let Some(template) = value.get("chat_template").and_then(|v| v.as_str()) {
                return Some(("tokenizer_config.json".to_string(), template.to_string()));
            }
        }
    }

    None
}
