use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptTemplate {
    ChatMl {
        default_system_prompt: Option<String>,
    },
    Llama3,
}

impl PromptTemplate {
    pub fn detect(dir: &Path, config: &Value) -> Self {
        if let Some(text) = read_template_text(dir) {
            if let Some(template) = detect_template_from_text(&text) {
                return template;
            }
        }

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

        if model_type.starts_with("qwen")
            || architectures.iter().any(|value| value.contains("qwen"))
        {
            return PromptTemplate::ChatMl {
                default_system_prompt: Some("You are a helpful assistant.".to_string()),
            };
        }

        PromptTemplate::Llama3
    }

    pub fn render_messages(&self, messages: &[Value]) -> String {
        match self {
            PromptTemplate::ChatMl {
                default_system_prompt,
            } => render_chatml(messages, default_system_prompt.as_deref()),
            PromptTemplate::Llama3 => render_llama3(messages),
        }
    }
}

fn read_template_text(dir: &Path) -> Option<String> {
    for filename in ["chat_template.json", "tokenizer_config.json"] {
        let path = dir.join(filename);
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let value: Value = serde_json::from_str(&text).ok()?;
        if let Some(template) = extract_template_text(&value) {
            return Some(template);
        }
    }
    None
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

fn detect_template_from_text(text: &str) -> Option<PromptTemplate> {
    if text.contains("<|start_header_id|>")
        && text.contains("<|end_header_id|>")
        && text.contains("<|eot_id|>")
    {
        return Some(PromptTemplate::Llama3);
    }

    if text.contains("<|im_start|>") && text.contains("<|im_end|>") {
        return Some(PromptTemplate::ChatMl {
            default_system_prompt: detect_default_system_prompt(text),
        });
    }

    None
}

fn detect_default_system_prompt(text: &str) -> Option<String> {
    if text.contains("You are a helpful assistant.") {
        Some("You are a helpful assistant.".to_string())
    } else if text.contains("You are a helpful assistant") {
        Some("You are a helpful assistant".to_string())
    } else {
        None
    }
}

fn render_chatml(messages: &[Value], default_system_prompt: Option<&str>) -> String {
    let mut prompt = String::new();
    if let Some(default_system_prompt) = default_system_prompt {
        let starts_with_system = messages
            .first()
            .and_then(|message| message.get("role"))
            .and_then(|role| role.as_str())
            == Some("system");
        if !starts_with_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(default_system_prompt);
            prompt.push_str("<|im_end|>\n");
        }
    }

    for message in messages {
        let role = message
            .get("role")
            .and_then(|role| role.as_str())
            .unwrap_or("user");
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message_content_text(message));
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_llama3(messages: &[Value]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        let role = message
            .get("role")
            .and_then(|role| role.as_str())
            .unwrap_or("user");
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message_content_text(message));
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn message_content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                Value::Object(map) => map.get("text").and_then(|value| value.as_str()),
                Value::String(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chatml_from_tokenizer_config() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-template-chatml-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
        assert_eq!(
            template,
            PromptTemplate::ChatMl {
                default_system_prompt: Some("You are a helpful assistant.".to_string())
            }
        );
    }

    #[test]
    fn renders_llama3_prompt_from_hf_template() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-template-llama3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("chat_template.json"),
            serde_json::json!({
                "chat_template": "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
        let prompt = template.render_messages(&[serde_json::json!({
            "role": "user",
            "content": "hello"
        })]);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn renders_array_content_text() {
        let prompt = PromptTemplate::ChatMl {
            default_system_prompt: None,
        }
        .render_messages(&[serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]
        })]);
        assert!(prompt.contains("hello world"));
    }
}
