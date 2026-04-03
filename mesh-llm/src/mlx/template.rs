use anyhow::{Context, Result};
use chrono::Local;
use minijinja::{Environment, ErrorKind, UndefinedBehavior};
use serde_json::Value;
use std::path::Path;

#[cfg(test)]
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptTemplate {
    HuggingFace {
        template: String,
        special_tokens: SpecialTokens,
        source_file: String,
        behavior: crate::models::ModelPromptBehavior,
        fallback: Box<PromptTemplate>,
    },
    ChatMl {
        default_system_prompt: Option<String>,
    },
    Gemma3,
    Llama3,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SpecialTokens {
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
}

impl PromptTemplate {
    pub fn detect(dir: &Path, config: &Value) -> Self {
        let fallback = heuristic_prompt_template(config);
        if let Some((source_file, template)) = read_template_text(dir) {
            if let Err(err) = validate_hf_template(&template) {
                tracing::warn!(
                    "MLX prompt template: failed to compile HF template from {}: {err}; falling back to {:?}",
                    source_file,
                    fallback.behavior().prompt_template
                );
                return fallback;
            }
            let behavior = crate::models::infer_prompt_behavior_for_dir(dir)
                .unwrap_or_else(|| fallback.behavior());
            tracing::info!(
                "MLX prompt template: loaded HF template from {} (kind={}, source={})",
                source_file,
                behavior
                    .prompt_template
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                behavior
                    .template_source
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
            );
            return PromptTemplate::HuggingFace {
                template,
                special_tokens: read_special_tokens(dir),
                source_file,
                behavior,
                fallback: Box::new(fallback),
            };
        }
        let behavior = fallback.behavior();
        tracing::info!(
            "MLX prompt template: no HF template found, using {} fallback",
            behavior
                .prompt_template
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        );
        fallback
    }

    pub fn behavior(&self) -> crate::models::ModelPromptBehavior {
        match self {
            PromptTemplate::HuggingFace { behavior, .. } => behavior.clone(),
            PromptTemplate::ChatMl {
                default_system_prompt,
            } => crate::models::ModelPromptBehavior {
                prompt_template: Some("chatml".to_string()),
                default_system_prompt: default_system_prompt.clone(),
                template_source: Some("fallback".to_string()),
            },
            PromptTemplate::Gemma3 => crate::models::ModelPromptBehavior {
                prompt_template: Some("gemma3".to_string()),
                default_system_prompt: None,
                template_source: Some("fallback".to_string()),
            },
            PromptTemplate::Llama3 => crate::models::ModelPromptBehavior {
                prompt_template: Some("llama3".to_string()),
                default_system_prompt: None,
                template_source: Some("fallback".to_string()),
            },
        }
    }

    pub fn render_request(&self, req: &Value) -> Result<String> {
        match self {
            PromptTemplate::HuggingFace {
                template,
                special_tokens,
                source_file,
                fallback,
                ..
            } => match render_hf_template(template, special_tokens, req) {
                Ok(prompt) => Ok(prompt),
                Err(err) => {
                    tracing::warn!(
                        "MLX prompt template: failed to render HF template from {}: {err}; falling back to {:?}",
                        source_file,
                        fallback.behavior().prompt_template
                    );
                    fallback.render_request(req)
                }
            },
            PromptTemplate::ChatMl {
                default_system_prompt,
            } => {
                let messages = req["messages"]
                    .as_array()
                    .context("missing messages array")?;
                Ok(render_chatml(messages, default_system_prompt.as_deref()))
            }
            PromptTemplate::Gemma3 => {
                let messages = req["messages"]
                    .as_array()
                    .context("missing messages array")?;
                Ok(render_gemma3(messages))
            }
            PromptTemplate::Llama3 => {
                let messages = req["messages"]
                    .as_array()
                    .context("missing messages array")?;
                Ok(render_llama3(messages))
            }
        }
    }
}

fn validate_hf_template(template: &str) -> Result<()> {
    let mut env = build_hf_environment();
    env.add_template("chat", template)
        .context("compile HF chat template")?;
    Ok(())
}

fn heuristic_prompt_template(config: &Value) -> PromptTemplate {
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
        return PromptTemplate::ChatMl {
            default_system_prompt: Some("You are a helpful assistant.".to_string()),
        };
    }
    if model_type.starts_with("gemma") || architectures.iter().any(|value| value.contains("gemma"))
    {
        return PromptTemplate::Gemma3;
    }

    PromptTemplate::Llama3
}

fn render_hf_template(
    template: &str,
    special_tokens: &SpecialTokens,
    req: &Value,
) -> Result<String> {
    let mut env = build_hf_environment();
    env.add_template("chat", template)
        .context("compile HF chat template")?;

    let tmpl = env.get_template("chat").context("load HF chat template")?;
    let messages = req
        .get("messages")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let tools = req.get("tools").cloned();
    let custom_tools = req.get("custom_tools").cloned();
    let add_generation_prompt = req
        .get("add_generation_prompt")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".to_string(), messages);
    ctx.insert("tools".to_string(), tools.unwrap_or(Value::Null));
    ctx.insert(
        "documents".to_string(),
        req.get("documents").cloned().unwrap_or(Value::Null),
    );
    ctx.insert(
        "builtin_tools".to_string(),
        req.get("builtin_tools").cloned().unwrap_or(Value::Null),
    );
    ctx.insert(
        "add_generation_prompt".to_string(),
        Value::Bool(add_generation_prompt),
    );
    if let Some(custom_tools) = custom_tools {
        ctx.insert("custom_tools".to_string(), custom_tools);
    }
    for (key, value) in [
        (
            "tools_in_user_message",
            req.get("tools_in_user_message").cloned(),
        ),
        ("keep_past_thinking", req.get("keep_past_thinking").cloned()),
        ("date_string", req.get("date_string").cloned()),
        ("reasoning_effort", req.get("reasoning_effort").cloned()),
        ("enable_thinking", req.get("enable_thinking").cloned()),
    ] {
        if let Some(value) = value {
            ctx.insert(key.to_string(), value);
        }
    }
    if let Some(token) = &special_tokens.bos_token {
        ctx.insert("bos_token".to_string(), Value::String(token.clone()));
    }
    if let Some(token) = &special_tokens.eos_token {
        ctx.insert("eos_token".to_string(), Value::String(token.clone()));
    }
    if let Some(token) = &special_tokens.pad_token {
        ctx.insert("pad_token".to_string(), Value::String(token.clone()));
    }
    if let Some(token) = &special_tokens.unk_token {
        ctx.insert("unk_token".to_string(), Value::String(token.clone()));
    }

    let rendered = tmpl.render(Value::Object(ctx))?;

    Ok(rendered)
}

fn build_hf_environment<'a>() -> Environment<'a> {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_function(
        "strftime_now",
        |format: String| -> std::result::Result<String, minijinja::Error> {
            Ok(Local::now().format(&format).to_string())
        },
    );
    env
}

fn read_template_text(dir: &Path) -> Option<(String, String)> {
    for filename in ["chat_template.json", "tokenizer_config.json"] {
        let path = dir.join(filename);
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let value: Value = serde_json::from_str(&text).ok()?;
        if let Some(template) = extract_template_text(&value) {
            return Some((filename.to_string(), template));
        }
    }
    None
}

fn read_special_tokens(dir: &Path) -> SpecialTokens {
    let mut tokens = SpecialTokens::default();
    let path = dir.join("tokenizer_config.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return tokens;
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return tokens;
    };

    tokens.bos_token = extract_token_string(value.get("bos_token"));
    tokens.eos_token = extract_token_string(value.get("eos_token"));
    tokens.pad_token = extract_token_string(value.get("pad_token"));
    tokens.unk_token = extract_token_string(value.get("unk_token"));
    tokens
}

fn extract_token_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Object(map)) => map
            .get("content")
            .and_then(|content| content.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    }
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

fn render_gemma3(messages: &[Value]) -> String {
    let mut prompt = String::from("<bos>");
    let mut loop_messages = messages;
    let mut first_user_prefix = String::new();

    if let Some(first) = messages.first() {
        if first.get("role").and_then(|role| role.as_str()) == Some("system") {
            first_user_prefix = message_content_text(first);
            if !first_user_prefix.is_empty() {
                first_user_prefix.push_str("\n\n");
            }
            loop_messages = &messages[1..];
        }
    }

    for (index, message) in loop_messages.iter().enumerate() {
        let role = match message.get("role").and_then(|role| role.as_str()) {
            Some("assistant") => "model",
            Some("user") => "user",
            Some(other) => other,
            None => "user",
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        if index == 0 && !first_user_prefix.is_empty() {
            prompt.push_str(&first_user_prefix);
        }
        prompt.push_str(&gemma_message_content_text(message));
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
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

fn gemma_message_content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.trim().to_string(),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    Value::Object(map) => match map.get("type").and_then(|value| value.as_str()) {
                        Some("image") => out.push_str("<start_of_image>"),
                        Some("text") => {
                            if let Some(text) = map.get("text").and_then(|value| value.as_str()) {
                                out.push_str(text.trim());
                            }
                        }
                        _ => {}
                    },
                    Value::String(text) => out.push_str(text.trim()),
                    _ => {}
                }
            }
            out
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct HfTemplateFixture {
        repo: String,
        source_file: String,
        expect_hf_render: bool,
        family: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
        pad_token: Option<String>,
        unk_token: Option<String>,
        template: String,
    }

    fn hf_template_corpus() -> Vec<HfTemplateFixture> {
        serde_json::from_str(include_str!("testdata/hf_template_corpus.json"))
            .expect("valid HF template corpus")
    }

    fn fixture_config(family: &str) -> Value {
        match family {
            "llama" => json!({"model_type":"llama","architectures":["LlamaForCausalLM"]}),
            "qwen" | "qwen3" | "qwen3_coder_next" | "deepseek_qwen3" => {
                json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]})
            }
            "gemma3" => {
                json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]})
            }
            "mistral" => {
                json!({"model_type":"mistral","architectures":["MistralForCausalLM"]})
            }
            "lfm2" => json!({"model_type":"lfm2","architectures":["LlamaForCausalLM"]}),
            other => panic!("unknown fixture family: {other}"),
        }
    }

    fn fixture_request(family: &str) -> Value {
        match family {
            "llama" => json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"}
                ],
                "add_generation_prompt": true
            }),
            "qwen" => json!({
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
                "add_generation_prompt": true
            }),
            "gemma3" => json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "look "},
                        {"type": "image"},
                        {"type": "text", "text": "here"}
                    ]}
                ],
                "add_generation_prompt": true
            }),
            "mistral" => json!({
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "again"}
                ],
                "add_generation_prompt": true
            }),
            "lfm2" => json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "<think>\ninternal\n</think>\nhi"},
                    {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]}
                ],
                "keep_past_thinking": false,
                "add_generation_prompt": true
            }),
            "deepseek_qwen3" => json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"}
                ],
                "add_generation_prompt": true
            }),
            "qwen3" | "qwen3_coder_next" => json!({
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": true
            }),
            other => panic!("unknown fixture family: {other}"),
        }
    }

    fn write_hf_fixture_dir(fixture: &HfTemplateFixture) -> std::path::PathBuf {
        let slug = fixture
            .repo
            .replace('/', "-")
            .replace('.', "-")
            .replace('_', "-");
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-hf-template-corpus-{}-{}",
            slug,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        match fixture.source_file.as_str() {
            "chat_template.json" => {
                std::fs::write(
                    root.join("chat_template.json"),
                    serde_json::to_string(&fixture.template).unwrap(),
                )
                .unwrap();
            }
            "tokenizer_config.json" => {
                std::fs::write(
                    root.join("tokenizer_config.json"),
                    serde_json::json!({
                        "chat_template": fixture.template,
                        "bos_token": fixture.bos_token,
                        "eos_token": fixture.eos_token,
                        "pad_token": fixture.pad_token,
                        "unk_token": fixture.unk_token,
                    })
                    .to_string(),
                )
                .unwrap();
                return root;
            }
            other => panic!("unknown template source: {other}"),
        }

        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": fixture.bos_token,
                "eos_token": fixture.eos_token,
                "pad_token": fixture.pad_token,
                "unk_token": fixture.unk_token,
            })
            .to_string(),
        )
        .unwrap();

        root
    }

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
        match template {
            PromptTemplate::HuggingFace { fallback, .. } => {
                assert_eq!(
                    *fallback,
                    PromptTemplate::ChatMl {
                        default_system_prompt: Some("You are a helpful assistant.".to_string())
                    }
                );
            }
            other => panic!("expected huggingface template, got {other:?}"),
        }
    }

    #[test]
    fn renders_llama3_prompt_from_hf_template() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-template-llama3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": "<|begin_of_text|>",
                "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
        let prompt = template
            .render_request(&json!({
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap();
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn renders_qwen_tools_template_with_minijinja() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-template-qwen-tools-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "chat_template": "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{{- messages[0]['content'] }}{%- else %}{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}{%- endif %}{{- '\\n\\n# Tools\\n\\n<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools><|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
        let prompt = template
            .render_request(&json!({
                "messages": [{"role": "user", "content": "use a tool"}],
                "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}]
            }))
            .unwrap();

        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\":\"run\""));
        assert!(prompt.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen_prompt_parity_fixture_matches_expected_output() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-template-qwen-fixture-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{%- if messages[0]['role'] != 'system' -%}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
        let prompt = template
            .render_request(&json!({
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap();

        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nhello<|im_end|><|im_start|>assistant"
        );
    }

    #[test]
    fn llama3_prompt_parity_fixture_matches_expected_output() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-template-llama3-fixture-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": "<|begin_of_text|>",
                "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
        let prompt = template
            .render_request(&json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"}
                ]
            }))
            .unwrap();

        assert_eq!(
            prompt,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        );
    }

    #[test]
    fn gemma3_prompt_parity_fixture_matches_expected_output() {
        let root = std::env::temp_dir().join(format!(
            "mesh-llm-template-gemma3-fixture-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": "<bos>",
                "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n    {{ '<end_of_turn>\\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\\n'}}\n{%- endif -%}\n"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"gemma3"}));
        let prompt = template
            .render_request(&json!({
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "look "},
                        {"type": "image"},
                        {"type": "text", "text": "here"}
                    ]}
                ]
            }))
            .unwrap();

        assert_eq!(
            prompt,
            "<bos><start_of_turn>user\nBe concise.\n\nhello<end_of_turn>\n<start_of_turn>model\nhi<end_of_turn>\n<start_of_turn>user\nlook<start_of_image>here<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn heuristic_fallback_uses_gemma3_for_gemma_models() {
        let template = PromptTemplate::detect(
            Path::new("/tmp/does-not-need-to-exist"),
            &serde_json::json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]}),
        );
        assert_eq!(template, PromptTemplate::Gemma3);
    }

    #[test]
    fn renders_when_template_uses_strftime_now() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-template-fallback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{{ strftime_now('%Y-%m-%d') }}"
            })
            .to_string(),
        )
        .unwrap();

        let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
        let prompt = template
            .render_request(&json!({
                "messages": [{"role": "user", "content": "hello world"}]
            }))
            .unwrap();

        assert_eq!(prompt.len(), 10);
        assert_eq!(prompt.chars().filter(|c| *c == '-').count(), 2);
    }

    #[test]
    fn real_hf_template_corpus_behaves_as_expected() {
        for fixture in hf_template_corpus() {
            let root = write_hf_fixture_dir(&fixture);
            let config = fixture_config(&fixture.family);
            let req = fixture_request(&fixture.family);
            let special_tokens = read_special_tokens(&root);

            validate_hf_template(&fixture.template)
                .unwrap_or_else(|err| panic!("{} should compile: {err}", fixture.repo));

            if fixture.expect_hf_render {
                let prompt = render_hf_template(&fixture.template, &special_tokens, &req)
                    .unwrap_or_else(|err| {
                        panic!("{} should render via HF path: {err}", fixture.repo)
                    });
                assert!(
                    !prompt.trim().is_empty(),
                    "{} rendered an empty prompt",
                    fixture.repo
                );
            } else {
                render_hf_template(&fixture.template, &special_tokens, &req)
                    .expect_err("fixture should still require fallback");

                let prompt = PromptTemplate::detect(&root, &config)
                    .render_request(&req)
                    .unwrap_or_else(|render_err| {
                        panic!("{} should render via fallback: {render_err}", fixture.repo)
                    });
                assert!(
                    !prompt.trim().is_empty(),
                    "{} fallback rendered an empty prompt",
                    fixture.repo
                );
            }
        }
    }
}
