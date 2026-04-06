use anyhow::{Context, Result};
use chrono::Local;
use minijinja::{Environment, ErrorKind, UndefinedBehavior};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptTemplate {
    HuggingFace {
        template: String,
        special_tokens: SpecialTokens,
        source_file: String,
        behavior: crate::models::ModelPromptBehavior,
        reasoning_defaults: ReasoningDefaults,
        reasoning_template: ReasoningTemplate,
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReasoningDefaults {
    enable_thinking: Option<bool>,
    thinking: Option<bool>,
    keep_past_thinking: Option<bool>,
    reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReasoningTemplate {
    pub supports_explicit_reasoning: bool,
    pub tagged_reasoning: Vec<TaggedReasoningBlock>,
    pub default_stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaggedReasoningBlock {
    pub start: String,
    pub end: String,
}

impl PromptTemplate {
    pub fn detect(dir: &Path, config: &Value) -> Self {
        let fallback = heuristic_prompt_template(config);
        if let Some((source_file, template)) = read_template_text(dir) {
            let template = normalize_hf_template(&template);
            let reasoning_template = detect_reasoning_template(&template);
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
                reasoning_defaults: reasoning_defaults(config),
                reasoning_template,
                fallback: Box::new(fallback),
            };
        }
        let behavior = fallback.behavior();
        tracing::info!(
            "MLX prompt template: no HF template found in {} (chat_template.jinja={}, chat_template.json={}, tokenizer_config.json={}), using {} fallback",
            dir.display(),
            dir.join("chat_template.jinja").exists(),
            dir.join("chat_template.json").exists(),
            dir.join("tokenizer_config.json").exists(),
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
                reasoning_defaults,
                source_file,
                fallback,
                ..
            } => match render_hf_template(template, special_tokens, reasoning_defaults, req) {
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

    pub fn reasoning_template(&self) -> ReasoningTemplate {
        match self {
            PromptTemplate::HuggingFace {
                reasoning_template, ..
            } => reasoning_template.clone(),
            PromptTemplate::ChatMl { .. } | PromptTemplate::Gemma3 | PromptTemplate::Llama3 => {
                ReasoningTemplate::default()
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
    reasoning_defaults: &ReasoningDefaults,
    req: &Value,
) -> Result<String> {
    let mut env = build_hf_environment();
    env.add_template("chat", template)
        .context("compile HF chat template")?;

    let tmpl = env.get_template("chat").context("load HF chat template")?;
    let messages = normalize_hf_messages(
        template,
        req.get("messages")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    );
    let tools = req.get("tools").cloned();
    let custom_tools = req.get("custom_tools").cloned();
    let add_generation_prompt = req
        .get("add_generation_prompt")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".to_string(), messages);
    ctx.insert(
        "tools".to_string(),
        tools.unwrap_or_else(|| absent_tools_value(template)),
    );
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
            template_kwarg(req, "tools_in_user_message"),
        ),
        (
            "keep_past_thinking",
            template_kwarg(req, "keep_past_thinking"),
        ),
        ("date_string", template_kwarg(req, "date_string")),
        ("reasoning_effort", template_kwarg(req, "reasoning_effort")),
        ("thinking", template_kwarg(req, "thinking")),
    ] {
        if let Some(value) = value {
            ctx.insert(key.to_string(), value);
        }
    }
    match template_kwarg(req, "enable_thinking") {
        Some(value) => {
            ctx.insert("enable_thinking".to_string(), value);
        }
        None => {
            if let Some(default_enable_thinking) = reasoning_defaults.enable_thinking {
                ctx.insert(
                    "enable_thinking".to_string(),
                    Value::Bool(default_enable_thinking),
                );
            }
        }
    }
    if !ctx.contains_key("thinking") {
        if let Some(value) = template_kwarg(req, "enable_thinking") {
            ctx.insert("thinking".to_string(), value);
        } else if let Some(default_thinking) = reasoning_defaults.thinking {
            ctx.insert("thinking".to_string(), Value::Bool(default_thinking));
        }
    }
    if !ctx.contains_key("keep_past_thinking") {
        if let Some(value) = template_kwarg(req, "enable_thinking") {
            ctx.insert("keep_past_thinking".to_string(), value);
        } else if let Some(default_keep_past_thinking) = reasoning_defaults.keep_past_thinking {
            ctx.insert(
                "keep_past_thinking".to_string(),
                Value::Bool(default_keep_past_thinking),
            );
        }
    }
    if !ctx.contains_key("reasoning_effort") {
        if let Some(value) = template_kwarg(req, "enable_thinking") {
            if value == Value::Bool(false) {
                ctx.insert(
                    "reasoning_effort".to_string(),
                    Value::String("low".to_string()),
                );
            } else if value == Value::Bool(true) {
                ctx.insert(
                    "reasoning_effort".to_string(),
                    Value::String("medium".to_string()),
                );
            }
        } else if let Some(default_reasoning_effort) = &reasoning_defaults.reasoning_effort {
            ctx.insert(
                "reasoning_effort".to_string(),
                Value::String(default_reasoning_effort.clone()),
            );
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

fn reasoning_defaults(config: &Value) -> ReasoningDefaults {
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

    if model_type == "qwen3" || architectures.iter().any(|value| value.contains("qwen3")) {
        return ReasoningDefaults {
            enable_thinking: Some(false),
            ..ReasoningDefaults::default()
        };
    }
    if model_type.starts_with("glm") || architectures.iter().any(|value| value.contains("glm")) {
        return ReasoningDefaults {
            enable_thinking: Some(false),
            ..ReasoningDefaults::default()
        };
    }
    if model_type == "kimi" || architectures.iter().any(|value| value.contains("kimi")) {
        return ReasoningDefaults {
            thinking: Some(false),
            ..ReasoningDefaults::default()
        };
    }
    if model_type == "gpt_oss" || architectures.iter().any(|value| value.contains("gptoss")) {
        return ReasoningDefaults {
            reasoning_effort: Some("low".to_string()),
            ..ReasoningDefaults::default()
        };
    }
    if model_type == "lfm2" || architectures.iter().any(|value| value.contains("lfm2")) {
        return ReasoningDefaults {
            keep_past_thinking: Some(false),
            ..ReasoningDefaults::default()
        };
    }

    ReasoningDefaults::default()
}

fn detect_reasoning_template(template: &str) -> ReasoningTemplate {
    let mut tagged_reasoning = Vec::new();

    if is_old_qwen_reasoning_template(template) || template_mentions_think_tags(template) {
        tagged_reasoning.push(TaggedReasoningBlock {
            start: "<think>".to_string(),
            end: "</think>".to_string(),
        });
    }

    if template.contains("<|channel>thought") && template.contains("<channel|>") {
        tagged_reasoning.push(TaggedReasoningBlock {
            start: "<|channel>thought".to_string(),
            end: "<channel|>".to_string(),
        });
    }

    tagged_reasoning
        .sort_by(|left, right| left.start.cmp(&right.start).then(left.end.cmp(&right.end)));
    tagged_reasoning.dedup();

    ReasoningTemplate {
        supports_explicit_reasoning: template_supports_explicit_reasoning(template)
            || !tagged_reasoning.is_empty(),
        tagged_reasoning,
        default_stop_sequences: detect_default_stop_sequences(template),
    }
}

fn detect_default_stop_sequences(template: &str) -> Vec<String> {
    let mut stops = Vec::new();

    for stop in [
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "<turn|>",
    ] {
        if template.contains(stop) {
            stops.push(stop.to_string());
        }
    }

    stops.sort();
    stops.dedup();
    stops
}

fn template_supports_explicit_reasoning(template: &str) -> bool {
    [
        "enable_thinking",
        "thinking",
        "keep_past_thinking",
        "reasoning_effort",
        "reasoning_content",
    ]
    .into_iter()
    .any(|needle| template.contains(needle))
}

fn template_mentions_think_tags(template: &str) -> bool {
    template.contains("<think>") && template.contains("</think>")
}

fn is_old_qwen_reasoning_template(template: &str) -> bool {
    let splits_on_end_think = [
        "split('</think>')",
        "split(\"</think>\")",
        "| split('</think>')",
        "| split(\"</think>\")",
    ]
    .into_iter()
    .any(|needle| template.contains(needle));

    splits_on_end_think
        && !template.contains("reasoning_content")
        && !template.contains("<SPECIAL_12>")
}

fn template_kwarg(req: &Value, key: &str) -> Option<Value> {
    req.get(key).cloned().or_else(|| {
        req.get("chat_template_kwargs")
            .and_then(|value| value.get(key))
            .cloned()
    })
}

fn template_references_message_field(template: &str, field: &str) -> bool {
    template.contains(&format!("message['{field}']"))
        || template.contains(&format!("message[\"{field}\"]"))
        || template.contains(&format!("message.{field}"))
}

fn template_checks_message_field_presence(template: &str, field: &str) -> bool {
    template.contains(&format!("'{field}' in message"))
        || template.contains(&format!("\"{field}\" in message"))
}

fn normalize_hf_messages(template: &str, messages: Value) -> Value {
    let Some(messages) = messages.as_array() else {
        return messages;
    };
    let fill_tool_calls = template_references_message_field(template, "tool_calls")
        && !template_checks_message_field_presence(template, "tool_calls");
    let fill_tool_call_id = template_references_message_field(template, "tool_call_id");
    let fill_name = template_references_message_field(template, "name");
    let fill_tool_responses = template_references_message_field(template, "tool_responses")
        && !template_checks_message_field_presence(template, "tool_responses");

    Value::Array(
        messages
            .iter()
            .map(|message| {
                let Some(object) = message.as_object() else {
                    return message.clone();
                };
                let mut normalized = object.clone();
                for (key, should_fill, value) in [
                    ("tool_calls", fill_tool_calls, Value::Null),
                    ("tool_call_id", fill_tool_call_id, Value::Null),
                    ("name", fill_name, Value::Null),
                    ("tool_responses", fill_tool_responses, Value::Null),
                ] {
                    if should_fill {
                        normalized.entry(key.to_string()).or_insert(value);
                    }
                }
                Value::Object(normalized)
            })
            .collect(),
    )
}

fn absent_tools_value(template: &str) -> Value {
    let uses_length = template.contains("tools|length")
        || template.contains("tools | length")
        || template.contains("tools|count")
        || template.contains("tools | count");
    if template.contains("tools is not none") && !uses_length {
        Value::Null
    } else {
        Value::Array(Vec::new())
    }
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
    env.add_filter("startswith", |value: String, prefix: String| {
        value.starts_with(&prefix)
    });
    env.add_filter("endswith", |value: String, suffix: String| {
        value.ends_with(&suffix)
    });
    env.add_filter("split", |value: String, separator: String| {
        value
            .split(&separator)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    });
    env.add_filter("strip", |value: String, chars: Option<String>| {
        strip_chars(&value, chars.as_deref(), true, true)
    });
    env.add_filter("lstrip", |value: String, chars: Option<String>| {
        strip_chars(&value, chars.as_deref(), true, false)
    });
    env.add_filter("rstrip", |value: String, chars: Option<String>| {
        strip_chars(&value, chars.as_deref(), false, true)
    });
    env
}

fn normalize_hf_template(template: &str) -> String {
    let single_get_re =
        regex_lite::Regex::new(r#"\.get\(\s*'([^']+)'\s*(?:,\s*([^)]+?))?\s*\)"#).unwrap();
    let double_get_re =
        regex_lite::Regex::new(r#"\.get\(\s*\"([^\"]+)\"\s*(?:,\s*([^)]+?))?\s*\)"#).unwrap();
    let split_index_re =
        regex_lite::Regex::new(r#"\.split\(([^()]*)\)\s*\[\s*(-?1|0)\s*\]"#).unwrap();
    let mut normalized = single_get_re
        .replace_all(template, |caps: &regex_lite::Captures<'_>| {
            let key = &caps[1];
            let default = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("none");
            format!(r#"["{key}"]|default({default})"#)
        })
        .to_string();
    normalized = double_get_re
        .replace_all(&normalized, |caps: &regex_lite::Captures<'_>| {
            let key = &caps[1];
            let default = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("none");
            format!(r#"["{key}"]|default({default})"#)
        })
        .to_string();
    normalized = split_index_re
        .replace_all(&normalized, |caps: &regex_lite::Captures<'_>| {
            let args = caps[1].trim();
            let selector = if &caps[2] == "-1" { "last" } else { "first" };
            format!(" | split({args}) | {selector}")
        })
        .to_string();
    for (from, to) in [
        (".lstrip(", " | lstrip("),
        (".rstrip(", " | rstrip("),
        (".startswith(", " | startswith("),
        (".endswith(", " | endswith("),
        (".split(", " | split("),
        (".strip(", " | strip("),
        (".keys()", " | items | map(attribute=0)"),
        ("|items", "| items"),
    ] {
        normalized = normalized.replace(from, to);
    }

    strip_tojson_kwargs(&normalized)
}

fn strip_chars(value: &str, chars: Option<&str>, left: bool, right: bool) -> String {
    match chars {
        Some(chars) => {
            let predicate = |c: char| chars.contains(c);
            match (left, right) {
                (true, true) => value.trim_matches(predicate).to_string(),
                (true, false) => value.trim_start_matches(predicate).to_string(),
                (false, true) => value.trim_end_matches(predicate).to_string(),
                (false, false) => value.to_string(),
            }
        }
        None => match (left, right) {
            (true, true) => value.trim().to_string(),
            (true, false) => value.trim_start().to_string(),
            (false, true) => value.trim_end().to_string(),
            (false, false) => value.to_string(),
        },
    }
}

fn strip_tojson_kwargs(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut cursor = 0usize;

    while let Some(rel) = template[cursor..].find("tojson(") {
        let start = cursor + rel;
        out.push_str(&template[cursor..start]);

        let args_start = start + "tojson(".len();
        let bytes = template.as_bytes();
        let mut i = args_start;
        let mut depth = 1usize;

        while i < bytes.len() && depth > 0 {
            match bytes[i] as char {
                '(' => depth += 1,
                ')' => depth -= 1,
                '"' | '\'' => {
                    let quote = bytes[i];
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                            continue;
                        }
                        if bytes[i] == quote {
                            break;
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if depth != 0 {
            out.push_str(&template[start..]);
            return out;
        }

        let args = &template[args_start..i - 1];
        if args.contains("separators") || args.contains("ensure_ascii") {
            out.push_str("tojson");
        } else {
            out.push_str(&template[start..i]);
        }
        cursor = i;
    }

    out.push_str(&template[cursor..]);
    out
}

fn read_template_text(dir: &Path) -> Option<(String, String)> {
    crate::models::prompt::find_template_with_source(dir)
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
mod tests;
