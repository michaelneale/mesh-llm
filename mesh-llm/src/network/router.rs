/// Smart model router — classifies requests and picks the best model.
use serde_json::Value;

// ── Request categories ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Code,
    Reasoning,
    Chat,
    ToolCall,
    Creative,
    /// Factual lookup, summarization, knowledge retrieval
    Info,
    /// Image generation or analysis (future: multimodal models)
    Image,
}

/// How complex/heavy the request appears to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Quick,    // simple fact, short answer, casual
    Moderate, // normal conversation, standard code
    Deep,     // long reasoning, complex analysis, architecture
}

/// Full classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub category: Category,
    pub complexity: Complexity,
    pub needs_tools: bool,
    pub has_media_inputs: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaRequirements {
    pub has_media: bool,
    pub needs_vision: bool,
    pub needs_audio: bool,
}

// ── Model capabilities for routing ──────────────────────────────────

/// Strip split GGUF suffix like "-00001-of-00004" from a model name.
pub fn strip_split_suffix(name: &str) -> &str {
    // Pattern: -NNNNN-of-NNNNN at the end
    if let Some(idx) = name.rfind("-of-") {
        // Check that what follows is digits and what precedes is -digits
        let after = &name[idx + 4..];
        if after.chars().all(|c| c.is_ascii_digit()) && !after.is_empty() {
            // Find the preceding -NNNNN
            if let Some(dash) = name[..idx].rfind('-') {
                let between = &name[dash + 1..idx];
                if between.chars().all(|c| c.is_ascii_digit()) && !between.is_empty() {
                    return &name[..dash];
                }
            }
        }
    }
    name
}

/// Owned version of strip_split_suffix for contexts that need a String.
pub fn strip_split_suffix_owned(name: &str) -> String {
    strip_split_suffix(name).to_string()
}

// ── Request classification ──────────────────────────────────────────

/// Classify a chat completion request body using heuristics.
/// No LLM call, just pattern matching on the request structure.
/// Classify a request body into category + complexity + needs_tools.
/// Tools presence is an attribute, not a category override — a code request
/// with tools is still Code (with needs_tools=true), not ToolCall.
pub fn classify(body: &Value) -> Classification {
    // Collect all text from messages for keyword analysis
    let text = collect_message_text(body);
    let lower = text.to_lowercase();
    let media = media_requirements(body);

    // Check if the request actually needs tool execution.
    // If the client sends a tools schema, this is an agentic session (Claude Code,
    // Goose, etc.) — always prefer the strongest tool-capable model regardless of
    // what the first message says.  Keyword matching on content is a secondary signal
    // but not required when tools are present.
    let has_tools_schema = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    // Anthropic-style requests may include structured content blocks with
    // explicit tool_use/tool_result blocks — definitely tool-driven.
    let has_tool_blocks = body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| {
            msgs.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_array())
                    .map(|blocks| {
                        blocks.iter().any(|b| {
                            matches!(
                                b.get("type").and_then(|t| t.as_str()),
                                Some("tool_use") | Some("tool_result")
                            )
                        })
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let needs_tools = has_tools_schema || has_tool_blocks;

    // Count last user message tokens (rough proxy for complexity)
    let last_user_len = last_user_message_len(body);

    // Code signals
    let code_signals = [
        "```",
        "def ",
        "fn ",
        "func ",
        "class ",
        "import ",
        "function",
        "const ",
        "let ",
        "var ",
        "return ",
        "write a program",
        "write code",
        "implement",
        "refactor",
        "debug",
        "fix the bug",
        "write a script",
        "code review",
        "pull request",
        "git ",
        "compile",
        "syntax",
        "python",
        "javascript",
        "typescript",
        " rust ",
        "golang",
        "java ",
        "c++",
        " ruby ",
        " swift ",
        "kotlin",
        "algorithm",
        "binary search",
        " sort ",
        "regex",
        " api ",
        " http ",
        " sql ",
        "database",
        " query ",
    ];
    let code_score: usize = code_signals.iter().filter(|s| lower.contains(*s)).count();

    // Reasoning signals
    let reasoning_signals = [
        "prove",
        "explain why",
        "step by step",
        "calculate",
        "solve",
        "derive",
        "what is the probability",
        "how many",
        "analyze",
        "compare and contrast",
        "evaluate",
        "mathematical",
        "theorem",
        "equation",
        "logic",
        "think carefully",
        "reason about",
    ];
    let reasoning_score: usize = reasoning_signals
        .iter()
        .filter(|s| lower.contains(*s))
        .count();

    // Creative signals
    let creative_signals = [
        "write a story",
        "write a poem",
        "creative",
        "imagine",
        "fiction",
        "narrative",
        "compose",
        "brainstorm",
        "write a song",
        "screenplay",
        "dialogue",
    ];
    let creative_score: usize = creative_signals
        .iter()
        .filter(|s| lower.contains(*s))
        .count();

    // Info/knowledge signals — factual lookup, summarization
    let info_signals = [
        "what is",
        "who is",
        "when did",
        "where is",
        "how does",
        "define ",
        "explain ",
        "summarize",
        "summary",
        "overview",
        "tell me about",
        "describe ",
        "what are the",
        "list the",
        "difference between",
        "compare ",
        "history of",
    ];
    let info_score: usize = info_signals.iter().filter(|s| lower.contains(*s)).count();

    // Image signals — generation or analysis (future)
    let image_signals = [
        "image",
        "picture",
        "photo",
        "draw",
        "generate an image",
        "visualize",
        "diagram",
        "screenshot",
        "describe this image",
    ];
    let image_score: usize = image_signals.iter().filter(|s| lower.contains(*s)).count();

    // Deep-thinking signals (want the biggest brain)
    let deep_signals = [
        "architect",
        "design a system",
        "trade-off",
        "tradeoff",
        "in depth",
        "comprehensive",
        "thorough",
        "detailed analysis",
        "long-term",
        "strategy",
        "plan for",
        "review this codebase",
        "rewrite",
        "from scratch",
    ];
    let deep_score: usize = deep_signals.iter().filter(|s| lower.contains(*s)).count();

    // System prompt hints
    let mut system_code = false;
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    let sys = content.to_lowercase();
                    if sys.contains("developer")
                        || sys.contains("coding")
                        || sys.contains("programmer")
                    {
                        system_code = true;
                    }
                }
            }
        }
    }

    // Pick category — tools don't override, content wins
    let category = if system_code
        || code_score >= 2
        || (code_score >= 1 && reasoning_score == 0 && creative_score == 0)
    {
        Category::Code
    } else if reasoning_score >= 2 {
        Category::Reasoning
    } else if creative_score >= 1 {
        Category::Creative
    } else if media.needs_vision || image_score >= 1 {
        Category::Image
    } else if needs_tools && code_score == 0 && reasoning_score == 0 && creative_score == 0 {
        // Only ToolCall if tools present AND no other signal dominates
        Category::ToolCall
    } else if info_score >= 2 && code_score == 0 {
        Category::Info
    } else {
        Category::Chat
    };

    // Complexity: Quick / Moderate / Deep
    let total_messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let complexity = if deep_score >= 1 || last_user_len > 500 || total_messages > 10 {
        Complexity::Deep
    } else if last_user_len < 60 && total_messages <= 2 && reasoning_score == 0 && deep_score == 0 {
        Complexity::Quick
    } else {
        Complexity::Moderate
    };

    Classification {
        category,
        complexity,
        needs_tools,
        has_media_inputs: media.has_media,
    }
}

pub fn media_requirements(body: &Value) -> MediaRequirements {
    let mut requirements = MediaRequirements::default();
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return requirements;
    };

    for msg in messages {
        let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            match block_type {
                "image_url" | "input_image" | "image" => {
                    requirements.has_media = true;
                    requirements.needs_vision = true;
                }
                "audio_url" | "input_audio" | "audio" => {
                    requirements.has_media = true;
                    requirements.needs_audio = true;
                }
                "file" | "input_file" => {
                    requirements.has_media = true;
                }
                _ => {
                    if block.get("image_url").is_some() || block.get("image").is_some() {
                        requirements.has_media = true;
                        requirements.needs_vision = true;
                    }
                    if block.get("audio_url").is_some() || block.get("audio").is_some() {
                        requirements.has_media = true;
                        requirements.needs_audio = true;
                    }
                }
            }
        }
    }

    requirements
}

/// Length of last user message in characters (rough complexity proxy).
fn last_user_message_len(body: &Value) -> usize {
    body.get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| {
            msgs.iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .map(message_text)
        .map(|s| s.len())
        .unwrap_or(0)
}

fn collect_message_text(body: &Value) -> String {
    let mut text = String::new();
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let content = message_text(msg);
            if !content.is_empty() {
                text.push_str(&content);
                text.push('\n');
            }
        }
    }
    text
}

/// Extract message text for both OpenAI-style and Anthropic-style payloads.
fn message_text(msg: &Value) -> String {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return s.to_string();
    }

    // Anthropic content blocks: [{"type":"text","text":"..."}, ...]
    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for b in blocks {
            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                out.push_str(t);
                out.push('\n');
            }
        }
        return out;
    }

    String::new()
}

// ── Model selection ─────────────────────────────────────────────────

/// Pick the best model using full classification (category + complexity + tools).
/// Pick the best model for a classified request using gossiped capabilities.
///
/// Filtering:
///   - `needs_tools` → prefer models with `tool_use != None`
///   - `Reasoning`   → prefer models with `reasoning != None`
///   - `Image`       → prefer models with `vision != None`
///   - anything else → no capability filter
///
/// Falls back to all models if the filter matches nothing.
/// Among candidates, pick randomly to spread load.
pub fn pick_model_classified<'a>(
    classification: &Classification,
    available_models: &[(&'a str, f64, crate::models::ModelCapabilities)],
) -> Option<&'a str> {
    use crate::models::CapabilityLevel;

    if available_models.is_empty() {
        return None;
    }
    if available_models.len() == 1 {
        return Some(available_models[0].0);
    }

    // Capability filter based on what the request needs
    let filtered: Vec<&(&str, f64, crate::models::ModelCapabilities)> =
        match classification.category {
            _ if classification.needs_tools => available_models
                .iter()
                .filter(|(_, _, caps)| caps.tool_use != CapabilityLevel::None)
                .collect(),
            Category::Reasoning => available_models
                .iter()
                .filter(|(_, _, caps)| caps.reasoning != CapabilityLevel::None)
                .collect(),
            Category::Image => available_models
                .iter()
                .filter(|(_, _, caps)| caps.vision != CapabilityLevel::None)
                .collect(),
            _ => Vec::new(),
        };

    // Fall back to all models if filter matched nothing
    let candidates: Vec<&(&str, f64, crate::models::ModelCapabilities)> = if filtered.is_empty() {
        available_models.iter().collect()
    } else {
        filtered
    };

    // Pick randomly to spread load
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as usize;
    Some(candidates[nanos % candidates.len()].0)
}

/// Legacy wrapper for tests that have category + tools but no complexity.
// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_classify_tool_call() {
        // Content that implies tool use + tools schema = ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Run the tests and check the output"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        assert_eq!(classify(&body).category, Category::ToolCall);
    }

    #[test]
    fn test_classify_code() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a Python function to implement binary search and debug any issues"}
            ]
        });
        assert_eq!(classify(&body).category, Category::Code);
    }

    #[test]
    fn test_classify_reasoning() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Prove that the square root of 2 is irrational. Explain step by step."}
            ]
        });
        assert_eq!(classify(&body).category, Category::Reasoning);
    }

    #[test]
    fn test_classify_creative() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a story about a robot who learns to paint"}
            ]
        });
        assert_eq!(classify(&body).category, Category::Creative);
    }

    #[test]
    fn test_classify_chat_default() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ]
        });
        let cl = classify(&body);
        assert_eq!(cl.category, Category::Chat);
        assert_eq!(cl.complexity, Complexity::Quick); // short simple question
        assert!(!cl.needs_tools);
        assert!(!cl.has_media_inputs);
    }

    #[test]
    fn test_classify_deep_analysis() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Design a system architecture for a distributed database with strong consistency guarantees. Provide a detailed analysis of the trade-offs between CAP theorem constraints and explain how to handle network partitions in depth."}
            ]
        });
        let cl = classify(&body);
        assert_eq!(cl.complexity, Complexity::Deep);
    }

    #[test]
    fn test_classify_code_with_tools() {
        // Code request that happens to have tools — should be Code, not ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Write a Python function to sort a list and debug it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert_eq!(cl.category, Category::Code);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_always_needs_tools() {
        // Tools schema present = agentic session, always needs_tools
        // even if the message content is plain chat
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_with_tool_content() {
        // Tools in schema AND content implies tool use — needs tools
        let body = json!({
            "messages": [{"role": "user", "content": "Read the file and fix the bug"}],
            "tools": [{"type": "function", "function": {"name": "read"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_anthropic_text_blocks_with_tools() {
        // Anthropic-style content blocks should still be parsed as text
        // and trigger needs_tools when tool-intent is present.
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
        assert!(matches!(cl.category, Category::Code | Category::ToolCall));
    }

    #[test]
    fn test_classify_anthropic_tool_use_block_sets_needs_tools() {
        // If an explicit tool_use/tool_result block is present, mark as needs_tools.
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "shell", "input": {"command": "ls"}}
                    ]
                }
            ]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_anthropic_tool_request_sets_needs_tools() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_system_prompt_code() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are a senior developer and coding assistant."},
                {"role": "user", "content": "Help me with this."}
            ]
        });
        assert_eq!(classify(&body).category, Category::Code);
    }

    #[test]
    fn test_media_requirements_detect_audio_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this clip"},
                        {"type": "audio_url", "audio_url": {"url": "mesh://blob/client-1/example"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_audio);
        assert!(!media.needs_vision);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_media_requirements_detect_image_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_vision);
        assert!(!media.needs_audio);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_pick_tools_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let tool_caps = ModelCapabilities {
            tool_use: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            ("reasoning-model", 10.0, no_caps),
            ("tool-model", 10.0, tool_caps),
        ];
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Moderate,
            needs_tools: true,
            has_media_inputs: false,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("tool-model"));
    }

    #[test]
    fn test_pick_reasoning_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let reasoning_caps = ModelCapabilities {
            reasoning: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            ("chat-model", 10.0, no_caps),
            ("reasoning-model", 10.0, reasoning_caps),
        ];
        let cl = Classification {
            category: Category::Reasoning,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("reasoning-model"));
    }

    #[test]
    fn test_pick_vision_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            ("text-model", 10.0, no_caps),
            ("vision-model", 10.0, vision_caps),
        ];
        let cl = Classification {
            category: Category::Image,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: true,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("vision-model"));
    }

    #[test]
    fn test_pick_falls_back_when_no_capability_match() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![("model-a", 10.0, no_caps), ("model-b", 10.0, no_caps)];
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Moderate,
            needs_tools: true,
            has_media_inputs: false,
        };
        // No tool-capable model — falls back to all
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_pick_empty_returns_none() {
        let available: Vec<(&str, f64, crate::models::ModelCapabilities)> = vec![];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), None);
    }

    #[test]
    fn test_pick_single_model() {
        use crate::models::ModelCapabilities;

        let available = vec![("only-model", 10.0, ModelCapabilities::default())];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), Some("only-model"));
    }

    #[test]
    fn test_pick_chat_no_filter() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![("model-a", 10.0, no_caps), ("model-b", 10.0, no_caps)];
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
            has_media_inputs: false,
        };
        // Chat with no special needs — any model is valid
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_strip_split_suffix() {
        assert_eq!(
            strip_split_suffix("MiniMax-M2.5-Q4_K_M-00001-of-00004"),
            "MiniMax-M2.5-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Qwen3-Coder-Next-Q4_K_M-00001-of-00004"),
            "Qwen3-Coder-Next-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Hermes-2-Pro-Mistral-7B-Q4_K_M"),
            "Hermes-2-Pro-Mistral-7B-Q4_K_M"
        );
        assert_eq!(strip_split_suffix(""), "");
    }
}
