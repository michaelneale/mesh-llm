//! Normalize dirty worker outputs into structured envelopes.
//!
//! Workers are asked to produce structured output but models are unreliable.
//! The normalizer tries multiple parse strategies in order:
//! 1. JSON object with kind/confidence/tool/payload fields
//! 2. Line-based key: value extraction
//! 3. Heuristic classification from raw text
//!
//! Anything the model returns is treated as dirty input.

use crate::worker::WorkerRole;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Answer,
    ToolProposal,
    Critique,
    Uncertainty,
}

/// Normalized worker output.
#[derive(Debug, Clone)]
pub struct WorkerOutput {
    pub kind: OutputKind,
    pub confidence: f32,
    pub tool_name: Option<String>,
    pub tool_arguments: Option<Value>,
    pub payload: String,
    pub model: String,
    pub role: WorkerRole,
    pub elapsed_ms: u64,
}

/// Normalize raw worker text into a structured output.
pub fn normalize_worker_output(
    raw: &str,
    model: &str,
    role: WorkerRole,
    elapsed_ms: u64,
) -> WorkerOutput {
    // Pre-clean: strip thinking tags so downstream parsers see clean text.
    let cleaned = strip_thinking_tags(raw);
    let text = if cleaned.is_empty() { raw } else { &cleaned };

    // Strategy 1: try JSON parse
    // Strategy 2: try line-based key:value extraction
    // Strategy 3: heuristic classification
    //
    // Whichever strategy wins, we then run a single sanitize pass over the
    // result so the invariants hold regardless of parse path. The previous
    // shape returned early on Strategy 1 / 2 and only sanitized Strategy 3,
    // which let non-finite confidences (e.g. KV `confidence: NaN` parsed via
    // `parse::<f32>()`) escape into the arbiter where `partial_cmp/total_cmp`
    // can panic on `unwrap`.
    let output = try_json_parse(text, model, role, elapsed_ms)
        .or_else(|| try_kv_parse(text, model, role, elapsed_ms))
        .unwrap_or_else(|| {
            let mut heuristic = heuristic_classify(text, model, role, elapsed_ms);
            // Heuristic path can leak stray KV envelope lines into the payload
            // when the model output contains a partial `kind:/confidence:/`
            // block; structured parsers already isolate the payload field.
            heuristic.payload = strip_kv_envelope(&heuristic.payload);
            heuristic
        });

    sanitize_worker_output(output)
}

/// Enforce the invariants every arbiter / reducer call site assumes.
///
/// * `confidence` is finite (NaN/Inf clamp to 0.5 so comparisons never panic).
/// * `tool_arguments`, when `Some`, is an object — callers serialize it as a
///   JSON object string; `Value::Null` collapses to an empty object, and bare
///   primitives collapse to `{}` rather than producing invalid `arguments`.
fn sanitize_worker_output(mut output: WorkerOutput) -> WorkerOutput {
    if !output.confidence.is_finite() {
        output.confidence = 0.5;
    }
    if let Some(args) = output.tool_arguments.as_ref() {
        if args.is_null() {
            output.tool_arguments = Some(serde_json::json!({}));
        } else if !args.is_object() {
            // Primitive or array payloads cannot be re-serialized as a valid
            // OpenAI tool-call `arguments` object string. Replace with an
            // empty object so downstream callers always see a well-formed
            // structure rather than `"null"` / `"\"foo\""`.
            output.tool_arguments = Some(serde_json::json!({}));
        }
    }
    output
}

/// Remove KV envelope metadata lines from text.  Models sometimes include
/// `kind: answer\nconfidence: 1.0\npayload: ...` as part of their prose when
/// the structured output wasn't parsed.
fn strip_kv_envelope(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut cleaned = Vec::new();
    let mut in_kv_block = false;
    let mut payload_text: Option<String> = None;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("kind:")
            || trimmed.starts_with("confidence:")
            || trimmed.starts_with("tool:")
            || trimmed.starts_with("arguments:")
        {
            in_kv_block = true;
            continue;
        }
        if trimmed.starts_with("payload:") {
            in_kv_block = true;
            let val = trimmed.strip_prefix("payload:").unwrap().trim();
            if !val.is_empty() {
                payload_text = Some(val.to_string());
            }
            continue;
        }
        if in_kv_block {
            // Lines after payload: are part of the payload value
            if let Some(ref mut pt) = payload_text {
                pt.push('\n');
                pt.push_str(line);
            }
            continue;
        }
        cleaned.push(*line);
    }

    // If we found a payload: value in the KV block, prefer that
    if let Some(pt) = payload_text {
        let pt = pt.trim().to_string();
        if !pt.is_empty() {
            return pt;
        }
    }

    // Otherwise use the non-KV lines
    let result = cleaned.join("\n").trim().to_string();
    if result.is_empty() {
        // Everything was KV — return original
        text.trim().to_string()
    } else {
        result
    }
}

/// Try parsing as a JSON envelope.
fn try_json_parse(
    raw: &str,
    model: &str,
    role: WorkerRole,
    elapsed_ms: u64,
) -> Option<WorkerOutput> {
    // Find JSON object in the text (models often wrap in markdown)
    let json_str = extract_json_object(raw)?;
    let obj: Value = serde_json::from_str(&json_str).ok()?;

    // First, recognise the OpenAI tool-call shape that models commonly
    // emit even without our `kind`/`confidence` envelope:
    //
    //   {"function": "read_file", "arguments": {"path": "README.md"}}
    //   {"name": "read_file",     "arguments": {...}}
    //   {"tool": "read_file",     "arguments": {...}}
    //
    // Agent harnesses (Goose, OpenCode) only act on real `tool_calls`
    // — if the worker writes inline tool JSON and we miss it, MoA leaks
    // the JSON back as `content` and the agent does nothing. This is
    // the failure mode PR #566 review called out.
    if obj.get("kind").is_none() {
        let openai_tool_name = obj
            .get("function")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("name").and_then(|v| v.as_str()))
            .or_else(|| obj.get("tool").and_then(|v| v.as_str()));
        if let Some(tname) = openai_tool_name {
            let args = extract_tool_arguments(obj.get("arguments"));
            return Some(WorkerOutput {
                kind: OutputKind::ToolProposal,
                // OpenAI-shape tool calls have no native confidence
                // marker, but a structurally well-formed proposal is a
                // stronger signal than a heuristic catch — score it
                // higher than the heuristic's 0.6 so the arbiter
                // prefers it on tie.
                confidence: 0.75,
                tool_name: Some(tname.to_string()),
                tool_arguments: args,
                payload: raw.to_string(),
                model: model.to_string(),
                role,
                elapsed_ms,
            });
        }
    }

    let kind = match obj.get("kind").and_then(|k| k.as_str()) {
        Some("tool_proposal") => OutputKind::ToolProposal,
        Some("critique") => OutputKind::Critique,
        Some("uncertainty") => OutputKind::Uncertainty,
        Some("answer") => OutputKind::Answer,
        _ => return None,
    };

    let confidence = obj
        .get("confidence")
        .and_then(|c| c.as_f64())
        .unwrap_or(0.5) as f32;

    let tool_name = obj
        .get("tool")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string());

    let tool_arguments = extract_tool_arguments(obj.get("arguments"));

    let payload = obj
        .get("payload")
        .and_then(|p| p.as_str())
        .unwrap_or(raw)
        .to_string();

    Some(WorkerOutput {
        kind,
        confidence,
        tool_name,
        tool_arguments,
        payload,
        model: model.to_string(),
        role,
        elapsed_ms,
    })
}

/// Pull a tool-call `arguments` value out of a parsed JSON envelope.
///
/// Models emit `arguments` in two shapes:
///
/// 1. As a JSON object: `"arguments": {"path": "README.md"}` — use as-is.
/// 2. As a JSON-encoded string: `"arguments": "{\"path\":\"README.md\"}"`
///    — the OpenAI wire shape. We need to `from_str` the inner string so
///    the downstream `Value::Object` invariant holds.
///
/// The original code wrote
///     obj.get("arguments").cloned().or_else(|| obj.get("arguments").and_then(as_str)...)
/// which is logically dead: `.cloned()` on a `Some(Value::String("…"))` is
/// already `Some`, so the `or_else` branch never ran and string-encoded
/// arguments leaked through unparsed. We branch explicitly here so each
/// shape gets the right handling.
///
/// Missing or `Value::Null` arguments return `None` from this helper;
/// downstream `tool_call_response` substitutes `"{}"` when serializing
/// the OpenAI tool-call wire shape, so the literal string `"null"`
/// never reaches the client. The `WorkerOutput::tool_arguments`
/// invariant is therefore "`None` or an object", with `None` meaning
/// "emit empty-object args at wire time".
fn extract_tool_arguments(value: Option<&Value>) -> Option<Value> {
    match value {
        Some(Value::String(s)) => {
            // String form: parse the inner JSON. If it doesn't parse,
            // fall through to `{}` rather than carrying a bare string.
            serde_json::from_str(s)
                .ok()
                .or_else(|| Some(serde_json::json!({})))
        }
        Some(Value::Null) | None => None,
        Some(other) if other.is_object() => Some(other.clone()),
        // Bare arrays / numbers / bools cannot be a valid OpenAI tool-call
        // arguments object — collapse to `{}`. The sanitize pass in
        // `normalize_worker_output` also enforces this defensively.
        Some(_) => Some(serde_json::json!({})),
    }
}

/// Try line-based `key: value` extraction.
fn try_kv_parse(raw: &str, model: &str, role: WorkerRole, elapsed_ms: u64) -> Option<WorkerOutput> {
    let mut kind = None;
    let mut confidence = None;
    let mut tool = None;
    let mut arguments = None;
    let mut payload_lines = Vec::new();
    let mut in_payload = false;

    for line in raw.lines() {
        let trimmed = line.trim();
        if in_payload {
            payload_lines.push(line);
            continue;
        }

        if let Some(v) = trimmed.strip_prefix("kind:") {
            let v = v.trim().trim_matches('"');
            kind = Some(match v {
                "tool_proposal" => OutputKind::ToolProposal,
                "critique" => OutputKind::Critique,
                "uncertainty" => OutputKind::Uncertainty,
                _ => OutputKind::Answer,
            });
        } else if let Some(v) = trimmed.strip_prefix("confidence:") {
            confidence = v.trim().parse::<f32>().ok();
        } else if let Some(v) = trimmed.strip_prefix("tool:") {
            let v = v.trim().trim_matches('"');
            if !v.is_empty() && v != "null" && v != "none" {
                tool = Some(v.to_string());
            }
        } else if let Some(v) = trimmed.strip_prefix("arguments:") {
            let v = v.trim();
            arguments = serde_json::from_str(v).ok();
        } else if trimmed.starts_with("payload:") {
            in_payload = true;
            let v = trimmed.strip_prefix("payload:").unwrap().trim();
            if !v.is_empty() {
                payload_lines.push(v);
            }
        }
    }

    // Need at least kind to count as structured
    let mut kind = kind?;

    // If the model said "kind: answer" but also named a tool, it's actually
    // a tool proposal — models frequently mislabel these.
    if tool.is_some() && kind == OutputKind::Answer {
        kind = OutputKind::ToolProposal;
    }

    // If payload was found, use it.  If not (e.g. truncated by max_tokens),
    // try to extract the last substantive line before the KV block, or fall
    // back to the confidence value as a signal that KV was found but incomplete.
    let payload = if !payload_lines.is_empty() {
        payload_lines.join("\n")
    } else {
        // KV envelope was found but payload: line was missing or truncated.
        // Use lines before the KV block as the payload — they're often the
        // model's natural-language answer before it started formatting.
        let mut pre_kv = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("kind:")
                || trimmed.starts_with("confidence:")
                || trimmed.starts_with("tool:")
                || trimmed.starts_with("arguments:")
            {
                break;
            }
            if !trimmed.is_empty() {
                pre_kv.push(trimmed);
            }
        }
        if pre_kv.is_empty() {
            raw.to_string()
        } else {
            // Take the last meaningful line(s) — skip reasoning preamble
            pre_kv.last().unwrap_or(&raw).to_string()
        }
    };

    Some(WorkerOutput {
        kind,
        confidence: confidence.unwrap_or(0.5),
        tool_name: tool,
        tool_arguments: arguments,
        payload,
        model: model.to_string(),
        role,
        elapsed_ms,
    })
}

/// Heuristic: classify raw text by content patterns.
fn heuristic_classify(raw: &str, model: &str, role: WorkerRole, elapsed_ms: u64) -> WorkerOutput {
    let lower = raw.to_lowercase();

    // Check for tool call patterns
    if looks_like_tool_proposal(&lower, raw) {
        let (name, args) = extract_tool_proposal(raw);
        return WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.6,
            tool_name: name,
            tool_arguments: args,
            payload: raw.to_string(),
            model: model.to_string(),
            role,
            elapsed_ms,
        };
    }

    // Check for critique patterns
    if looks_like_critique(&lower) {
        return WorkerOutput {
            kind: OutputKind::Critique,
            confidence: 0.5,
            tool_name: None,
            tool_arguments: None,
            payload: raw.to_string(),
            model: model.to_string(),
            role,
            elapsed_ms,
        };
    }

    // Check for uncertainty
    if looks_like_uncertainty(&lower) {
        return WorkerOutput {
            kind: OutputKind::Uncertainty,
            confidence: 0.3,
            tool_name: None,
            tool_arguments: None,
            payload: raw.to_string(),
            model: model.to_string(),
            role,
            elapsed_ms,
        };
    }

    // Default: answer
    WorkerOutput {
        kind: OutputKind::Answer,
        confidence: 0.5,
        tool_name: None,
        tool_arguments: None,
        payload: strip_thinking_tags(raw),
        model: model.to_string(),
        role,
        elapsed_ms,
    }
}

/// Known tool names that models might reference in prose.  These are
/// matched against the lowercased text to detect tool proposals that
/// weren't formatted as structured output.
const KNOWN_TOOLS: &[&str] = &[
    "read_file",
    "edit_file",
    "run_command",
    "search_code",
    "web_search",
    "get_weather",
    "create_file",
    "delete_file",
    "list_files",
];

fn looks_like_tool_proposal(lower: &str, _raw: &str) -> bool {
    // Explicit structured markers
    let has_structured = lower.contains("tool_call")
        || lower.contains("function_call")
        || lower.contains("i would call")
        || lower.contains("i propose calling")
        || lower.contains("tool_proposal");

    if has_structured && !lower.contains("i would not") {
        return true;
    }

    // Agentic patterns: model describes using a tool by name
    let mentions_tool = KNOWN_TOOLS.iter().any(|t| lower.contains(t));
    if mentions_tool {
        // Must also have an action verb — not just mentioning the tool in discussion
        let has_action = lower.contains("i'll use")
            || lower.contains("i will use")
            || lower.contains("let me use")
            || lower.contains("i need to use")
            || lower.contains("use the")
            || lower.contains("using the")
            || lower.contains("should use")
            || lower.contains("call the")
            || lower.contains("calling")
            || lower.contains("propose")
            || lower.contains("**tool**")
            || lower.contains("tool:")
            || lower.contains("identify the tool");
        if has_action {
            return true;
        }
    }

    false
}

fn looks_like_critique(lower: &str) -> bool {
    let markers = [
        "however,",
        "but actually",
        "correction:",
        "that's incorrect",
        "this is wrong",
        "i disagree",
        "the correct answer",
        "actually,",
    ];
    markers.iter().filter(|m| lower.contains(**m)).count() >= 2
}

fn looks_like_uncertainty(lower: &str) -> bool {
    let markers = [
        "i'm not sure",
        "i don't know",
        "uncertain",
        "hard to say",
        "it depends",
        "i cannot determine",
        "insufficient information",
    ];
    markers.iter().any(|m| lower.contains(m))
}

/// Try to extract a tool name and arguments from messy text.
fn extract_tool_proposal(raw: &str) -> (Option<String>, Option<Value>) {
    // Strategy 1: Look for structured JSON in the text
    if let Some(json_str) = extract_json_object(raw) {
        if let Ok(obj) = serde_json::from_str::<Value>(&json_str) {
            // {"function": "name", "arguments": {...}}
            if let Some(name) = obj.get("function").and_then(|f| f.as_str()) {
                let args = obj.get("arguments").cloned();
                return (Some(name.to_string()), args);
            }
            // {"name": "...", "arguments": {...}}
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                let args = obj.get("arguments").cloned();
                return (Some(name.to_string()), args);
            }
            // {"tool": "...", "arguments": {...}}
            if let Some(name) = obj.get("tool").and_then(|t| t.as_str()) {
                let args = obj.get("arguments").cloned();
                return (Some(name.to_string()), args);
            }
            // Could be the arguments themselves (e.g. {"path": "src/auth.py"})
            // Look for a tool name in the surrounding text
            let lower = raw.to_lowercase();
            for tool in KNOWN_TOOLS {
                if lower.contains(tool) {
                    return (Some(tool.to_string()), Some(obj));
                }
            }
        }
    }

    // Strategy 2: Find a known tool name in prose and try to extract args
    let lower = raw.to_lowercase();
    for tool in KNOWN_TOOLS {
        if lower.contains(tool) {
            // Try to find JSON arguments nearby
            let args =
                extract_json_object(raw).and_then(|s| serde_json::from_str::<Value>(&s).ok());
            return (Some(tool.to_string()), args);
        }
    }

    (None, None)
}

/// Find the first JSON object in text (handles markdown fences, etc.).
fn extract_json_object(text: &str) -> Option<String> {
    // Try the whole thing first
    let trimmed = text.trim();
    if trimmed.starts_with('{') && serde_json::from_str::<Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    // Look inside markdown code blocks
    for block in text.split("```") {
        let block = block.trim().strip_prefix("json").unwrap_or(block).trim();
        if block.starts_with('{') && serde_json::from_str::<Value>(block).is_ok() {
            return Some(block.to_string());
        }
    }

    // Find first { ... } substring
    if let Some(start) = text.find('{') {
        let mut depth = 0;
        for (i, c) in text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let candidate = &text[start..start + i + 1];
                        if serde_json::from_str::<Value>(candidate).is_ok() {
                            return Some(candidate.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    None
}

/// Clean passthrough content for display: strip think tags, orphan </think>,
/// and any KV envelope lines (kind:/confidence:/payload:) that leaked.
pub fn strip_passthrough_content(text: &str) -> String {
    let stripped = strip_thinking_tags(text);
    strip_kv_envelope(&stripped)
}

/// Strip `<think>...</think>` tags that reasoning models emit.
/// Thin wrapper over the canonical implementation in `worker`.
fn strip_thinking_tags(text: &str) -> String {
    crate::worker::strip_thinking(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_envelope() {
        let raw = r#"{"kind": "tool_proposal", "tool": "read_file", "arguments": {"path": "foo.rs"}, "confidence": 0.85, "payload": "Need to read the file"}"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
        assert!((out.confidence - 0.85).abs() < 0.01);
    }

    #[test]
    fn kv_envelope() {
        let raw = "kind: answer\nconfidence: 0.7\npayload: The answer is 42.";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Specialist, 200);
        assert_eq!(out.kind, OutputKind::Answer);
        assert!((out.confidence - 0.7).abs() < 0.01);
        assert_eq!(out.payload, "The answer is 42.");
    }

    #[test]
    fn heuristic_answer() {
        let raw = "Paris is the capital of France. It has been since the 10th century.";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Strong, 300);
        assert_eq!(out.kind, OutputKind::Answer);
    }

    #[test]
    fn heuristic_uncertainty() {
        let raw = "I'm not sure about this. It could be either way.";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 50);
        assert_eq!(out.kind, OutputKind::Uncertainty);
    }

    #[test]
    fn strip_think_tags() {
        let raw = "<think>Let me think about this...</think>The answer is 42.";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Strong, 100);
        assert_eq!(out.payload, "The answer is 42.");
    }

    #[test]
    fn kv_answer_with_tool_is_proposal() {
        // Models frequently say "kind: answer" but also name a tool — this
        // should be classified as a tool proposal.
        let raw = "kind: answer\nconfidence: 0.9\ntool: read_file\narguments: {\"path\": \"src/auth.py\"}";
        let out = normalize_worker_output(raw, "glm", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
    }

    #[test]
    fn prose_tool_proposal() {
        // Small models often describe tool usage in prose instead of structured output
        let raw = "I'll use the read_file tool to examine the code:\n```json\n{\"path\": \"src/auth.py\"}\n```";
        let out = normalize_worker_output(raw, "small-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
    }

    #[test]
    fn prose_edit_proposal() {
        let raw = "I need to use the edit_file tool to fix this bug. The arguments would be:\n{\"path\": \"src/auth.py\", \"old_text\": \"== password\", \"new_text\": \"== hash(password)\"}";
        let out = normalize_worker_output(raw, "qwen3:4b", WorkerRole::Specialist, 200);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("edit_file"));
        assert!(out.tool_arguments.is_some());
    }

    #[test]
    fn think_tags_then_kv() {
        let raw = "<think>\nThe user is asking a simple question.\nMultiple workers agree the answer is Canberra.\nI should provide a direct answer.\n</think>\nkind: answer\nconfidence: 1.0\npayload: Canberra is the capital of Australia.";
        let out = normalize_worker_output(raw, "glm", WorkerRole::Reducer, 500);
        assert_eq!(out.kind, OutputKind::Answer);
        assert_eq!(out.payload, "Canberra is the capital of Australia.");
        assert!(!out.payload.contains("think"));
        assert!(!out.payload.contains("kind:"));
    }

    #[test]
    fn heuristic_strips_trailing_kv() {
        // Model outputs reasoning text then KV envelope at the end
        let raw = "Let me analyze this.\n\n1. The answer is 4.\n2. This is simple math.\n\nkind: answer\nconfidence: 1.0\npayload: 4";
        let out = normalize_worker_output(raw, "glm", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::Answer);
        // Payload should be "4" (from the payload: line), not the full reasoning
        assert_eq!(out.payload, "4");
    }

    #[test]
    fn heuristic_strips_kv_no_payload() {
        // Model outputs reasoning then truncated KV (no payload line)
        let raw = "The capital of Australia is Canberra.\n\nkind: answer\nconfidence: 1.0";
        let out = normalize_worker_output(raw, "qwen", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::Answer);
        // Should use the pre-KV text
        assert_eq!(out.payload, "The capital of Australia is Canberra.");
    }

    #[test]
    fn json_inside_markdown() {
        let raw = r#"Here is my response:
```json
{"kind": "answer", "confidence": 0.9, "payload": "The sky is blue"}
```"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::Answer);
        assert!((out.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn nan_confidence_clamped() {
        // A model that outputs NaN confidence should not cause panics
        // in arbiter comparisons. The sanitizer clamps to 0.5.
        let raw = r#"{"kind": "answer", "confidence": "NaN", "payload": "hello"}"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert!(out.confidence.is_finite());
    }

    #[test]
    fn kv_path_clamps_nan_confidence() {
        // Regression for PR #566 review: the JSON path was being clamped
        // but the KV path was not, so `confidence: NaN` in the KV envelope
        // returned `f32::NAN` straight to the arbiter where `partial_cmp`
        // panicked. After the refactor, sanitize runs on every parse path.
        let raw = "kind: answer\nconfidence: NaN\npayload: hello";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert!(
            out.confidence.is_finite(),
            "KV NaN must be sanitized; got {}",
            out.confidence
        );
        assert_eq!(out.confidence, 0.5);
    }

    #[test]
    fn kv_path_clamps_inf_confidence() {
        // `parse::<f32>()` happily accepts `inf` / `-inf` too.
        let raw = "kind: answer\nconfidence: inf\npayload: hello";
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert!(out.confidence.is_finite());
        assert_eq!(out.confidence, 0.5);
    }

    #[test]
    fn json_string_encoded_arguments_are_parsed_to_object() {
        // Regression: the original `obj.get("arguments").cloned().or_else(…)`
        // chain had a dead `or_else` branch — string-encoded JSON arguments
        // never made it through the inner `from_str` and leaked as a bare
        // string into the tool-call wire shape. With `extract_tool_arguments`,
        // the string is parsed into a real JSON object.
        let raw = r#"{"function": "read_file", "arguments": "{\"path\": \"README.md\"}"}"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
        let args = out.tool_arguments.expect("args parsed");
        assert!(args.is_object(), "args should be object, got {args}");
        assert_eq!(args["path"], "README.md");
    }

    #[test]
    fn null_tool_arguments_become_none() {
        // `"arguments": null` previously parsed as `Some(Value::Null)`,
        // which then serialized as the literal string `"null"` in the
        // OpenAI tool-call wire shape. Now it becomes `None`, and the
        // response builder substitutes `"{}"`.
        let raw = r#"{"function": "list", "arguments": null}"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert!(out.tool_arguments.is_none());
    }

    #[test]
    fn primitive_tool_arguments_collapse_to_empty_object() {
        // Defensive: a model that emits `"arguments": 42` should not
        // produce a wire-invalid tool call.
        let raw = r#"{"function": "list", "arguments": 42}"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        let args = out.tool_arguments.expect("sanitize produced an object");
        assert!(args.is_object());
        assert_eq!(args.as_object().unwrap().len(), 0);
    }
}
