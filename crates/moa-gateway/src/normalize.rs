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
    // Strategy 1: try JSON parse
    if let Some(output) = try_json_parse(raw, model, role, elapsed_ms) {
        return output;
    }

    // Strategy 2: try line-based key:value extraction
    if let Some(output) = try_kv_parse(raw, model, role, elapsed_ms) {
        return output;
    }

    // Strategy 3: heuristic classification
    heuristic_classify(raw, model, role, elapsed_ms)
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

    let tool_arguments = obj.get("arguments").cloned().or_else(|| {
        // Sometimes models put args as a string
        obj.get("arguments")
            .and_then(|a| a.as_str())
            .and_then(|s| serde_json::from_str(s).ok())
    });

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
    let kind = kind?;

    let payload = if payload_lines.is_empty() {
        raw.to_string()
    } else {
        payload_lines.join("\n")
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

fn looks_like_tool_proposal(lower: &str, _raw: &str) -> bool {
    (lower.contains("tool_call")
        || lower.contains("function_call")
        || lower.contains("i would call")
        || lower.contains("i propose calling")
        || lower.contains("tool_proposal"))
        && !lower.contains("i would not")
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
    // Look for function_call or tool_call JSON
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
        }
    }
    (None, None)
}

/// Find the first JSON object in text (handles markdown fences, etc.).
fn extract_json_object(text: &str) -> Option<String> {
    // Try the whole thing first
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        if let Ok(_) = serde_json::from_str::<Value>(trimmed) {
            return Some(trimmed.to_string());
        }
    }

    // Look inside markdown code blocks
    for block in text.split("```") {
        let block = block.trim().strip_prefix("json").unwrap_or(block).trim();
        if block.starts_with('{') {
            if let Ok(_) = serde_json::from_str::<Value>(block) {
                return Some(block.to_string());
            }
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

/// Strip `<think>...</think>` tags that reasoning models emit.
fn strip_thinking_tags(text: &str) -> String {
    let mut result = text.to_string();
    // Remove <think>...</think> blocks (greedy)
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</think>".len()..]
            );
        } else {
            // Unclosed think tag — remove from <think> to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
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
    fn json_inside_markdown() {
        let raw = r#"Here is my response:
```json
{"kind": "answer", "confidence": 0.9, "payload": "The sky is blue"}
```"#;
        let out = normalize_worker_output(raw, "test-model", WorkerRole::Fast, 100);
        assert_eq!(out.kind, OutputKind::Answer);
        assert!((out.confidence - 0.9).abs() < 0.01);
    }
}
