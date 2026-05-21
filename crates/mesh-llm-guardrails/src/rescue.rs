//! Tool-call rescue parser.
//!
//! When the model emits a tool call as free text (instead of through
//! the structured `tool_calls` channel), try to parse it back out.
//! Ported from forge `src/forge/prompts/templates.py::rescue_tool_call`
//! (v0.6.0).
//!
//! Strategies, tried in order:
//!
//! 1. **JSON** — `{"tool": "...", "args": {...}}` or
//!    `{"name": "...", "arguments": {...}}` (OpenAI / Granite-4 style,
//!    optionally wrapped in `<tool_call>…</tool_call>` or code fences).
//! 2. **Rehearsal** — `tool_name[ARGS]{...}` (reasoning-model thinking
//!    tokens).
//! 3. **Qwen3-Coder XML** — `<function=name><parameter=key>val</parameter></function>`.
//!
//! Think tags (`<think>…</think>`, `[THINK]…[/THINK]`) are stripped
//! first so the rescue parser can see calls emitted after thinking.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::types::ToolCall;

static THINK_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    // `(?s)` = DOTALL so `.` matches newlines (matches Python `re.DOTALL`).
    Regex::new(r"(?s)\[THINK\].*?\[/THINK\]|<think>.*?</think>").unwrap()
});

static CODE_FENCE_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```(?:json)?\s*\n?").unwrap());
static CODE_FENCE_CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"```").unwrap());

// Rehearsal syntax: tool_name[ARGS]{...}
static REHEARSAL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)(\w+)\[ARGS\](\{.*\})").unwrap());

// Qwen Coder XML.
static QWEN_FUNCTION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<function=([^>\s]+)>(.*?)</function>").unwrap());
// Match the *open* tag only — `<parameter=name>`. The body is
// extracted manually below because the Rust `regex` crate doesn't
// support lookahead, and we need the body to terminate at the next
// `<parameter=` / `</function>` boundary (not just `</parameter>`).
static QWEN_PARAMETER_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<parameter=([^>\s]+)>").unwrap());

/// Top-level entry point. Returns an empty vec if nothing parseable
/// was found — the caller falls through to the normal retry nudge.
pub fn rescue_tool_call(text: &str, available_tools: &[String]) -> Vec<ToolCall> {
    let cleaned = THINK_TAG_RE.replace_all(text, "");
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return Vec::new();
    }

    // Strategy 1: JSON extraction (handles code fences, embedded JSON).
    let found = extract_json_tool_calls(cleaned, available_tools);
    if !found.is_empty() {
        return found;
    }

    // Strategy 2: rehearsal syntax — tool_name[ARGS]{...}.
    let mut found = Vec::new();
    for caps in REHEARSAL_RE.captures_iter(cleaned) {
        let tool_name = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let args_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        if !available_tools.iter().any(|t| t == tool_name) {
            continue;
        }
        if let Ok(args) = serde_json::from_str::<Value>(args_str) {
            if args.is_object() {
                found.push(ToolCall::new(tool_name, args));
            }
        }
    }
    if !found.is_empty() {
        return found;
    }

    // Strategy 3: Qwen Coder XML.
    parse_qwen_xml(cleaned, available_tools)
}

/// JSON-extraction strategy. Strips code fences, then scans for
/// balanced `{...}` blocks and tries to parse each one. Accepts both
/// forge style (`tool`/`args`) and OpenAI / Granite-4 style
/// (`name`/`arguments`). The optional `<tool_call>…</tool_call>`
/// wrapper used by Granite is handled implicitly because we scan for
/// JSON objects within the surrounding text.
pub(crate) fn extract_json_tool_calls(text: &str, available_tools: &[String]) -> Vec<ToolCall> {
    let cleaned = CODE_FENCE_OPEN_RE.replace_all(text, "");
    let cleaned = CODE_FENCE_CLOSE_RE.replace_all(&cleaned, "");
    let bytes = cleaned.as_bytes();

    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        // Scan for the matching `}` accounting for depth — operating
        // on bytes is safe because `{` / `}` are ASCII and any UTF-8
        // continuation bytes have the high bit set.
        let mut depth = 0i32;
        let mut end = None;
        for (j, b) in bytes.iter().enumerate().skip(i) {
            match *b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(j) => {
                let candidate = &cleaned[i..=j];
                if let Some(call) = try_parse_tool_call(candidate, available_tools) {
                    found.push(call);
                }
                i = j + 1;
            }
            None => break,
        }
    }
    found
}

fn try_parse_tool_call(json_str: &str, available_tools: &[String]) -> Option<ToolCall> {
    let data: Value = serde_json::from_str(json_str).ok()?;
    let obj = data.as_object()?;

    // Forge style: `tool`, `args`
    // OpenAI / Granite-4 style: `name`, `arguments`
    let tool_name = obj
        .get("tool")
        .and_then(Value::as_str)
        .or_else(|| obj.get("name").and_then(Value::as_str))?;

    if !available_tools.iter().any(|t| t == tool_name) {
        return None;
    }

    let args = obj
        .get("args")
        .cloned()
        .or_else(|| {
            obj.get("arguments").map(|v| {
                // Some models (Granite, OpenAI native) emit args as a
                // *stringified* JSON blob. Try to parse it; if that
                // fails, keep the original value.
                if let Some(s) = v.as_str() {
                    serde_json::from_str(s).unwrap_or_else(|_| v.clone())
                } else {
                    v.clone()
                }
            })
        })
        .unwrap_or_else(|| Value::Object(Map::new()));

    Some(ToolCall::new(tool_name, args))
}

/// Qwen3-Coder XML parser. Whitespace behavior matches Qwen's
/// reference parser: one leading and one trailing newline are
/// stripped from each parameter value. Values stay as strings; the
/// downstream tool is expected to coerce.
fn parse_qwen_xml(text: &str, available_tools: &[String]) -> Vec<ToolCall> {
    let mut found = Vec::new();
    for fn_caps in QWEN_FUNCTION_RE.captures_iter(text) {
        let tool_name = fn_caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !available_tools.iter().any(|t| t == tool_name) {
            continue;
        }
        let body = fn_caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let args = parse_qwen_parameters(body);
        found.push(ToolCall::new(tool_name, Value::Object(args)));
    }
    found
}

/// Walk `<parameter=name>...</parameter>` blocks manually so the body
/// can terminate at the next `<parameter=` / `</function>` boundary
/// (Rust `regex` has no lookahead, so we can't express that in a
/// single capture group like forge does).
fn parse_qwen_parameters(body: &str) -> Map<String, Value> {
    let mut args = Map::new();

    // Collect (open_start, open_end, name) for each parameter tag.
    let opens: Vec<(usize, usize, &str)> = QWEN_PARAMETER_OPEN_RE
        .captures_iter(body)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            let name = caps.get(1)?.as_str().trim();
            Some((m.start(), m.end(), name))
        })
        .collect();

    for (i, (_, value_start, name)) in opens.iter().enumerate() {
        // Body of this parameter runs from value_start up to whichever
        // comes first: the next `<parameter=`, an explicit
        // `</parameter>`, an explicit `</function>`, or end-of-body.
        let next_open = opens.get(i + 1).map(|n| n.0);
        let close_param = body[*value_start..]
            .find("</parameter>")
            .map(|p| value_start + p);
        let close_fn = body[*value_start..]
            .find("</function>")
            .map(|p| value_start + p);

        let mut end = body.len();
        for candidate in [next_open, close_param, close_fn].into_iter().flatten() {
            if candidate < end {
                end = candidate;
            }
        }

        let mut value = &body[*value_start..end];
        if let Some(stripped) = value.strip_prefix('\n') {
            value = stripped;
        }
        if let Some(stripped) = value.strip_suffix('\n') {
            value = stripped;
        }
        args.insert(name.to_string(), Value::String(value.to_string()));
    }

    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<String> {
        ["read_file", "write_file", "run_tests", "search", "respond"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    #[test]
    fn rescues_plain_json() {
        let text = r#"{"tool": "read_file", "args": {"path": "src/lib.rs"}}"#;
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[0].args, json!({"path": "src/lib.rs"}));
    }

    #[test]
    fn rescues_json_in_code_fence() {
        let text = "```json\n{\"tool\": \"read_file\", \"args\": {\"path\": \"x\"}}\n```";
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
    }

    #[test]
    fn rescues_openai_style_with_stringified_arguments() {
        // OpenAI native FC sometimes leaks the function-call form as text
        // with `arguments` as a JSON-encoded string.
        let text = r#"{"name": "run_tests", "arguments": "{\"path\": \"tests/\"}"}"#;
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "run_tests");
        assert_eq!(calls[0].args, json!({"path": "tests/"}));
    }

    #[test]
    fn rescues_granite_tool_call_wrapper() {
        // Granite-4 emits OpenAI-style inside <tool_call>…</tool_call>.
        // The JSON scanner picks the inner object regardless of wrapper.
        let text = r#"<tool_call>{"name": "search", "arguments": {"q": "rust"}}</tool_call>"#;
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "search");
        assert_eq!(calls[0].args, json!({"q": "rust"}));
    }

    #[test]
    fn rescues_rehearsal_syntax() {
        let text = r#"I'll do read_file[ARGS]{"path": "src/lib.rs"} now."#;
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
    }

    #[test]
    fn rescues_qwen_xml() {
        let text = "<function=read_file>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n</function>";
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[0].args, json!({"path": "src/lib.rs"}));
    }

    #[test]
    fn strips_think_tags_before_rescue() {
        let text =
            "<think>Hmm, I should read the file.</think>{\"tool\": \"read_file\", \"args\": {\"path\": \"x\"}}";
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn unknown_tool_in_json_is_rejected() {
        let text = r#"{"tool": "delete_universe", "args": {}}"#;
        let calls = rescue_tool_call(text, &tools());
        assert!(calls.is_empty());
    }

    #[test]
    fn pure_prose_returns_empty() {
        let text = "Sure! Let me help you with that.";
        let calls = rescue_tool_call(text, &tools());
        assert!(calls.is_empty());
    }

    #[test]
    fn rescues_embedded_json_in_prose() {
        let text =
            "Sure, here you go: {\"tool\": \"read_file\", \"args\": {\"path\": \"x\"}}. Done!";
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn qwen_xml_with_multiple_parameters() {
        let text = "<function=write_file>\n<parameter=path>\nfoo.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>";
        let calls = rescue_tool_call(text, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].args,
            json!({"path": "foo.txt", "content": "hello"})
        );
    }
}
