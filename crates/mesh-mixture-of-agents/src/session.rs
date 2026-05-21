//! Canonical session state.
//!
//! The gateway owns the transcript for one request. It tracks what messages
//! were received and what tool calls/results are present so workers and the
//! reducer can be packed with the right slice of context.
//!
//! ## Lifetime: request-scoped, not conversation-scoped
//!
//! `handle_turn` constructs a fresh `Session` per inbound request and ingests
//! the caller's `messages` array. The caller (Goose, OpenCode, an SDK) owns
//! the multi-turn conversation; we trust that array as the authoritative
//! history. There is intentionally no cross-request state.
//!
//! Earlier iterations carried `turns`, `accepted_facts`, and a deterministic
//! `running_summary` on the assumption that the gateway would persist a
//! `Session` across requests — but the gateway never invokes
//! `record_assistant_response` / `record_turn_outcome` in production, so
//! "Continuation" turns never fired and the summary was never built.
//! PR #566 review (Copilot) called this out as silently-dead code. We have
//! removed it. Continuation context comes from the caller's `messages`.

use serde_json::Value;

/// What kind of turn this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnType {
    /// First message of this request, or a normal user follow-up. The
    /// gateway fans out to workers and arbitrates.
    Fresh,
    /// Agent client is returning a tool result. The gateway skips fan-out
    /// and goes straight to the reducer with the tool output in context.
    ToolResult,
}

/// A tool call observed in the caller's message history. Carried so the
/// reducer-side context packer can pair tool calls with their results.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub call_id: String,
    pub function_name: String,
    pub arguments: Value,
    pub result: Option<String>,
}

/// Canonical session state for one request.
pub struct Session {
    /// Full message history as received from the caller.
    messages: Vec<Value>,
    /// Tool schemas from the caller.
    tools: Option<Value>,
    /// Tool calls observed in the caller's history, paired with their
    /// results when present.
    pending_tools: Vec<PendingToolCall>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            pending_tools: Vec::new(),
        }
    }

    /// Ingest incoming messages from the caller.
    ///
    /// The caller sends the full conversation each time (OpenAI convention),
    /// so we replace our history with theirs — they're authoritative. We
    /// scan the new history for `role: "assistant"` messages carrying
    /// `tool_calls` and `role: "tool"` messages carrying `tool_call_id`,
    /// pairing them into `pending_tools` so the reducer-side context packer
    /// can surface tool outputs.
    pub fn ingest(&mut self, messages: &[Value], tools: &Option<Value>) {
        self.tools = tools.clone();
        self.messages = messages.to_vec();

        // Rebuild `pending_tools` from the canonical history. Previously
        // this was deltas-since-last-ingest, which only worked when the
        // gateway persisted Sessions across requests. With per-request
        // sessions, we scan the full caller-provided history every time.
        self.pending_tools.clear();
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "assistant" => {
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            // Skip malformed tool_calls. Both `id` and
                            // `function.name` are required by the OpenAI
                            // wire shape; defaulting them to empty
                            // strings would let two malformed calls
                            // share `call_id == ""` and later cross-pair
                            // with a `role: "tool"` message whose
                            // `tool_call_id` is also missing. Drop the
                            // entry with a warning so the rest of the
                            // history still ingests cleanly.
                            let Some(call_id) = tc
                                .get("id")
                                .and_then(|id| id.as_str())
                                .filter(|s| !s.is_empty())
                            else {
                                tracing::warn!(
                                    "moa session: ignoring tool_call with missing/empty `id`"
                                );
                                continue;
                            };
                            let Some(function_name) = tc
                                .pointer("/function/name")
                                .and_then(|n| n.as_str())
                                .filter(|s| !s.is_empty())
                            else {
                                tracing::warn!(
                                    "moa session: ignoring tool_call `{call_id}` with missing/empty `function.name`"
                                );
                                continue;
                            };
                            let arguments = tc
                                .pointer("/function/arguments")
                                .cloned()
                                .unwrap_or(Value::Null);
                            self.pending_tools.push(PendingToolCall {
                                call_id: call_id.to_string(),
                                function_name: function_name.to_string(),
                                arguments,
                                result: None,
                            });
                        }
                    }
                }
                "tool" => {
                    // Skip tool results without a `tool_call_id` rather
                    // than letting them match an empty-id placeholder.
                    let Some(call_id) = msg
                        .get("tool_call_id")
                        .and_then(|id| id.as_str())
                        .filter(|s| !s.is_empty())
                    else {
                        tracing::warn!(
                            "moa session: ignoring tool result with missing/empty `tool_call_id`"
                        );
                        continue;
                    };
                    let content = msg
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    if let Some(pending) =
                        self.pending_tools.iter_mut().find(|p| p.call_id == call_id)
                    {
                        pending.result = Some(content);
                        tracing::info!(
                            "moa session: tool result received for {}({})",
                            pending.function_name,
                            call_id,
                        );
                    }
                }
                _ => {}
            }
        }
    }

    /// Classify what kind of turn this is.
    ///
    /// Anything that ends with an unsynthesised tool result is a
    /// `ToolResult` turn: the gateway must skip fan-out and go
    /// straight to the reducer, so the workers don't re-broadcast
    /// the same tool call whose result we already have in context.
    ///
    /// We scan from the end of the conversation backwards. The first
    /// non-user role we hit decides:
    ///
    ///   * `role: "tool"` first — OpenAI canonical: classify as
    ///     `ToolResult`.
    ///   * `role: "assistant"` first — the assistant has already
    ///     spoken since the last tool result. Hand the next turn
    ///     to fan-out normally.
    ///   * `role: "user"` first — keep scanning past it. A user
    ///     nudge after an unsynthesised tool result is still a
    ///     tool-result turn; the model needs to consume the tool
    ///     output and answer the nudge in one synthesis pass. A
    ///     user message that predates any tool result reaches the
    ///     start of the history and we fall through to `Fresh`.
    pub fn classify_turn(&self) -> TurnType {
        for msg in self.messages.iter().rev() {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "tool" => return TurnType::ToolResult,
                "assistant" => break,
                _ => continue,
            }
        }
        TurnType::Fresh
    }

    // ── Accessors for context packing ─────────────────────────────

    /// Full message history.
    pub fn messages(&self) -> &[Value] {
        &self.messages
    }

    /// Clone all messages for passthrough to workers that need the full
    /// original conversation (e.g. when tools are present).
    pub fn all_messages(&self) -> Vec<Value> {
        self.messages.clone()
    }

    /// Tool schemas (if any).
    pub fn tools(&self) -> Option<&Value> {
        self.tools.as_ref()
    }

    /// The last user message text.
    pub fn last_user_text(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            .and_then(extract_text_content)
            .unwrap_or_default()
    }

    /// System prompt (first system message).
    pub fn system_prompt(&self) -> Option<String> {
        self.messages
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            .and_then(|m| m.get("content").and_then(|c| c.as_str()))
            .map(|s| s.to_string())
    }

    /// Tool names only (for compact worker context).
    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|t| {
                        t.pointer("/function/name")
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compact tool descriptions: name + first line of description.
    pub fn tool_summaries(&self) -> Vec<String> {
        self.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|t| {
                        let name = t.pointer("/function/name")?.as_str()?;
                        let desc = t
                            .pointer("/function/description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        let first_line = desc.lines().next().unwrap_or(desc);
                        let truncated = if first_line.len() > 80 {
                            format!("{}...", crate::worker::truncate_chars(first_line, 77))
                        } else {
                            first_line.to_string()
                        };
                        Some(format!("{name}: {truncated}"))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Recent context: last N messages (for specialist workers).
    pub fn recent_messages(&self, n: usize) -> Vec<Value> {
        let start = self.messages.len().saturating_sub(n);
        self.messages[start..].to_vec()
    }

    /// Pending tool calls and their results.
    pub fn pending_tool_calls(&self) -> &[PendingToolCall] {
        &self.pending_tools
    }

    /// Most recent tool results (for reducer context).
    pub fn recent_tool_results(&self) -> Vec<(String, String)> {
        self.pending_tools
            .iter()
            .filter_map(|p| {
                p.result
                    .as_ref()
                    .map(|r| (p.function_name.clone(), r.clone()))
            })
            .collect()
    }
}

/// Extract text content from a message (handles both string and multipart).
fn extract_text_content(msg: &Value) -> Option<String> {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return Some(s.to_string());
    }
    if let Some(parts) = msg.get("content").and_then(|c| c.as_array()) {
        let texts: Vec<&str> = parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect();
        if !texts.is_empty() {
            return Some(texts.join("\n"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fresh_turn() {
        let mut s = Session::new();
        s.ingest(&[json!({"role": "user", "content": "hello"})], &None);
        assert_eq!(s.classify_turn(), TurnType::Fresh);
        assert_eq!(s.last_user_text(), "hello");
    }

    #[test]
    fn tool_result_turn() {
        let mut s = Session::new();
        // Per-request session: the caller sends the full history each
        // request. A history ending in a `role: "tool"` message must
        // classify as a ToolResult turn so the gateway routes to the
        // reducer with the tool output in context.
        s.ingest(
            &[
                json!({"role": "user", "content": "what's the weather?"}),
                json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"}}]}),
                json!({"role": "tool", "tool_call_id": "call_123", "content": "22°C, sunny"}),
            ],
            &Some(json!([{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}])),
        );
        assert_eq!(s.classify_turn(), TurnType::ToolResult);
        // pending_tools is rebuilt from the caller's history, so a fresh
        // session can still pair the call with its result.
        let pairs = s.recent_tool_results();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "get_weather");
        assert_eq!(pairs[0].1, "22°C, sunny");
    }

    #[test]
    fn tool_result_context_includes_content() {
        // Regression: a fresh session receiving a tool-result turn must
        // still surface the tool output in packed context, even though
        // pending_tools is empty (session is stateless per request).
        use crate::context;
        let mut s = Session::new();
        s.ingest(
            &[
                json!({"role": "user", "content": "read the file"}),
                json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]}),
                json!({"role": "tool", "tool_call_id": "call_abc", "content": "# Hello World"}),
            ],
            &Some(json!([{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {}}}])),
        );
        assert_eq!(s.classify_turn(), TurnType::ToolResult);

        let (messages, _tools) = context::pack_for_tool_result_turn(&s, true);
        // The packed messages must contain the actual tool output content
        let serialized = serde_json::to_string(&messages).unwrap();
        assert!(
            serialized.contains("# Hello World"),
            "tool result content must be in packed context, got: {serialized}"
        );
        // Must also contain the tool_call for context
        assert!(
            serialized.contains("read_file"),
            "tool call function name must be in packed context, got: {serialized}"
        );
    }

    #[test]
    fn tool_names_extraction() {
        let mut s = Session::new();
        s.ingest(
            &[json!({"role": "user", "content": "hi"})],
            &Some(json!([
                {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk"}},
                {"type": "function", "function": {"name": "web_search", "description": "Search the web"}},
            ])),
        );
        assert_eq!(s.tool_names(), vec!["read_file", "web_search"]);
    }

    #[test]
    fn malformed_tool_calls_are_dropped_not_collapsed_to_empty_id() {
        // Regression for PR #612 review (Copilot): missing/empty `id` or
        // `function.name` used to default to "" and still push a
        // PendingToolCall. Two such malformed entries would then share
        // `call_id == ""` and any `role: "tool"` with a missing
        // `tool_call_id` would match the first one. We now skip
        // malformed tool_calls outright.
        let mut s = Session::new();
        s.ingest(
            &[
                json!({"role": "user", "content": "do two things"}),
                json!({
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "good", "arguments": "{}"}},
                        // Missing id — must be dropped.
                        {"type": "function", "function": {"name": "no_id", "arguments": "{}"}},
                        // Missing function.name — must be dropped.
                        {"id": "call_c", "type": "function", "function": {"arguments": "{}"}},
                        // Empty id — must be dropped.
                        {"id": "", "type": "function", "function": {"name": "empty_id", "arguments": "{}"}},
                    ],
                }),
                // A malformed tool result (no `tool_call_id`) must not
                // attach to any pending call.
                json!({"role": "tool", "content": "orphaned"}),
            ],
            &None,
        );
        // Only the one well-formed call survives.
        let pending = s.pending_tool_calls();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_a");
        assert_eq!(pending[0].function_name, "good");
        // The orphaned tool result did not attach — result still None.
        assert!(pending[0].result.is_none());
    }
}
