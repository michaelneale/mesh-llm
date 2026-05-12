//! Canonical session state.
//!
//! The gateway owns the transcript.  It tracks what messages have been seen,
//! what tool calls were emitted, what tool results came back, and how to
//! summarize prior context for workers that can't see the full history.

use serde_json::Value;

/// What kind of turn this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnType {
    /// First message or new topic.
    Fresh,
    /// Continuation of ongoing conversation.
    Continuation,
    /// Agent client is returning a tool result.
    ToolResult,
}

/// A tool call the gateway emitted and is tracking.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub call_id: String,
    pub function_name: String,
    pub arguments: Value,
    pub result: Option<String>,
}

/// Canonical session state across turns.
pub struct Session {
    /// Full message history as received/emitted.
    messages: Vec<Value>,
    /// Tool schemas from the client (updated each turn).
    tools: Option<Value>,
    /// Tool calls the gateway has emitted, keyed by call_id.
    pending_tools: Vec<PendingToolCall>,
    /// Turn counter.
    turns: usize,
    /// Whether the last thing we emitted was a tool_call.
    last_was_tool_call: bool,
}

impl Session {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            pending_tools: Vec::new(),
            turns: 0,
            last_was_tool_call: false,
        }
    }

    /// Ingest incoming messages from the client.
    ///
    /// The client sends the full conversation each time (OpenAI convention).
    /// We replace our history with theirs — they're authoritative — but we
    /// track deltas for tool result detection.
    pub fn ingest(&mut self, messages: &[Value], tools: &Option<Value>) {
        self.tools = tools.clone();

        // Detect if new messages include tool results
        let new_count = messages.len();
        let old_count = self.messages.len();

        // Replace with client's view (they own the outer loop)
        self.messages = messages.to_vec();
        self.turns += 1;

        // Check for tool results in the new messages
        if new_count > old_count {
            for msg in &messages[old_count..] {
                if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
                    let call_id = msg
                        .get("tool_call_id")
                        .and_then(|id| id.as_str())
                        .unwrap_or("");
                    let content = msg
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Match to pending tool call
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
            }
        }
    }

    /// Classify what kind of turn this is.
    pub fn classify_turn(&self) -> TurnType {
        // If the last message is a tool result, this is a tool-result turn
        if let Some(last) = self.messages.last() {
            if last.get("role").and_then(|r| r.as_str()) == Some("tool") {
                return TurnType::ToolResult;
            }
        }

        // Also check if we just got tool results back after our tool_call
        if self.last_was_tool_call && self.has_unprocessed_tool_results() {
            return TurnType::ToolResult;
        }

        if self.turns <= 1 {
            TurnType::Fresh
        } else {
            TurnType::Continuation
        }
    }

    /// Record an assistant response we're about to emit.
    pub fn record_assistant_response(&mut self, response: &Value) {
        // Check if this response has tool_calls
        let has_tool_calls = response
            .pointer("/choices/0/message/tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

        self.last_was_tool_call = has_tool_calls;

        if has_tool_calls {
            if let Some(tool_calls) = response
                .pointer("/choices/0/message/tool_calls")
                .and_then(|tc| tc.as_array())
            {
                for tc in tool_calls {
                    let call_id = tc
                        .get("id")
                        .and_then(|id| id.as_str())
                        .unwrap_or("")
                        .to_string();
                    let function_name = tc
                        .pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = tc
                        .pointer("/function/arguments")
                        .cloned()
                        .unwrap_or(Value::Null);

                    self.pending_tools.push(PendingToolCall {
                        call_id,
                        function_name,
                        arguments,
                        result: None,
                    });
                }
            }
        }

        // Add the assistant message to our canonical history
        if let Some(msg) = response.pointer("/choices/0/message") {
            self.messages.push(msg.clone());
        }
    }

    // ── Accessors for context packing ─────────────────────────────

    /// Full message history.
    pub fn messages(&self) -> &[Value] {
        &self.messages
    }

    /// Tool schemas (if any).
    pub fn tools(&self) -> Option<&Value> {
        self.tools.as_ref()
    }

    /// Current turn count.
    pub fn turn_count(&self) -> usize {
        self.turns
    }

    /// The last user message text.
    pub fn last_user_text(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            .and_then(|m| extract_text_content(m))
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
                            format!("{}...", &first_line[..77])
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

    fn has_unprocessed_tool_results(&self) -> bool {
        self.pending_tools.iter().any(|p| p.result.is_some())
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
        // First turn: user message
        s.ingest(
            &[json!({"role": "user", "content": "what's the weather?"})],
            &Some(json!([{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}])),
        );
        // Gateway emitted a tool call
        s.record_assistant_response(&json!({
            "choices": [{"message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"}}]
            }}]
        }));
        // Client sends back the tool result
        s.ingest(
            &[
                json!({"role": "user", "content": "what's the weather?"}),
                json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"}}]}),
                json!({"role": "tool", "tool_call_id": "call_123", "content": "22°C, sunny"}),
            ],
            &Some(json!([{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}])),
        );
        assert_eq!(s.classify_turn(), TurnType::ToolResult);
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
}
