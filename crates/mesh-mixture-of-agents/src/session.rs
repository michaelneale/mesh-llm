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

/// A resolved fact or decision from a prior turn.
#[derive(Debug, Clone)]
pub struct AcceptedFact {
    /// Which turn produced this fact.
    pub turn: usize,
    /// Short description of what was established.
    pub fact: String,
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
    /// Progressive summary — grows slowly, captures accepted facts and
    /// decisions from prior turns so workers don't need raw history.
    accepted_facts: Vec<AcceptedFact>,
    /// Compact summary of the conversation so far (deterministic, not model-generated).
    running_summary: String,
}

impl Session {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            pending_tools: Vec::new(),
            turns: 0,
            last_was_tool_call: false,
            accepted_facts: Vec::new(),
            running_summary: String::new(),
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

    /// Record what the gateway decided this turn — used to build the
    /// running summary for future turns' workers.
    pub fn record_turn_outcome(&mut self, outcome: &str) {
        if outcome.is_empty() {
            return;
        }
        // Truncate individual facts — generous enough to capture a useful
        // summary of the turn but not the full response.
        let truncated = if outcome.len() > 500 {
            format!("{}...", &outcome[..497])
        } else {
            outcome.to_string()
        };
        self.accepted_facts.push(AcceptedFact {
            turn: self.turns,
            fact: truncated,
        });
        self.rebuild_summary();
    }

    /// Deterministic summary rebuild.  No model calls.
    fn rebuild_summary(&mut self) {
        // Keep the last N facts — older ones are compressed into a single line.
        // With ~500 chars per fact + tool history, 15 facts gives a summary
        // budget of roughly 2000 tokens — enough for real agent sessions.
        const MAX_RECENT_FACTS: usize = 15;
        let total = self.accepted_facts.len();
        let mut parts = Vec::new();

        if total > MAX_RECENT_FACTS {
            let old_count = total - MAX_RECENT_FACTS;
            parts.push(format!("[{old_count} earlier facts omitted]"));
        }

        let start = total.saturating_sub(MAX_RECENT_FACTS);
        for fact in &self.accepted_facts[start..] {
            parts.push(format!("Turn {}: {}", fact.turn, fact.fact));
        }

        // Also include tool call history compactly
        let tool_history: Vec<String> = self
            .pending_tools
            .iter()
            .map(|p| {
                if let Some(ref result) = p.result {
                    let short_result = if result.len() > 300 {
                        format!("{}...", &result[..297])
                    } else {
                        result.clone()
                    };
                    format!("{}() → {}", p.function_name, short_result)
                } else {
                    format!("{}() → pending", p.function_name)
                }
            })
            .collect();
        if !tool_history.is_empty() {
            parts.push(format!("Tools used: {}", tool_history.join("; ")));
        }

        self.running_summary = parts.join("\n");
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

    /// Running summary of prior turns — compact, deterministic, no model calls.
    /// Use this instead of raw history for workers that can't see everything.
    pub fn running_summary(&self) -> &str {
        &self.running_summary
    }

    /// Accepted facts from prior turns.
    pub fn accepted_facts(&self) -> &[AcceptedFact] {
        &self.accepted_facts
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
