//! Pin the contract: a tool-call-shaped worker reply must produce a
//! real `tool_calls` field in the response, not get returned as
//! free-form `content`.
//!
//! Background — PR #566 review feedback (Apr 2026):
//!
//! > In the read-tool probe, the model wrote text that looked like a
//! > tool call instead of actually invoking the read tool.
//!
//! Agent harnesses (Goose, OpenCode, pi) only act on `tool_calls`.
//! If a worker writes "I'll use read_file to inspect README.md" and
//! that text leaks out as `choices[0].message.content` instead of
//! `choices[0].message.tool_calls[*]`, the harness sees a text reply
//! and takes no action. This is the core blocker for agent loops.
//!
//! This test drives `handle_turn` with mock workers that all return
//! the same "I'll use read_file" prose, and asserts that the
//! response carries a structured tool call. The contract:
//!
//!   * `choices[0].message.tool_calls` is a non-empty array, OR
//!   * `choices[0].finish_reason == "tool_calls"`, OR
//!   * the response is an error / reducer-escalation (i.e. MoA
//!     refused to return prose when the request had `tools` and
//!     workers proposed using one).
//!
//! What is NOT acceptable is a response with `content: "I'll use
//! read_file..."` and no `tool_calls` \u2014 the agent harness would
//! silently do nothing.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

/// Backend that returns a fixed text on every call.
struct FixedTextBackend {
    text: String,
}

impl FixedTextBackend {
    fn new(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self { text: text.into() })
    }
}

#[async_trait]
impl moa::ModelBackend for FixedTextBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        // Modest delay so the runtime doesn't optimize the whole turn into
        // a single sync poll.
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(json!({
            "choices": [{"message": {"content": self.text}}],
        }))
    }
}

fn config_with_three_workers_returning(text: &str) -> moa::GatewayConfig {
    let a = FixedTextBackend::new(text);
    let b = FixedTextBackend::new(text);
    let c = FixedTextBackend::new(text);

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![a, b, c];
    let models = vec![
        moa::ModelEntry {
            name: "worker-a-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "worker-b-13b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "worker-c-32b".into(),
            backend_index: 2,
        },
    ];

    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
    }
}

fn user_request_with_read_file_tool(content: &str) -> Value {
    json!({
        "model": "mesh",
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                }
            }
        }],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 128,
    })
}

/// Helper: does the response body have a real, non-empty `tool_calls`
/// array?
fn has_tool_calls(body: &Value) -> bool {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

/// Helper: does the response signal an explicit failure (the all-workers-
/// fail or reducer-failed path)?
fn is_explicit_error(body: &Value) -> bool {
    if body.get("error").is_some() {
        return true;
    }
    body.pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        .map(|s| s == "error" || s == "moa_failed")
        .unwrap_or(false)
}

/// Helper: did the response just smuggle the worker prose back to the
/// agent as `content`? That's the failure shape from the PR review.
fn returned_prose_to_agent(body: &Value) -> bool {
    body.pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false)
        && !has_tool_calls(body)
        && !is_explicit_error(body)
}

#[tokio::test]
async fn workers_describing_tool_call_must_emit_structured_tool_call() {
    // Three workers ALL reply with the same agentic-prose. Today the
    // heuristic classifier marks this as ToolProposal (good), but
    // `extract_tool_proposal` has no JSON in the text to pull
    // arguments from, so the gateway returns chat_response(payload)
    // — i.e. the prose — instead of a tool_call_response.
    let config = config_with_three_workers_returning(
        "I'll use read_file to inspect README.md and report what it contains.",
    );
    let body = user_request_with_read_file_tool("Read README.md and tell me what it says.");

    let result = moa::handle_turn(&config, &body).await;
    let body = &result.response_body;

    assert!(
        has_tool_calls(body) || is_explicit_error(body),
        "tool-flavored worker prose must produce either a real tool_calls field \
         or a clear failure response. Returning the prose as plain content is the \
         agent-harness failure mode the PR review called out. \
         turn_kind={:?}, reducer_used={}, body={body}",
        result.turn_kind,
        result.reducer_used,
    );

    assert!(
        !returned_prose_to_agent(body),
        "must not smuggle prose to agent as `content` without `tool_calls`; body={body}"
    );
}

#[tokio::test]
async fn workers_with_inline_tool_json_emit_real_tool_call() {
    // Counterpart: when worker output IS structurally a tool proposal
    // (JSON with function+arguments), MoA should emit a real
    // `tool_calls`. This already works today; the test is here to
    // pin the success case so the fix to the previous test doesn't
    // regress the normal path.
    let config = config_with_three_workers_returning(
        r#"I'll read the README. {"function": "read_file", "arguments": {"path": "README.md"}}"#,
    );
    let body = user_request_with_read_file_tool("Read README.md.");

    let result = moa::handle_turn(&config, &body).await;
    let body = &result.response_body;

    assert!(
        has_tool_calls(body),
        "well-formed inline tool JSON must produce tool_calls; \
         turn_kind={:?}, body={body}",
        result.turn_kind,
    );
    let name = body
        .pointer("/choices/0/message/tool_calls/0/function/name")
        .and_then(|v| v.as_str());
    assert_eq!(
        name,
        Some("read_file"),
        "tool_call function name must be the proposed tool"
    );
}
