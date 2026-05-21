//! Pin the tool-result turn classification.
//!
//! Background — PR #566 review feedback (Apr 2026):
//!
//! > The tool-result path isn't ready for agent loops:
//! > - A tool-result follow-up was treated like another fanout turn.
//! > - It wasn't handled like a controlled reducer/synthesis turn.
//! > - Tool results should be handled carefully and predictably, not
//! >   sprayed back through the whole fanout path.
//!
//! When the conversation has a recent tool result that has not yet
//! been synthesized into an assistant answer, the gateway must take
//! the reducer-only path (`TurnKind::ToolResult`), not fan-out to all
//! workers. Fanning out wastes a round-trip per worker, drowns the
//! reducer in worker outputs that ignore the tool result, and \u2014 most
//! dangerously \u2014 invites a worker to re-propose the same tool call
//! whose result we already have in-context.
//!
//! Two shapes of agent conversation must classify as ToolResult:
//!
//! 1. **OpenAI canonical shape.** The last message has role `tool`
//!    (the harness sent the tool result and expects the next assistant
//!    turn to interpret it). This is the simplest and most explicit
//!    shape. Classifying this is straightforward.
//!
//! 2. **Trailing-user shape.** The conversation ends with assistant
//!    tool_calls + tool result + a user message that just nudges
//!    ("continue", "what did you find?"). The harness has left the
//!    tool result in-context for the model to consume. There is no
//!    new tool result in *this* turn, but the previous one was never
//!    synthesized into an assistant message. Today this classifies
//!    as `Continuation` and fans out \u2014 wrong per the review.
//!
//! This file pins both shapes.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Backend that records every call it receives so the test can assert
/// fan-out did or did not happen.
struct RecordingBackend {
    text: String,
    delay: Duration,
    calls: AtomicUsize,
}

impl RecordingBackend {
    fn new(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            delay: Duration::from_millis(10),
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl moa::ModelBackend for RecordingBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok(json!({
            "choices": [{"message": {"content": self.text}}],
        }))
    }
}

fn config_with_three_recording_workers() -> (
    moa::GatewayConfig,
    Arc<RecordingBackend>,
    Arc<RecordingBackend>,
    Arc<RecordingBackend>,
) {
    let fast = RecordingBackend::new("synthesised: README says 'Hello World'");
    let mid = RecordingBackend::new("synthesised: README says 'Hello World'");
    let strong = RecordingBackend::new("synthesised: README says 'Hello World'");

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![fast.clone(), mid.clone(), strong.clone()];
    let models = vec![
        moa::ModelEntry {
            name: "fast-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "mid-13b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "strong-32b".into(),
            backend_index: 2,
        },
    ];

    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
    };
    (config, fast, mid, strong)
}

fn read_file_tool() -> Value {
    json!([{
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
    }])
}

#[tokio::test]
async fn last_message_role_tool_classifies_as_tool_result() {
    // OpenAI canonical shape. This is already supposed to work today
    // and pins the existing behavior so we don't regress when we
    // tighten the "trailing user" shape below.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "tools": read_file_tool(),
        "messages": [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "# Hello World\n"},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(
        result.turn_kind,
        moa::TurnKind::ToolResult,
        "last-msg-role=tool must classify as TurnKind::ToolResult; got {:?}",
        result.turn_kind
    );
    assert!(
        result.reducer_used,
        "tool-result turn must invoke the reducer"
    );
    // No fanout: only the reducer (a single backend in the candidate
    // ladder) should have been called.
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert_eq!(
        total_calls,
        1,
        "tool-result turn must not fan out — expected 1 reducer call, got {total_calls} \
         (fast={}, mid={}, strong={})",
        fast.calls(),
        mid.calls(),
        strong.calls()
    );
}

#[tokio::test]
async fn trailing_user_after_unsynthesised_tool_result_classifies_as_tool_result() {
    // The bug from the PR review. The harness has appended a `user`
    // message AFTER an unsynthesised tool result. The last message is
    // now `user`, not `tool`. The gateway today classifies this as
    // Continuation and fans out to every worker — wasting a round-trip
    // per worker and risking duplicate tool calls. It must instead
    // take the reducer-only path: synthesize the tool result, address
    // the user nudge, return one coherent response.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "tools": read_file_tool(),
        "messages": [
            {"role": "user", "content": "Read README.md and tell me what it says."},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "# Hello World\n"},
            // Harness leaves the tool result in-context and asks the
            // model to continue. There is NO new assistant message
            // synthesizing the tool result yet.
            {"role": "user", "content": "Go on."},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(
        result.turn_kind,
        moa::TurnKind::ToolResult,
        "trailing-user after unsynthesised tool result must classify as \
         TurnKind::ToolResult to avoid spraying through fanout; got {:?}",
        result.turn_kind
    );
    assert!(
        result.reducer_used,
        "unsynthesised-tool-result turn must invoke the reducer"
    );
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert_eq!(
        total_calls,
        1,
        "tool-result follow-up must not fan out — expected 1 reducer call, got \
         {total_calls} (fast={}, mid={}, strong={})",
        fast.calls(),
        mid.calls(),
        strong.calls()
    );
}

#[tokio::test]
async fn fresh_user_question_still_classifies_as_fan_out() {
    // Counterpart: a fresh conversation with just a user message must
    // continue to fan out. We must not over-trigger the tool-result
    // path and drop fan-out for normal questions.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "messages": [
            {"role": "user", "content": "What is the capital of Japan? One word only."},
        ],
        "max_tokens": 32,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert!(
        matches!(
            result.turn_kind,
            moa::TurnKind::Fanout | moa::TurnKind::EarlyExit
        ),
        "fresh user question must fan out (Fanout or EarlyExit); got {:?}",
        result.turn_kind
    );
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert!(
        total_calls >= 1,
        "fresh user question must reach at least one worker; got 0 calls"
    );
}
