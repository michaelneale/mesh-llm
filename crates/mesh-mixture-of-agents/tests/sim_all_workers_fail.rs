//! Simulated-mesh integration test for the "all workers failed" path.
//!
//! Pin the client contract for the case where every fan-out worker
//! fails: the MoA gateway must return a response body that a client
//! can _distinguish_ from a successful chat completion without parsing
//! the model's free-form output.
//!
//! Background — PR #566 review feedback (Apr 2026):
//!
//! > One concurrency request returned HTTP 200 even though the response
//! > body said all MoA workers failed. That's a bad client contract.
//! > If all workers fail, the API should probably return a proper
//! > error, not a successful-looking response with failure text inside
//! > it.
//!
//! Today `handle_turn` returns a body shaped like a successful
//! `chat.completion` with the error text in `choices[0].message.content`
//! and `finish_reason: "stop"`. There is no top-level `error` field, no
//! HTTP status carried in-band, and no way for an unsophisticated client
//! to know the call failed without string-matching the assistant text.
//!
//! This test drives `handle_turn` with three mock backends that all
//! error, then asserts the contract we want clients to be able to rely
//! on. It is expected to fail against the current implementation — that
//! failure is what we then fix.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

/// Backend that returns a configured error on every call. Counts calls
/// per model so a test can also assert the dispatch fanned out as
/// expected.
struct AlwaysErrBackend {
    err: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl AlwaysErrBackend {
    fn new(err: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            err: err.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl moa::ModelBackend for AlwaysErrBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(self.err.clone())
    }
}

fn three_failing_backends() -> moa::GatewayConfig {
    let alpha = AlwaysErrBackend::new("HTTP 502 from peer alpha");
    let beta = AlwaysErrBackend::new("connect timeout to peer beta");
    let gamma = AlwaysErrBackend::new("stream closed unexpectedly from peer gamma");

    let backends: Vec<Arc<dyn moa::ModelBackend>> =
        vec![alpha.clone(), beta.clone(), gamma.clone()];
    let models = vec![
        moa::ModelEntry {
            name: "alpha-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "beta-13b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "gamma-32b".into(),
            backend_index: 2,
        },
    ];

    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(200),
        reducer_timeout: Duration::from_secs(2),
    }
}

fn user_turn(content: &str) -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64,
    })
}

#[tokio::test]
async fn all_workers_fail_returns_distinguishable_error_body() {
    let config = three_failing_backends();
    let body = user_turn("What is the capital of Japan?");

    let result = moa::handle_turn(&config, &body).await;

    // First, lock in the path classification we already have.
    assert_eq!(
        result.turn_kind,
        moa::TurnKind::Failed,
        "all-workers-fail must classify as TurnKind::Failed; got {:?}",
        result.turn_kind
    );
    assert!(
        !result.reducer_used,
        "reducer must not be invoked when no worker output is available"
    );
    assert_eq!(result.reducer_attempts, 0, "no reducer attempts expected");
    assert_eq!(
        result.worker_summaries.len(),
        3,
        "all three workers should appear in worker_summaries"
    );
    assert!(
        result.worker_summaries.iter().all(|w| !w.succeeded),
        "every worker should be marked succeeded=false"
    );

    // The actual contract bug: the response body must be distinguishable
    // from a successful chat completion. Clients should not need to
    // parse `choices[0].message.content` to discover that the call
    // failed.
    let body = &result.response_body;
    let object = body.get("object").and_then(|v| v.as_str());

    // Either the top-level shape is not a chat.completion, OR there is
    // an explicit top-level `error` field. Both are acceptable; what is
    // NOT acceptable is a body that looks exactly like a successful
    // completion with the error text smuggled into `content`.
    let looks_like_success = object == Some("chat.completion");
    let has_top_level_error = body.get("error").is_some();
    let finish_reason = body
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str());
    let finish_reason_signals_error =
        finish_reason == Some("error") || finish_reason == Some("moa_failed");

    assert!(
        !looks_like_success || has_top_level_error || finish_reason_signals_error,
        "all-workers-fail body must be distinguishable from a successful chat.completion. \
         Got: object={object:?}, finish_reason={finish_reason:?}, has top-level error={has_top_level_error}, \
         full body: {body}"
    );
}
