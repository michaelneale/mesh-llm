//! Mixture-of-Agents (MoA) gateway.
//!
//! Fan out to N heterogeneous LLM backends in parallel, arbitrate their
//! outputs with deterministic logic, and return one coherent OpenAI-
//! compatible response.  The client thinks it talks to one model.
//!
//! Transport is abstracted behind the [`ModelBackend`] trait (see
//! [`backend`]). The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that
//! dispatch local models via direct HTTP and remote models via QUIC
//! tunnel.
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  POST /v1/chat/completions { "model": "mesh" }
//!     ▼
//!  MoA Gateway  (handle_turn)
//!   ├─ session / context packing (role-shaped)        — context::*
//!   ├─ parallel fan-out via ModelBackend              — fanout::gather_workers_incremental
//!   ├─ incremental gathering with early-exit          — arbiter::try_early_decision
//!   ├─ deterministic arbiter (code, not models)       — arbiter::arbitrate
//!   └─ reducer escalation only on genuine conflict    — reducer::hedged_reducer_call
//! ```
//!
//! Modules:
//! - [`backend`] — `ModelBackend` trait, `HttpBackend`, `SamplingParams`,
//!   `ModelEntry`
//! - [`reducer`] — reducer candidate ordering, hedged ladder
//! - [`fanout`] — incremental worker gathering with early-exit
//! - [`arbiter`] — deterministic arbitration + early-exit consensus
//! - [`normalize`] — 3-tier dirty-output parsing
//! - [`session`] — canonical transcript + turn classification
//! - [`context`] — role-shaped context packing
//! - [`worker`] — role assignment, think-tag stripping

pub mod arbiter;
pub mod backend;
pub mod context;
mod fanout;
pub mod normalize;
mod reducer;
pub mod session;
mod tool_guard;
pub mod worker;

pub use backend::{HttpBackend, ModelBackend, ModelEntry, SamplingParams};

use backend::call_backend;
use fanout::gather_workers_incremental;
use normalize::WorkerOutput;
use reducer::{hedged_reducer_call, reducer_candidates};
use serde_json::{json, Value};
use session::Session;
use std::time::{Duration, Instant};
use tool_guard::enforce_allowed_tools;
use worker::WorkerRole;
pub use worker::{strip_thinking, truncate_chars};

/// The virtual model name that triggers MoA routing.
pub const VIRTUAL_MODEL_NAME: &str = "mesh";

// ─── Configuration ───────────────────────────────────────────────────

/// Gateway configuration.
pub struct GatewayConfig {
    /// Available backends.  Models reference these by index.
    pub backends: Vec<std::sync::Arc<dyn ModelBackend>>,
    /// Available models for fan-out.
    pub models: Vec<ModelEntry>,
    /// Per-worker timeout.
    pub worker_timeout: Duration,
    /// Per-candidate wait before hedging a second reducer candidate. When the
    /// primary candidate is slow (e.g. cold KV) we don't want to wait the full
    /// reducer_timeout before kicking off candidate 2 — start the next one
    /// after hedge_delay and race them. Cost: up to 2× tokens for the rare
    /// slow case; zero cost on the happy path (candidate 1 returns first).
    pub hedge_delay: Duration,
    /// Reducer timeout.
    pub reducer_timeout: Duration,
}

// ─── Turn result ─────────────────────────────────────────────────────

/// Which gateway path produced this turn's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnKind {
    /// Fan-out path: arbiter decided from full worker outputs.
    Fanout,
    /// Fan-out path with early-exit consensus before all workers returned.
    EarlyExit,
    /// Tool-result turn: skipped fan-out, went straight to reducer.
    ToolResult,
    /// All workers failed and no reducer recovery happened.
    Failed,
}

impl TurnKind {
    /// Lowercase header-friendly label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fanout => "fanout",
            Self::EarlyExit => "early-exit",
            Self::ToolResult => "tool-result",
            Self::Failed => "failed",
        }
    }
}

/// What the gateway returns for a single turn.
#[derive(Debug)]
pub struct TurnResult {
    /// OpenAI chat.completion response body.
    pub response_body: Value,
    /// Per-worker details for observability.
    pub worker_summaries: Vec<WorkerSummary>,
    /// Whether the reducer was invoked.
    pub reducer_used: bool,
    /// How many reducer candidates were spawned (0 if reducer didn't run,
    /// 1 on the happy reducer path, ≥2 if the hedge fired or a fast-fail
    /// cascaded to the next candidate).
    pub reducer_attempts: u32,
    /// Which gateway path produced this response.
    pub turn_kind: TurnKind,
    /// Wall-clock time for this turn.
    pub elapsed_ms: u64,
}

#[derive(Debug)]
pub struct WorkerSummary {
    pub model: String,
    pub role: WorkerRole,
    pub succeeded: bool,
    pub elapsed_ms: u64,
    pub output_kind: Option<normalize::OutputKind>,
    pub confidence: Option<f32>,
}

// ─── Gateway entry point ─────────────────────────────────────────────

/// Process one MoA turn.
///
/// Stateless per request.  Multi-turn state is managed by the agent client
/// which sends the full conversation on each request.
pub async fn handle_turn(config: &GatewayConfig, body: &Value) -> TurnResult {
    let start = Instant::now();

    let mut session = Session::new();
    let incoming_messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let tools = body.get("tools").cloned();
    let has_tools = tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    session.ingest(&incoming_messages, &tools);

    let turn_type = session.classify_turn();
    tracing::info!(
        "moa: turn={:?}, {} models, tools={}",
        turn_type,
        config.models.len(),
        has_tools,
    );

    let allowed_tools = session.tool_names();

    match turn_type {
        session::TurnType::ToolResult => {
            handle_tool_result(config, &session, has_tools, &allowed_tools, start).await
        }
        session::TurnType::Fresh | session::TurnType::Continuation => {
            handle_query(config, &session, has_tools, &allowed_tools, start).await
        }
    }
}

// ─── Query handling ──────────────────────────────────────────────────

async fn handle_query(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    start: Instant,
) -> TurnResult {
    let assignments = worker::assign_roles(&config.models);

    tracing::info!(
        "moa: dispatching to {} workers: [{}]",
        assignments.len(),
        assignments
            .iter()
            .map(|a| format!("{}({})", a.model_name, a.role.label()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut join_set = tokio::task::JoinSet::new();
    let mut dispatched: Vec<fanout::DispatchedWorker> = Vec::with_capacity(assignments.len());

    for assignment in &assignments {
        let packed = context::pack_for_worker(session, assignment.role, has_tools);
        let model_name = assignment.model_name.clone();
        let role = assignment.role;
        let backend = config.backends[assignment.backend_index].clone();
        let timeout = config.worker_timeout;

        dispatched.push(fanout::DispatchedWorker {
            model: model_name.clone(),
            role,
        });

        join_set.spawn(async move {
            let t0 = Instant::now();
            let result = call_backend(
                &*backend,
                &model_name,
                &packed.messages,
                packed.tools.as_ref(),
                packed.max_tokens,
                timeout,
                SamplingParams::worker(),
            )
            .await;
            let elapsed = t0.elapsed().as_millis() as u64;
            (model_name, role, result, elapsed)
        });
    }

    let (outputs, summaries, early_decision) =
        gather_workers_incremental(&mut join_set, &dispatched, has_tools, allowed_tools).await;

    if outputs.is_empty() {
        return TurnResult {
            response_body: error_response("All MoA workers failed"),
            worker_summaries: summaries,
            reducer_used: false,
            reducer_attempts: 0,
            turn_kind: TurnKind::Failed,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    // Capture whether we took the early-exit path BEFORE we resolve the
    // decision: the arbiter never runs when early_decision is Some.
    let took_early_exit = early_decision.is_some();
    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs, has_tools));
    let (response_body, reducer_used, reducer_attempts) = resolve_decision(
        config,
        session,
        decision,
        &outputs,
        has_tools,
        allowed_tools,
    )
    .await;

    // turn_kind is "early-exit" only when we genuinely short-circuited via
    // consensus AND didn't need to escalate to the reducer. A reducer-
    // escalated turn is "fanout" even if early_decision was set, because
    // we still did the expensive serial call.
    let turn_kind = if took_early_exit && !reducer_used {
        TurnKind::EarlyExit
    } else {
        TurnKind::Fanout
    };

    TurnResult {
        response_body,
        worker_summaries: summaries,
        reducer_used,
        reducer_attempts,
        turn_kind,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

// ─── Tool result handling ────────────────────────────────────────────

async fn handle_tool_result(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    start: Instant,
) -> TurnResult {
    let candidates = reducer_candidates(config);
    let candidate_count = candidates.len();
    let (messages, tools) = context::pack_for_tool_result_turn(session, has_tools);

    // Hedged ladder: start candidate 0, hedge to candidate 1 after hedge_delay
    // (or immediately on candidate 0 error), race for the first OK. Rescues
    // tool-result turns when the first strong peer is broken (e.g. stale
    // binary that 502s on tool grammars) without paying N×timeout serially.
    tracing::info!("moa: tool result → hedged reducer over {candidate_count} candidate(s)");
    let hedge_result = hedged_reducer_call(
        &config.backends,
        candidates.clone(),
        messages,
        tools,
        config.reducer_timeout,
        config.hedge_delay,
    )
    .await;

    let mut last_err: Option<String> = None;
    let (attempts, chosen): (u32, Option<(String, normalize::WorkerOutput)>) = match hedge_result {
        Ok(reducer::HedgedReducerOk {
            winner,
            text,
            attempts: spawned,
        }) => {
            let mut reduced =
                normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
            enforce_allowed_tools(&mut reduced, allowed_tools, &winner);
            (spawned, Some((winner, reduced)))
        }
        Err(reducer::HedgedReducerErr {
            err,
            attempts: spawned,
        }) => {
            last_err = Some(err);
            (spawned, None)
        }
    };

    let (reducer_name, succeeded, response_body) = match chosen {
        Some((name, reduced)) => {
            let body = match reduced.kind {
                normalize::OutputKind::ToolProposal => {
                    if let (Some(tname), Some(args)) =
                        (reduced.tool_name.as_ref(), reduced.tool_arguments.as_ref())
                    {
                        tool_call_response(tname, args)
                    } else {
                        chat_response(&reduced.payload)
                    }
                }
                _ => chat_response(&reduced.payload),
            };
            (name, true, body)
        }
        None => {
            let err = last_err.unwrap_or_else(|| "no reducer candidates".into());
            tracing::warn!("moa: all {attempts} reducer candidates failed");
            (
                candidates.first().map(|c| c.0.clone()).unwrap_or_default(),
                false,
                error_response(&format!("Reducer failed (tried {attempts}): {err}")),
            )
        }
    };

    TurnResult {
        response_body,
        worker_summaries: vec![WorkerSummary {
            model: reducer_name,
            role: WorkerRole::Reducer,
            succeeded,
            elapsed_ms: start.elapsed().as_millis() as u64,
            output_kind: None,
            confidence: None,
        }],
        reducer_used: true,
        reducer_attempts: attempts,
        turn_kind: TurnKind::ToolResult,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

// ─── Decision resolution ─────────────────────────────────────────────

/// Returns (response body, reducer_used, reducer_attempts).
async fn resolve_decision(
    config: &GatewayConfig,
    session: &Session,
    decision: arbiter::Decision,
    outputs: &[WorkerOutput],
    has_tools: bool,
    allowed_tools: &[String],
) -> (Value, bool, u32) {
    match decision {
        arbiter::Decision::Answer(text) => (chat_response(&text), false, 0),
        arbiter::Decision::ToolCall { name, arguments } => {
            (tool_call_response(&name, &arguments), false, 0)
        }
        arbiter::Decision::NeedsReducer { reason } => {
            tracing::info!("moa: reducer — {reason}");
            let candidates = reducer_candidates(config);
            let (messages, tools) = context::pack_for_reducer(session, outputs, &reason, has_tools);

            // Hedged ladder over the ordered candidates (see hedged_reducer_call).
            let hedge_result = hedged_reducer_call(
                &config.backends,
                candidates,
                messages,
                tools,
                config.reducer_timeout,
                config.hedge_delay,
            )
            .await;

            let (attempts, chosen): (u32, Option<normalize::WorkerOutput>) = match hedge_result {
                Ok(reducer::HedgedReducerOk {
                    winner,
                    text,
                    attempts: spawned,
                }) => {
                    let mut reduced =
                        normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
                    enforce_allowed_tools(&mut reduced, allowed_tools, &winner);
                    (spawned, Some(reduced))
                }
                Err(reducer::HedgedReducerErr {
                    err: _,
                    attempts: spawned,
                }) => (spawned, None),
            };

            match chosen {
                Some(reduced) => match reduced.kind {
                    normalize::OutputKind::ToolProposal => {
                        if let (Some(name), Some(args)) =
                            (reduced.tool_name.as_ref(), reduced.tool_arguments.as_ref())
                        {
                            (tool_call_response(name, args), true, attempts)
                        } else {
                            (chat_response(&reduced.payload), true, attempts)
                        }
                    }
                    _ => (chat_response(&reduced.payload), true, attempts),
                },
                None => {
                    tracing::warn!("moa: all reducer candidates failed, using best worker");
                    // reducer_used=false here because the reducer did NOT
                    // produce the output we're returning — we fell back to
                    // a worker. attempts still reflects what was spawned so
                    // observability can see "we tried N times and all failed".
                    (chat_response(&best_answer(outputs)), false, attempts)
                }
            }
        }
    }
}

// ─── Response builders ───────────────────────────────────────────────

fn best_answer(outputs: &[WorkerOutput]) -> String {
    outputs
        .iter()
        .filter(|o| matches!(o.kind, normalize::OutputKind::Answer))
        // `total_cmp` is total over all f32 (including NaN/Inf); `partial_cmp`
        // can return `None` on NaN, which would panic on `unwrap`.
        // `normalize_worker_output` now sanitizes non-finite confidences
        // before they reach here, but using `total_cmp` keeps this site
        // panic-free even if a future caller skips the normalizer.
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .or(outputs.first())
        .map(|o| o.payload.clone())
        .unwrap_or_default()
}

/// Build a response body that signals MoA-level failure to the client.
///
/// Distinguishable from a successful `chat.completion` in three ways:
///
///   * Top-level `error` object (OpenAI error-shape) so SDKs that read
///     `response.error` see the failure without parsing `choices`.
///   * `choices[0].finish_reason == "error"` (instead of `"stop"`) so
///     SDKs that branch on `finish_reason` see the failure too.
///   * The error text is still placed in `choices[0].message.content`
///     so unstructured clients still surface a useful string to the
///     human, just not as a successful assistant reply.
///
/// The ingress layer is responsible for choosing the HTTP status; this
/// body is the in-band signal.
fn error_response(message: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "error": {
            "message": message,
            "type": "moa_failure",
            "code": "all_workers_failed",
        },
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": message },
            "finish_reason": "error"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })
}

fn chat_response(content: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })
}

fn tool_call_response(name: &str, arguments: &Value) -> Value {
    // OpenAI tool-call `arguments` is a JSON-object *string*. Three input
    // shapes have to collapse to a valid object string here:
    //
    //   * String form: trust the caller's JSON (worker already passed
    //     through `extract_tool_arguments` so the inner shape is sane).
    //   * Null / non-object: emit `"{}"` rather than `"null"` or
    //     `"\"foo\""`. The previous shape would serialize `Value::Null`
    //     to the literal four-char string `"null"`, which downstream
    //     OpenAI tool-call consumers reject.
    //   * Object: serialize as JSON.
    let args_str = match arguments {
        Value::String(s) => {
            // Validate that the string is itself a JSON object; if not,
            // fall back to `{}` to keep wire-shape sane.
            if serde_json::from_str::<Value>(s)
                .map(|v| v.is_object())
                .unwrap_or(false)
            {
                s.clone()
            } else {
                "{}".to_string()
            }
        }
        Value::Null => "{}".to_string(),
        v if v.is_object() => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
        // Bare primitives / arrays — not a valid tool-call arguments object.
        _ => "{}".to_string(),
    };

    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_{}", short_id()),
                    "type": "function",
                    "function": { "name": name, "arguments": args_str }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", t)
}

#[cfg(test)]
mod response_builder_tests {
    use super::*;
    use crate::normalize::{OutputKind, WorkerOutput};
    use crate::worker::WorkerRole;

    fn answer(model: &str, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Fast,
            elapsed_ms: 1,
        }
    }

    #[test]
    fn best_answer_does_not_panic_on_nan_confidence() {
        // Regression for PR #566 review: `partial_cmp(...).unwrap()` could
        // panic if any confidence reached this site as NaN. After switching
        // to `total_cmp`, this is safe even if normalize is bypassed.
        let outputs = vec![
            answer("a", f32::NAN, "nan-answer"),
            answer("b", 0.7, "good-answer"),
            answer("c", f32::NAN, "another-nan"),
        ];
        let picked = best_answer(&outputs);
        // `total_cmp` treats NaN as greater than any finite; the assertion
        // here is *not* about which specific answer wins, only that we do
        // not panic and we return *some* answer.
        assert!(!picked.is_empty());
    }

    #[test]
    fn tool_call_response_emits_object_args_for_null() {
        // Regression: `Value::Null` previously serialized to the literal
        // string "null", which downstream OpenAI tool-call consumers reject.
        let resp = tool_call_response("list", &Value::Null);
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_emits_object_args_for_primitive() {
        let resp = tool_call_response("list", &Value::from(42));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_passes_through_string_form_when_valid() {
        let resp = tool_call_response(
            "read_file",
            &Value::String("{\"path\":\"README.md\"}".to_string()),
        );
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["path"], "README.md");
    }

    #[test]
    fn tool_call_response_rejects_invalid_string_form() {
        // If the caller hands us a bare non-JSON string, fall back to `{}`.
        let resp = tool_call_response("x", &Value::String("not json at all".to_string()));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }
}
