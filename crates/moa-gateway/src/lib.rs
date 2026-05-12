//! Mixture-of-Agents (MoA) gateway.
//!
//! A stateful gateway that owns the session between an agent client and N
//! heterogeneous LLM backends.  The client thinks it's talking to one model.
//! The gateway fans out to workers, arbitrates, manages tool lifecycle, and
//! maintains the canonical context across turns.
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  OpenAI chat completions
//!     ▼
//!  MoA Gateway (stateful, owns session)
//!   ├─ canonical transcript
//!   ├─ tool call / result tracking
//!   ├─ context packer (role-shaped)
//!   ├─ deterministic arbiter
//!   └─ worker dispatcher
//!         │
//!         ├──► endpoint A  (fast worker)
//!         ├──► endpoint B  (specialist)
//!         └──► endpoint C  (strong / reducer)
//! ```
//!
//! Key design choices:
//!
//! - **Deterministic logic first.** Parsing, normalizing, voting, thresholding,
//!   tool validation are all code.  Models are semantic coprocessors.
//! - **Gateway owns tool lifecycle.** Workers propose tools; only the gateway
//!   emits tool_calls.  Tool results go to the reducer, not re-broadcast.
//! - **Context is role-shaped.** Full context enters the gateway but each
//!   worker gets a tailored packet.  Short-context models are useful workers.
//! - **Transport-agnostic.** Backends are just OpenAI-compatible HTTP URLs.

pub mod arbiter;
pub mod context;
pub mod normalize;
pub mod session;
pub mod worker;

use normalize::WorkerOutput;
use serde_json::{json, Value};
use session::Session;
use std::time::{Duration, Instant};
use worker::WorkerRole;

// ─── Configuration ───────────────────────────────────────────────────

/// An endpoint the gateway can dispatch to.
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Base URL, e.g. `http://localhost:11434/v1`
    pub base_url: String,
    /// Model name to put in the request body.
    pub model: String,
}

/// Gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Available endpoints.
    pub endpoints: Vec<Endpoint>,
    /// Per-worker timeout.
    pub worker_timeout: Duration,
    /// Reducer timeout.
    pub reducer_timeout: Duration,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![],
            worker_timeout: Duration::from_secs(30),
            reducer_timeout: Duration::from_secs(45),
        }
    }
}

// ─── Turn result ─────────────────────────────────────────────────────

/// What the gateway returns for a single turn.
#[derive(Debug)]
pub struct TurnResult {
    /// OpenAI chat.completion response body.
    pub response_body: Value,
    /// Per-worker details for observability.
    pub worker_summaries: Vec<WorkerSummary>,
    /// Whether the reducer was invoked.
    pub reducer_used: bool,
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

// ─── Gateway ─────────────────────────────────────────────────────────

/// The stateful MoA gateway.
///
/// Create once, call `turn()` for each incoming request.  The gateway
/// tracks the canonical session across turns — tool calls it emitted,
/// tool results it received, context summaries it computed.
pub struct Gateway {
    config: GatewayConfig,
    session: Session,
    http: reqwest::Client,
}

impl Gateway {
    pub fn new(config: GatewayConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.worker_timeout + Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        Self {
            config,
            session: Session::new(),
            http,
        }
    }

    /// Process one turn.  `body` is the OpenAI request from the agent client.
    ///
    /// The gateway:
    /// 1. Records the incoming messages into the canonical session
    /// 2. Decides the turn type (fresh query, tool result, continuation)
    /// 3. Fans out to workers (or just the reducer for tool results)
    /// 4. Arbitrates
    /// 5. Returns one response and records it in the session
    pub async fn turn(&mut self, body: &Value) -> TurnResult {
        let start = Instant::now();

        // Ingest new messages into canonical session
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

        self.session.ingest(&incoming_messages, &tools);

        let turn_type = self.session.classify_turn();
        tracing::info!(
            "moa: turn type = {:?}, session turns = {}",
            turn_type,
            self.session.turn_count()
        );

        match turn_type {
            session::TurnType::ToolResult => {
                // Tool result came back — don't fan out broadly.
                // Send to reducer/finalizer only with the original task + result.
                self.handle_tool_result(has_tools, start).await
            }
            session::TurnType::Fresh | session::TurnType::Continuation => {
                // Normal turn — full fan-out.
                self.handle_query(has_tools, start).await
            }
        }
    }

    /// Full fan-out: dispatch to all workers, gather, arbitrate.
    async fn handle_query(&mut self, has_tools: bool, start: Instant) -> TurnResult {
        let assignments = worker::assign_roles(&self.config.endpoints, has_tools);

        tracing::info!(
            "moa: dispatching to {} workers: [{}]",
            assignments.len(),
            assignments
                .iter()
                .map(|a| format!("{}({})", a.endpoint.model, a.role.label()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut join_set = tokio::task::JoinSet::new();

        for assignment in &assignments {
            let packed = context::pack_for_worker(&self.session, assignment.role, has_tools);
            let endpoint = assignment.endpoint.clone();
            let role = assignment.role;
            let http = self.http.clone();
            let timeout = self.config.worker_timeout;

            join_set.spawn(async move {
                let t0 = Instant::now();
                let result = worker::call(
                    &http,
                    &endpoint,
                    &packed.messages,
                    packed.max_tokens,
                    timeout,
                )
                .await;
                let elapsed = t0.elapsed().as_millis() as u64;
                (endpoint.model.clone(), role, result, elapsed)
            });
        }

        let (outputs, summaries) = gather_workers(&mut join_set).await;

        if outputs.is_empty() {
            return TurnResult {
                response_body: error_response("All MoA workers failed"),
                worker_summaries: summaries,
                reducer_used: false,
                elapsed_ms: start.elapsed().as_millis() as u64,
            };
        }

        let decision = arbiter::arbitrate(&outputs, has_tools);
        let (response_body, reducer_used) =
            self.resolve_decision(decision, &outputs, has_tools).await;

        // Record what we emitted
        self.session.record_assistant_response(&response_body);

        TurnResult {
            response_body,
            worker_summaries: summaries,
            reducer_used,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Tool result turn — send to reducer only, don't re-broadcast.
    async fn handle_tool_result(&mut self, has_tools: bool, start: Instant) -> TurnResult {
        let reducer_endpoint = self.pick_reducer_endpoint();

        tracing::info!(
            "moa: tool result turn → reducer only ({})",
            reducer_endpoint.model,
        );

        let reducer_messages = context::pack_for_tool_result_turn(&self.session, has_tools);

        let result = worker::call(
            &self.http,
            &reducer_endpoint,
            &reducer_messages,
            2048,
            self.config.reducer_timeout,
        )
        .await;

        let succeeded = result.is_ok();
        let response_body = match result {
            Ok(text) => {
                let reduced = normalize::normalize_worker_output(
                    &text,
                    &reducer_endpoint.model,
                    WorkerRole::Reducer,
                    0,
                );
                match reduced.kind {
                    normalize::OutputKind::ToolProposal => {
                        if let (Some(name), Some(args)) =
                            (reduced.tool_name.as_ref(), reduced.tool_arguments.as_ref())
                        {
                            tool_call_response(name, args)
                        } else {
                            chat_response(&reduced.payload)
                        }
                    }
                    _ => chat_response(&reduced.payload),
                }
            }
            Err(e) => {
                tracing::warn!("moa: reducer failed on tool result: {e}");
                error_response(&format!("Reducer failed: {e}"))
            }
        };

        self.session.record_assistant_response(&response_body);

        TurnResult {
            response_body,
            worker_summaries: vec![WorkerSummary {
                model: reducer_endpoint.model.clone(),
                role: WorkerRole::Reducer,
                succeeded,
                elapsed_ms: start.elapsed().as_millis() as u64,
                output_kind: None,
                confidence: None,
            }],
            reducer_used: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Resolve an arbiter decision — may invoke the reducer.
    async fn resolve_decision(
        &self,
        decision: arbiter::Decision,
        outputs: &[WorkerOutput],
        has_tools: bool,
    ) -> (Value, bool) {
        match decision {
            arbiter::Decision::Answer(text) => (chat_response(&text), false),
            arbiter::Decision::ToolCall { name, arguments } => {
                (tool_call_response(&name, &arguments), false)
            }
            arbiter::Decision::NeedsReducer { reason } => {
                tracing::info!("moa: invoking reducer — {reason}");
                let endpoint = self.pick_reducer_endpoint();
                let messages =
                    context::pack_for_reducer(&self.session, outputs, &reason, has_tools);

                match worker::call(
                    &self.http,
                    &endpoint,
                    &messages,
                    2048,
                    self.config.reducer_timeout,
                )
                .await
                {
                    Ok(text) => {
                        let reduced = normalize::normalize_worker_output(
                            &text,
                            &endpoint.model,
                            WorkerRole::Reducer,
                            0,
                        );
                        match reduced.kind {
                            normalize::OutputKind::ToolProposal => {
                                if let (Some(name), Some(args)) =
                                    (reduced.tool_name.as_ref(), reduced.tool_arguments.as_ref())
                                {
                                    (tool_call_response(name, args), true)
                                } else {
                                    (chat_response(&reduced.payload), true)
                                }
                            }
                            _ => (chat_response(&reduced.payload), true),
                        }
                    }
                    Err(e) => {
                        tracing::warn!("moa: reducer failed: {e}, using best worker");
                        (chat_response(&best_answer(outputs)), false)
                    }
                }
            }
        }
    }

    /// Pick the best endpoint for the reducer role.
    fn pick_reducer_endpoint(&self) -> Endpoint {
        // Last endpoint is typically the strongest (by assignment order).
        // Could be smarter with capability metadata later.
        self.config
            .endpoints
            .last()
            .cloned()
            .unwrap_or_else(|| self.config.endpoints[0].clone())
    }
}

// ─── Worker gathering ────────────────────────────────────────────────

async fn gather_workers(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
) -> (Vec<WorkerOutput>, Vec<WorkerSummary>) {
    let mut outputs = Vec::new();
    let mut summaries = Vec::new();

    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok((model, role, Ok(text), elapsed)) => {
                let normalized = normalize::normalize_worker_output(&text, &model, role, elapsed);
                tracing::info!(
                    "moa: worker {} ({}) → {:?} conf={:.2} ({}ms, {} chars)",
                    model,
                    role.label(),
                    normalized.kind,
                    normalized.confidence,
                    elapsed,
                    text.len(),
                );
                summaries.push(WorkerSummary {
                    model: model.clone(),
                    role,
                    succeeded: true,
                    elapsed_ms: elapsed,
                    output_kind: Some(normalized.kind),
                    confidence: Some(normalized.confidence),
                });
                outputs.push(normalized);
            }
            Ok((model, role, Err(e), elapsed)) => {
                tracing::warn!(
                    "moa: worker {} ({}) failed after {}ms: {}",
                    model,
                    role.label(),
                    elapsed,
                    e,
                );
                summaries.push(WorkerSummary {
                    model,
                    role,
                    succeeded: false,
                    elapsed_ms: elapsed,
                    output_kind: None,
                    confidence: None,
                });
            }
            Err(e) => {
                tracing::warn!("moa: worker task panicked: {e}");
            }
        }
    }

    (outputs, summaries)
}

// ─── Response builders ───────────────────────────────────────────────

fn best_answer(outputs: &[WorkerOutput]) -> String {
    outputs
        .iter()
        .filter(|o| matches!(o.kind, normalize::OutputKind::Answer))
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
        .or(outputs.first())
        .map(|o| o.payload.clone())
        .unwrap_or_default()
}

fn error_response(message: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": "moa",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": message },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })
}

fn chat_response(content: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": "moa",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })
}

fn tool_call_response(name: &str, arguments: &Value) -> Value {
    let args_str = if arguments.is_string() {
        arguments.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())
    };

    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": "moa",
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
