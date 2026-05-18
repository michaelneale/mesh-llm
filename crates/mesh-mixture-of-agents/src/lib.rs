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
pub mod worker;

pub use backend::{HttpBackend, ModelBackend, ModelEntry, SamplingParams};

use backend::call_backend;
use fanout::gather_workers_incremental;
use normalize::WorkerOutput;
use reducer::{hedged_reducer_call, reducer_candidates};
use serde_json::{json, Value};
use session::Session;
use std::time::{Duration, Instant};
use worker::WorkerRole;

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

/// Demote a `ToolProposal` whose `tool_name` is not in `allowed_tools`
/// to `Uncertainty`. This guards downstream arbitration from worker
/// hallucinations like proposing `execute_typescript` when only `shell`
/// is declared.
///
/// If `allowed_tools` is empty (no tools were declared on the request
/// body) we leave the proposal as-is — the arbiter will route to the
/// reducer which has the same call site policy.
pub(crate) fn enforce_allowed_tools(
    output: &mut normalize::WorkerOutput,
    allowed_tools: &[String],
    model: &str,
) {
    if allowed_tools.is_empty() {
        return;
    }
    if output.kind != normalize::OutputKind::ToolProposal {
        return;
    }
    let Some(ref name) = output.tool_name else {
        return;
    };
    if allowed_tools.iter().any(|t| t == name) {
        return;
    }
    tracing::warn!(
        "moa: worker {model} proposed unknown tool {name:?}, demoting to uncertainty \
         (allowed: {allowed_tools:?})"
    );
    output.kind = normalize::OutputKind::Uncertainty;
    output.tool_name = None;
    output.tool_arguments = None;
    // Drop confidence so this proposal doesn't outrank real ones in any
    // tie-breaking path that still inspects it.
    output.confidence = 0.0;
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

    for assignment in &assignments {
        let packed = context::pack_for_worker(session, assignment.role, has_tools);
        let model_name = assignment.model_name.clone();
        let role = assignment.role;
        let backend = config.backends[assignment.backend_index].clone();
        let timeout = config.worker_timeout;

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

    let total_workers = join_set.len();
    let (outputs, summaries, early_decision) =
        gather_workers_incremental(&mut join_set, total_workers, has_tools, allowed_tools).await;

    if outputs.is_empty() {
        return TurnResult {
            response_body: error_response("All MoA workers failed"),
            worker_summaries: summaries,
            reducer_used: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs, has_tools));
    let (response_body, reducer_used) = resolve_decision(
        config,
        session,
        decision,
        &outputs,
        has_tools,
        allowed_tools,
    )
    .await;

    TurnResult {
        response_body,
        worker_summaries: summaries,
        reducer_used,
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
    let chosen: Option<(String, normalize::WorkerOutput)> = match hedge_result {
        Ok((name, text)) => {
            let mut reduced =
                normalize::normalize_worker_output(&text, &name, WorkerRole::Reducer, 0);
            enforce_allowed_tools(&mut reduced, allowed_tools, &name);
            Some((name, reduced))
        }
        Err(e) => {
            last_err = Some(e);
            None
        }
    };
    let attempts = candidate_count;

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
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

// ─── Decision resolution ─────────────────────────────────────────────

async fn resolve_decision(
    config: &GatewayConfig,
    session: &Session,
    decision: arbiter::Decision,
    outputs: &[WorkerOutput],
    has_tools: bool,
    allowed_tools: &[String],
) -> (Value, bool) {
    match decision {
        arbiter::Decision::Answer(text) => (chat_response(&text), false),
        arbiter::Decision::ToolCall { name, arguments } => {
            (tool_call_response(&name, &arguments), false)
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

            let chosen: Option<normalize::WorkerOutput> = match hedge_result {
                Ok((name, text)) => {
                    let mut reduced =
                        normalize::normalize_worker_output(&text, &name, WorkerRole::Reducer, 0);
                    enforce_allowed_tools(&mut reduced, allowed_tools, &name);
                    Some(reduced)
                }
                Err(_) => None,
            };

            match chosen {
                Some(reduced) => match reduced.kind {
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
                },
                None => {
                    tracing::warn!("moa: all reducer candidates failed, using best worker");
                    (chat_response(&best_answer(outputs)), false)
                }
            }
        }
    }
}

// ─── Endpoint discovery (for standalone/test use) ────────────────────

/// An endpoint for the HTTP backend (convenience for test harnesses).
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub base_url: String,
    pub model: String,
}

/// Discover models from an OpenAI-compatible `/v1/models` endpoint.
pub async fn discover_endpoints(base_url: &str) -> Result<Vec<Endpoint>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/models"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("can't reach {base_url}/models: {e}"))?;
    let body: Value = resp.json().await.map_err(|e| format!("bad json: {e}"))?;

    let empty = vec![];
    let mut entries: Vec<&Value> = body["data"].as_array().unwrap_or(&empty).iter().collect();
    entries.sort_by_key(|m| m["id"].as_str().unwrap_or("").len());

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut endpoints = Vec::new();

    for m in entries {
        let id = match m["id"].as_str() {
            Some(id) => id,
            None => continue,
        };
        if id.contains("cloud") || id == VIRTUAL_MODEL_NAME {
            continue;
        }
        let display = m["display_name"].as_str().unwrap_or(id).to_string();
        let base_name = display
            .split("-Q")
            .next()
            .unwrap_or(&display)
            .to_lowercase()
            .replace("-gguf", "")
            .replace("unsloth/", "")
            .replace("meshllm/", "");
        if seen.contains(&base_name) {
            continue;
        }
        seen.insert(base_name);
        endpoints.push(Endpoint {
            base_url: base_url.to_string(),
            model: id.to_string(),
        });
    }
    Ok(endpoints)
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
        "model": VIRTUAL_MODEL_NAME,
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
    let args_str = if arguments.is_string() {
        arguments.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())
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
