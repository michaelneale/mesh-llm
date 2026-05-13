//! Mixture-of-Agents (MoA) gateway.
//!
//! Fan out to N heterogeneous LLM backends in parallel, arbitrate their
//! outputs with deterministic logic, and return one coherent OpenAI-
//! compatible response.  The client thinks it talks to one model.
//!
//! Transport is abstracted behind the [`ModelBackend`] trait:
//! - Default [`HttpBackend`] talks to any OpenAI-compatible HTTP endpoint
//! - The mesh host-runtime provides a mesh-native backend that dispatches
//!   local models via direct HTTP and remote models via QUIC tunnel
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  POST /v1/chat/completions { "model": "mesh" }
//!     ▼
//!  MoA Gateway
//!   ├─ session / context packing (role-shaped)
//!   ├─ parallel fan-out via ModelBackend
//!   ├─ incremental gathering with early-exit on consensus
//!   ├─ deterministic arbiter (code, not models)
//!   └─ reducer escalation only on genuine conflict
//! ```

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

/// The virtual model name that triggers MoA routing.
pub const VIRTUAL_MODEL_NAME: &str = "mesh";

// ─── Backend trait ───────────────────────────────────────────────────

/// Abstraction for calling a model.  The gateway doesn't care whether
/// the model is local HTTP, remote QUIC, or something else entirely.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    /// Call the model with the given messages (and optionally tools).
    /// Returns the full JSON response body from the model.
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
    ) -> Result<Value, String>;
}

/// Default HTTP backend — calls any OpenAI-compatible endpoint.
pub struct HttpBackend {
    pub base_url: String,
    http: reqwest::Client,
}

impl HttpBackend {
    pub fn new(base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self { base_url, http }
    }
}

#[async_trait::async_trait]
impl ModelBackend for HttpBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
    ) -> Result<Value, String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {status}: {}", &text[..text.len().min(200)]));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| format!("response parse: {e}"))
    }
}

// ─── Model entry ─────────────────────────────────────────────────────

/// A model available for MoA fan-out.
#[derive(Clone)]
pub struct ModelEntry {
    /// Model name (as used in the API).
    pub name: String,
    /// Index into the backends vec.  Multiple models can share a backend
    /// (e.g. all models behind the same proxy) or each have their own.
    pub backend_index: usize,
}

// ─── Configuration ───────────────────────────────────────────────────

/// Gateway configuration.
pub struct GatewayConfig {
    /// Available backends.  Models reference these by index.
    pub backends: Vec<std::sync::Arc<dyn ModelBackend>>,
    /// Available models for fan-out.
    pub models: Vec<ModelEntry>,
    /// Per-worker timeout.
    pub worker_timeout: Duration,
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

    match turn_type {
        session::TurnType::ToolResult => {
            handle_tool_result(config, &session, has_tools, start).await
        }
        session::TurnType::Fresh | session::TurnType::Continuation => {
            handle_query(config, &session, has_tools, start).await
        }
    }
}

// ─── Query handling ──────────────────────────────────────────────────

async fn handle_query(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
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
            )
            .await;
            let elapsed = t0.elapsed().as_millis() as u64;
            (model_name, role, result, elapsed)
        });
    }

    let total_workers = join_set.len();
    let (outputs, summaries, early_decision) =
        gather_workers_incremental(&mut join_set, total_workers, has_tools).await;

    if outputs.is_empty() {
        return TurnResult {
            response_body: error_response("All MoA workers failed"),
            worker_summaries: summaries,
            reducer_used: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs, has_tools));
    let (response_body, reducer_used) =
        resolve_decision(config, session, decision, &outputs, has_tools).await;

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
    start: Instant,
) -> TurnResult {
    let (reducer_name, reducer_backend_idx) = pick_reducer(config);
    tracing::info!("moa: tool result → reducer only ({reducer_name})");

    let (messages, tools) = context::pack_for_tool_result_turn(session, has_tools);
    let backend = &*config.backends[reducer_backend_idx];

    let result = call_backend(
        backend,
        &reducer_name,
        &messages,
        tools.as_ref(),
        2048,
        config.reducer_timeout,
    )
    .await;

    let succeeded = result.is_ok();
    let response_body = match result {
        Ok(text) => {
            let reduced =
                normalize::normalize_worker_output(&text, &reducer_name, WorkerRole::Reducer, 0);
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
            tracing::warn!("moa: reducer failed: {e}");
            error_response(&format!("Reducer failed: {e}"))
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
) -> (Value, bool) {
    match decision {
        arbiter::Decision::Answer(text) => (chat_response(&text), false),
        arbiter::Decision::ToolCall { name, arguments } => {
            (tool_call_response(&name, &arguments), false)
        }
        arbiter::Decision::NeedsReducer { reason } => {
            tracing::info!("moa: reducer — {reason}");
            let (reducer_name, reducer_backend_idx) = pick_reducer(config);
            let (messages, tools) = context::pack_for_reducer(session, outputs, &reason, has_tools);
            let backend = &*config.backends[reducer_backend_idx];

            match call_backend(
                backend,
                &reducer_name,
                &messages,
                tools.as_ref(),
                2048,
                config.reducer_timeout,
            )
            .await
            {
                Ok(text) => {
                    let reduced = normalize::normalize_worker_output(
                        &text,
                        &reducer_name,
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

/// Pick the reducer — prefers first model (typically local, zero RTT).
fn pick_reducer(config: &GatewayConfig) -> (String, usize) {
    config
        .models
        .first()
        .map(|m| (m.name.clone(), m.backend_index))
        .unwrap_or_else(|| ("unknown".into(), 0))
}

// ─── Backend call + text extraction ──────────────────────────────────

/// Call a backend and extract the assistant text from the response.
async fn call_backend(
    backend: &dyn ModelBackend,
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
    max_tokens: u32,
    timeout: Duration,
) -> Result<String, String> {
    let resp = backend
        .chat_completion(model, messages, tools, max_tokens, timeout)
        .await?;
    extract_text_from_response(&resp)
}

/// Extract assistant text from a chat completion response body.
fn extract_text_from_response(resp: &Value) -> Result<String, String> {
    let message = &resp["choices"][0]["message"];

    // Native tool_calls → KV format for normalizer
    if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
        if let Some(tc) = tool_calls.first() {
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            return Ok(format!(
                "kind: tool_proposal\ntool: {name}\narguments: {args}\nconfidence: 0.9\npayload: calling {name}",
            ));
        }
    }

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let stripped = worker::strip_thinking(&content);
    if !stripped.is_empty() {
        return Ok(stripped);
    }

    let thinking = worker::extract_thinking(&content);
    if !thinking.is_empty() {
        return Ok(thinking);
    }

    let reasoning = message
        .get("reasoning")
        .and_then(|r| r.as_str())
        .unwrap_or("");
    if !reasoning.is_empty() {
        return Ok(reasoning.to_string());
    }

    Err("empty response".into())
}

// ─── Worker gathering ────────────────────────────────────────────────

async fn gather_workers_incremental(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    total_workers: usize,
    has_tools: bool,
) -> (
    Vec<WorkerOutput>,
    Vec<WorkerSummary>,
    Option<arbiter::Decision>,
) {
    let mut outputs = Vec::new();
    let mut summaries = Vec::new();
    let mut total_finished: usize = 0;

    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok((model, role, Ok(text), elapsed)) => {
                total_finished += 1;
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

                if let Some(decision) =
                    arbiter::try_early_decision(&outputs, total_workers, total_finished, has_tools)
                {
                    join_set.abort_all();
                    while let Some(leftover) = join_set.join_next().await {
                        if let Ok((m, r, result, el)) = leftover {
                            summaries.push(WorkerSummary {
                                model: m,
                                role: r,
                                succeeded: result.is_ok(),
                                elapsed_ms: el,
                                output_kind: None,
                                confidence: None,
                            });
                        }
                    }
                    return (outputs, summaries, Some(decision));
                }
            }
            Ok((model, role, Err(e), elapsed)) => {
                total_finished += 1;
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

                if let Some(decision) =
                    arbiter::try_early_decision(&outputs, total_workers, total_finished, has_tools)
                {
                    join_set.abort_all();
                    while let Some(leftover) = join_set.join_next().await {
                        if let Ok((m, r, result, el)) = leftover {
                            summaries.push(WorkerSummary {
                                model: m,
                                role: r,
                                succeeded: result.is_ok(),
                                elapsed_ms: el,
                                output_kind: None,
                                confidence: None,
                            });
                        }
                    }
                    return (outputs, summaries, Some(decision));
                }
            }
            Err(e) => {
                total_finished += 1;
                tracing::warn!("moa: worker task panicked: {e}");
            }
        }
    }

    (outputs, summaries, None)
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
