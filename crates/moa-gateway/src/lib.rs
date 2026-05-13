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
        // ── Passthrough mode ─────────────────────────────────────
        // When tools are present (agentic use), pass the original messages
        // to workers and return the first successful raw response.  The MoA
        // envelope (normalize → arbitrate → reduce) actively harms tool-use
        // because it replaces the client's system prompt and tools, confusing
        // small models.  In passthrough mode MoA still provides redundancy:
        // if the first worker fails, the next one answers.
        if has_tools {
            return self.handle_passthrough(start).await;
        }

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

        // Record what we emitted + accepted fact for future context
        self.session.record_assistant_response(&response_body);
        let outcome = extract_turn_outcome(&response_body);
        self.session.record_turn_outcome(&outcome);

        TurnResult {
            response_body,
            worker_summaries: summaries,
            reducer_used,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Passthrough mode: fan out with original messages, return the first
    /// successful raw response.  Used when tools are present so native
    /// tool_call format is preserved.
    async fn handle_passthrough(&mut self, start: Instant) -> TurnResult {
        let messages = self.session.all_messages();
        let tools = self.session.tools().cloned();
        let mut join_set = tokio::task::JoinSet::new();

        tracing::info!(
            "moa: passthrough mode (tools present), {} endpoints",
            self.config.endpoints.len(),
        );

        for endpoint in &self.config.endpoints {
            let ep = endpoint.clone();
            let msgs = messages.clone();
            let t = tools.clone();
            let http = self.http.clone();
            let timeout = self.config.worker_timeout;

            join_set.spawn(async move {
                let t0 = Instant::now();
                let result =
                    worker::call_raw_with_tools(&http, &ep, &msgs, t.as_ref(), 4096, timeout).await;
                let elapsed = t0.elapsed().as_millis() as u64;
                (ep.model.clone(), result, elapsed)
            });
        }

        let mut summaries = Vec::new();
        let mut best_response: Option<Value> = None;

        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((model, Ok(resp), elapsed)) => {
                    tracing::info!(
                        "moa: passthrough worker {} succeeded ({}ms)",
                        model,
                        elapsed,
                    );
                    summaries.push(WorkerSummary {
                        model,
                        role: WorkerRole::Generalist,
                        succeeded: true,
                        elapsed_ms: elapsed,
                        output_kind: None,
                        confidence: None,
                    });
                    // Take the first successful response
                    if best_response.is_none() {
                        best_response = Some(resp);
                    }
                }
                Ok((model, Err(e), elapsed)) => {
                    tracing::warn!(
                        "moa: passthrough worker {} failed ({}ms): {}",
                        model,
                        elapsed,
                        e,
                    );
                    summaries.push(WorkerSummary {
                        model,
                        role: WorkerRole::Generalist,
                        succeeded: false,
                        elapsed_ms: elapsed,
                        output_kind: None,
                        confidence: None,
                    });
                }
                Err(e) => {
                    tracing::warn!("moa: passthrough task panicked: {e}");
                }
            }
        }

        let response_body =
            best_response.unwrap_or_else(|| error_response("All MoA workers failed"));

        // Clean up the passthrough response:
        // - Rewrite model field to "moa"
        // - Strip <think>...</think> tags from content
        let mut response_body = response_body;
        if let Some(obj) = response_body.as_object_mut() {
            obj.insert("model".to_string(), json!("moa"));
        }
        clean_passthrough_content(&mut response_body);

        self.session.record_assistant_response(&response_body);
        let outcome = extract_turn_outcome(&response_body);
        self.session.record_turn_outcome(&outcome);

        TurnResult {
            response_body,
            worker_summaries: summaries,
            reducer_used: false,
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
        let outcome = extract_turn_outcome(&response_body);
        self.session.record_turn_outcome(&outcome);

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
    ///
    /// Prefers the first endpoint — for mesh deployments this is typically the
    /// local model which has zero network RTT.  Remote models are fine as
    /// parallel workers but terrible as the sequential reducer because every
    /// relay hop adds latency to the critical path.
    fn pick_reducer_endpoint(&self) -> Endpoint {
        self.config
            .endpoints
            .first()
            .cloned()
            .unwrap_or_else(|| self.config.endpoints.last().unwrap().clone())
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

/// Extract a compact description of what this turn decided.
fn extract_turn_outcome(response: &Value) -> String {
    // Tool call?
    if let Some(tool_calls) = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|tc| tc.as_array())
    {
        if let Some(tc) = tool_calls.first() {
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or("?");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let short_args = if args.len() > 200 {
                format!("{}...", &args[..197])
            } else {
                args.to_string()
            };
            return format!("Called tool {name}({short_args})");
        }
    }

    // Text answer — take first 2-3 sentences or 400 chars, enough to
    // capture the substance of the response for future context.
    if let Some(content) = response
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
    {
        let truncated = if content.len() > 400 {
            format!("{}...", &content[..397])
        } else {
            content.to_string()
        };
        return format!("Answered: {truncated}");
    }

    String::new()
}

/// Discover models from an OpenAI-compatible `/v1/models` endpoint.
///
/// Deduplicates by display name so the same model under multiple aliases
/// (e.g. `unsloth/GLM-4.7-Flash-GGUF` and `unsloth/GLM-4.7-Flash-GGUF@main:Q4_K_M`)
/// only appears once.  Keeps the shorter ID as canonical.
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
    // Sort by ID length so shorter aliases win dedup
    entries.sort_by_key(|m| m["id"].as_str().unwrap_or("").len());

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut endpoints = Vec::new();

    for m in entries {
        let id = match m["id"].as_str() {
            Some(id) => id,
            None => continue,
        };
        if id.contains("cloud") || id == "moa" {
            continue;
        }
        let display = m["display_name"].as_str().unwrap_or(id).to_string();
        // Normalize: strip quant suffix, GGUF, org prefix
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

/// Clean up a passthrough response: strip think tags from content,
/// remove stray KV envelope lines (kind:/confidence:/payload:).
fn clean_passthrough_content(response: &mut Value) {
    if let Some(content) = response
        .pointer_mut("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    {
        let cleaned = normalize::strip_passthrough_content(&content);
        if let Some(msg) = response.pointer_mut("/choices/0/message") {
            msg.as_object_mut()
                .map(|o| o.insert("content".to_string(), json!(cleaned)));
        }
    }
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", t)
}
