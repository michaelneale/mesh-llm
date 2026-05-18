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

// ─── Sampling params ─────────────────────────────────────────────────

/// Sampling hyperparameters sent to backend models.
/// Workers get higher temperature for diversity; reducer gets lower for precision.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
}

impl SamplingParams {
    /// High-diversity settings for MoA workers — encourages each model
    /// to explore different parts of the solution space.
    pub fn worker() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
        }
    }

    /// Low-variance settings for the reducer — precise synthesis.
    pub fn reducer() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::reducer()
    }
}

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
        sampling: SamplingParams,
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
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
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
fn enforce_allowed_tools(
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

/// Pick the reducer — prefers first model (typically local, zero RTT).
/// Reducer candidates in priority order: big-tier models first (multi-
/// digit B, or names with no size like MiniMax), then small-tier models
/// as last-resort fallback. Callers should try each in order and stop
/// on the first that succeeds, so a broken big-tier peer (e.g. a peer
/// running a stale binary that 502s on tool calls) doesn't take down
/// the whole reducer step.
fn reducer_candidates(config: &GatewayConfig) -> Vec<(String, usize)> {
    let mut big = Vec::new();
    let mut small = Vec::new();
    for m in &config.models {
        let entry = (m.name.clone(), m.backend_index);
        if worker::is_single_digit_b_name(&m.name) {
            small.push(entry);
        } else {
            big.push(entry);
        }
    }
    big.extend(small);
    if big.is_empty() {
        big.push(("unknown".into(), 0));
    }
    big
}

/// Call the ordered reducer candidates with hedging.
///
/// Starts the first candidate immediately. If it hasn't returned within
/// `hedge_delay`, the next candidate is started in parallel without
/// cancelling the in-flight one — we race for the first OK. If a candidate
/// errors, the next one is started immediately (no hedge wait).
///
/// Returns the first successful (name, response_text). If every candidate
/// fails, returns the last error encountered.
///
/// Cost shape:
/// - Happy path (cand 0 OK in <hedge_delay): exactly 1 backend call.
/// - Slow happy path (cand 0 OK in hedge_delay..reducer_timeout): up to 2
///   overlapping calls, accept whichever wins, cancel the loser.
/// - Fast-fail (cand 0 errors quickly): immediate move to cand 1, 1 call.
/// - All fail: at most N calls, capped at reducer_timeout + (N-1)·hedge_delay
///   end-to-end (vs N·reducer_timeout sequentially).
async fn hedged_reducer_call(
    backends: &[std::sync::Arc<dyn ModelBackend>],
    candidates: Vec<(String, usize)>,
    messages: Vec<Value>,
    tools: Option<Value>,
    timeout: Duration,
    hedge_delay: Duration,
) -> Result<(String, String), String> {
    use tokio::task::JoinSet;

    if candidates.is_empty() {
        return Err("no reducer candidates".into());
    }

    let mut join_set: JoinSet<(String, Result<String, String>)> = JoinSet::new();
    let mut remaining = candidates.into_iter();
    let mut last_err: Option<String> = None;

    // Spawn a single candidate.
    fn spawn(
        join_set: &mut JoinSet<(String, Result<String, String>)>,
        backends: &[std::sync::Arc<dyn ModelBackend>],
        name: String,
        backend_idx: usize,
        messages: Vec<Value>,
        tools: Option<Value>,
        timeout: Duration,
    ) {
        let backend = backends[backend_idx].clone();
        tracing::info!("moa: reducer hedge → {name}");
        join_set.spawn(async move {
            let result = call_backend(
                &*backend,
                &name,
                &messages,
                tools.as_ref(),
                2048,
                timeout,
                SamplingParams::reducer(),
            )
            .await;
            (name, result)
        });
    }

    // Start candidate 0.
    if let Some((name, idx)) = remaining.next() {
        spawn(
            &mut join_set,
            backends,
            name,
            idx,
            messages.clone(),
            tools.clone(),
            timeout,
        );
    }

    // Race in-flight calls against a hedge timer.
    while !join_set.is_empty() {
        let hedge_sleep = tokio::time::sleep(hedge_delay);
        tokio::pin!(hedge_sleep);

        tokio::select! {
            // A candidate finished.
            joined = join_set.join_next() => {
                match joined {
                    Some(Ok((name, Ok(text)))) => {
                        // First success wins. Cancel the rest.
                        join_set.abort_all();
                        // Drain so cancellations complete cleanly.
                        while join_set.join_next().await.is_some() {}
                        return Ok((name, text));
                    }
                    Some(Ok((name, Err(e)))) => {
                        tracing::warn!(
                            "moa: reducer {name} failed: {e}, trying next candidate"
                        );
                        last_err = Some(e);
                        // Start the next candidate immediately on failure.
                        if let Some((next_name, next_idx)) = remaining.next() {
                            spawn(
                                &mut join_set,
                                backends,
                                next_name,
                                next_idx,
                                messages.clone(),
                                tools.clone(),
                                timeout,
                            );
                        }
                    }
                    Some(Err(join_err)) => {
                        tracing::warn!("moa: reducer task join error: {join_err}");
                        if let Some((next_name, next_idx)) = remaining.next() {
                            spawn(
                                &mut join_set,
                                backends,
                                next_name,
                                next_idx,
                                messages.clone(),
                                tools.clone(),
                                timeout,
                            );
                        }
                    }
                    None => break,
                }
            }
            // Hedge timer fires: start another candidate alongside in-flight ones.
            _ = &mut hedge_sleep => {
                if let Some((next_name, next_idx)) = remaining.next() {
                    spawn(
                        &mut join_set,
                        backends,
                        next_name,
                        next_idx,
                        messages.clone(),
                        tools.clone(),
                        timeout,
                    );
                }
                // If no more to start, just wait on the JoinSet without the
                // hedge timer racing again (next loop iteration's sleep will
                // simply never fire because we'll take the join branch).
            }
        }
    }

    Err(last_err.unwrap_or_else(|| "all reducer candidates failed".into()))
}

// ─── Backend call + text extraction ──────────────────────────────────

/// Call a backend and extract the assistant text from the response.
/// Retries once on HTTP 429 (rate limit) after the server's `retry-after`
/// delay (default 1s).
async fn call_backend(
    backend: &dyn ModelBackend,
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
    max_tokens: u32,
    timeout: Duration,
    sampling: SamplingParams,
) -> Result<String, String> {
    match backend
        .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
        .await
    {
        Ok(resp) => extract_text_from_response(&resp),
        Err(e) if e.contains("429") => {
            // Parse retry-after from error message if present, default 1s
            let delay = parse_retry_after(&e).unwrap_or(1);
            tracing::info!("moa: 429 from {model}, retrying after {delay}s");
            tokio::time::sleep(Duration::from_secs(delay)).await;
            let resp = backend
                .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
                .await?;
            extract_text_from_response(&resp)
        }
        Err(e) => Err(e),
    }
}

/// Extract retry-after seconds from an error message containing "retry-after: N".
fn parse_retry_after(err: &str) -> Option<u64> {
    let lower = err.to_lowercase();
    lower
        .find("retry-after:")
        .map(|i| &err[i + 12..])
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.trim().parse::<u64>().ok())
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
    allowed_tools: &[String],
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
                let mut normalized =
                    normalize::normalize_worker_output(&text, &model, role, elapsed);
                enforce_allowed_tools(&mut normalized, allowed_tools, &model);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_from_header() {
        let err = "HTTP 429: retry-after: 2\r\ncontent-type: application/json";
        assert_eq!(parse_retry_after(err), Some(2));
    }

    #[test]
    fn parse_retry_after_missing() {
        let err = "HTTP 429: Too Many Requests";
        assert_eq!(parse_retry_after(err), None);
    }

    #[test]
    fn parse_retry_after_case_insensitive() {
        let err = "Retry-After: 5";
        assert_eq!(parse_retry_after(err), Some(5));
    }

    #[test]
    fn worker_sampling_high_diversity() {
        let s = SamplingParams::worker();
        assert!(s.temperature > 0.5, "workers need high temp for diversity");
        assert!(s.top_p > 0.9, "workers need high top_p for diversity");
    }

    #[test]
    fn reducer_sampling_low_variance() {
        let s = SamplingParams::reducer();
        assert!(s.temperature <= 0.4, "reducer needs low temp for precision");
        assert!(s.top_p <= 0.95, "reducer needs bounded top_p");
    }

    #[test]
    fn default_sampling_is_reducer() {
        let d = SamplingParams::default();
        let r = SamplingParams::reducer();
        assert!((d.temperature - r.temperature).abs() < f32::EPSILON);
        assert!((d.top_p - r.top_p).abs() < f32::EPSILON);
    }

    // ── hedged_reducer_call ──────────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    enum FakeBehavior {
        OkAfter(Duration, String),
        ErrAfter(Duration, String),
    }

    struct FakeBackend {
        behaviors: std::sync::Mutex<std::collections::HashMap<String, FakeBehavior>>,
        calls: AtomicUsize,
    }

    impl FakeBackend {
        fn new(behaviors: Vec<(&str, FakeBehavior)>) -> std::sync::Arc<Self> {
            let mut map = std::collections::HashMap::new();
            for (n, b) in behaviors {
                map.insert(n.to_string(), b);
            }
            std::sync::Arc::new(FakeBackend {
                behaviors: std::sync::Mutex::new(map),
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ModelBackend for FakeBackend {
        async fn chat_completion(
            &self,
            model: &str,
            _messages: &[Value],
            _tools: Option<&Value>,
            _max_tokens: u32,
            _timeout: Duration,
            _sampling: SamplingParams,
        ) -> Result<Value, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let behavior = self.behaviors.lock().unwrap().get(model).cloned();
            match behavior {
                Some(FakeBehavior::OkAfter(d, body)) => {
                    tokio::time::sleep(d).await;
                    Ok(serde_json::json!({
                        "choices": [{"message": {"content": body}}],
                    }))
                }
                Some(FakeBehavior::ErrAfter(d, msg)) => {
                    tokio::time::sleep(d).await;
                    Err(msg)
                }
                None => Err(format!("unconfigured model: {model}")),
            }
        }
    }

    #[tokio::test]
    async fn hedged_reducer_happy_path_calls_only_first() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::OkAfter(Duration::from_millis(50), "alpha-resp".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(50), "beta-resp".into()),
            ),
        ]);
        let backends: Vec<std::sync::Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_secs(5),
        )
        .await;

        let (name, _) = res.expect("happy path returns Ok");
        assert_eq!(name, "alpha", "first candidate should win");
        assert_eq!(fake.calls(), 1, "only one backend call on happy path");
    }

    #[tokio::test]
    async fn hedged_reducer_slow_first_hedges_to_second() {
        let fake = FakeBackend::new(vec![
            // alpha takes longer than hedge_delay; beta is fast.
            (
                "alpha",
                FakeBehavior::OkAfter(Duration::from_millis(800), "alpha-late".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(100), "beta-fast".into()),
            ),
        ]);
        let backends: Vec<std::sync::Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(100),
        )
        .await;

        let (name, body) = res.expect("hedge returns Ok");
        assert_eq!(
            name, "beta",
            "hedge winner should be the faster second candidate"
        );
        assert_eq!(body, "beta-fast");
        assert_eq!(fake.calls(), 2, "both candidates should have been issued");
    }

    #[tokio::test]
    async fn hedged_reducer_fast_fail_starts_next_immediately() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::ErrAfter(Duration::from_millis(50), "boom".into()),
            ),
            (
                "beta",
                FakeBehavior::OkAfter(Duration::from_millis(100), "beta-ok".into()),
            ),
        ]);
        let backends: Vec<std::sync::Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let start = tokio::time::Instant::now();
        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            // Large hedge_delay — the fast-fail path must not wait for it.
            Duration::from_secs(60),
        )
        .await;

        let (name, body) = res.expect("fail-then-recover returns Ok");
        let elapsed = start.elapsed();
        assert_eq!(name, "beta");
        assert_eq!(body, "beta-ok");
        assert_eq!(fake.calls(), 2);
        assert!(
            elapsed < Duration::from_secs(10),
            "fast-fail should not wait for hedge_delay; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn hedged_reducer_all_fail_returns_last_err() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::ErrAfter(Duration::from_millis(10), "alpha-boom".into()),
            ),
            (
                "beta",
                FakeBehavior::ErrAfter(Duration::from_millis(10), "beta-boom".into()),
            ),
        ]);
        let backends: Vec<std::sync::Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(200),
        )
        .await;

        let err = res.expect_err("all-fail returns Err");
        assert!(
            err.contains("boom"),
            "should surface a backend error: {err}"
        );
        assert_eq!(fake.calls(), 2);
    }
}
