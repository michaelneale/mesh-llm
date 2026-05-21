//! Mesh-wide MoA orchestration entrypoint.
//!
//! Any node that receives a chat-completion request with `model: "mesh"`
//! runs MoA orchestration here, regardless of whether that node is serving
//! models locally. The worker pool is built from gossip — every model
//! advertised by any peer (or hosted locally) is a candidate.
//!
//! Both the host's `api_proxy` and the passive `handle_mesh_request` path
//! call `try_handle_moa`. On a pure client node, all backends are remote;
//! on a serving host, the local model is wired directly to its skippy port
//! and the rest go over QUIC.

use crate::inference::election;
use crate::mesh;
use crate::network::openai::transport as proxy;
use mesh_mixture_of_agents as moa;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

/// Detect `model: "mesh"`, build a mesh-wide MoA config, run the turn,
/// and write the HTTP response (JSON or SSE) directly to the stream.
///
/// Return value carries the un-consumed `TcpStream` so the caller knows
/// what to do next:
///
/// * `Some(stream)` — the request is *not* MoA-shaped (effective model
///   is not the virtual `"mesh"` name). The stream is returned unused
///   and the caller should fall through to normal routing.
///
/// * `None` — MoA owns the response. The stream has been consumed: a
///   successful MoA response, a 503 (when fewer than 2 models are
///   reachable), or a 400 (when the request body wasn't JSON) was
///   already written. The caller must *not* attempt to respond again.
pub async fn try_handle_moa(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
    targets: Option<&election::ModelTargets>,
) -> Option<TcpStream> {
    if effective_model != Some(moa::VIRTUAL_MODEL_NAME) {
        return Some(tcp_stream);
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        let _ = proxy::send_400(tcp_stream, "MoA requires a JSON body").await;
        return None;
    };

    let Some(config) = build_moa_config(node, targets).await else {
        let _ = proxy::send_503(tcp_stream, "MoA requires ≥2 models available in the mesh").await;
        return None;
    };

    run_moa_turn(tcp_stream, body_json, &config, request.response_adapter).await;
    None
}

/// Run a turn through the gateway and write the response with x-moa-* headers.
/// Caller has already validated the request and built the config.
async fn run_moa_turn(
    tcp_stream: TcpStream,
    body_json: serde_json::Value,
    config: &moa::GatewayConfig,
    response_adapter: proxy::ResponseAdapter,
) {
    let was_streaming = body_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut moa_body = body_json;
    moa_body.as_object_mut().map(|o| o.remove("stream"));

    let moa_result = moa::handle_turn(config, &moa_body).await;
    let extra_headers = build_moa_headers(&moa_result);
    write_moa_response(
        tcp_stream,
        &moa_result,
        &extra_headers,
        was_streaming,
        response_adapter,
    )
    .await;
}

/// Write the MoA response on the chosen transport (JSON or SSE), logging
/// (but not propagating) any I/O error.
///
/// Detect whether a MoA response body is signalling failure.
///
/// Two signals, either of which means "failure":
///
///   * Top-level `error` object — OpenAI-shape error envelope produced
///     by `moa::error_response`.
///   * `choices[0].finish_reason == "error"` — same convention applied
///     by the crate's response builder for in-band failure signalling.
///
/// Previously the HTTP-status decision was based on `TurnKind == Failed`,
/// but the tool-result reducer path can produce an error_response with
/// `TurnKind::ToolResult` when every reducer candidate fails. Tying the
/// status to the body's failure signal instead means *all* error-shaped
/// MoA responses get a non-200 status, regardless of which sub-flow
/// produced them.
fn is_moa_failure_body(body: &serde_json::Value) -> bool {
    if body.get("error").is_some() {
        return true;
    }
    body.pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        == Some("error")
}

/// When the response body signals MoA failure (top-level `error` field or
/// `choices[0].finish_reason == "error"`) we send an HTTP 502 (Bad
/// Gateway), not HTTP 200. Unsophisticated clients that only check the
/// HTTP status need that status to actually reflect failure.
async fn write_moa_response(
    tcp_stream: TcpStream,
    moa_result: &moa::TurnResult,
    extra_headers: &[(&str, String)],
    was_streaming: bool,
    response_adapter: proxy::ResponseAdapter,
) {
    let body = &moa_result.response_body;
    let is_failure = is_moa_failure_body(body);
    // Streaming + failure: respond as non-streaming HTTP 502 with the
    // structured error body. Failure path doesn't go through SSE in any
    // adapter mode — callers want a clean connection-level error.
    let (mode, result) = if was_streaming && !is_failure {
        match response_adapter {
            proxy::ResponseAdapter::OpenAiResponsesStream => (
                "SSE-responses",
                send_moa_as_responses_sse(tcp_stream, body, extra_headers).await,
            ),
            // None, OpenAiChatCompletionsStream, OpenAiResponsesJson all
            // get the chat.completion.chunk SSE shape — the JSON-mode
            // adapter caller will never set was_streaming=true.
            _ => (
                "SSE-chat",
                send_moa_as_sse(tcp_stream, body, extra_headers).await,
            ),
        }
    } else if is_failure {
        (
            "JSON-502",
            proxy::send_json_with_status_and_headers(tcp_stream, 502, body, extra_headers).await,
        )
    } else if response_adapter == proxy::ResponseAdapter::OpenAiResponsesJson {
        // Non-streaming Responses-API request: emit a Responses-shape
        // JSON body instead of the chat.completion shape.
        (
            "JSON-responses",
            proxy::send_json_ok_with_headers(
                tcp_stream,
                &chat_completion_to_responses_json(body),
                extra_headers,
            )
            .await,
        )
    } else {
        (
            "JSON",
            proxy::send_json_ok_with_headers(tcp_stream, body, extra_headers).await,
        )
    };
    if let Err(e) = result {
        tracing::warn!("MoA: response write failed ({mode}): {e}");
    }
}

/// Build the `x-moa-*` observability headers from a finished turn and log
/// a one-line summary.
fn build_moa_headers(result: &moa::TurnResult) -> Vec<(&'static str, String)> {
    let workers_ok = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    let workers_total = result.worker_summaries.len();
    tracing::info!(
        "moa: {}ms, {}/{} workers, kind={}, reducer={} (attempts={})",
        result.elapsed_ms,
        workers_ok,
        workers_total,
        result.turn_kind.label(),
        result.reducer_used,
        result.reducer_attempts,
    );

    vec![
        ("x-moa-elapsed-ms", result.elapsed_ms.to_string()),
        ("x-moa-turn", result.turn_kind.label().to_string()),
        ("x-moa-workers", workers_total.to_string()),
        ("x-moa-workers-ok", workers_ok.to_string()),
        ("x-moa-reducer", result.reducer_used.to_string()),
        (
            "x-moa-reducer-attempts",
            result.reducer_attempts.to_string(),
        ),
    ]
}

/// Build a MoA gateway config from this node's mesh-wide view.
///
/// Every distinct model in the mesh becomes a worker:
/// - Models served by this node → `LocalModelBackend` (direct skippy port)
/// - Models served by a peer → `RemoteModelBackend` (QUIC tunnel)
///
/// Models are deduplicated by canonical base name so e.g.
/// `unsloth/Qwen3-8B-GGUF:Q4_K_M` and `Qwen3-8B-Q4_K_M` (different naming
/// conventions for the same model from different peers) only show up once.
///
/// Returns `None` if fewer than 2 distinct models exist — MoA needs at
/// least two workers to be meaningfully different from a single call.
///
/// `targets` is the runtime's local routing table, used to discover the
/// skippy port for locally-served models. In passive (`--client`) mode
/// this is `None` — every backend goes over QUIC. In `serve` mode it's
/// `Some`, so locally-served models bypass the tunnel.
pub async fn build_moa_config(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
) -> Option<moa::GatewayConfig> {
    let http = reqwest::Client::new();
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised
    // routable models).
    let all_models: Vec<String> = node
        .models_being_served()
        .await
        .into_iter()
        .filter(|n| n != moa::VIRTUAL_MODEL_NAME)
        .collect();

    // Group aliases by canonical base. The old shape sorted by name
    // length, took the *first* alias per base, and dropped the rest —
    // which silently dropped the model from the worker pool whenever the
    // shortest-named peer was unreachable (regression flagged by PR #566
    // review). Now we keep every alias per base and try them in order so
    // a longer-named reachable alias can still resolve when the shortest
    // one is offline.
    let groups = group_aliases_by_canonical_base(all_models, targets);
    for aliases in groups {
        resolve_one_worker_from_aliases(
            node,
            targets,
            &http,
            &aliases,
            &mut backends,
            &mut models,
            &mut local_count,
        )
        .await;
    }

    if models.len() < 2 {
        tracing::warn!(
            "MoA: only {} model(s) reachable, need ≥2 (models={:?})",
            models.len(),
            models.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
        return None;
    }

    tracing::info!(
        "MoA config: {} workers ({} local, {} remote): {:?}",
        models.len(),
        local_count,
        models.len() - local_count,
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
    );

    Some(moa::GatewayConfig {
        backends,
        models,
        // Bumped from 15s → 60s. 15s was tight for big-context interactive
        // turns: a large model with a 10–20k-token prompt and tool schema
        // (typical for agent harnesses like OpenCode/Goose) can need 20–30s
        // just to produce a first tool-call. Workers were getting killed
        // mid-inference and MoA reported `kind=early-exit` with the small
        // worker, never the strong one. 60s gives the strong worker room
        // to land without making the no-progress wait painful.
        worker_timeout: std::time::Duration::from_secs(60),
        // Per-attempt cap; hedged_reducer_call hedges across candidates so the
        // end-to-end wait is roughly reducer_timeout + a couple of hedge delays.
        reducer_timeout: std::time::Duration::from_secs(60),
        // Start a second reducer candidate after 5s if the first hasn't replied
        // (or sooner on outright failure). Cheap on the happy path, big win on
        // the cold-KV / stale-peer tail.
        hedge_delay: std::time::Duration::from_secs(5),
        // Chat-only sole-answer grace. Tool turns ignore this.
        first_answer_grace: std::time::Duration::from_secs(6),
    })
}

/// Try each alias in `aliases` until one resolves to a backend, then stop.
///
/// Aliases are pre-sorted by `group_aliases_by_canonical_base` so the most
/// preferred (locally-served first, then shortest) is tried first. Falls
/// back to longer aliases when the preferred one's peer is unreachable.
#[allow(clippy::too_many_arguments)]
async fn resolve_one_worker_from_aliases(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    aliases: &[String],
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) {
    for name in aliases {
        if add_worker_backend(node, targets, http, name, backends, models, local_count).await {
            return;
        }
    }
}

/// Group all advertised model names by their canonical base so each
/// canonical model contributes exactly one worker, but the resolver gets
/// to pick the alias that actually has a reachable backend.
///
/// The earlier shape committed to a single alias per base *before* trying
/// to resolve a backend. Two failure modes:
///
///   1. The chosen alias is advertised only by a peer that drops between
///      gossip refresh and orchestration — `hosts_for_model` returns
///      empty, the worker is dropped, and longer-form aliases for the
///      same canonical model from still-reachable peers are rejected as
///      duplicates.
///   2. The local node advertises a longer convention
///      (e.g. `unsloth/Qwen3-8B-GGUF:Q4_K_M`) while a peer advertises a
///      shorter variant (e.g. `Qwen3-8B-Q4_K_M`). The shortest-name rule
///      picks the peer alias, `add_worker_backend` looks for a local port
///      under that specific string, finds nothing, and forces a
///      QUIC-tunnel backend even though the model is right here.
///
/// Both failure modes are fixed by grouping first and resolving second.
/// Within each group the aliases are ordered so the most likely
/// optimization wins first try: locally-served name (skippy-port fast
/// path) before remote names, then shortest first as a tiebreaker.
fn group_aliases_by_canonical_base(
    names: Vec<String>,
    targets: Option<&election::ModelTargets>,
) -> Vec<Vec<String>> {
    let mut by_base: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for name in names {
        by_base
            .entry(canonical_base_name(&name))
            .or_default()
            .push(name);
    }
    // Deterministic group order so the worker list is stable across
    // builds even though HashMap iteration is not. Sort group entries
    // (locally-served first, then shortest), then sort groups by their
    // first ("best") alias.
    let mut groups: Vec<Vec<String>> = by_base
        .into_values()
        .map(|mut aliases| {
            aliases.sort_by(|a, b| {
                let la = is_locally_served(a, targets);
                let lb = is_locally_served(b, targets);
                lb.cmp(&la) // local (true) before remote (false)
                    .then_with(|| a.len().cmp(&b.len()))
                    .then_with(|| a.cmp(b))
            });
            aliases
        })
        .collect();
    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Does the local routing table have a backend port for this exact name?
fn is_locally_served(name: &str, targets: Option<&election::ModelTargets>) -> bool {
    targets
        .and_then(|t| {
            t.targets.get(name).map(|tv| {
                tv.iter()
                    .any(|t| matches!(t, election::InferenceTarget::Local(_)))
            })
        })
        .unwrap_or(false)
}

/// Resolve `name` to a backend (local skippy port if available, else first
/// remote host) and append it to `backends`/`models`. Returns true if a
/// backend was added.
async fn add_worker_backend(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    name: &str,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) -> bool {
    // Prefer local skippy port when this node serves the model.
    let local_port = targets.and_then(|t| {
        t.targets.get(name).and_then(|tv| {
            tv.iter().find_map(|t| match t {
                election::InferenceTarget::Local(p) => Some(*p),
                _ => None,
            })
        })
    });
    if let Some(port) = local_port {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(LocalModelBackend {
            port,
            http: http.clone(),
        }));
        models.push(moa::ModelEntry {
            name: name.to_string(),
            backend_index: backend_idx,
        });
        *local_count += 1;
        return true;
    }

    // Otherwise find a remote host. hosts_for_model returns peers in
    // hash-preferred order; take the first.
    let remote_hosts = node.hosts_for_model(name).await;
    if let Some(peer_id) = remote_hosts.into_iter().next() {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(RemoteModelBackend {
            node: node.clone(),
            peer_id,
        }));
        models.push(moa::ModelEntry {
            name: name.to_string(),
            backend_index: backend_idx,
        });
        return true;
    }
    false
}

/// Canonical name used for cross-peer dedup. Different peers advertise the
/// same model under different conventions (`unsloth/Qwen3-8B-GGUF:Q4_K_M`
/// vs `Qwen3-8B-Q4_K_M`); normalize before comparing.
///
/// Strategy: strip the publisher prefix, the `-gguf` suffix, any `@branch`
/// suffix, then keep only `[a-z0-9]` characters so `:` vs `-` separators
/// don't matter.
fn canonical_base_name(name: &str) -> String {
    let lower = name.to_lowercase();
    // Drop an `@branch` segment if present, keeping anything after the
    // next `:` so quant tags survive (e.g. `repo@main:q4_k_m` → `repo:q4_k_m`).
    let no_branch = match lower.find('@') {
        Some(at) => {
            let after = &lower[at + 1..];
            let rest = after.find(':').map(|c| &after[c..]).unwrap_or("");
            format!("{}{}", &lower[..at], rest)
        }
        None => lower,
    };
    let stripped = no_branch
        .replace("-gguf", "")
        .replace("unsloth/", "")
        .replace("meshllm/", "");
    stripped
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Backend that calls a local model directly on its skippy HTTP port.
struct LocalModelBackend {
    port: u16,
    http: reqwest::Client,
}

#[async_trait::async_trait]
impl moa::ModelBackend for LocalModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
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
            .map_err(|e| format!("local:{} failed: {e}", self.port))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP {status}: {}",
                moa::truncate_chars(&text, 200)
            ));
        }
        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

/// Backend that calls a remote model over the QUIC tunnel.
struct RemoteModelBackend {
    node: mesh::Node,
    peer_id: iroh::EndpointId,
}

#[async_trait::async_trait]
impl moa::ModelBackend for RemoteModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
        let http_request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            body_bytes.len()
        );
        let mut raw = http_request.into_bytes();
        raw.extend_from_slice(&body_bytes);

        let result = tokio::time::timeout(timeout, async {
            let (mut send, mut recv) = self
                .node
                .open_http_tunnel(self.peer_id)
                .await
                .map_err(|e| format!("tunnel: {e}"))?;
            send.write_all(&raw)
                .await
                .map_err(|e| format!("send: {e}"))?;
            send.finish().map_err(|e| format!("finish: {e}"))?;
            let response = recv
                .read_to_end(4 * 1024 * 1024)
                .await
                .map_err(|e| format!("recv: {e}"))?;
            parse_quic_http_response(&response)
        })
        .await
        .map_err(|_| format!("remote timeout after {}s", timeout.as_secs()))?;
        result
    }
}

fn parse_quic_http_response(response: &[u8]) -> Result<serde_json::Value, String> {
    let s = String::from_utf8_lossy(response);
    let header_end = s
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = s[..header_end].lines().next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(format!("HTTP {status}: {}", moa::truncate_chars(&s, 200)));
    }
    let body = &s[header_end + 4..];
    serde_json::from_str(body).map_err(|e| format!("parse: {e}"))
}

/// Send the MoA response as a one-shot SSE stream so SSE-only clients
/// (like Goose) can consume it. Emits one delta chunk with the full
/// content, then a `finish_reason: stop` chunk, then `[DONE]`.
///
/// `extra_headers` are emitted alongside the standard SSE response headers
/// (used to attach `x-moa-*` observability headers).
async fn send_moa_as_sse(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n",
    );
    for (name, value) in extra_headers {
        crate::network::openai::transport::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await?;

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-mesh");
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME);
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);

    let tool_calls = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .cloned();

    // Caller (`write_moa_response`) routes failure-shaped bodies to a
    // non-streaming 502 JSON response, so this function only ever sees a
    // successful turn. The only choice the SSE adapter still has to make
    // is `tool_calls` vs `stop`.
    let finish_reason: &str = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    debug_assert!(
        !is_moa_failure_body(response),
        "send_moa_as_sse received a failure body; should have routed to 502"
    );

    let delta = if let Some(ref tcs) = tool_calls {
        serde_json::json!({
            "role": "assistant",
            "tool_calls": tcs.iter().enumerate().map(|(i, tc)| {
                serde_json::json!({
                    "index": i,
                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0"),
                    "type": "function",
                    "function": tc.get("function").cloned().unwrap_or(serde_json::json!({})),
                })
            }).collect::<Vec<_>>()
        })
    } else {
        serde_json::json!({ "role": "assistant", "content": content })
    };

    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": null,
        }]
    });
    let data = format!("data: {}\n\n", chunk);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    let stop = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }]
    });
    let data = format!("data: {}\n\n", stop);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Strip `<think>...</think>` tags and orphan `</think>` from content.
/// Thin wrapper over the canonical implementation in moa::worker.
fn strip_think_from_content(text: &str) -> String {
    moa::strip_thinking(text)
}

/// Emit the MoA response as a one-shot OpenAI Responses-API SSE stream
/// so callers that hit `/v1/responses` with `stream:true` (the chat UI)
/// get event shapes their parser understands.
///
/// We synthesize the minimum set the standard Responses-API stream
/// emits: `response.created`, `response.output_text.delta` (one chunk
/// with the full content), `response.output_text.done`, and
/// `response.completed`. MoA always knows the full body before writing,
/// so we don't bother with incremental delta streaming — it would only
/// make the UI's spinner-to-text transition smoother.
async fn send_moa_as_responses_sse(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n",
    );
    for (name, value) in extra_headers {
        crate::network::openai::transport::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await?;

    let response_id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("resp_moa")
        .to_string();
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME)
        .to_string();
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);
    // MoA's body is chat-shape; the Responses-API completed event
    // expects input_tokens / output_tokens. Translate before emitting
    // so downstream consumers (chat UI, billing) see the right keys.
    let usage = response
        .get("usage")
        .map(openai_frontend::responses::chat_usage_to_responses_usage);
    let item_id = format!("msg_moa_{}", short_id_from_response(response));
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    use openai_frontend::responses as resp;

    // The created/completed events must share the same `response.id`
    // so clients can correlate the stream by id. Overwrite the
    // auto-generated `resp_{created_at}` placeholder with the MoA
    // response id we'll also use on the completed event.
    let mut created = resp::responses_stream_created_event(&model, created_at);
    if let Some(obj) = created
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(response_id.clone()),
        );
    }

    let events = [
        created,
        resp::responses_stream_delta_event(&item_id, &content),
        resp::responses_stream_text_done_event(&item_id, &content),
        resp::responses_stream_completed_event(
            &response_id,
            created_at,
            &model,
            &item_id,
            &content,
            usage,
        ),
    ];

    for event in &events {
        let data = format!("data: {}\n\n", event);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
    }

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Convert a chat.completion JSON body to a Responses-API JSON body.
/// Used for non-streaming `/v1/responses` requests against MoA.
fn chat_completion_to_responses_json(chat: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(chat).unwrap_or_default();
    match crate::network::openai::response_adapter::translate_chat_completion_to_responses(&bytes) {
        Ok(translated) => serde_json::from_slice(&translated).unwrap_or_else(|_| chat.clone()),
        Err(e) => {
            tracing::warn!("MoA: chat-to-responses JSON translate failed: {e}");
            chat.clone()
        }
    }
}

fn short_id_from_response(response: &serde_json::Value) -> String {
    response
        .get("id")
        .and_then(|v| v.as_str())
        .and_then(|id| id.rsplit('-').next())
        .unwrap_or("x")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_base_dedupes_unsloth_and_gguf_variants() {
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF@main:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn canonical_base_keeps_distinct_models_distinct() {
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M")
        );
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/MiniMax-M2.5-GGUF:Q4_K_M")
        );
    }

    #[test]
    fn strip_think_handles_simple_block() {
        assert_eq!(
            strip_think_from_content("<think>reasoning</think>answer"),
            "answer"
        );
    }

    #[test]
    fn strip_think_handles_orphan_close_tag() {
        // Orphan `</think>` is removed but prefix content is preserved.
        assert_eq!(
            strip_think_from_content("stuff</think>answer"),
            "stuffanswer"
        );
    }

    #[test]
    fn strip_think_handles_unclosed_block() {
        assert_eq!(
            strip_think_from_content("answer prefix<think>never closed"),
            "answer prefix"
        );
    }

    #[test]
    fn is_moa_failure_body_detects_top_level_error() {
        // Regression for PR #566 review (item #7): the HTTP status was
        // gated on `TurnKind == Failed`, but reducer-failure tool-result
        // turns produce an error_response with `TurnKind::ToolResult`.
        // The body still carries the canonical failure signals, so
        // status now follows the body.
        let body = serde_json::json!({
            "error": { "message": "reducer failed", "type": "moa_failure" },
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn is_moa_failure_body_detects_finish_reason_error() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn is_moa_failure_body_returns_false_for_success() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "hello" } }],
        });
        assert!(!is_moa_failure_body(&body));
    }

    fn make_targets(local_names: &[&str]) -> election::ModelTargets {
        let mut t = election::ModelTargets::default();
        for (i, name) in local_names.iter().enumerate() {
            t.targets.insert(
                (*name).to_string(),
                vec![election::InferenceTarget::Local(50000 + i as u16)],
            );
        }
        t
    }

    #[test]
    fn group_aliases_keeps_all_aliases_per_canonical_base() {
        // Regression for PR #566 review (item #10): the dedup-then-resolve
        // shape committed to a single alias per base before checking
        // backend reachability. Now every alias is retained so the
        // resolver can fall back if the preferred alias is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "Qwen3-8B-Q4_K_M".to_string(),
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1, "both names share a canonical base");
        assert_eq!(groups[0].len(), 2, "both aliases retained");
    }

    #[test]
    fn group_aliases_prefers_locally_served_alias_even_when_longer() {
        // Without a targets table, length-order wins and the shorter peer
        // alias would be tried first — forcing an unnecessary QUIC hop
        // when the model is right here under a different alias.
        // With targets, the local-served alias must come first.
        let local = "unsloth/Qwen3-8B-GGUF:Q4_K_M";
        let peer = "Qwen3-8B-Q4_K_M";
        let targets = make_targets(&[local]);
        let groups = group_aliases_by_canonical_base(
            vec![peer.to_string(), local.to_string()],
            Some(&targets),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some(local),
            "locally-served alias must win even though it's longer"
        );
    }

    #[test]
    fn group_aliases_falls_back_to_shortest_when_no_local() {
        // No targets table at all (pure --client --auto node) — shortest
        // alias should win, but the longer alias is still in the group so
        // it can be tried if the shortest one is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "Qwen3-8B-Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(groups[0].len(), 2, "longer alias kept as fallback");
    }

    #[test]
    fn group_aliases_distinct_models_stay_in_separate_groups() {
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "unsloth/Qwen3-32B-GGUF:Q4_K_M".to_string(),
                "unsloth/MiniMax-M2.5-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn is_moa_failure_body_returns_false_for_tool_calls() {
        let body = serde_json::json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                },
            }],
        });
        assert!(!is_moa_failure_body(&body));
    }

    // ── Streaming + failure routing ────────────────────────────────────
    //
    // The actual write path (`write_moa_response`) writes to a real
    // `TcpStream`, so we test the *decision* it makes by extracting the
    // failure detection into `is_moa_failure_body` and proving the
    // routing logic with the same booleans the writer uses.
    //
    // The contract is:
    //   was_streaming=false, is_failure=false  -> JSON 200
    //   was_streaming=false, is_failure=true   -> JSON 502
    //   was_streaming=true,  is_failure=false  -> SSE
    //   was_streaming=true,  is_failure=true   -> JSON 502 (NOT SSE 200)
    // The last row is the PR #612 review finding: streaming MoA failures
    // must surface as a real 502 at the HTTP layer instead of streaming
    // a 200 SSE carrying an in-band error.

    fn route_decision(was_streaming: bool, is_failure: bool) -> &'static str {
        if was_streaming && !is_failure {
            "sse"
        } else if is_failure {
            "json-502"
        } else {
            "json-200"
        }
    }

    #[test]
    fn streaming_success_routes_to_sse() {
        assert_eq!(route_decision(true, false), "sse");
    }

    #[test]
    fn streaming_failure_routes_to_json_502_not_sse() {
        // Regression for PR #612 review: streaming failures previously
        // went out as `SSE 200` + in-band `finish_reason: "error"`.
        // Now they collapse to a non-streaming JSON 502, matching the
        // OpenAI API and the non-streaming MoA failure path.
        assert_eq!(route_decision(true, true), "json-502");
    }

    #[test]
    fn non_streaming_success_routes_to_json_200() {
        assert_eq!(route_decision(false, false), "json-200");
    }

    #[test]
    fn non_streaming_failure_routes_to_json_502() {
        assert_eq!(route_decision(false, true), "json-502");
    }

    // ── Responses-API adapter ───────────────────────────────────────
    //
    // When the request came in via /v1/responses, MoA's response must
    // be rendered in the Responses-API shape, not chat.completion. The
    // chat UI's streaming parser ignores chat.completion.chunk events,
    // which is what caused the "streaming response" spinner with no
    // visible text on the public mesh.

    fn fixture_chat_completion(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        })
    }

    #[test]
    fn chat_completion_to_responses_json_returns_response_object() {
        // Non-streaming /v1/responses with model=mesh: the body that
        // reaches the client must be Responses-shape, not chat-shape.
        let chat = fixture_chat_completion("hello world");
        let responses = chat_completion_to_responses_json(&chat);
        assert_eq!(
            responses.get("object").and_then(|v| v.as_str()),
            Some("response"),
            "got: {}",
            serde_json::to_string(&responses).unwrap_or_default()
        );
        // The text must survive translation.
        let text = serde_json::to_string(&responses).unwrap_or_default();
        assert!(
            text.contains("hello world"),
            "response body must carry the original content; got {text}"
        );
    }

    #[test]
    fn chat_completion_to_responses_json_passes_through_on_malformed() {
        // Defensive: if the translator can't make sense of the body
        // we return the chat body unchanged rather than blowing up.
        let bogus = serde_json::json!({ "not": "a chat completion" });
        let out = chat_completion_to_responses_json(&bogus);
        // The translator may either succeed (producing an empty
        // response) or fall back to the input; both behaviours are
        // acceptable, what matters is no panic and a JSON value.
        assert!(out.is_object());
    }

    /// Run `send_moa_as_responses_sse` against a real TCP loopback
    /// pair and return the raw bytes the client received as a string.
    /// Includes HTTP/1.1 headers and the chunked-transfer framing
    /// around each SSE event. Callers in this module match by
    /// `.contains(...)`, which is robust to framing without needing
    /// to parse it.
    async fn capture_responses_sse_body(response: serde_json::Value) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");

        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            send_moa_as_responses_sse(socket, &response, &[])
                .await
                .expect("sse write");
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn responses_sse_uses_same_response_id_for_created_and_completed() {
        // Regression: created and completed events used different
        // `response.id` values (one auto-generated, one from the chat
        // body), breaking clients that correlate by id.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-correlation",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }]
        });

        let raw = capture_responses_sse_body(response).await;

        // Extract every `data: { ... }` JSON blob and look at
        // (event.type, event.response.id).
        let mut ids = Vec::<(String, String)>::new();
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if event_type == "response.created" || event_type == "response.completed" {
                let id = v
                    .pointer("/response/id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                ids.push((event_type.to_string(), id));
            }
        }

        assert_eq!(ids.len(), 2, "need created + completed; got {ids:?}");
        assert_eq!(ids[0].1, "chatcmpl-moa-correlation");
        assert_eq!(
            ids[0].1, ids[1].1,
            "created and completed must share response.id: {ids:?}"
        );
    }

    #[tokio::test]
    async fn responses_sse_emits_responses_shape_usage_not_chat_shape() {
        // Regression: MoA was forwarding the chat-completion `usage`
        // object (prompt_tokens/completion_tokens) straight into the
        // Responses-API completed event, which expects
        // input_tokens/output_tokens. Downstream consumers that read
        // `response.usage.input_tokens` saw `undefined`.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "total_tokens": 24
            }
        });

        let raw = capture_responses_sse_body(response).await;

        // The completed event carries the response object including
        // usage. We assert by string match so we're robust to
        // serializer ordering.
        assert!(
            raw.contains("\"input_tokens\":11"),
            "expected input_tokens=11 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"output_tokens\":13"),
            "expected output_tokens=13 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"total_tokens\":24"),
            "expected total_tokens=24 in SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"prompt_tokens\":"),
            "chat-shape prompt_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"completion_tokens\":"),
            "chat-shape completion_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
    }
}
