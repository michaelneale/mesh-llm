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
/// send the HTTP response (JSON or SSE), and return `true` if the request
/// was handled. Returns `false` if the request is not for MoA, so the
/// caller can fall through to normal routing.
///
/// On MoA failure (e.g. <2 models in the mesh) sends a 503 and still
/// returns `true` — the caller must not also try to respond.
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

    run_moa_turn(tcp_stream, body_json, &config).await;
    None
}

/// Run a turn through the gateway and write the response with x-moa-* headers.
/// Caller has already validated the request and built the config.
async fn run_moa_turn(
    tcp_stream: TcpStream,
    body_json: serde_json::Value,
    config: &moa::GatewayConfig,
) {
    let was_streaming = body_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut moa_body = body_json;
    moa_body.as_object_mut().map(|o| o.remove("stream"));

    let moa_result = moa::handle_turn(config, &moa_body).await;
    let extra_headers = build_moa_headers(&moa_result);
    write_moa_response(tcp_stream, &moa_result, &extra_headers, was_streaming).await;
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
) {
    let body = &moa_result.response_body;
    let is_failure = is_moa_failure_body(body);
    let result = if was_streaming {
        // SSE always uses 200: once the headers are flushed we cannot
        // change the status code. The failure signal rides inside the
        // emitted SSE chunks (`finish_reason: "error"`) — see
        // `send_moa_as_sse` for the propagation.
        send_moa_as_sse(tcp_stream, body, extra_headers).await
    } else if is_failure {
        proxy::send_json_with_status_and_headers(tcp_stream, 502, body, extra_headers).await
    } else {
        proxy::send_json_ok_with_headers(tcp_stream, body, extra_headers).await
    };
    if let Err(e) = result {
        tracing::warn!(
            "MoA: response write failed ({}): {e}",
            if was_streaming { "SSE" } else { "JSON" }
        );
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
    let mut seen_bases = std::collections::HashSet::new();
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised
    // routable models). Sorted by name length so the shorter (canonical)
    // form wins dedup.
    let mut all_models: Vec<String> = node.models_being_served().await;
    all_models.sort_by_key(|n| n.len());

    for name in all_models {
        if !accept_for_dedup(&name, &mut seen_bases) {
            continue;
        }
        add_worker_backend(
            node,
            targets,
            &http,
            &name,
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
    })
}

/// Filter out the virtual `"mesh"` name and de-dup by canonical base.
/// Returns true if `name` is a fresh model that should be considered.
fn accept_for_dedup(name: &str, seen_bases: &mut std::collections::HashSet<String>) -> bool {
    if name == moa::VIRTUAL_MODEL_NAME {
        return false;
    }
    seen_bases.insert(canonical_base_name(name))
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
        // Strip CR/LF defensively against header-injection bugs creeping in later.
        let safe_value: String = value.chars().filter(|c| *c != '\r' && *c != '\n').collect();
        header.push_str(name);
        header.push_str(": ");
        header.push_str(&safe_value);
        header.push_str("\r\n");
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

    // Propagate the actual finish_reason rather than always emitting `stop`.
    // For failure-shaped bodies the inner `finish_reason` is `"error"` and
    // the body carries a top-level `error` object — SSE clients keyed on
    // `finish_reason` (Goose, OpenAI SDKs) need that signal to detect
    // failure. Previously the SSE adapter hard-coded `"stop"`, which made
    // MoA failures look like successful completions to streaming consumers.
    let inner_finish_reason = response
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str());
    let finish_reason: &str = match inner_finish_reason {
        Some("error") => "error",
        _ if tool_calls.is_some() => "tool_calls",
        Some(other) => other,
        None => "stop",
    };
    let is_failure = is_moa_failure_body(response);

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

    // For failure-shaped bodies, emit an explicit error chunk before the
    // final `finish_reason` chunk so SSE clients that scan deltas for an
    // `error` field can see the failure signal even if they ignore the
    // finish_reason itself. This mirrors the OpenAI-compatible shape used
    // by some upstream servers (an in-stream chunk carrying a structured
    // `error` payload).
    if is_failure {
        if let Some(err) = response.get("error") {
            let err_chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "error": err,
            });
            let data = format!("data: {}\n\n", err_chunk);
            let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
            stream.write_all(framed.as_bytes()).await?;
        }
    }

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
}
