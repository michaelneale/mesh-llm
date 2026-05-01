//! Peer consultation — ask another model in the mesh for help.
//!
//! This is the core mechanism behind the virtual LLM engine. When a hook
//! fires and decides to consult another model, it calls into this module
//! to find a suitable peer and send it a request over the mesh's QUIC
//! transport.
//!
//! Three consultation patterns:
//!
//! - **Caption** — send an image to a vision-capable peer, get a text description
//! - **Summarize** — send conversation history, get a condensed summary
//! - **Second opinion** — send the same question to a different model, get its answer

use crate::mesh;
use anyhow::Result;
use iroh::EndpointId;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Peer discovery
// ---------------------------------------------------------------------------

/// Find a peer that can handle vision (images).
/// Returns None if no vision-capable peer exists in the mesh.
pub async fn find_vision_peer(node: &mesh::Node, exclude_model: &str) -> Option<EndpointId> {
    let peers = node.peers().await;
    peers
        .iter()
        .filter(|p| {
            p.served_model_descriptors.iter().any(|d| {
                d.capabilities.supports_vision_runtime() && d.identity.model_name != exclude_model
            })
        })
        .min_by_key(|p| p.rtt_ms.unwrap_or(u32::MAX))
        .map(|p| p.id)
}

/// Find a peer that can handle audio.
/// Returns None if no audio-capable peer exists in the mesh.
pub async fn find_audio_peer(node: &mesh::Node, exclude_model: &str) -> Option<EndpointId> {
    let peers = node.peers().await;
    peers
        .iter()
        .filter(|p| {
            p.served_model_descriptors.iter().any(|d| {
                d.capabilities.supports_audio_runtime() && d.identity.model_name != exclude_model
            })
        })
        .min_by_key(|p| p.rtt_ms.unwrap_or(u32::MAX))
        .map(|p| p.id)
}

/// Find up to `n` peers serving a *different* model from the current one,
/// ranked by score (best first).
///
/// Picks peers running a different model for diversity. Prefers reasoning-capable
/// models, then lower RTT. Deduplicates by model name — two nodes running the
/// same model don't give diversity, just redundancy.
pub async fn find_different_model_peers(
    node: &mesh::Node,
    current_model: &str,
    n: usize,
) -> Vec<(EndpointId, String)> {
    use crate::models::CapabilityLevel;

    let peers = node.peers().await;

    let mut candidates: Vec<_> = peers
        .iter()
        .filter_map(|p| {
            let different = p.served_model_descriptors.iter().find(|d| {
                d.identity.model_name != current_model && !d.identity.model_name.is_empty()
            });
            different.map(|d| {
                let rtt = p.rtt_ms.unwrap_or(500);
                let has_reasoning = d.capabilities.reasoning != CapabilityLevel::None;
                // Sort key: reasoning models first (0), then non-reasoning (1), then RTT
                let score = if has_reasoning { rtt } else { 10_000 + rtt };
                (p.id, d.identity.model_name.clone(), score)
            })
        })
        .collect();

    candidates.sort_by_key(|(_, _, score)| *score);
    // Deduplicate by model name — keep the best-scored peer for each model.
    let mut seen_models = std::collections::HashSet::new();
    candidates.retain(|(_, model, _)| seen_models.insert(model.clone()));
    candidates.truncate(n);
    candidates.into_iter().map(|(id, m, _)| (id, m)).collect()
}

// ---------------------------------------------------------------------------
// Consultation requests
// ---------------------------------------------------------------------------

/// Consultation timeout — 20s for all hooks. Triggers are rare enough that
/// a pause is acceptable, and mesh peers often need 6-10s to respond.
pub const TIMEOUT_CONSULTATION: std::time::Duration = std::time::Duration::from_secs(20);

/// Send a chat completion request to a peer over the mesh QUIC tunnel.
/// Returns the assistant message content, or an error.
pub async fn chat_completion(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
    timeout: std::time::Duration,
) -> Result<String> {
    match tokio::time::timeout(
        timeout,
        chat_completion_inner(node, peer_id, model, messages, max_tokens),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => anyhow::bail!("consultation timed out after {}s", timeout.as_secs()),
    }
}

async fn chat_completion_inner(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
) -> Result<String> {
    let request_body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": false,
        // Disable hooks on the peer — prevent recursive consultation loops.
        // Without this, the peer could consult another peer about our request,
        // which could consult another, etc.
        "mesh_hooks": false,
    });
    let body_bytes = serde_json::to_vec(&request_body)?;

    // Build a minimal HTTP request
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

    // Open QUIC tunnel to peer and send request
    let (mut send, mut recv) = node.open_http_tunnel(peer_id).await?;
    send.write_all(&raw).await?;
    send.finish()?;

    // Read the full HTTP response
    let response_bytes = recv.read_to_end(64 * 1024).await?;
    let response_str = String::from_utf8_lossy(&response_bytes);

    // Parse HTTP status line
    let header_end = response_str
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP response: no header terminator"))?;
    let headers = &response_str[..header_end];
    let status_line = headers.lines().next().unwrap_or("");
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status_code != 200 {
        anyhow::bail!(
            "peer returned HTTP {status_code}: {}",
            &response_str[..response_str.len().min(200)]
        );
    }

    let body = &response_str[header_end + 4..];
    let parsed: Value = serde_json::from_str(body).map_err(|e| {
        anyhow::anyhow!(
            "failed to parse peer response body: {e}\nraw: {}",
            &body[..body.len().min(200)]
        )
    })?;

    // Extract the assistant message content
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if content.is_empty() {
        anyhow::bail!("peer returned empty response");
    }

    Ok(content)
}

// ---------------------------------------------------------------------------
// High-level consultation patterns
// ---------------------------------------------------------------------------

/// Ask a vision peer to caption an image.
/// `image_url` should be the full data URL (data:image/png;base64,...).
pub async fn caption_image(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    image_url: &str,
    user_text: &str,
) -> Result<String> {
    let prompt = if user_text.is_empty() {
        "Describe this image concisely in one paragraph.".to_string()
    } else {
        format!("The user asked: \"{user_text}\"\n\nDescribe this image concisely, focusing on details relevant to the user's question.")
    };

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })];

    chat_completion(node, peer_id, model, messages, 256, TIMEOUT_CONSULTATION).await
}

/// Ask a peer for a second opinion on the user's question.
///
/// Sends only the last user message (not the full conversation) and asks
/// for a short, direct answer. The result is injected into the uncertain
/// model's KV cache as context — it should be concise (a fact, a key point,
/// a starting direction), not a full essay.
pub async fn second_opinion(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: &[Value],
    timeout: std::time::Duration,
) -> Result<String> {
    // Extract just the last user message text
    let last_user_text = messages
        .iter()
        .rev()
        .find(|m| m["role"].as_str() == Some("user"))
        .and_then(|m| {
            // Handle both string content and multimodal array content
            if let Some(s) = m["content"].as_str() {
                Some(s.to_string())
            } else if let Some(parts) = m["content"].as_array() {
                parts
                    .iter()
                    .find(|p| p["type"].as_str() == Some("text"))
                    .and_then(|p| p["text"].as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    if last_user_text.is_empty() {
        anyhow::bail!("no user message found for second opinion");
    }

    // Truncate very long user messages — we want a fast answer
    let user_text = if last_user_text.len() > 2000 {
        let end = last_user_text
            .char_indices()
            .take_while(|(i, _)| *i < 2000)
            .last()
            .map_or(0, |(i, c)| i + c.len_utf8());
        format!("{}...", &last_user_text[..end])
    } else {
        last_user_text
    };

    let ask_messages = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Answer this briefly and directly in 2-3 sentences:\n\n{user_text}"
        )
    })];

    chat_completion(node, peer_id, model, ask_messages, 192, timeout).await
}

/// Fan out a second-opinion request to up to 2 peers, return the first
/// response. If only one peer is available, falls back to a single call.
pub async fn race_second_opinion(
    node: &mesh::Node,
    peers: &[(EndpointId, String)],
    messages: &[Value],
    timeout: std::time::Duration,
) -> Option<(String, EndpointId, String)> {
    if peers.is_empty() {
        return None;
    }

    if peers.len() == 1 {
        let (id, model) = &peers[0];
        return match second_opinion(node, *id, model, messages, timeout).await {
            Ok(text) => Some((text, *id, model.clone())),
            Err(e) => {
                tracing::warn!(
                    "virtual: second opinion from {} failed: {e}",
                    id.fmt_short()
                );
                None
            }
        };
    }

    // Race two peers — fire both via JoinSet, take first Ok, abort the rest.
    let mut set = tokio::task::JoinSet::new();

    for (id, model) in peers.iter().skip(1).take(1) {
        let node = node.clone();
        let msgs = messages.to_vec();
        let id = *id;
        let model = model.clone();
        let t = timeout;
        set.spawn(async move {
            second_opinion(&node, id, &model, &msgs, t)
                .await
                .map(|text| (text, id, model))
        });
    }
    // Spawn the best peer last so it appears in the set too
    {
        let node = node.clone();
        let msgs = messages.to_vec();
        let id = peers[0].0;
        let model = peers[0].1.clone();
        let t = timeout;
        set.spawn(async move {
            second_opinion(&node, id, &model, &msgs, t)
                .await
                .map(|text| (text, id, model))
        });
    }

    while let Some(result) = set.join_next().await {
        if let Ok(Ok((text, id, model))) = result {
            tracing::info!("virtual: peer {} ({model}) won the race", id.fmt_short());
            set.abort_all();
            return Some((text, id, model));
        }
    }

    tracing::warn!("virtual: all peers failed");
    None
}
