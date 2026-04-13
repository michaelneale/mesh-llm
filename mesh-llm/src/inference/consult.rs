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

/// Find a peer serving a *different* model from the current one.
/// Prefers peers with lower RTT. Does not require a specific capability —
/// the value is in getting a different architecture's perspective.
pub async fn find_different_model_peer(
    node: &mesh::Node,
    current_model: &str,
) -> Option<(EndpointId, String)> {
    let peers = node.peers().await;
    peers
        .iter()
        .filter_map(|p| {
            // Find the first served model that's different from the current one
            let different = p.served_model_descriptors.iter().find(|d| {
                d.identity.model_name != current_model && !d.identity.model_name.is_empty()
            });
            different.map(|d| {
                (
                    p.id,
                    d.identity.model_name.clone(),
                    p.rtt_ms.unwrap_or(u32::MAX),
                )
            })
        })
        .min_by_key(|(_, _, rtt)| *rtt)
        .map(|(id, model, _)| (id, model))
}

/// Find any peer serving a model (including same model on a different node).
/// Used for summarization where we just need compute, not a specific capability.
pub async fn find_any_peer(node: &mesh::Node) -> Option<(EndpointId, String)> {
    let peers = node.peers().await;
    peers
        .iter()
        .filter_map(|p| {
            p.served_model_descriptors.first().map(|d| {
                (
                    p.id,
                    d.identity.model_name.clone(),
                    p.rtt_ms.unwrap_or(u32::MAX),
                )
            })
        })
        .min_by_key(|(_, _, rtt)| *rtt)
        .map(|(id, model, _)| (id, model))
}

// ---------------------------------------------------------------------------
// Consultation requests
// ---------------------------------------------------------------------------

/// Send a chat completion request to a peer over the mesh QUIC tunnel.
/// Returns the assistant message content, or an error.
///
/// This is a blocking call (from the caller's perspective) — it opens a
/// tunnel, sends the HTTP request, reads the full response. Suitable for
/// hook handlers where C++ is waiting on our response.
pub async fn chat_completion(
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

    // Parse HTTP response — find the body after \r\n\r\n
    let body_start = response_str.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &response_str[body_start..];

    let parsed: Value = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("failed to parse peer response: {e}"))?;

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

    chat_completion(node, peer_id, model, messages, 256).await
}

/// Ask a peer to summarize conversation history.
/// Takes the messages array and returns a condensed summary.
pub async fn summarize_conversation(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: &[Value],
) -> Result<String> {
    let conversation = messages
        .iter()
        .filter_map(|m| {
            let role = m["role"].as_str()?;
            let content = m["content"].as_str()?;
            Some(format!("{role}: {content}"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    let summary_messages = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Summarize the following conversation concisely. \
             Keep key facts, decisions, and context. Be brief.\n\n{conversation}"
        )
    })];

    chat_completion(node, peer_id, model, summary_messages, 512).await
}

/// Ask a peer for a second opinion on a question.
/// Sends the user's messages to a different model and returns its response.
pub async fn second_opinion(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: &[Value],
) -> Result<String> {
    // Send the conversation as-is (minus images/audio to keep it fast)
    let text_messages: Vec<Value> = messages
        .iter()
        .map(|m| {
            // If content is an array (multimodal), extract just the text parts
            if let Some(parts) = m["content"].as_array() {
                let text_parts: Vec<&Value> = parts
                    .iter()
                    .filter(|p| p["type"].as_str() == Some("text"))
                    .collect();
                if text_parts.len() == 1 {
                    serde_json::json!({
                        "role": m["role"],
                        "content": text_parts[0]["text"]
                    })
                } else {
                    m.clone()
                }
            } else {
                m.clone()
            }
        })
        .collect();

    chat_completion(node, peer_id, model, text_messages, 1024).await
}

/// Ask a peer to verify whether generated text is accurate.
/// Returns the peer's assessment.
pub async fn verify_response(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    user_messages: &[Value],
    generated_text: &str,
) -> Result<String> {
    // Build the last user message text for context
    let last_user = user_messages
        .iter()
        .rev()
        .find(|m| m["role"].as_str() == Some("user"))
        .and_then(|m| m["content"].as_str())
        .unwrap_or("(unknown question)");

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "A language model was asked: \"{last_user}\"\n\n\
             It responded: \"{generated_text}\"\n\n\
             Is the ending of this response accurate and complete, \
             or does it trail off into nonsense? \
             If the ending is wrong, provide a corrected version of \
             just the last sentence or two. If it's fine, say \"LOOKS_GOOD\"."
        )
    })];

    chat_completion(node, peer_id, model, messages, 256).await
}
