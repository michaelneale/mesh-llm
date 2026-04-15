//! Virtual LLM — consult other models in the mesh during inference.
//!
//! llama-server POSTs to /mesh/hook at key points. Each hook blocks the
//! slot. We can ask mesh peers for help and tell C++ to inject context
//! or replace the response.
//!
//! | Function          | When                      | Action                    |
//! |-------------------|---------------------------|---------------------------|
//! | handle_image      | Image on text-only model  | Caption via vision peer   |
//! | handle_uncertain  | High entropy at start     | Hint from different model |
//! | handle_drift      | Entropy spike mid-gen     | Hint from different model |

use crate::inference::consult;
use crate::mesh;
use serde_json::{json, Value};

// ===========================================================================
// handle_image — model can't see media, get a caption/transcript
// ===========================================================================

/// The model received media it can't process (image on text-only model,
/// audio on non-audio model). Finds a capable peer, gets a text
/// description, and returns it for injection before tokenization.
///
/// `trigger`: `"images_no_multimodal"` or `"audio_no_support"`
/// `image_url`: data URL for the image (empty if audio trigger)
/// `user_text`: the user's text alongside the media
///
/// Returns `{"action": "inject", "text": "[Image description: ...]"}` or
/// `{"action": "none"}` if no capable peer or captioning fails.
pub async fn handle_image(
    node: &mesh::Node,
    trigger: &str,
    model: &str,
    image_url: &str,
    user_text: &str,
) -> Value {
    tracing::info!("virtual: handle_image trigger={trigger} model={model}");

    match trigger {
        "images_no_multimodal" => {
            if image_url.is_empty() {
                tracing::warn!("virtual: images trigger but no image URL");
                return json!({ "action": "none" });
            }
            caption_image(node, model, image_url, user_text).await
        }
        "audio_no_support" => transcribe_audio(node, model).await,
        "video_no_support" => {
            // TODO: extract keyframes, caption via vision peer
            tracing::info!("virtual: video — not yet implemented");
            json!({ "action": "none" })
        }
        _ => json!({ "action": "none" }),
    }
}

async fn caption_image(
    node: &mesh::Node,
    current_model: &str,
    image_url: &str,
    user_text: &str,
) -> Value {
    let peer_id = match consult::find_vision_peer(node, current_model).await {
        Some(id) => id,
        None => {
            tracing::info!("virtual: no vision peer available");
            return json!({ "action": "none" });
        }
    };

    let vision_model =
        peer_model_with_capability(node, peer_id, |d| d.capabilities.supports_vision_runtime())
            .await;

    tracing::info!(
        "virtual: captioning via {} model={vision_model}",
        peer_id.fmt_short()
    );

    match consult::caption_image(node, peer_id, &vision_model, image_url, user_text).await {
        Ok(caption) => {
            tracing::info!("virtual: caption ({} chars)", caption.len());
            json!({
                "action": "inject",
                "text": format!("[Image description: {caption}]\n\n"),
            })
        }
        Err(e) => {
            tracing::warn!("virtual: caption failed: {e}");
            json!({ "action": "none" })
        }
    }
}

async fn transcribe_audio(node: &mesh::Node, current_model: &str) -> Value {
    let peer_id = match consult::find_audio_peer(node, current_model).await {
        Some(id) => id,
        None => {
            tracing::info!("virtual: no audio peer available");
            return json!({ "action": "none" });
        }
    };

    let audio_model =
        peer_model_with_capability(node, peer_id, |d| d.capabilities.supports_audio_runtime())
            .await;

    // TODO: extract audio data from payload and send to peer
    // For now, log that we found a peer but can't extract audio yet
    tracing::info!(
        "virtual: found audio peer {} model={audio_model}, but audio extraction not yet wired",
        peer_id.fmt_short()
    );
    json!({ "action": "none" })
}

/// Look up a peer's model name matching a capability predicate.
async fn peer_model_with_capability(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    predicate: impl Fn(&crate::mesh::ServedModelDescriptor) -> bool,
) -> String {
    let peers = node.peers().await;
    peers
        .iter()
        .find(|p| p.id == peer_id)
        .and_then(|p| {
            p.served_model_descriptors
                .iter()
                .find(|d| predicate(d))
                .map(|d| d.identity.model_name.clone())
        })
        .unwrap_or_default()
}

// ===========================================================================
// handle_uncertain — model stuck at start, get a hint from a peer
// ===========================================================================

/// Model doesn't know how to start its answer — first token has high
/// entropy after prefill. Asks a different-architecture peer the same
/// question and injects the answer so the model reads it before generating.
///
/// `entropy`: first token entropy (higher = more uncertain)
/// `margin`: gap between top two token probabilities (lower = more uncertain)
///
/// Returns `{"action": "inject", "text": "\n[Context: ...]\n\n"}` or
/// `{"action": "none"}` if no peers available or consultation fails.
pub async fn handle_uncertain(
    node: &mesh::Node,
    model: &str,
    messages: &[Value],
    entropy: f64,
    margin: f64,
) -> Value {
    tracing::info!(
        "virtual: handle_uncertain entropy={entropy:.2} margin={margin:.3} model={model}"
    );

    if messages.is_empty() {
        tracing::debug!("virtual: no messages, skipping");
        return json!({ "action": "none" });
    }

    // Pre-generation: user is waiting for first token anyway, can afford longer timeout
    get_peer_hint(node, model, messages, consult::TIMEOUT_CONSULTATION).await
}

// ===========================================================================
// handle_drift — model losing coherence mid-generation
// ===========================================================================

/// Model is losing coherence mid-generation — sustained entropy spike
/// over the last 16 tokens. Asks a peer the original question and injects
/// the answer at the current KV position so the model course-corrects.
///
/// `n_decoded`: tokens generated so far (for logging)
///
/// Returns `{"action": "inject", "text": "..."}` or `{"action": "none"}`.
pub async fn handle_drift(
    node: &mesh::Node,
    model: &str,
    messages: &[Value],
    n_decoded: i64,
) -> Value {
    tracing::info!("virtual: handle_drift n_decoded={n_decoded} model={model}");

    if messages.is_empty() {
        tracing::debug!("virtual: no messages, skipping");
        return json!({ "action": "none" });
    }

    // Mid-generation: user sees a stall, keep it short
    get_peer_hint(node, model, messages, consult::TIMEOUT_CONSULTATION).await
}

// ===========================================================================
// get_peer_hint — race 2 peers, inject winner's answer
// ===========================================================================

/// Shared by handle_uncertain and handle_drift. Finds up to 2 peers
/// serving a different model, races them for a second opinion, returns
/// an inject action with the winner's answer.
///
async fn get_peer_hint(
    node: &mesh::Node,
    current_model: &str,
    messages: &[Value],
    timeout: std::time::Duration,
) -> Value {
    let peers = consult::find_different_model_peers(node, current_model, 2).await;
    if peers.is_empty() {
        tracing::info!("virtual: no different model available");
        return json!({ "action": "none" });
    }

    let peer_names: Vec<_> = peers
        .iter()
        .map(|(id, m)| format!("{}={m}", id.fmt_short()))
        .collect();
    tracing::info!(
        "virtual: racing {} peers: [{}]",
        peers.len(),
        peer_names.join(", ")
    );

    match consult::race_second_opinion(node, &peers, messages, timeout).await {
        Some((opinion, winner_id, winner_model)) => {
            let trimmed = if opinion.len() > 512 {
                let end = opinion
                    .char_indices()
                    .take_while(|(i, _)| *i < 512)
                    .last()
                    .map_or(0, |(i, c)| i + c.len_utf8());
                format!("{}...", &opinion[..end])
            } else {
                opinion
            };
            tracing::info!(
                "virtual: hint from {} ({}) ({} chars)",
                winner_id.fmt_short(),
                winner_model,
                trimmed.len()
            );
            json!({
                "action": "inject",
                "text": format!("\n\nReference answer: {trimmed}\n\nUse the reference above to provide an accurate response.\n"),
            })
        }
        None => {
            tracing::warn!("virtual: all peers failed");
            json!({ "action": "none" })
        }
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

pub fn extract_image(payload: &Value) -> (String, String) {
    let messages = match payload["messages"].as_array() {
        Some(m) => m,
        None => return (String::new(), String::new()),
    };

    for msg in messages.iter().rev() {
        if msg["role"].as_str() != Some("user") {
            continue;
        }
        if let Some(parts) = msg["content"].as_array() {
            let mut image_url = String::new();
            let mut text = String::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("image_url") => {
                        if image_url.is_empty() {
                            image_url = part["image_url"]["url"].as_str().unwrap_or("").to_string();
                        }
                    }
                    Some("text") => {
                        // Check for mesh_image_url — preserved by llama-server's
                        // content parser when mesh hooks strip unsupported images.
                        if image_url.is_empty() {
                            if let Some(url) = part["mesh_image_url"]["url"].as_str() {
                                image_url = url.to_string();
                            }
                        }
                        text = part["text"].as_str().unwrap_or("").to_string();
                    }
                    _ => {}
                }
            }
            if !image_url.is_empty() {
                return (image_url, text);
            }
        }
    }

    (String::new(), String::new())
}
