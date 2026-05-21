//! Virtual LLM — consult other models in the mesh during inference.
//!
//! The serving runtime POSTs to /mesh/hook at key points. Each hook blocks the
//! slot. We can ask mesh peers for help and tell the runtime to inject context
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
/// `media_url`: data URL or URL for the media
/// `user_text`: the user's text alongside the media
///
/// Returns `{"action": "inject", "text": "[Image description: ...]"}` (or
/// audio context) or `{"action": "none"}` if no capable peer is available or
/// the consultation fails.
pub async fn handle_image(
    node: &mesh::Node,
    trigger: &str,
    model: &str,
    media_url: &str,
    user_text: &str,
) -> Value {
    tracing::info!("virtual: handle_image trigger={trigger} model={model}");

    match trigger {
        "images_no_multimodal" => handle_image_caption(node, model, media_url, user_text).await,
        "audio_no_support" => handle_audio_rescue(node, model, media_url, user_text).await,
        "video_no_support" => video_not_supported(),
        _ => no_virtual_action(),
    }
}

async fn handle_image_caption(
    node: &mesh::Node,
    model: &str,
    image_url: &str,
    user_text: &str,
) -> Value {
    if image_url.is_empty() {
        tracing::warn!("virtual: images trigger but no image URL");
        return no_virtual_action();
    }

    caption_image(node, model, image_url, user_text).await
}

fn video_not_supported() -> Value {
    // TODO: extract keyframes, caption via vision peer
    tracing::info!("virtual: video — not yet implemented");
    no_virtual_action()
}

async fn caption_image(
    node: &mesh::Node,
    current_model: &str,
    image_url: &str,
    user_text: &str,
) -> Value {
    let Some((peer_id, vision_model)) = vision_peer_model(node, current_model).await else {
        tracing::info!("virtual: no vision peer available");
        return no_virtual_action();
    };

    tracing::info!(
        "virtual: captioning via {} model={vision_model}",
        peer_id.fmt_short()
    );

    let Some(caption) =
        request_image_caption(node, peer_id, &vision_model, image_url, user_text).await
    else {
        return no_virtual_action();
    };

    image_caption_response(caption)
}

async fn vision_peer_model(
    node: &mesh::Node,
    current_model: &str,
) -> Option<(iroh::EndpointId, String)> {
    let peer_id = consult::find_vision_peer(node, current_model).await?;
    let vision_model =
        peer_model_with_capability(node, peer_id, |d| d.capabilities.supports_vision_runtime())
            .await;
    Some((peer_id, vision_model))
}

async fn request_image_caption(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    vision_model: &str,
    image_url: &str,
    user_text: &str,
) -> Option<String> {
    match consult::caption_image(node, peer_id, vision_model, image_url, user_text).await {
        Ok(caption) => Some(caption),
        Err(e) => {
            tracing::warn!("virtual: caption failed: {e}");
            None
        }
    }
}

fn image_caption_response(caption: String) -> Value {
    tracing::info!("virtual: caption ({} chars)", caption.len());
    json!({
        "action": "inject",
        "text": format!("[Image description: {caption}]\n\n"),
    })
}

async fn handle_audio_rescue(
    node: &mesh::Node,
    model: &str,
    audio_url: &str,
    user_text: &str,
) -> Value {
    if audio_url.is_empty() {
        tracing::warn!("virtual: audio trigger but no audio URL");
        return no_virtual_action();
    }

    transcribe_audio(node, model, audio_url, user_text).await
}

async fn transcribe_audio(
    node: &mesh::Node,
    current_model: &str,
    audio_url: &str,
    user_text: &str,
) -> Value {
    let Some((peer_id, audio_model)) = audio_peer_model(node, current_model).await else {
        tracing::info!("virtual: no audio peer available");
        return no_virtual_action();
    };

    tracing::info!(
        "virtual: audio rescue via {} model={audio_model}",
        peer_id.fmt_short()
    );

    let Some(context) =
        request_audio_context(node, peer_id, &audio_model, audio_url, user_text).await
    else {
        return no_virtual_action();
    };

    audio_context_response(context)
}

async fn audio_peer_model(
    node: &mesh::Node,
    current_model: &str,
) -> Option<(iroh::EndpointId, String)> {
    let peer_id = consult::find_audio_peer(node, current_model).await?;
    let audio_model =
        peer_model_with_capability(node, peer_id, |d| d.capabilities.supports_audio_runtime())
            .await;
    Some((peer_id, audio_model))
}

async fn request_audio_context(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    audio_model: &str,
    audio_url: &str,
    user_text: &str,
) -> Option<String> {
    match consult::transcribe_audio(node, peer_id, audio_model, audio_url, user_text).await {
        Ok(context) => Some(context),
        Err(e) => {
            tracing::warn!("virtual: audio rescue failed: {e}");
            None
        }
    }
}

fn audio_context_response(context: String) -> Value {
    let context = context.trim();
    if context.is_empty() {
        tracing::warn!("virtual: audio peer returned empty context");
        return no_virtual_action();
    }
    tracing::info!("virtual: audio context ({} chars)", context.len());
    json!({
        "action": "inject",
        "text": format!("[Audio context: {context}]\n\n"),
    })
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
        return no_virtual_action();
    }

    log_peer_race(&peers);

    match consult::race_second_opinion(node, &peers, messages, timeout).await {
        Some((opinion, winner_id, winner_model)) => {
            peer_hint_response(opinion, winner_id, winner_model)
        }
        None => {
            tracing::warn!("virtual: all peers failed");
            no_virtual_action()
        }
    }
}

fn log_peer_race(peers: &[(iroh::EndpointId, String)]) {
    let peer_names: Vec<_> = peers
        .iter()
        .map(|(id, m)| format!("{}={m}", id.fmt_short()))
        .collect();
    tracing::info!(
        "virtual: racing {} peers: [{}]",
        peers.len(),
        peer_names.join(", ")
    );
}

fn peer_hint_response(opinion: String, winner_id: iroh::EndpointId, winner_model: String) -> Value {
    let trimmed = trim_reference_opinion(opinion);
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

fn trim_reference_opinion(opinion: String) -> String {
    if opinion.len() <= 512 {
        return opinion;
    }

    let end = opinion
        .char_indices()
        .take_while(|(i, _)| *i < 512)
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8());
    format!("{}...", &opinion[..end])
}

fn no_virtual_action() -> Value {
    json!({ "action": "none" })
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
                    Some("image_url") if image_url.is_empty() => {
                        image_url = part["image_url"]["url"].as_str().unwrap_or("").to_string();
                    }
                    Some("image_url") => {}
                    Some("text") => {
                        // Check for mesh_image_url preserved by the OpenAI surface
                        // when mesh hooks strip unsupported images.
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

pub fn extract_audio(payload: &Value) -> (String, String) {
    let messages = match payload["messages"].as_array() {
        Some(m) => m,
        None => return (String::new(), String::new()),
    };

    for msg in messages.iter().rev() {
        if msg["role"].as_str() != Some("user") {
            continue;
        }
        if let Some(parts) = msg["content"].as_array() {
            let mut audio_url = String::new();
            let mut text = Vec::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("input_audio") if audio_url.is_empty() => {
                        audio_url = media_container_url(part, "input_audio").unwrap_or_default();
                    }
                    Some("audio_url") if audio_url.is_empty() => {
                        audio_url = media_container_url(part, "audio_url").unwrap_or_default();
                    }
                    Some("audio") if audio_url.is_empty() => {
                        audio_url = media_container_url(part, "audio").unwrap_or_default();
                    }
                    Some("text") => {
                        if audio_url.is_empty() {
                            if let Some(url) = part["mesh_audio_url"]["url"].as_str() {
                                audio_url = url.to_string();
                            }
                        }
                        if let Some(part_text) = part["text"].as_str() {
                            text.push(part_text);
                        }
                    }
                    _ => {}
                }
            }
            if !audio_url.is_empty() {
                return (audio_url, text.join("\n"));
            }
        }
    }

    (String::new(), String::new())
}

fn media_container_url(part: &Value, key: &str) -> Option<String> {
    let value = part.get(key)?;
    if let Some(url) = value.as_str() {
        return Some(url.to_string());
    }
    if let Some(url) = value.get("url").and_then(Value::as_str) {
        return Some(url.to_string());
    }
    inline_audio_data_url(value)
}

fn inline_audio_data_url(value: &Value) -> Option<String> {
    let data = value.get("data").and_then(Value::as_str)?;
    if data.trim_start().starts_with("data:") {
        return Some(data.to_string());
    }
    let mime_type = value
        .get("mime_type")
        .or_else(|| value.get("media_type"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("format")
                .and_then(Value::as_str)
                .and_then(audio_mime_type_from_format)
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "audio/wav".to_string());
    Some(format!("data:{mime_type};base64,{data}"))
}

fn audio_mime_type_from_format(format: &str) -> Option<&'static str> {
    let format = format.trim().trim_start_matches('.').to_ascii_lowercase();
    match format.as_str() {
        "wav" => Some("audio/wav"),
        "mp3" | "mpeg" | "mpga" => Some("audio/mpeg"),
        "m4a" | "mp4" => Some("audio/mp4"),
        "flac" => Some("audio/flac"),
        "ogg" | "opus" => Some("audio/ogg"),
        "webm" => Some("audio/webm"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn extract_audio_reads_audio_url_and_user_text() {
        let payload = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "please transcribe this"},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,abc"}}
                ]
            }]
        });

        let (audio_url, user_text) = extract_audio(&payload);

        assert_eq!(audio_url, "data:audio/wav;base64,abc");
        assert_eq!(user_text, "please transcribe this");
    }

    #[test]
    fn extract_audio_converts_inline_input_audio_data() {
        let payload = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is said here?"},
                    {"type": "input_audio", "input_audio": {
                        "data": "YWJj",
                        "format": "mp3"
                    }}
                ]
            }]
        });

        let (audio_url, user_text) = extract_audio(&payload);

        assert_eq!(audio_url, "data:audio/mpeg;base64,YWJj");
        assert_eq!(user_text, "what is said here?");
    }

    #[test]
    fn extract_audio_reads_mesh_audio_url_fallback_from_text_part() {
        let payload = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "fallback text", "mesh_audio_url": {"url": "mesh://audio/ref"}}
                ]
            }]
        });

        let (audio_url, user_text) = extract_audio(&payload);

        assert_eq!(audio_url, "mesh://audio/ref");
        assert_eq!(user_text, "fallback text");
    }
}
