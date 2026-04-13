//! Virtual LLM decision engine.
//!
//! This module is the "brain" behind mesh hook callbacks. When llama-server
//! detects something interesting during inference (context pressure, high
//! entropy, truncated response, etc.) it POSTs to `/mesh/hook` on the
//! management API. The route handler in `api/routes/mesh_hook.rs` parses
//! the payload and calls into this module to decide what to do.
//!
//! # Hook lifecycle
//!
//! ```text
//! llama-server                    mesh-llm
//!     │                               │
//!     │ ── POST /mesh/hook ──────────>│
//!     │    { hook: "pre_inference",   │
//!     │      trigger: "context_pres." }│── handle_pre_inference()
//!     │                               │     → action: inject / none
//!     │ <──────────── response ───────│
//!     │                               │
//!     │   (generation proceeds)       │
//!     │                               │
//!     │ ── POST /mesh/hook ──────────>│
//!     │    { hook: "post_prefill",    │
//!     │      signals: { entropy, … }} │── handle_post_prefill()
//!     │                               │     → action: inject / none
//!     │ <──────────── response ───────│
//!     │   (inject → tokenize + decode │
//!     │    into KV cache, model "sees"│
//!     │    the context before gen.)   │
//!     │                               │
//!     │ ── POST /mesh/hook ──────────>│
//!     │    { hook: "pre_response",    │
//!     │      generated_text, signals }│── handle_pre_response()
//!     │                               │     → action: inject / none
//!     │ <──────────── response ───────│
//! ```
//!
//! # Background work pattern
//!
//! All hooks are synchronous from the C++ side — each is a blocking POST.
//! But mesh-llm can start background work on Hook 1 (e.g. send the prompt
//! to a stronger model for verification) and collect results on Hook 3
//! (which always fires before the response leaves). A DashMap keyed by
//! `request_id` bridges the two hooks. If the background work finishes
//! in time, Hook 3 uses it. If not, it lets the response go as-is.
//!
//! # Model awareness
//!
//! The C++ hook payload only contains the model filename (e.g.
//! `"Qwen3-8B-Q4_K_M"`). mesh-llm enriches this with mesh peer state:
//! which other models are available, their capabilities, their RTT.
//! The consultation module (`inference/consult.rs`) handles finding
//! suitable peers and sending them requests over the QUIC mesh.

use crate::inference::consult;
use crate::mesh;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Hook 1: pre-inference
// ---------------------------------------------------------------------------

/// Called when llama-server detects a structural trigger before inference:
///
/// | Trigger                  | Meaning                                    |
/// |--------------------------|--------------------------------------------|
/// | `images_no_multimodal`   | Request has images but model is text-only  |
/// | `context_pressure`       | Prompt fills >75% of context window        |
/// | `long_session`           | Conversation has >10 turns                 |
/// | `large_user_message`     | Last user message is unusually large       |
///
/// The response can:
/// - Inject context (e.g. image caption, summary of early turns)
/// - Set `entropy_threshold` to enable Hook 2
/// - Set `verify: true` to enable Hook 3's verify trigger
pub async fn handle_pre_inference(node: &mesh::Node, payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let model = payload["model"].as_str().unwrap_or("");
    let n_prompt = payload["n_prompt_tokens"].as_i64().unwrap_or(0);
    let n_ctx = payload["n_ctx"].as_i64().unwrap_or(0);

    tracing::info!(
        "virtual: pre_inference trigger={trigger} model={model} \
         prompt={n_prompt}/{n_ctx}"
    );

    match trigger {
        "images_no_multimodal" => handle_image_caption(node, payload, model).await,
        "context_pressure" => handle_context_pressure(node, payload, model, n_prompt, n_ctx).await,
        "long_session" => {
            // Long sessions benefit from summarization too
            handle_context_pressure(node, payload, model, n_prompt, n_ctx).await
        }
        _ => {
            tracing::debug!("virtual: pre_inference trigger={trigger}, no action");
            // Arm Hook 2 with a moderate threshold so we catch uncertain starts
            json!({ "action": "none", "entropy_threshold": 5.0 })
        }
    }
}

/// Image captioning: find a vision peer, send the image, inject the caption.
async fn handle_image_caption(node: &mesh::Node, payload: &Value, current_model: &str) -> Value {
    // Find a vision-capable peer
    let vision_peer = consult::find_vision_peer(node, current_model).await;
    let peer_id = match vision_peer {
        Some(id) => id,
        None => {
            tracing::info!("virtual: no vision peer available, skipping image caption");
            return json!({ "action": "none", "entropy_threshold": 5.0 });
        }
    };

    // Find the vision model name on that peer
    let peers = node.peers().await;
    let vision_model = peers
        .iter()
        .find(|p| p.id == peer_id)
        .and_then(|p| {
            p.served_model_descriptors
                .iter()
                .find(|d| d.capabilities.supports_vision_runtime())
                .map(|d| d.identity.model_name.clone())
        })
        .unwrap_or_default();

    // Extract image URL and user text from the request_id's stored request
    // For now, extract from the messages in the payload if available.
    // TODO: look up stored request by mesh_request_id once that's wired
    let (image_url, user_text) = extract_image_from_payload(payload);

    if image_url.is_empty() {
        tracing::warn!("virtual: images_no_multimodal trigger but no image found in payload");
        return json!({ "action": "none", "entropy_threshold": 5.0 });
    }

    tracing::info!(
        "virtual: captioning image via vision peer {} model={vision_model}",
        peer_id.fmt_short()
    );

    match consult::caption_image(node, peer_id, &vision_model, &image_url, &user_text).await {
        Ok(caption) => {
            tracing::info!(
                "virtual: got caption ({} chars): {}...",
                caption.len(),
                &caption[..caption.len().min(80)]
            );
            json!({
                "action": "inject",
                "text": format!("[Image description: {caption}]\n\n"),
                "entropy_threshold": 5.0,
            })
        }
        Err(e) => {
            tracing::warn!("virtual: image caption failed: {e}");
            json!({ "action": "none", "entropy_threshold": 5.0 })
        }
    }
}

/// Context pressure: summarize early turns to free context space.
async fn handle_context_pressure(
    node: &mesh::Node,
    payload: &Value,
    _current_model: &str,
    n_prompt: i64,
    n_ctx: i64,
) -> Value {
    let pressure_pct = if n_ctx > 0 {
        (n_prompt as f64 / n_ctx as f64 * 100.0) as u32
    } else {
        0
    };
    tracing::info!("virtual: context at {pressure_pct}%, looking for peer to summarize");

    // Find any peer to do the summarization
    let peer = consult::find_any_peer(node).await;
    let (peer_id, peer_model) = match peer {
        Some(p) => p,
        None => {
            tracing::info!("virtual: no peers available for summarization");
            return json!({ "action": "none", "entropy_threshold": 5.0 });
        }
    };

    // Extract messages from payload
    let messages = match payload["messages"].as_array() {
        Some(m) if m.len() > 4 => m,
        _ => {
            tracing::debug!("virtual: not enough messages to summarize");
            return json!({ "action": "none", "entropy_threshold": 5.0 });
        }
    };

    // Summarize all but the last 2 messages (keep recent context intact)
    let to_summarize = &messages[..messages.len().saturating_sub(2)];

    tracing::info!(
        "virtual: summarizing {} early messages via peer {} model={peer_model}",
        to_summarize.len(),
        peer_id.fmt_short()
    );

    match consult::summarize_conversation(node, peer_id, &peer_model, to_summarize).await {
        Ok(summary) => {
            tracing::info!("virtual: got summary ({} chars)", summary.len());
            json!({
                "action": "inject",
                "text": format!("[Conversation summary: {summary}]\n\n"),
                "entropy_threshold": 5.0,
            })
        }
        Err(e) => {
            tracing::warn!("virtual: summarization failed: {e}");
            json!({ "action": "none", "entropy_threshold": 5.0 })
        }
    }
}

// ---------------------------------------------------------------------------
// Hook 2: post-prefill
// ---------------------------------------------------------------------------

/// Called after prompt evaluation when the first predicted token shows
/// high uncertainty. The model is unsure how to begin its response.
///
/// We ask a different model in the mesh the same question and inject its
/// answer. The value is diversity — a different architecture might be
/// confident where this one isn't. Not necessarily a "stronger" model,
/// just a different perspective.
///
/// If this returns `{"action": "inject", "text": "..."}`, the C++ side
/// tokenizes the text and decodes it into the KV cache. The model processes
/// it as if it were part of the original prompt, then generates from that
/// informed state.
pub async fn handle_post_prefill(node: &mesh::Node, payload: &Value) -> Value {
    let entropy = payload["signals"]["first_token_entropy"]
        .as_f64()
        .unwrap_or(0.0);
    let margin = payload["signals"]["first_token_margin"]
        .as_f64()
        .unwrap_or(1.0);
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!("virtual: post_prefill entropy={entropy:.2} margin={margin:.3} model={model}");

    // Find a different model to get a second opinion
    let peer = consult::find_different_model_peer(node, model).await;
    let (peer_id, peer_model) = match peer {
        Some(p) => p,
        None => {
            tracing::info!("virtual: no different model available for second opinion");
            return json!({ "action": "none" });
        }
    };

    // Extract messages from the payload
    let messages = match payload["messages"].as_array() {
        Some(m) if !m.is_empty() => m.clone(),
        _ => {
            tracing::debug!("virtual: no messages in post_prefill payload");
            return json!({ "action": "none" });
        }
    };

    tracing::info!(
        "virtual: getting second opinion from peer {} model={peer_model} (current model={model} has entropy={entropy:.2})",
        peer_id.fmt_short()
    );

    match consult::second_opinion(node, peer_id, &peer_model, &messages).await {
        Ok(opinion) => {
            // Inject the other model's answer as context, not as the answer itself.
            // The current model will incorporate this and generate its own response.
            let trimmed = if opinion.len() > 1024 {
                format!("{}...", &opinion[..1024])
            } else {
                opinion
            };
            tracing::info!(
                "virtual: injecting second opinion ({} chars)",
                trimmed.len()
            );
            json!({
                "action": "inject",
                "text": format!("\n[Another model's perspective: {trimmed}]\n\n"),
            })
        }
        Err(e) => {
            tracing::warn!("virtual: second opinion failed: {e}");
            json!({ "action": "none" })
        }
    }
}

// ---------------------------------------------------------------------------
// Hook 3: pre-response
// ---------------------------------------------------------------------------

/// Called just before the response is sent to the client. Fires when the
/// generation shows signs of distress:
///
/// | Trigger              | Meaning                                      |
/// |----------------------|----------------------------------------------|
/// | `max_tokens`         | Hit the token limit — response may be cut    |
/// | `very_short`         | Very short response to a long prompt         |
/// | `high_uncertainty`   | >30% of tokens had high entropy              |
/// | `tail_entropy_spike` | Entropy spiked in the last 16 tokens         |
/// | `verify`             | Hook 1 requested verification                |
///
/// `generated_text` is the key piece — the actual model output that only
/// C++ has. Combined with the original request (via mesh_request_id), this
/// gives mesh-llm everything needed to verify, correct, or annotate.
pub async fn handle_pre_response(node: &mesh::Node, payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let generated_text = payload["generated_text"].as_str().unwrap_or("");
    let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
    let stop_reason = payload["stop_reason"].as_str().unwrap_or("");
    let model = payload["model"].as_str().unwrap_or("");
    let mean_entropy = payload["signals"]["mean_entropy"].as_f64().unwrap_or(0.0);

    tracing::info!(
        "virtual: pre_response trigger={trigger} n_decoded={n_decoded} \
         stop={stop_reason} mean_entropy={mean_entropy:.2}"
    );

    match trigger {
        "max_tokens" => {
            tracing::info!("virtual: response truncated at {n_decoded} tokens");
            json!({
                "action": "inject",
                "text": "\n\n[Note: This response was truncated due to length limits.]"
            })
        }
        "very_short" => {
            tracing::info!("virtual: suspiciously short response ({n_decoded} tokens)");
            // Short responses might be refusals — not much we can do at Hook 3
            // since generation is already complete. Log it for now.
            json!({ "action": "none" })
        }
        "tail_entropy_spike" | "high_uncertainty" | "verify" => {
            handle_tail_verification(node, payload, model, trigger, generated_text).await
        }
        _ => {
            tracing::debug!("virtual: pre_response trigger={trigger}, no action");
            json!({ "action": "none" })
        }
    }
}

/// Verify the ending of a response by asking another model.
/// Used for tail_entropy_spike (model went off the rails),
/// high_uncertainty (many uncertain tokens throughout), and
/// verify (Hook 1 explicitly requested verification).
///
/// Same pattern as Hook 2 — the value is a different perspective,
/// not necessarily a "stronger" model checking a weaker one.
async fn handle_tail_verification(
    node: &mesh::Node,
    payload: &Value,
    current_model: &str,
    trigger: &str,
    generated_text: &str,
) -> Value {
    // Find a different model for verification
    let peer = consult::find_different_model_peer(node, current_model).await;
    let (peer_id, peer_model) = match peer {
        Some(p) => p,
        None => {
            tracing::info!("virtual: no different model available for verification");
            return json!({ "action": "none" });
        }
    };

    // Extract messages from payload for context
    let messages = match payload["messages"].as_array() {
        Some(m) => m.as_slice(),
        None => {
            tracing::debug!("virtual: no messages in pre_response payload for verification");
            return json!({ "action": "none" });
        }
    };

    tracing::info!(
        "virtual: verifying response ({trigger}) via peer {} model={peer_model}",
        peer_id.fmt_short()
    );

    match consult::verify_response(node, peer_id, &peer_model, messages, generated_text).await {
        Ok(verdict) => {
            if verdict.contains("LOOKS_GOOD") {
                tracing::info!("virtual: verification passed");
                json!({ "action": "none" })
            } else {
                tracing::info!(
                    "virtual: verification suggests correction ({} chars)",
                    verdict.len()
                );
                let trimmed = if verdict.len() > 512 {
                    format!("{}...", &verdict[..512])
                } else {
                    verdict
                };
                json!({
                    "action": "inject",
                    "text": format!("\n\n[Correction: {trimmed}]"),
                })
            }
        }
        Err(e) => {
            tracing::warn!("virtual: verification failed: {e}");
            json!({ "action": "none" })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the first image URL and accompanying text from a hook payload's messages.
fn extract_image_from_payload(payload: &Value) -> (String, String) {
    let messages = match payload["messages"].as_array() {
        Some(m) => m,
        None => return (String::new(), String::new()),
    };

    // Look at the last user message
    for msg in messages.iter().rev() {
        if msg["role"].as_str() != Some("user") {
            continue;
        }

        // Content might be a string or an array of parts
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

// ---------------------------------------------------------------------------
// Background work tracking
// ---------------------------------------------------------------------------

// Tracks in-flight background consultations started by Hook 1 and collected
// by Hook 3. Keyed by `request_id` so Hook 3 can find results from Hook 1.
//
// TODO: implement with DashMap<String, BackgroundResult> where result is
// either Pending (with a JoinHandle) or Ready (with the text/verdict).
// Hook 1 inserts Pending + spawns a tokio task.
// Hook 3 checks: if Ready, use it. If Pending, let the response go.
// Entries are removed after Hook 3 checks (or after a timeout).
