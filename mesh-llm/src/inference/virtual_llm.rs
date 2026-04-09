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
//!     │                               │     → action: inject / pending / none
//!     │ <──────────── response ───────│
//!     │                               │
//!     │   (generation proceeds)       │
//!     │                               │
//!     │ ── POST /mesh/hook ──────────>│
//!     │    { hook: "post_prefill",    │
//!     │      signals: { entropy, … }} │── handle_post_prefill()
//!     │                               │     → action: none (usually)
//!     │ <──────────── response ───────│
//!     │                               │
//!     │ ── POST /mesh/hook ──────────>│
//!     │    { hook: "pre_response",    │
//!     │      generated_text, signals }│── handle_pre_response()
//!     │                               │     → action: inject / none
//!     │ <──────────── response ───────│
//! ```
//!
//! # Async consultations
//!
//! When a hook needs to consult another model in the mesh (e.g. captioning
//! an image, summarizing context), it can return `{ "action": "pending",
//! "async_id": "..." }`. The C++ side polls `GET /mesh/hook/poll/{id}`
//! every 16 tokens during generation. When the consultation finishes, the
//! poll returns the inject text.
//!
//! Pending consultations are tracked in `AsyncConsultations` (a DashMap).
//!
//! # Current status
//!
//! All handlers are stubs that log and return `none`. The hook plumbing
//! (C++ triggers → HTTP callback → this module) is tested end-to-end.
//! Next step: implement actual consultations against mesh peers.

use serde_json::{json, Value};

/// Response actions returned to llama-server.
///
/// The C++ side understands four actions:
/// - `none`    — do nothing, continue normally
/// - `inject`  — insert `text` into the prompt (Hook 1) or response (Hook 3)
/// - `pending` — async consultation started, poll `async_id` for result
/// - `stop`    — abort generation (not yet implemented in C++)
#[derive(Debug)]
#[allow(dead_code)] // variants used when consultations are implemented
pub enum HookAction {
    /// No intervention needed.
    None,
    /// Inject text into the prompt or response.
    Inject { text: String },
    /// Async consultation in progress — poll for result.
    Pending { async_id: String },
}

// ---------------------------------------------------------------------------
// Hook 1: pre-inference
// ---------------------------------------------------------------------------

/// Called when llama-server detects a structural trigger before inference:
///
/// | Trigger                  | Meaning                                    |
/// |--------------------------|--------------------------------------------|
/// | `images_no_multimodal`   | Request has images but model is text-only  |
/// | `audio_no_support`       | Request has audio but model can't process  |
/// | `context_pressure`       | Prompt fills >75% of context window        |
/// | `long_session`           | Conversation has >10 turns                 |
/// | `large_user_message`     | Last user message is unusually large       |
///
/// The response can:
/// - Inject context (e.g. image caption, summary of early turns)
/// - Start an async consultation and return `pending`
/// - Set `entropy_threshold` to enable Hook 2
/// - Set `verify: true` to enable Hook 3's verify trigger
pub fn handle_pre_inference(payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let model = payload["model"].as_str().unwrap_or("");
    let n_prompt = payload["n_prompt_tokens"].as_i64().unwrap_or(0);
    let n_ctx = payload["n_ctx"].as_i64().unwrap_or(0);

    tracing::info!(
        "virtual: pre_inference trigger={trigger} model={model} \
         prompt={n_prompt}/{n_ctx}"
    );

    match trigger {
        "images_no_multimodal" => {
            // TODO: find a vision-capable model in the mesh, send the images,
            // get a caption, return it as inject text.
            tracing::info!("virtual: would consult vision model for image caption");
            json!({ "action": "none", "entropy_threshold": 5.0 })
        }
        "context_pressure" => {
            // TODO: summarize early conversation turns using a model in the
            // mesh, inject the summary to free context space.
            let pressure_pct = if n_ctx > 0 {
                (n_prompt as f64 / n_ctx as f64 * 100.0) as u32
            } else {
                0
            };
            tracing::info!("virtual: context at {pressure_pct}%, would summarize early turns");
            json!({ "action": "none", "entropy_threshold": 5.0 })
        }
        "long_session" => {
            // TODO: consider summarizing or trimming the conversation.
            tracing::info!("virtual: long session, would consider summarization");
            json!({ "action": "none" })
        }
        _ => {
            tracing::debug!("virtual: pre_inference trigger={trigger}, no action");
            json!({ "action": "none" })
        }
    }
}

// ---------------------------------------------------------------------------
// Hook 2: post-prefill
// ---------------------------------------------------------------------------

/// Called after prompt evaluation when the first predicted token shows
/// high uncertainty. This means the model is unsure how to begin its
/// response — it might benefit from consulting another model.
///
/// The payload includes:
/// - `signals.first_token_entropy` — entropy of the first token's distribution
/// - `signals.first_token_margin`  — probability gap between top-1 and top-2
/// - `signals.top_tokens`          — top 5 tokens with probabilities
///
/// This hook fires only if Hook 1 set an `entropy_threshold`.
pub fn handle_post_prefill(payload: &Value) -> Value {
    let entropy = payload["signals"]["first_token_entropy"]
        .as_f64()
        .unwrap_or(0.0);
    let margin = payload["signals"]["first_token_margin"]
        .as_f64()
        .unwrap_or(1.0);

    tracing::info!("virtual: post_prefill entropy={entropy:.2} margin={margin:.3}");

    // TODO: if entropy is very high, consider:
    // - Asking a stronger model for a "seed" response to guide generation
    // - Injecting a clarifying system note
    // - Running self-consistency (multiple completions via n_cmpl)

    json!({ "action": "none" })
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
/// | `very_short`         | Very short response to a long prompt (maybe a refusal) |
/// | `high_uncertainty`   | >30% of tokens had high entropy              |
/// | `tail_entropy_spike` | Entropy spiked in the last 16 tokens         |
/// | `verify`             | Hook 1 requested verification of every response |
///
/// The payload includes the full `generated_text` and signal summary.
/// The response can inject text that gets appended to the output.
pub fn handle_pre_response(payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
    let stop_reason = payload["stop_reason"].as_str().unwrap_or("");
    let mean_entropy = payload["signals"]["mean_entropy"].as_f64().unwrap_or(0.0);

    tracing::info!(
        "virtual: pre_response trigger={trigger} n_decoded={n_decoded} \
         stop={stop_reason} mean_entropy={mean_entropy:.2}"
    );

    match trigger {
        "max_tokens" => {
            // TODO: consider continuing generation by consulting the mesh,
            // or appending a "response was truncated" note.
            tracing::info!("virtual: response truncated at {n_decoded} tokens");
            json!({ "action": "none" })
        }
        "very_short" => {
            // TODO: might be a refusal. Could re-prompt with different
            // framing or consult a different model.
            tracing::info!("virtual: suspiciously short response ({n_decoded} tokens)");
            json!({ "action": "none" })
        }
        "high_uncertainty" => {
            // TODO: run self-consistency check — generate multiple completions
            // and pick the best, or flag low confidence to the caller.
            tracing::info!("virtual: high uncertainty in response");
            json!({ "action": "none" })
        }
        "tail_entropy_spike" => {
            // TODO: the model may have started hallucinating at the end.
            // Consider truncating at the spike point.
            tracing::info!("virtual: tail entropy spike detected");
            json!({ "action": "none" })
        }
        "verify" => {
            // TODO: send the generated text to another model for verification.
            tracing::info!("virtual: would verify response with another model");
            json!({ "action": "none" })
        }
        _ => {
            tracing::debug!("virtual: pre_response trigger={trigger}, no action");
            json!({ "action": "none" })
        }
    }
}

// ---------------------------------------------------------------------------
// Async consultation tracking
// ---------------------------------------------------------------------------

/// Tracks in-flight async consultations (e.g. image captioning, summarization).
///
/// When a hook returns `pending`, the async_id is stored here. The C++ side
/// polls every 16 tokens. When the consultation finishes, the result is
/// stored and the next poll returns it.
///
/// TODO: implement with DashMap<String, ConsultationState> where state is
/// either Pending (with a JoinHandle) or Ready (with the inject text).
#[allow(dead_code)] // used when async consultations are implemented
pub struct AsyncConsultations;

#[allow(dead_code)] // used when async consultations are implemented
impl AsyncConsultations {
    /// Look up an async consultation result.
    ///
    /// Returns `Some(text)` if the consultation is complete, `None` if still
    /// pending. A completed consultation is removed from the map.
    pub fn poll(&self, _async_id: &str) -> Option<String> {
        // TODO: look up in DashMap, return Ready result, remove entry
        None
    }
}
