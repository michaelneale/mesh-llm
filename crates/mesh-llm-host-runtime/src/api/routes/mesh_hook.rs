use super::MeshApi;
use crate::api::http;
use crate::inference::virtual_llm;
use serde_json::Value;
use tokio::net::TcpStream;

/// Handle mesh hook callbacks from the serving runtime.
///
/// Parses the JSON payload once, dispatches to typed handler functions.
/// Each hook blocks the C++ slot until we respond.
///
/// Only accepts connections from loopback because hook callbacks are local-only.
/// This prevents remote callers from triggering costly peer consultations even
/// when the management API is bound to 0.0.0.0 via `--listen-all`.
pub async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    _method: &str,
    _path: &str,
    body: &str,
) -> anyhow::Result<()> {
    if reject_non_loopback_caller(stream).await? {
        return Ok(());
    }

    let Some(payload) = parse_hook_payload(stream, body).await? else {
        return Ok(());
    };

    let response = dispatch_hook(state, payload).await;
    http::respond_json(stream, 200, &response).await
}

async fn reject_non_loopback_caller(stream: &mut TcpStream) -> anyhow::Result<bool> {
    let Ok(addr) = stream.peer_addr() else {
        return Ok(false);
    };
    if addr.ip().is_loopback() {
        return Ok(false);
    }

    tracing::warn!("mesh hook: rejected non-loopback caller {addr}");
    http::respond_json(
        stream,
        403,
        &serde_json::json!({"error": "mesh hooks only accept localhost connections"}),
    )
    .await?;
    Ok(true)
}

async fn parse_hook_payload(stream: &mut TcpStream, body: &str) -> anyhow::Result<Option<Value>> {
    match serde_json::from_str(body) {
        Ok(payload) => Ok(Some(payload)),
        Err(e) => {
            tracing::warn!("mesh hook: invalid JSON: {e}");
            http::respond_json(stream, 400, &serde_json::json!({"error": "invalid JSON"})).await?;
            Ok(None)
        }
    }
}

async fn dispatch_hook(state: &MeshApi, payload: Value) -> Value {
    let hook = payload["hook"].as_str().unwrap_or("unknown");
    let node = state.node().await;

    let model = payload["model"].as_str().unwrap_or("").to_string();
    let messages: Vec<Value> = payload["messages"].as_array().cloned().unwrap_or_default();

    match hook {
        "pre_inference" => dispatch_pre_inference(&node, &payload, &model).await,
        "post_prefill" => {
            let entropy = payload["signals"]["first_token_entropy"]
                .as_f64()
                .unwrap_or(0.0);
            let margin = payload["signals"]["first_token_margin"]
                .as_f64()
                .unwrap_or(1.0);
            virtual_llm::handle_uncertain(&node, &model, &messages, entropy, margin).await
        }
        "mid_generation" => {
            let trigger = payload["trigger"].as_str().unwrap_or("unknown");
            let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
            tracing::info!("mesh hook 2b: trigger={trigger} n_decoded={n_decoded} model={model}");
            virtual_llm::handle_drift(&node, &model, &messages, n_decoded).await
        }
        _ => {
            tracing::warn!("mesh hook: unknown hook type: {hook}");
            serde_json::json!({ "action": "none" })
        }
    }
}

async fn dispatch_pre_inference(node: &crate::mesh::Node, payload: &Value, model: &str) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let (media_url, user_text) = pre_inference_media(payload, trigger);
    virtual_llm::handle_image(node, trigger, model, &media_url, &user_text).await
}

fn pre_inference_media(payload: &Value, trigger: &str) -> (String, String) {
    if trigger == "audio_no_support" {
        virtual_llm::extract_audio(payload)
    } else {
        virtual_llm::extract_image(payload)
    }
}
