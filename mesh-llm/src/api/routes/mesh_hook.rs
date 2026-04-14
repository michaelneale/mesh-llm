use super::MeshApi;
use crate::api::http;
use crate::inference::virtual_llm;
use serde_json::Value;
use tokio::net::TcpStream;

/// Handle mesh hook callbacks from llama-server.
///
/// Parses the JSON payload once, dispatches to typed handler functions.
/// Each hook blocks the C++ slot until we respond.
///
/// Only accepts connections from loopback — llama-server is always on localhost.
/// This prevents remote callers from triggering costly peer consultations even
/// when the management API is bound to 0.0.0.0 via `--listen-all`.
pub async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    _method: &str,
    _path: &str,
    body: &str,
) -> anyhow::Result<()> {
    if let Ok(addr) = stream.peer_addr() {
        if !addr.ip().is_loopback() {
            tracing::warn!("mesh hook: rejected non-loopback caller {addr}");
            return http::respond_json(
                stream,
                403,
                &serde_json::json!({"error": "mesh hooks only accept localhost connections"}),
            )
            .await;
        }
    }

    let payload: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("mesh hook: invalid JSON: {e}");
            return http::respond_json(stream, 400, &serde_json::json!({"error": "invalid JSON"}))
                .await;
        }
    };

    let hook = payload["hook"].as_str().unwrap_or("unknown");
    let node = state.node().await;

    let model = payload["model"].as_str().unwrap_or("").to_string();
    let messages: Vec<Value> = payload["messages"].as_array().cloned().unwrap_or_default();

    let response = match hook {
        "pre_inference" => {
            let trigger = payload["trigger"].as_str().unwrap_or("unknown");
            let (image_url, user_text) = virtual_llm::extract_image(&payload);
            virtual_llm::handle_image(&node, trigger, &model, &image_url, &user_text).await
        }
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
    };

    http::respond_json(stream, 200, &response).await
}
