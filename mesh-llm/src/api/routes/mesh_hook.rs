use super::MeshApi;
use crate::api::http;
use tokio::net::TcpStream;

/// Handle mesh hook callbacks from llama-server.
///
/// POST /mesh/hook           — hook callback (pre_inference, post_prefill, pre_response)
/// GET  /mesh/hook/poll/{id} — poll for async result
pub async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path) {
        ("POST", "/mesh/hook") => handle_hook(stream, state, body).await,
        ("GET", p) if p.starts_with("/mesh/hook/poll/") => {
            let async_id = p.strip_prefix("/mesh/hook/poll/").unwrap_or("");
            handle_poll(stream, state, async_id).await
        }
        _ => http::respond_json(stream, 404, &serde_json::json!({"error": "not found"})).await,
    }
}

async fn handle_hook(stream: &mut TcpStream, _state: &MeshApi, body: &str) -> anyhow::Result<()> {
    let payload: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("mesh hook: invalid JSON: {e}");
            return http::respond_json(stream, 400, &serde_json::json!({"error": "invalid JSON"}))
                .await;
        }
    };

    let hook = payload["hook"].as_str().unwrap_or("unknown");
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let request_id = payload["request_id"].as_str().unwrap_or("");
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!(
        "mesh hook: hook={hook} trigger={trigger} model={model} request_id={request_id}"
    );

    // For now, return none for all hooks — the plumbing is what we're testing.
    // TODO: implement decision logic per hook type in inference/virtual.rs
    let response = match hook {
        "pre_inference" => {
            // Enable Hook 2 with default entropy threshold
            serde_json::json!({
                "action": "none",
                "entropy_threshold": 5.0
            })
        }
        "post_prefill" => {
            let entropy = payload["signals"]["first_token_entropy"]
                .as_f64()
                .unwrap_or(0.0);
            tracing::info!("mesh hook: post_prefill entropy={entropy:.2}");
            serde_json::json!({ "action": "none" })
        }
        "pre_response" => {
            let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
            let stop_reason = payload["stop_reason"].as_str().unwrap_or("");
            tracing::info!(
                "mesh hook: pre_response n_decoded={n_decoded} stop_reason={stop_reason}"
            );
            serde_json::json!({ "action": "none" })
        }
        _ => {
            tracing::warn!("mesh hook: unknown hook type: {hook}");
            serde_json::json!({ "action": "none" })
        }
    };

    http::respond_json(stream, 200, &response).await
}

async fn handle_poll(
    stream: &mut TcpStream,
    _state: &MeshApi,
    async_id: &str,
) -> anyhow::Result<()> {
    tracing::debug!("mesh hook poll: async_id={async_id}");

    // TODO: look up async_id in a DashMap of pending consultations
    // For now, always return 202 (not ready)
    http::respond_json(stream, 202, &serde_json::json!({"status": "pending"})).await
}
