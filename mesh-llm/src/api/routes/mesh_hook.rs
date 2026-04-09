use super::MeshApi;
use crate::api::http;
use crate::inference::virtual_llm;
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

    let response = match hook {
        "pre_inference" => virtual_llm::handle_pre_inference(&payload),
        "post_prefill" => virtual_llm::handle_post_prefill(&payload),
        "pre_response" => virtual_llm::handle_pre_response(&payload),
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

    // TODO: look up async_id in AsyncConsultations
    // For now, always return 202 (not ready)
    http::respond_json(stream, 202, &serde_json::json!({"status": "pending"})).await
}
