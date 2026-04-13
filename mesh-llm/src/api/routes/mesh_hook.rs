use super::MeshApi;
use crate::api::http;
use crate::inference::virtual_llm;
use tokio::net::TcpStream;

/// Handle mesh hook callbacks from llama-server.
///
/// POST /mesh/hook — hook callback (pre_inference, post_prefill, pre_response)
///
/// All hooks are synchronous from the C++ side — each is a blocking POST.
/// But mesh-llm can do async work (consulting peers over QUIC) before
/// responding. The hook blocks llama-server's slot until we reply.
pub async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    _method: &str,
    _path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let payload: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("mesh hook: invalid JSON: {e}");
            return http::respond_json(stream, 400, &serde_json::json!({"error": "invalid JSON"}))
                .await;
        }
    };

    let hook = payload["hook"].as_str().unwrap_or("unknown");
    let node = state.node().await;

    let response = match hook {
        "pre_inference" => virtual_llm::handle_pre_inference(&node, &payload).await,
        "post_prefill" => virtual_llm::handle_post_prefill(&node, &payload).await,
        "pre_response" => virtual_llm::handle_pre_response(&node, &payload).await,
        _ => {
            tracing::warn!("mesh hook: unknown hook type: {hook}");
            serde_json::json!({ "action": "none" })
        }
    };

    http::respond_json(stream, 200, &response).await
}
