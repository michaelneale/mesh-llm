use super::super::{http::respond_error, MeshApi};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    req: &str,
) -> anyhow::Result<()> {
    if method != "POST" {
        return respond_error(stream, 405, "Method Not Allowed").await;
    }

    let upstream_path = if path_only.starts_with("/api/chat") {
        "/v1/chat/completions"
    } else if path_only.starts_with("/api/responses") {
        "/v1/responses"
    } else {
        return Ok(());
    };

    let inner = state.inner.lock().await;
    if !inner.llama_ready && !inner.is_client {
        drop(inner);
        return respond_error(stream, 503, "LLM not ready").await;
    }
    let port = inner.api_port;
    drop(inner);

    let target = format!("127.0.0.1:{port}");
    if let Ok(mut upstream) = TcpStream::connect(&target).await {
        let rewritten = if path_only.starts_with("/api/chat") {
            req.replacen("/api/chat", upstream_path, 1)
        } else {
            req.replacen("/api/responses", upstream_path, 1)
        };
        upstream.write_all(rewritten.as_bytes()).await?;
        tokio::io::copy_bidirectional(stream, &mut upstream).await?;
    } else {
        respond_error(stream, 502, "Cannot reach LLM server").await?;
    }
    Ok(())
}
