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
    let is_openai_passthrough = path_only.starts_with("/v1/") || path_only == "/models";
    if method == "OPTIONS" && is_openai_passthrough {
        stream
            .write_all(
                b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type, authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\n\r\n",
            )
            .await?;
        return Ok(());
    }

    if method != "POST" && !(method == "GET" && is_openai_passthrough) {
        return respond_error(stream, 405, "Method Not Allowed").await;
    }

    let upstream_path = if is_openai_passthrough {
        path_only
    } else if path_only.starts_with("/api/chat") {
        "/v1/chat/completions"
    } else if path_only.starts_with("/api/responses") {
        "/v1/responses"
    } else {
        return Ok(());
    };

    let port = state.inner.lock().await.api_port;

    let target = format!("127.0.0.1:{port}");
    if let Ok(mut upstream) = TcpStream::connect(&target).await {
        let rewritten = if is_openai_passthrough {
            req.to_string()
        } else if path_only.starts_with("/api/chat") {
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
