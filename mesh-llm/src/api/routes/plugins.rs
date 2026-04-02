use super::super::{http::respond_error, MeshApi};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path_only) {
        ("GET", "/api/plugins") => handle_list(stream, state).await,
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            handle_tools(stream, state, p).await
        }
        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            handle_call(stream, state, p, body).await
        }
        _ => Ok(()),
    }
}

async fn handle_list(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let plugins = plugin_manager.list().await;
    let json = serde_json::to_string(&plugins)?;
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn handle_tools(stream: &mut TcpStream, state: &MeshApi, path: &str) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/tools");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.tools(plugin_name).await {
        Ok(tools) => {
            let json = serde_json::to_string(&tools)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Err(e) => {
            respond_error(stream, 404, &e.to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_call(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    if let Some((plugin_name, tool_name)) = rest.split_once("/tools/") {
        let payload = if body.trim().is_empty() { "{}" } else { body };
        let plugin_manager = state.inner.lock().await.plugin_manager.clone();
        match plugin_manager
            .call_tool(plugin_name, tool_name, payload)
            .await
        {
            Ok(result) if !result.is_error => {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    result.content_json.len(),
                    result.content_json
                );
                stream.write_all(resp.as_bytes()).await?;
            }
            Ok(result) => {
                respond_error(stream, 502, &result.content_json).await?;
            }
            Err(e) => {
                respond_error(stream, 502, &e.to_string()).await?;
            }
        }
    } else {
        respond_error(stream, 404, "Not found").await?;
    }
    Ok(())
}
