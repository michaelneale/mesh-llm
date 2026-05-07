use super::http::{respond_bytes, respond_bytes_cached};
use tokio::net::TcpStream;

pub(super) async fn respond_console_index(stream: &mut TcpStream) -> anyhow::Result<bool> {
    if let Some(asset) = mesh_llm_ui::index() {
        respond_bytes(stream, 200, "OK", asset.content_type, asset.contents).await?;
        return Ok(true);
    }
    Ok(false)
}

pub(super) async fn respond_console_asset(
    stream: &mut TcpStream,
    path: &str,
) -> anyhow::Result<bool> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return Ok(false);
    }
    let Some(asset) = mesh_llm_ui::asset(rel) else {
        return Ok(false);
    };
    respond_bytes_cached(
        stream,
        200,
        "OK",
        asset.content_type,
        asset.cache_control,
        asset.contents,
    )
    .await?;
    Ok(true)
}
