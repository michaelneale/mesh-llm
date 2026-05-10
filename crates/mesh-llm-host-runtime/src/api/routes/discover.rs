use super::super::{http::respond_error, MeshApi};
use crate::network::{discovery, nostr};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let (mode, relays) = {
        let inner = state.inner.lock().await;
        (inner.mesh_discovery_mode, inner.nostr_relays.clone())
    };
    let filter = nostr::MeshFilter::default();
    let json = match mode {
        discovery::MeshDiscoveryMode::Nostr => {
            match nostr::discover(&relays, &filter, None).await {
                Ok(meshes) => serde_json::to_string(&meshes),
                Err(e) => {
                    respond_error(stream, 500, &format!("Discovery failed: {e}")).await?;
                    return Ok(());
                }
            }
        }
        discovery::MeshDiscoveryMode::Mdns => {
            match discovery::discover_lan(&filter, None, std::time::Duration::from_secs(3)).await {
                Ok(meshes) => serde_json::to_string(&meshes),
                Err(e) => {
                    respond_error(stream, 500, &format!("Discovery failed: {e}")).await?;
                    return Ok(());
                }
            }
        }
    };

    match json {
        Ok(json) => {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Err(_) => respond_error(stream, 500, "Failed to serialize").await?,
    }
    Ok(())
}
