use super::super::{http::respond_error, MeshApi};
use crate::network::nostr;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let relays = state.inner.lock().await.nostr_relays.clone();
    let filter = nostr::MeshFilter::default();
    match nostr::discover(&relays, &filter, None).await {
        Ok(meshes) => {
            if let Ok(json) = serde_json::to_string(&meshes) {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    json.len(),
                    json
                );
                stream.write_all(resp.as_bytes()).await?;
            } else {
                respond_error(stream, 500, "Failed to serialize").await?;
            }
        }
        Err(e) => {
            respond_error(stream, 500, &format!("Discovery failed: {e}")).await?;
        }
    }
    Ok(())
}
