mod blackboard;
mod chat;
mod discover;
mod objects;
mod plugins;
mod runtime;

use super::MeshApi;
use tokio::net::TcpStream;

pub(super) async fn dispatch_request(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    req: &str,
) -> anyhow::Result<bool> {
    match (method, path_only) {
        ("GET", "/api/discover") => {
            discover::handle(stream, state).await?;
            Ok(true)
        }
        ("GET", "/api/status")
        | ("GET", "/api/models")
        | ("GET", "/api/runtime")
        | ("GET", "/api/runtime/processes")
        | ("POST", "/api/runtime/models")
        | ("GET", "/api/events") => {
            runtime::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        ("DELETE", p) if p.starts_with("/api/runtime/models/") => {
            runtime::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        ("GET", "/api/plugins") => {
            plugins::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            plugins::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            plugins::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        ("GET", "/api/blackboard/feed")
        | ("GET", "/api/blackboard/search")
        | ("POST", "/api/blackboard/post") => {
            blackboard::handle(stream, state, method, path, body).await?;
            Ok(true)
        }
        ("POST", "/api/objects")
        | ("POST", "/api/objects/complete")
        | ("POST", "/api/objects/abort") => {
            objects::handle(stream, state, method, path_only, body).await?;
            Ok(true)
        }
        (m, p)
            if m != "POST" && (p.starts_with("/api/chat") || p.starts_with("/api/responses")) =>
        {
            chat::handle(stream, state, method, path_only, req).await?;
            Ok(true)
        }
        ("POST", p) if p.starts_with("/api/chat") || p.starts_with("/api/responses") => {
            chat::handle(stream, state, method, path_only, req).await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
