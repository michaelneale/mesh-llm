mod chat;
mod discover;
mod mesh_hook;
mod objects;
mod plugins;
mod runtime;
mod search;

use super::MeshApi;
use std::future::Future;
use std::pin::Pin;
use tokio::net::TcpStream;

/// Legacy `/api/blackboard/*` routes are thin aliases onto the blackboard
/// plugin's declared HTTP bindings at `/api/plugins/blackboard/http/*`.
/// Rewrite the path and hand off to the plugin stapler.
async fn dispatch_blackboard(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    const PREFIX: &str = "/api/blackboard/";
    const PLUGIN_PREFIX: &str = "/api/plugins/blackboard/http/";
    let suffix = path_only.strip_prefix(PREFIX).unwrap_or("");
    let new_path_only = format!("{PLUGIN_PREFIX}{suffix}");
    let new_path = if let Some(query_start) = path.find('?') {
        format!("{new_path_only}{}", &path[query_start..])
    } else {
        new_path_only.clone()
    };
    plugins::handle(
        stream,
        state,
        method,
        &new_path,
        &new_path_only,
        body,
        raw_request,
    )
    .await
}

type DispatchRequestFn =
    for<'a> fn(
        &'a mut TcpStream,
        &'a MeshApi,
        &'a str,
        &'a str,
        &'a str,
        &'a str,
        &'a str,
        &'a [u8],
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<bool>> + Send + 'a>>;

pub(super) const DISPATCH_REQUEST: DispatchRequestFn =
    |stream, state, method, path, path_only, body, req, raw_request| {
        Box::pin(async move {
            match (method, path_only) {
                ("GET", "/api/discover") => {
                    discover::handle(stream, state).await?;
                    Ok(true)
                }
                ("GET", "/api/status")
                | ("GET", "/api/models")
                | ("GET", "/api/runtime")
                | ("GET", "/api/runtime/endpoints")
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
                ("GET", "/api/search") => {
                    search::handle(stream, path).await?;
                    Ok(true)
                }
                ("GET", "/api/plugins") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", "/api/plugins/endpoints") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", "/api/plugins/providers") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/providers/") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/manifest") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                (m, p)
                    if p.starts_with("/api/plugins/")
                        && matches!(m, "GET" | "POST" | "PUT" | "PATCH" | "DELETE") =>
                {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", "/api/blackboard/feed")
                | ("GET", "/api/blackboard/search")
                | ("POST", "/api/blackboard/post") => {
                    dispatch_blackboard(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                // Mesh hook callbacks from llama-server
                ("POST", "/mesh/hook") => {
                    mesh_hook::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("POST", "/api/objects")
                | ("POST", "/api/objects/complete")
                | ("POST", "/api/objects/abort") => {
                    objects::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                (m, p)
                    if m != "POST"
                        && (p.starts_with("/api/chat") || p.starts_with("/api/responses")) =>
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
        })
    };

pub(super) use DISPATCH_REQUEST as dispatch_request;
