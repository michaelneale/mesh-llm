use super::super::{
    http::{respond_error, respond_json},
    MeshApi,
};
use crate::plugin::stapler;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use url::form_urlencoded;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path_only) {
        ("GET", "/api/plugins") => handle_list(stream, state).await,
        ("GET", "/api/plugins/endpoints") => handle_endpoints(stream, state).await,
        ("GET", "/api/plugins/providers") => handle_providers(stream, state).await,
        ("GET", p) if p.starts_with("/api/plugins/providers/") => {
            handle_provider(stream, state, p).await
        }
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/manifest") => {
            handle_manifest(stream, state, p).await
        }
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            handle_tools(stream, state, p).await
        }
        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            handle_call(stream, state, p, body).await
        }
        (m, p)
            if p.starts_with("/api/plugins/")
                && matches!(m, "GET" | "POST" | "PUT" | "PATCH" | "DELETE") =>
        {
            handle_stapled_http(stream, state, method, path, path_only, body).await
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

async fn handle_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.endpoints().await {
        Ok(endpoints) => respond_json(stream, 200, &endpoints).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_providers(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.capability_providers().await {
        Ok(providers) => respond_json(stream, 200, &providers).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_provider(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let capability = &path["/api/plugins/providers/".len()..];
    let capability = urlencoding::decode(capability)
        .map(|value| value.into_owned())
        .unwrap_or_else(|_| capability.to_string());
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.provider_for_capability(&capability).await {
        Ok(Some(provider)) => respond_json(stream, 200, &provider).await?,
        Ok(None) => {
            respond_error(
                stream,
                404,
                &format!("No provider for capability '{}'", capability),
            )
            .await?
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
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

async fn handle_manifest(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/manifest");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.manifest_json(plugin_name).await {
        Ok(Some(manifest)) => {
            let json = serde_json::to_string(&manifest)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
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

async fn handle_stapled_http(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    let Some((plugin_name, route_path)) = parse_stapled_http_path(path_only) else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };

    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let manifest = match plugin_manager.manifest(plugin_name).await {
        Ok(Some(manifest)) => manifest,
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
            return Ok(());
        }
        Err(err) => {
            respond_error(stream, 404, &err.to_string()).await?;
            return Ok(());
        }
    };

    let Some(binding) = manifest.http_bindings.iter().find(|binding| {
        stapler::http_binding_route(plugin_name, binding)
            .map(|route| route.method == method && route.route_path == route_path)
            .unwrap_or(false)
    }) else {
        respond_error(stream, 404, "No matching plugin HTTP binding").await?;
        return Ok(());
    };

    let Some(operation_name) = binding.operation_name.as_deref() else {
        respond_error(
            stream,
            501,
            "HTTP binding does not declare an operation_name yet",
        )
        .await?;
        return Ok(());
    };

    let args = match build_http_arguments(path, body) {
        Ok(args) => args,
        Err(err) => {
            respond_error(stream, 400, &err).await?;
            return Ok(());
        }
    };

    match plugin_manager
        .call_tool(
            plugin_name,
            operation_name,
            &Value::Object(args).to_string(),
        )
        .await
    {
        Ok(result) if !result.is_error => match serde_json::from_str::<Value>(&result.content_json)
        {
            Ok(value) => respond_json(stream, 200, &value).await?,
            Err(_) => {
                respond_error(
                    stream,
                    502,
                    "Plugin returned a non-JSON response for a buffered HTTP binding",
                )
                .await?;
            }
        },
        Ok(result) => {
            respond_error(stream, 502, &result.content_json).await?;
        }
        Err(err) => {
            respond_error(stream, 502, &err.to_string()).await?;
        }
    }

    Ok(())
}

fn parse_stapled_http_path(path_only: &str) -> Option<(&str, &str)> {
    let rest = path_only.strip_prefix("/api/plugins/")?;
    let (plugin_name, remainder) = rest.split_once("/http")?;
    if plugin_name.is_empty() || remainder.is_empty() {
        return None;
    }
    Some((
        plugin_name,
        &path_only[.."/api/plugins/".len() + plugin_name.len() + "/http".len() + remainder.len()],
    ))
}

fn build_http_arguments(path: &str, body: &str) -> Result<Map<String, Value>, String> {
    let mut args = query_arguments(path);
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(args);
    }
    let body_value: Value =
        serde_json::from_str(trimmed).map_err(|err| format!("Invalid JSON body: {err}"))?;
    let Value::Object(body_map) = body_value else {
        return Err("Buffered plugin HTTP bindings currently require a JSON object body".into());
    };
    args.extend(body_map);
    Ok(args)
}

fn query_arguments(path: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let Some((_, query)) = path.split_once('?') else {
        return args;
    };
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        args.insert(key.into_owned(), Value::String(value.into_owned()));
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stapled_http_path() {
        let parsed = parse_stapled_http_path("/api/plugins/demo/http/feed").unwrap();
        assert_eq!(parsed.0, "demo");
        assert_eq!(parsed.1, "/api/plugins/demo/http/feed");
    }

    #[test]
    fn query_arguments_decode_values() {
        let args = query_arguments("/api/plugins/demo/http/feed?name=hello%20world&limit=10");
        assert_eq!(args.get("name"), Some(&Value::String("hello world".into())));
        assert_eq!(args.get("limit"), Some(&Value::String("10".into())));
    }

    #[test]
    fn build_http_arguments_merges_query_and_body() {
        let args = build_http_arguments(
            "/api/plugins/demo/http/feed?from=alice",
            r#"{"limit":10,"from":"bob"}"#,
        )
        .unwrap();
        assert_eq!(args.get("limit"), Some(&Value::Number(10.into())));
        assert_eq!(args.get("from"), Some(&Value::String("bob".into())));
    }
}
