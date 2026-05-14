use super::super::{
    http::{respond_error, respond_json},
    MeshApi,
};
use crate::plugin::stapler;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use url::form_urlencoded;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HttpBindingTransferMode {
    Buffered,
    StreamedRequest,
    StreamedResponse,
    StreamedBidirectional,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
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
            handle_stapled_http(stream, state, method, path, path_only, body, raw_request).await
        }
        _ => Ok(()),
    }
}

async fn handle_list(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugins = state.plugins().await;
    respond_json(stream, 200, &plugins).await?;
    Ok(())
}

async fn handle_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.runtime_endpoints().await {
        Ok(endpoints) => respond_json(stream, 200, &endpoints).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_providers(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.plugin_capability_providers().await {
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
    match state.plugin_provider_for_capability(&capability).await {
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
            respond_error(stream, 500, &e.to_string()).await?;
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
            .invoke_operation(plugin_name, tool_name, payload)
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
    raw_request: &[u8],
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
            respond_error(stream, 500, &err.to_string()).await?;
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

    if binding_transfer_mode(binding) != HttpBindingTransferMode::Buffered {
        return handle_streamed_http_binding(
            stream,
            &plugin_manager,
            plugin_name,
            binding,
            raw_request,
        )
        .await;
    }

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
        .invoke_operation(
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

async fn handle_streamed_http_binding(
    client_stream: &mut TcpStream,
    plugin_manager: &crate::plugin::PluginManager,
    plugin_name: &str,
    binding: &crate::plugin::proto::HttpBindingManifest,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let forwarded_request = rewrite_http_request_path(raw_request, &binding.path)?;
    let stream_id = format!("http-{}-{}", std::process::id(), rand::random::<u64>());
    let request = crate::plugin::proto::OpenStreamRequest {
        stream_id,
        purpose: crate::plugin::proto::StreamPurpose::Generic as i32,
        mode: crate::plugin::proto::StreamMode::Http1 as i32,
        bidirectional: true,
        content_type: Some("application/http".into()),
        correlation_id: None,
        metadata_json: Some(
            serde_json::json!({
                "binding_id": binding.binding_id,
                "method": method_name(binding.method),
                "path": binding.path,
            })
            .to_string(),
        ),
        expected_bytes: Some(forwarded_request.len() as u64),
        idle_timeout_ms: Some(30_000),
    };
    let mut plugin_stream = plugin_manager.connect_stream(plugin_name, request).await?;
    plugin_stream.write_all(&forwarded_request).await?;
    plugin_stream.shutdown().await?;

    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = plugin_stream.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        client_stream.write_all(&buf[..read]).await?;
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

fn rewrite_http_request_path(raw_request: &[u8], path: &str) -> anyhow::Result<Vec<u8>> {
    let header_end = raw_request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .ok_or_else(|| anyhow::anyhow!("HTTP request is missing a header terminator"))?;
    let mut headers_buf = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers_buf);
    req.parse(raw_request)
        .map_err(|err| anyhow::anyhow!("HTTP parse error while rewriting request path: {err}"))?;

    let method = req.method.unwrap_or("GET");
    let version = req.version.unwrap_or(1);
    let original_path = req.path.unwrap_or("/");
    let query = original_path
        .find('?')
        .map(|i| &original_path[i..])
        .unwrap_or("");
    let mut rebuilt = format!(
        "{method} {}{} HTTP/1.{version}\r\n",
        normalized_http_path(path),
        query
    );

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(&raw_request[header_end..]);
    Ok(forwarded)
}

fn normalized_http_path(path: &str) -> &str {
    if path.is_empty() {
        "/"
    } else {
        path
    }
}

fn method_name(value: i32) -> &'static str {
    match crate::plugin::proto::HttpMethod::try_from(value)
        .unwrap_or(crate::plugin::proto::HttpMethod::Unspecified)
    {
        crate::plugin::proto::HttpMethod::Get => "GET",
        crate::plugin::proto::HttpMethod::Post => "POST",
        crate::plugin::proto::HttpMethod::Put => "PUT",
        crate::plugin::proto::HttpMethod::Patch => "PATCH",
        crate::plugin::proto::HttpMethod::Delete => "DELETE",
        crate::plugin::proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn binding_transfer_mode(
    binding: &crate::plugin::proto::HttpBindingManifest,
) -> HttpBindingTransferMode {
    let request_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.request_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    let response_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.response_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    match (request_streamed, response_streamed) {
        (false, false) => HttpBindingTransferMode::Buffered,
        (true, false) => HttpBindingTransferMode::StreamedRequest,
        (false, true) => HttpBindingTransferMode::StreamedResponse,
        (true, true) => HttpBindingTransferMode::StreamedBidirectional,
    }
}

fn query_arguments(path: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let Some((_, query)) = path.split_once('?') else {
        return args;
    };
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        let json_value = if value == "true" {
            Value::Bool(true)
        } else if value == "false" {
            Value::Bool(false)
        } else if let Ok(n) = value.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(f) = value.parse::<f64>() {
            // NaN and Infinity are not valid JSON numbers; keep the raw string.
            match serde_json::Number::from_f64(f) {
                Some(n) => Value::Number(n),
                None => Value::String(value.into_owned()),
            }
        } else {
            Value::String(value.into_owned())
        };
        args.insert(key.into_owned(), json_value);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MeshApi;
    use crate::mesh::{Node, NodeRole};
    use crate::network::affinity;
    use crate::plugin::{self};
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

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
        assert_eq!(args.get("limit"), Some(&Value::Number(10.into())));
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

    #[test]
    fn rewrite_http_request_path_updates_request_line_only() {
        let raw = b"POST /api/plugins/demo/http/feed?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\nConnection: keep-alive\r\n\r\n{\"a\":1}";
        let rewritten = rewrite_http_request_path(raw, "/feed").unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.starts_with("POST /feed?x=1 HTTP/1.1\r\n"));
        assert!(text.contains("Host: localhost\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"));
    }

    #[test]
    fn rewrite_http_request_path_without_query_string() {
        let raw = b"GET /api/plugins/demo/http/items HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let rewritten = rewrite_http_request_path(raw, "/items").unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.starts_with("GET /items HTTP/1.1\r\n"));
    }

    #[test]
    fn binding_transfer_mode_covers_all_streaming_combinations() {
        let mut binding = crate::plugin::proto::HttpBindingManifest {
            binding_id: "demo".into(),
            method: crate::plugin::proto::HttpMethod::Post as i32,
            path: "/demo".into(),
            operation_name: Some("demo".into()),
            request_body_mode: crate::plugin::proto::HttpBodyMode::Buffered as i32,
            response_body_mode: crate::plugin::proto::HttpBodyMode::Buffered as i32,
            request_schema_json: None,
            response_schema_json: None,
        };
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::Buffered
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedRequest
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Buffered as i32;
        binding.response_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedResponse
        );

        binding.request_body_mode = crate::plugin::proto::HttpBodyMode::Streamed as i32;
        assert_eq!(
            binding_transfer_mode(&binding),
            HttpBindingTransferMode::StreamedBidirectional
        );
    }

    async fn connected_tcp_streams() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        (client, server)
    }

    async fn build_test_api_with_plugin_manager(plugin_manager: plugin::PluginManager) -> MeshApi {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let runtime_data_collector = node.runtime_data_collector();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "runtime",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });
        MeshApi::new(crate::api::MeshApiConfig {
            node,
            model_name: "test-model".into(),
            api_port: 3131,
            model_size_bytes: 0,
            owner_key_path: None,
            plugin_manager,
            affinity_router: affinity::AffinityRouter::default(),
            runtime_data_collector,
            runtime_data_producer,
        })
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamed_http_bindings_proxy_all_transfer_modes_over_side_streams() {
        struct NoopBridge;
        impl plugin::PluginRpcBridge for NoopBridge {
            fn handle_request(
                &self,
                _plugin_name: String,
                _method: String,
                _params_json: String,
            ) -> plugin::BridgeFuture<Result<plugin::RpcResult, crate::plugin::proto::ErrorResponse>>
            {
                Box::pin(async {
                    Err(crate::plugin::proto::ErrorResponse {
                        code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
                        message: "unexpected request".into(),
                        data_json: String::new(),
                    })
                })
            }

            fn handle_notification(
                &self,
                _plugin_name: String,
                _method: String,
                _params_json: String,
            ) -> plugin::BridgeFuture<()> {
                Box::pin(async {})
            }
        }

        let plugin_manager =
            plugin::PluginManager::for_test_bridge(&["demo"], std::sync::Arc::new(NoopBridge));
        let transfer_modes = [
            (
                crate::plugin::proto::HttpBodyMode::Buffered,
                crate::plugin::proto::HttpBodyMode::Streamed,
            ),
            (
                crate::plugin::proto::HttpBodyMode::Streamed,
                crate::plugin::proto::HttpBodyMode::Buffered,
            ),
            (
                crate::plugin::proto::HttpBodyMode::Streamed,
                crate::plugin::proto::HttpBodyMode::Streamed,
            ),
        ];
        for (request_mode, response_mode) in transfer_modes {
            plugin_manager
                .set_test_manifests(std::collections::BTreeMap::from([(
                    "demo".into(),
                    crate::plugin::proto::PluginManifest {
                        http_bindings: vec![crate::plugin::proto::HttpBindingManifest {
                            binding_id: "stream".into(),
                            method: crate::plugin::proto::HttpMethod::Post as i32,
                            path: "/stream".into(),
                            operation_name: Some("stream".into()),
                            request_body_mode: request_mode as i32,
                            response_body_mode: response_mode as i32,
                            request_schema_json: None,
                            response_schema_json: None,
                        }],
                        ..Default::default()
                    },
                )]))
                .await;
            plugin_manager
                .set_test_stream_handler("demo", move |request| {
                    Box::pin(async move {
                        let mut request = request;
                        request.stream_id = "s".into();
                        let listener =
                            mesh_llm_plugin::bind_side_stream("demo", &request.stream_id).await?;
                        let response = listener.open_stream_response(&request);
                        let endpoint = response.endpoint.clone().unwrap();
                        let transport_kind = response.transport_kind;
                        tokio::spawn(async move {
                            let mut plugin_stream = listener.accept().await.unwrap();
                            let mut request_bytes =
                                vec![0u8; request.expected_bytes.unwrap_or_default() as usize];
                            plugin_stream
                                .read_exact_bytes(&mut request_bytes)
                                .await
                                .unwrap();
                            let request_text = String::from_utf8_lossy(&request_bytes);
                            assert!(request_text.starts_with("POST /stream HTTP/1.1\r\n"));
                            assert!(request_text.contains("Connection: close\r\n"));
                            plugin_stream
                                .write_all_bytes(
                                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\n{\"ok\":true}\n",
                                )
                                .await
                                .unwrap();
                        });
                        crate::plugin::connect_test_side_stream(&endpoint, transport_kind).await
                    })
                })
                .await;
            let state = build_test_api_with_plugin_manager(plugin_manager.clone()).await;
            let (mut observed_client, mut response_stream) = connected_tcp_streams().await;
            let raw_request = b"POST /api/plugins/demo/http/stream HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 7\r\nConnection: keep-alive\r\n\r\n{\"a\":1}";
            handle_stapled_http(
                &mut response_stream,
                &state,
                "POST",
                "/api/plugins/demo/http/stream",
                "/api/plugins/demo/http/stream",
                "{\"a\":1}",
                raw_request,
            )
            .await
            .unwrap();
            response_stream.shutdown().await.unwrap();
            let mut response_bytes = Vec::new();
            observed_client
                .read_to_end(&mut response_bytes)
                .await
                .unwrap();
            let response_text = String::from_utf8_lossy(&response_bytes);
            assert!(response_text.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(response_text.contains("{\"ok\":true}"));
        }
    }
}
