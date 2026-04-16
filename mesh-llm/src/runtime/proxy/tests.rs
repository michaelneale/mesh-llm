use super::*;
use crate::inference::pipeline;
use crate::network::router;
use crate::plugin;
use crate::plugins::blobstore::BlobStore;
use base64::Engine;
use rmcp::model::ErrorCode;
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, watch};

async fn spawn_api_proxy_test_harness(
    targets: election::ModelTargets,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (_target_tx, target_rx) = watch::channel(targets);
    let (drop_tx, _drop_rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(api_proxy(
        node,
        addr.port(),
        target_rx,
        drop_tx,
        Some(listener),
        false,
        affinity::AffinityRouter::default(),
    ));
    (addr, handle)
}

async fn spawn_api_proxy_test_harness_with_plugin_manager(
    targets: election::ModelTargets,
    plugin_manager: plugin::PluginManager,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_plugin_manager(plugin_manager).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (_target_tx, target_rx) = watch::channel(targets);
    let (drop_tx, _drop_rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(api_proxy(
        node,
        addr.port(),
        target_rx,
        drop_tx,
        Some(listener),
        false,
        affinity::AffinityRouter::default(),
    ));
    (addr, handle)
}

#[derive(Clone)]
struct BlobstoreTestBridge {
    plugin_name: String,
    store: BlobStore,
}

#[derive(Clone, Default)]
struct NoopTestBridge;

impl BlobstoreTestBridge {
    fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

impl plugin::PluginRpcBridge for NoopTestBridge {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        Box::pin(async move {
            Err(plugin::proto::ErrorResponse {
                code: ErrorCode::METHOD_NOT_FOUND.0,
                message: format!("Noop test bridge cannot handle {plugin_name}:{method}"),
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

impl plugin::PluginRpcBridge for BlobstoreTestBridge {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        let expected_plugin_name = self.plugin_name.clone();
        let store = self.store.clone();
        Box::pin(async move {
            if plugin_name != expected_plugin_name {
                return Err(Self::error_response(format!(
                    "Unsupported test plugin '{}'",
                    plugin_name
                )));
            }

            if method == "tools/call" {
                let request: mesh_llm_plugin::OperationRequest = serde_json::from_str(&params_json)
                    .map_err(|err| Self::error_response(err.to_string()))?;
                let result_json = match request.name.as_str() {
                    crate::plugins::blobstore::PUT_REQUEST_OBJECT_TOOL => {
                        let request: crate::plugins::blobstore::PutRequestObjectRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .put_request_object(request)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    crate::plugins::blobstore::GET_REQUEST_OBJECT_TOOL => {
                        let request: crate::plugins::blobstore::GetRequestObjectRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .get_request_object(request)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    crate::plugins::blobstore::COMPLETE_REQUEST_TOOL
                    | crate::plugins::blobstore::ABORT_REQUEST_TOOL => {
                        let request: crate::plugins::blobstore::FinishRequestRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .finish_request(&request.request_id)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    _ => {
                        return Err(Self::error_response(format!(
                            "Unsupported blobstore tool '{}'",
                            request.name
                        )));
                    }
                };
                return Ok(plugin::RpcResult { result_json });
            }

            let result_json = match method.as_str() {
                crate::plugins::blobstore::PUT_REQUEST_OBJECT_METHOD => {
                    let request: crate::plugins::blobstore::PutRequestObjectRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .put_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::GET_REQUEST_OBJECT_METHOD => {
                    let request: crate::plugins::blobstore::GetRequestObjectRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .get_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::COMPLETE_REQUEST_METHOD => {
                    let request: crate::plugins::blobstore::FinishRequestRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::ABORT_REQUEST_METHOD => {
                    let request: crate::plugins::blobstore::FinishRequestRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                _ => {
                    return Err(Self::error_response(format!(
                        "Unsupported blobstore RPC '{}'",
                        method
                    )));
                }
            };

            Ok(plugin::RpcResult { result_json })
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

fn temp_blobstore_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mesh-llm-runtime-proxy-{name}-{}",
        rand::random::<u64>()
    ))
}

async fn start_blobstore_plugin_manager() -> (plugin::PluginManager, std::path::PathBuf) {
    start_blobstore_plugin_manager_for(
        plugin::BLOBSTORE_PLUGIN_ID,
        vec!["internal:blobstore".into(), "object-store.v1".into()],
    )
    .await
}

async fn start_blobstore_plugin_manager_for(
    plugin_name: &str,
    capabilities: Vec<String>,
) -> (plugin::PluginManager, std::path::PathBuf) {
    let root = temp_blobstore_root("blobstore");
    let bridge = BlobstoreTestBridge {
        plugin_name: plugin_name.to_string(),
        store: BlobStore::new(root.clone()),
    };
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
    let mut manifests = HashMap::new();
    manifests.insert(
        plugin_name.to_string(),
        mesh_llm_plugin::proto::PluginManifest {
            capabilities,
            ..Default::default()
        },
    );
    plugin_manager
        .set_test_manifests(manifests.into_iter().collect())
        .await;
    (plugin_manager, root)
}

async fn start_inference_endpoint_plugin_manager(
    address: String,
    models: Vec<String>,
) -> plugin::PluginManager {
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[], Arc::new(NoopTestBridge));
    plugin_manager
        .set_test_inference_endpoints(vec![plugin::InferenceEndpointRoute {
            plugin_name: plugin::LEMONADE_PLUGIN_ID.into(),
            endpoint_id: "lemonade".into(),
            address,
            models,
        }])
        .await;
    plugin_manager
}

async fn spawn_capturing_upstream(
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    spawn_status_upstream("200 OK", response_body).await
}

async fn spawn_status_upstream(
    status: &str,
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let status = status.to_string();
    let response = response_body.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let raw = read_raw_http_request(&mut stream).await;
        let _ = request_tx.send(raw);

        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        );
        stream.write_all(resp.as_bytes()).await.unwrap();
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

async fn spawn_streaming_upstream(
    content_type: &str,
    chunks: Vec<(Duration, Vec<u8>)>,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let content_type = content_type.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let raw = read_raw_http_request(&mut stream).await;
        let _ = request_tx.send(raw);

        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(header.as_bytes()).await.is_err() {
            return;
        }

        for (delay, chunk) in chunks {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let chunk_header = format!("{:x}\r\n", chunk.len());
            if stream.write_all(chunk_header.as_bytes()).await.is_err() {
                return;
            }
            if stream.write_all(&chunk).await.is_err() {
                return;
            }
            if stream.write_all(b"\r\n").await.is_err() {
                return;
            }
        }

        let _ = stream.write_all(b"0\r\n\r\n").await;
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

async fn read_raw_http_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut raw = Vec::new();
    loop {
        let mut chunk = [0u8; 8192];
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "unexpected EOF while reading test request");
        raw.extend_from_slice(&chunk[..n]);

        let Some(header_end) = find_header_end(&raw) else {
            continue;
        };
        let headers = std::str::from_utf8(&raw[..header_end]).unwrap();

        if header_has_token(headers, "transfer-encoding", "chunked") {
            if raw[header_end..]
                .windows(5)
                .any(|window| window == b"0\r\n\r\n")
            {
                return raw;
            }
            continue;
        }

        if let Some(content_length) = content_length(headers) {
            if raw.len() >= header_end + content_length {
                raw.truncate(header_end + content_length);
                return raw;
            }
            continue;
        }

        raw.truncate(header_end);
        return raw;
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim().eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

fn header_has_token(headers: &str, name: &str, token: &str) -> bool {
    header_value(headers, name)
        .map(|value| {
            value
                .split(',')
                .any(|part| part.trim().eq_ignore_ascii_case(token))
        })
        .unwrap_or(false)
}

fn content_length(headers: &str) -> Option<usize> {
    header_value(headers, "content-length")?.parse().ok()
}

fn local_targets(entries: &[(&str, u16)]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets = entries
        .iter()
        .map(|(model, port)| {
            (
                (*model).to_string(),
                vec![election::InferenceTarget::Local(*port)],
            )
        })
        .collect::<HashMap<_, _>>();
    targets
}

fn unavailable_targets(models: &[&str]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets = models
        .iter()
        .map(|model| ((*model).to_string(), vec![election::InferenceTarget::None]))
        .collect();
    targets
}

fn single_model_targets(model: &str, ports: &[u16]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        model.to_string(),
        ports
            .iter()
            .copied()
            .map(election::InferenceTarget::Local)
            .collect(),
    );
    targets
}

fn build_chunked_request(path: &str, body: &[u8], chunks: &[usize]) -> Vec<u8> {
    let mut out = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
    )
    .into_bytes();
    let mut pos = 0usize;
    for &chunk_len in chunks {
        let end = pos + chunk_len;
        out.extend_from_slice(format!("{chunk_len:x}\r\n").as_bytes());
        out.extend_from_slice(&body[pos..end]);
        out.extend_from_slice(b"\r\n");
        pos = end;
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

async fn read_until_contains(stream: &mut TcpStream, needle: &[u8], timeout: Duration) -> Vec<u8> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut response = Vec::new();
    while !contains_bytes(&response, needle) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "timed out waiting for {:?} in response: {}",
            String::from_utf8_lossy(needle),
            String::from_utf8_lossy(&response)
        );
        let mut chunk = [0u8; 8192];
        let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
            .await
            .expect("timed out waiting for response bytes")
            .unwrap();
        assert!(n > 0, "unexpected EOF while waiting for response bytes");
        response.extend_from_slice(&chunk[..n]);
    }
    response
}

async fn send_request_and_read_response(addr: SocketAddr, parts: Vec<Vec<u8>>) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    for part in parts {
        stream.write_all(&part).await.unwrap();
    }
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    String::from_utf8(response).unwrap()
}

async fn connected_tcp_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).await.unwrap();
    let (server, _) = listener.accept().await.unwrap();
    (client, server)
}

#[tokio::test]
async fn test_api_proxy_integration_fragmented_post_body() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );

    let response = send_request_and_read_response(
        proxy_addr,
        vec![
            headers.as_bytes()[..38].to_vec(),
            headers.as_bytes()[38..].to_vec(),
            body.as_bytes()[..12].to_vec(),
            body.as_bytes()[12..].to_vec(),
        ],
    )
    .await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(&body));
    assert!(raw.contains("Connection: close"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_chunked_body() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = br#"{"model":"test","messages":[{"role":"user","content":"chunked"}]}"#;
    let request = build_chunked_request("/v1/chat/completions", body, &[17, body.len() - 17]);

    let response = send_request_and_read_response(proxy_addr, vec![request]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("Transfer-Encoding: chunked"));
    assert!(raw.contains("\"model\":\"test\""));
    assert!(raw.contains("0\r\n\r\n"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_moe_remote_failure_removes_peer_for_faildown() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let peer_id = iroh::EndpointId::from(
        iroh::SecretKey::from_bytes(&{
            let mut bytes = [0u8; 32];
            bytes[0] = 42;
            bytes
        })
        .public(),
    );
    node.insert_test_peer(mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        tunnel_port: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        models: vec![],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        serving_models: vec!["Qwen3-Coder-Next-Q4_K_M".to_string()],
        hosted_models: vec!["Qwen3-Coder-Next-Q4_K_M".to_string()],
        hosted_models_known: true,
        available_models: vec![],
        requested_models: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        moe_recovered_at: None,
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
    })
    .await;

    let (client, mut observer) = connected_tcp_pair().await;
    let request =
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";

    let routed = crate::network::proxy::route_to_target(
        node.clone(),
        client,
        election::InferenceTarget::MoeRemote(peer_id),
        request,
        crate::network::proxy::ResponseAdapter::None,
    )
    .await;

    assert!(!routed);
    assert!(
        !node.has_test_peer(peer_id).await,
        "failed MoE shard should be removed so election can fail down"
    );

    let mut response = Vec::new();
    observer.read_to_end(&mut response).await.unwrap();
    let response = String::from_utf8(response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"));
}

#[tokio::test]
async fn test_api_proxy_rewrites_image_blob_url_to_data_url() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-image-smoke".into(),
            mime_type: "image/png".into(),
            file_name: Some("smoke.png".into()),
            bytes_base64: "aGVsbG8=".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-image-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(crate::plugins::blobstore::get_request_object(
        &plugin_manager,
        crate::plugins::blobstore::GetRequestObjectRequest {
            token: put.token.clone(),
            request_id: Some("req-image-smoke".into()),
        },
    )
    .await
    .is_err());

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_blobstore_helper_resolves_object_store_capability() {
    let (plugin_manager, blobstore_root) =
        start_blobstore_plugin_manager_for("alt-store", vec!["object-store.v1".into()]).await;

    let response = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-capability".into(),
            mime_type: "text/plain".into(),
            file_name: Some("note.txt".into()),
            bytes_base64: base64::engine::general_purpose::STANDARD.encode("hello"),
            expires_in_secs: Some(60),
            uses_remaining: Some(1),
        },
    )
    .await
    .unwrap();

    assert_eq!(response.request_id, "req-capability");

    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_routes_to_registered_inference_endpoint() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","choices":[]}"#)
            .await;
    let plugin_manager = start_inference_endpoint_plugin_manager(
        format!("http://127.0.0.1:{upstream_port}/api/v1"),
        vec!["lemonade-test".into()],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let body = json!({
        "model": "lemonade-test",
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /api/v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""model":"lemonade-test""#));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_lists_registered_inference_models() {
    let plugin_manager = start_inference_endpoint_plugin_manager(
        "http://127.0.0.1:8000/api/v1".into(),
        vec!["lemonade-test".into()],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let entries = json["data"].as_array().cloned().unwrap_or_default();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(entries.iter().any(|entry| entry["id"] == "lemonade-test"));

    proxy_handle.abort();
}

#[test]
fn test_callable_models_excludes_none_only_targets() {
    let mut targets = local_targets(&[("ready-model", 1234)]);
    targets
        .targets
        .extend(unavailable_targets(&["warming-model"]).targets);
    assert_eq!(callable_models(&targets), vec!["ready-model".to_string()]);
}

#[tokio::test]
async fn test_api_proxy_lemonade_integration_when_enabled() {
    if std::env::var("MESH_LLM_TEST_LEMONADE").ok().as_deref() != Some("1") {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let models_response = client
        .get("http://localhost:8000/api/v1/models")
        .send()
        .await
        .expect("Lemonade should be reachable when MESH_LLM_TEST_LEMONADE=1")
        .error_for_status()
        .expect("Lemonade /models should succeed")
        .json::<serde_json::Value>()
        .await
        .expect("Lemonade /models should return JSON");
    let models = models_response["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entry| entry["id"].as_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    assert!(
        !models.is_empty(),
        "Lemonade reported no models at http://localhost:8000/api/v1/models"
    );
    let model = models[0].clone();

    let plugin_manager = start_inference_endpoint_plugin_manager(
        "http://localhost:8000/api/v1".into(),
        models.clone(),
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let models_response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let models_body = models_response.split("\r\n\r\n").nth(1).unwrap_or_default();
    let models_json: serde_json::Value = serde_json::from_str(models_body).unwrap();
    let model_entries = models_json["data"].as_array().cloned().unwrap_or_default();
    assert!(model_entries.iter().any(|entry| entry["id"] == model));

    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "stream": false,
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected Lemonade proxy response: {response}"
    );

    proxy_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_rewrites_audio_blob_url_to_data_url() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-audio-smoke".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-audio-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "audio_url", "audio_url": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("data:audio/wav;base64,UklGRg=="));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(crate::plugins::blobstore::get_request_object(
        &plugin_manager,
        crate::plugins::blobstore::GetRequestObjectRequest {
            token: put.token.clone(),
            request_id: Some("req-audio-smoke".into()),
        },
    )
    .await
    .is_err());

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_rewrites_input_audio_blob_url_to_inline_audio() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-input-audio-smoke".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-input-audio-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "input_audio", "input_audio": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert!(raw.contains(r#""mime_type":"audio/wav""#));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(crate::plugins::blobstore::get_request_object(
        &plugin_manager,
        crate::plugins::blobstore::GetRequestObjectRequest {
            token: put.token.clone(),
            request_id: Some("req-input-audio-smoke".into()),
        },
    )
    .await
    .is_err());

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_translates_responses_image_request() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-responses-image".into(),
            mime_type: "image/png".into(),
            file_name: Some("smoke.png".into()),
            bytes_base64: "aGVsbG8=".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let upstream_response = serde_json::json!({
        "id": "chatcmpl_image",
        "object": "chat.completion",
        "created": 123,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "image ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9
        }
    })
    .to_string();
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(&upstream_response).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "request_id": "req-responses-image",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": format!("mesh://blob/{client_id}/{}", put.token)}
            ]
        }]
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    let response_body = response.split("\r\n\r\n").nth(1).unwrap();
    let response_json: serde_json::Value = serde_json::from_str(response_body).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["output_text"], "image ok");

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_translates_responses_audio_request() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-responses-audio".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let upstream_response = serde_json::json!({
        "id": "chatcmpl_audio",
        "object": "chat.completion",
        "created": 123,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "audio ok"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(&upstream_response).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "request_id": "req-responses-audio",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "transcribe this"},
                {"type": "input_audio", "audio_url": format!("mesh://blob/{client_id}/{}", put.token)}
            ]
        }]
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    let response_body = response.split("\r\n\r\n").nth(1).unwrap();
    let response_json: serde_json::Value = serde_json::from_str(response_body).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["output_text"], "audio ok");

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_integration_expect_continue() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = br#"{"model":"test","messages":[{"role":"user","content":"expect"}]}"#;
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\n\r\n",
        body.len()
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(headers.as_bytes()).await.unwrap();

    let mut interim = [0u8; 64];
    let n = stream.read(&mut interim).await.unwrap();
    assert_eq!(
        std::str::from_utf8(&interim[..n]).unwrap(),
        "HTTP/1.1 100 Continue\r\n\r\n"
    );

    stream.write_all(body).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(String::from_utf8(response)
        .unwrap()
        .starts_with("HTTP/1.1 200 OK"));
    assert!(!raw.contains("Expect: 100-continue"));
    assert!(raw.contains("Connection: close"));
    assert!(raw.contains(std::str::from_utf8(body).unwrap()));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_streaming_response_arrives_incrementally() {
    let chunks = vec![
        (Duration::ZERO, br#"data: {"delta":"one"}\n\n"#.to_vec()),
        (
            Duration::from_millis(1000),
            br#"data: {"delta":"two"}\n\n"#.to_vec(),
        ),
    ];
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_streaming_upstream("text/event-stream", chunks).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "stream directly"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let first = read_until_contains(
        &mut stream,
        br#"data: {"delta":"one"}\n\n"#,
        Duration::from_secs(2),
    )
    .await;
    let first_text = String::from_utf8_lossy(&first);
    assert!(first_text.contains("HTTP/1.1 200 OK"));
    assert!(first_text.contains("Content-Type: text/event-stream"));
    assert!(first_text.contains(r#"data: {"delta":"one"}\n\n"#));
    assert!(tokio::time::timeout(Duration::from_millis(200), async {
        let mut probe = [0u8; 32];
        stream.read(&mut probe).await
    })
    .await
    .is_err());

    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await.unwrap();
    let mut full = first;
    full.extend_from_slice(&rest);
    let full_text = String::from_utf8(full).unwrap();
    assert!(full_text.contains(r#"data: {"delta":"two"}\n\n"#));
    assert!(full_text.ends_with("0\r\n\r\n"));

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.contains("\"stream\":true"));
    assert!(raw.contains("Connection: close"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_translates_streaming_responses_events_incrementally() {
    let chunks = vec![
        (
            Duration::ZERO,
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"one"},"finish_reason":null}]}

"#
            .to_vec(),
        ),
        (
            Duration::from_millis(1000),
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"two"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

"#
            .to_vec(),
        ),
    ];
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_streaming_upstream("text/event-stream", chunks).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "input": "stream responses",
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let started_at = tokio::time::Instant::now();
    let first = read_until_contains(
        &mut stream,
        br#"event: response.output_text.delta
data: {"#,
        Duration::from_secs(2),
    )
    .await;
    let first_elapsed = started_at.elapsed();
    let first_text = String::from_utf8_lossy(&first);
    assert!(first_text.contains("HTTP/1.1 200 OK"));
    assert!(first_text.contains("Content-Type: text/event-stream"));
    assert!(first_text.contains("event: response.created"));
    assert!(first_text.contains("event: response.output_text.delta"));
    assert!(first_text.contains(r#""delta":"one""#));
    assert!(
        first_elapsed < Duration::from_millis(900),
        "first translated delta arrived too late: {first_elapsed:?}"
    );
    assert!(!first_text.contains(r#""delta":"two""#));
    assert!(!first_text.contains("event: response.output_text.done"));
    assert!(!first_text.contains("event: response.completed"));

    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await.unwrap();
    let mut full = first;
    full.extend_from_slice(&rest);
    let full_text = String::from_utf8(full).unwrap();
    assert!(full_text.contains(r#""delta":"two""#));
    assert!(full_text.contains("event: response.output_text.done"));
    assert!(full_text.contains("event: response.completed"));
    assert!(full_text.contains(r#""output_text":"onetwo""#));
    assert!(full_text.contains("event: done"));
    assert!(full_text.contains("data: [DONE]"));
    assert!(full_text.ends_with("0\r\n\r\n"));

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains("\"stream\":true"));
    assert!(raw.contains("\"messages\""));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipeline_fallback_uses_direct_proxy() {
    // Pipeline fallback test: when only one model is available, auto routes
    // to it directly without attempting a pipeline plan.
    let strong_model = "Qwen2.5-Coder-32B-Instruct-Q4_K_M";
    let body = json!({
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}
        ]
    });
    let classification = router::classify(&body);
    assert!(pipeline::should_pipeline(&classification));

    let (strong_port, strong_rx, strong_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;

    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[(strong_model, strong_port)])).await;

    let request_body = body.to_string();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        request_body.len()
    );

    let response = send_request_and_read_response(
        proxy_addr,
        vec![format!("{headers}{request_body}").into_bytes()],
    )
    .await;
    let raw = String::from_utf8(strong_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("\"model\":\"auto\""));
    assert!(!raw.contains("[Task Plan from"));
    assert!(raw.contains("\"Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse.\""));
    // model=auto must inject mesh_hooks flag so llama-server enables hook callbacks
    assert!(
        raw.contains("\"mesh_hooks\":true"),
        "model=auto should inject mesh_hooks:true into the forwarded body"
    );

    proxy_handle.abort();
    let _ = strong_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipeline_streaming_response_arrives_incrementally() {
    // With a single model, pipeline is skipped (needs 2 local models).
    // This tests that a streaming agentic request still gets proxied correctly.
    let model = "Qwen2.5-Coder-32B-Instruct-Q4_K_M";
    let body = json!({
        "model": "auto",
        "stream": true,
        "messages": [
            {"role": "user", "content": "Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}
        ]
    });
    let classification = router::classify(&body);
    assert!(pipeline::should_pipeline(&classification));

    let (port, _rx, handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (
                Duration::ZERO,
                br#"data: {"delta":"chunk-one"}\n\n"#.to_vec(),
            ),
            (
                Duration::from_millis(1000),
                br#"data: {"delta":"chunk-two"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;

    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[(model, port)])).await;

    let request_body = body.to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_body.len(),
        request_body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let full = read_until_contains(
        &mut stream,
        br#"data: {"delta":"chunk-two"}\n\n"#,
        Duration::from_secs(5),
    )
    .await;
    let full_text = String::from_utf8_lossy(&full);
    assert!(full_text.contains("HTTP/1.1 200 OK"));
    assert!(full_text.contains(r#"data: {"delta":"chunk-one"}\n\n"#));
    assert!(full_text.contains(r#"data: {"delta":"chunk-two"}\n\n"#));

    proxy_handle.abort();
    let _ = handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipelined_follow_up_is_not_forwarded() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "first"}],
    })
    .to_string();
    let first_request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let second_request = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";

    let response = send_request_and_read_response(
        proxy_addr,
        vec![format!("{first_request}{second_request}").into_bytes()],
    )
    .await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("\"content\":\"first\""));
    assert!(!raw.contains("GET /v1/models HTTP/1.1"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_streaming_client_disconnect_does_not_hang() {
    let (upstream_port, upstream_rx, upstream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (Duration::ZERO, br#"data: {"delta":"hello"}\n\n"#.to_vec()),
            (
                Duration::from_millis(150),
                br#"data: {"delta":"after-disconnect"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "disconnect me"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let first = read_until_contains(
        &mut stream,
        br#"data: {"delta":"hello"}\n\n"#,
        Duration::from_secs(2),
    )
    .await;
    assert!(String::from_utf8_lossy(&first).contains(r#"data: {"delta":"hello"}\n\n"#));
    drop(stream);

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.contains("\"disconnect me\""));
    tokio::time::timeout(Duration::from_secs(1), upstream_handle)
        .await
        .expect("streaming upstream hung after client disconnect")
        .unwrap();

    proxy_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_retries_context_overflow_bad_request_to_next_target() {
    let overflow_body =
        r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (small_port, small_rx, small_handle) =
        spawn_status_upstream("400 Bad Request", overflow_body).await;
    let (large_port, large_rx, large_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[small_port, large_port])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "overflow then retry"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(small_rx.await.unwrap()).unwrap();
    let second_raw = String::from_utf8(large_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains(r#"{"ok":true}"#));
    assert!(first_raw.contains("overflow then retry"));
    assert!(second_raw.contains("overflow then retry"));

    proxy_handle.abort();
    let _ = small_handle.await;
    let _ = large_handle.await;
}

#[tokio::test]
async fn test_api_proxy_preserves_context_overflow_bad_request_for_single_target() {
    let overflow_body =
        r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (port, upstream_rx, upstream_handle) =
        spawn_status_upstream("400 Bad Request", overflow_body).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "single target overflow should stay 400"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("context window"));
    assert!(raw.contains("single target overflow should stay 400"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_returns_last_context_overflow_bad_request_when_all_targets_overflow() {
    let first_body = r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=2048)"}}"#;
    let second_body = r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (first_port, first_rx, first_handle) =
        spawn_status_upstream("400 Bad Request", first_body).await;
    let (second_port, second_rx, second_handle) =
        spawn_status_upstream("400 Bad Request", second_body).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[first_port, second_port]))
            .await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "all targets overflow"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(first_rx.await.unwrap()).unwrap();
    let second_raw = String::from_utf8(second_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("n_ctx=4096"));
    assert!(first_raw.contains("all targets overflow"));
    assert!(second_raw.contains("all targets overflow"));

    proxy_handle.abort();
    let _ = first_handle.await;
    let _ = second_handle.await;
}

#[tokio::test]
async fn test_api_proxy_does_not_retry_generic_bad_request() {
    let bad_request_body = r#"{"error":{"message":"missing required field: messages"}}"#;
    let (bad_port, bad_rx, bad_handle) =
        spawn_status_upstream("400 Bad Request", bad_request_body).await;
    let (unused_port, unused_rx, unused_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[bad_port, unused_port])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "bad request should stop"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(bad_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("missing required field"));
    assert!(first_raw.contains("bad request should stop"));
    assert!(tokio::time::timeout(Duration::from_millis(250), unused_rx)
        .await
        .is_err());

    proxy_handle.abort();
    let _ = bad_handle.await;
    unused_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_normalizes_max_completion_tokens_for_upstream() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "max_completion_tokens": 32,
        "messages": [{"role": "user", "content": "normalize token alias"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("\"max_tokens\":32"));
    assert!(!raw.contains("max_completion_tokens"));
    assert!(raw.contains("normalize token alias"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_does_not_retry_after_successful_stream_starts() {
    let (stream_port, stream_rx, stream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (Duration::ZERO, br#"data: {"delta":"first"}\n\n"#.to_vec()),
            (
                Duration::from_millis(50),
                br#"data: {"delta":"second"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;
    let (unused_port, unused_rx, unused_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[stream_port, unused_port]))
            .await;

    let body = json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "stream wins immediately"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let first = read_until_contains(
        &mut stream,
        br#"data: {"delta":"first"}\n\n"#,
        Duration::from_secs(2),
    )
    .await;
    let first_text = String::from_utf8_lossy(&first);
    let raw = String::from_utf8(stream_rx.await.unwrap()).unwrap();

    assert!(first_text.contains("HTTP/1.1 200 OK"));
    assert!(first_text.contains(r#"data: {"delta":"first"}\n\n"#));
    assert!(raw.contains("stream wins immediately"));
    assert!(tokio::time::timeout(Duration::from_millis(250), unused_rx)
        .await
        .is_err());

    drop(stream);
    proxy_handle.abort();
    tokio::time::timeout(Duration::from_secs(1), stream_handle)
        .await
        .expect("streaming upstream hung")
        .unwrap();
    unused_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_passes_through_native_base64_image() {
    // A client that already has a base64-encoded image (data URI) and sends it
    // directly to /v1/chat/completions should have it forwarded unchanged.
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB"}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/jpeg;base64,/9j/4AAQSkZJRgAB"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_passes_through_native_base64_audio() {
    // A client that already has base64-encoded audio and sends it in the
    // input_audio format directly to /v1/chat/completions should have it
    // forwarded unchanged.
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": "UklGRg==",
                    "format": "wav"
                }}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}
