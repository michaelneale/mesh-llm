use base64::Engine;
use iroh::{Endpoint, EndpointAddr, SecretKey};
use mesh_client::proto::node::{
    ConfigApplyMode, NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry, OwnerControlEnvelope,
    OwnerControlErrorCode,
};
use mesh_client::protocol::{
    decode_owner_control_envelope, read_len_prefixed, write_len_prefixed, ALPN_CONTROL_V1,
    NODE_PROTOCOL_GENERATION,
};
use mesh_client::{
    ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneClientError, ControlPlaneConnection,
    InviteToken, OwnerControlClient, OwnerControlRemoteError, OwnerControlWatchEvent, OwnerKeypair,
};
use prost::Message;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};

#[derive(Clone)]
struct TestServerState {
    expected_owner_id: String,
    expected_owner_signing_key: Vec<u8>,
    received_apply: Arc<Mutex<Vec<mesh_client::proto::node::OwnerControlApplyConfigRequest>>>,
    watch_closed_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

fn test_owner_keypair(signing_seed: u8, encryption_seed: u8) -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[signing_seed; 32], &[encryption_seed; 32])
        .expect("test owner keypair must be valid")
}

fn control_endpoint_token(addr: &EndpointAddr) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(addr).expect("endpoint addr should serialize"))
}

fn test_snapshot(
    node_id: &[u8; 32],
    revision: u64,
    model: &str,
) -> mesh_client::proto::node::OwnerControlConfigSnapshot {
    mesh_client::proto::node::OwnerControlConfigSnapshot {
        node_id: node_id.to_vec(),
        revision,
        config_hash: vec![revision as u8; 32],
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: Some(NodeGpuConfig {
                assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
            }),
            models: vec![NodeModelEntry {
                model: model.to_string(),
                mmproj: None,
                ctx_size: Some(4096),
                gpu_id: None,
                model_ref: None,
                mmproj_ref: None,
            }],
            plugins: vec![],
        }),
        hostname: Some("control.test".to_string()),
    }
}

async fn make_client() -> mesh_client::MeshClient {
    ClientBuilder::new(
        test_owner_keypair(0x11, 0x12),
        InviteToken::from_str("mesh-test:control-plane").unwrap(),
    )
    .build()
    .expect("mesh client should build")
}

async fn spawn_success_server(
    owner_keypair: &OwnerKeypair,
) -> (Endpoint, String, TestServerState, oneshot::Receiver<()>) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let (watch_closed_tx, watch_closed_rx) = oneshot::channel();
    let state = TestServerState {
        expected_owner_id: owner_keypair.owner_id(),
        expected_owner_signing_key: owner_keypair.verifying_key().as_bytes().to_vec(),
        received_apply: Arc::new(Mutex::new(Vec::new())),
        watch_closed_tx: Arc::new(Mutex::new(Some(watch_closed_tx))),
    };
    let state_clone = state.clone();
    let endpoint_id = endpoint.id();
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        loop {
            let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                break;
            };
            let state = state_clone.clone();
            let stream_connection = connection.clone();
            tokio::spawn(async move {
                let handshake = read_control_envelope(&mut recv).await;
                let ownership = handshake
                    .handshake
                    .expect("first envelope should be handshake")
                    .ownership
                    .expect("handshake should include ownership");
                assert_eq!(ownership.owner_id, state.expected_owner_id);
                assert_eq!(
                    ownership.owner_sign_public_key,
                    state.expected_owner_signing_key
                );
                assert_eq!(
                    ownership.node_endpoint_id,
                    stream_connection.remote_id().as_bytes().to_vec()
                );

                let request = read_control_envelope(&mut recv)
                    .await
                    .request
                    .expect("second envelope should contain a request");
                if request.get_config.is_some() {
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: Some(
                                    mesh_client::proto::node::OwnerControlGetConfigResponse {
                                        snapshot: Some(test_snapshot(
                                            endpoint_id.as_bytes(),
                                            3,
                                            "get-model.gguf",
                                        )),
                                    },
                                ),
                                watch_config: None,
                                apply_config: None,
                                refresh_inventory: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if let Some(apply) = request.apply_config {
                    state.received_apply.lock().await.push(apply);
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: None,
                                apply_config: Some(
                                    mesh_client::proto::node::OwnerControlApplyConfigResponse {
                                        success: true,
                                        current_revision: 4,
                                        config_hash: vec![0x44; 32],
                                        error: None,
                                        apply_mode: ConfigApplyMode::Staged as i32,
                                    },
                                ),
                                refresh_inventory: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if request.refresh_inventory.is_some() {
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: None,
                                apply_config: None,
                                refresh_inventory: Some(mesh_client::proto::node::OwnerControlRefreshInventoryResponse {
                                    snapshot: Some(test_snapshot(endpoint_id.as_bytes(), 5, "refresh-model.gguf")),
                                }),
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = send.finish();
                    return;
                }
                if let Some(watch_config) = request.watch_config {
                    let watch_response = if watch_config.include_snapshot {
                        mesh_client::proto::node::OwnerControlWatchConfigResponse {
                            accepted: None,
                            snapshot: Some(test_snapshot(
                                endpoint_id.as_bytes(),
                                6,
                                "watch-model.gguf",
                            )),
                            update: None,
                        }
                    } else {
                        mesh_client::proto::node::OwnerControlWatchConfigResponse {
                            accepted: Some(mesh_client::proto::node::OwnerControlWatchAccepted {
                                target_node_id: endpoint_id.as_bytes().to_vec(),
                            }),
                            snapshot: None,
                            update: None,
                        }
                    };
                    write_control_envelope(
                        &mut send,
                        OwnerControlEnvelope {
                            r#gen: NODE_PROTOCOL_GENERATION,
                            handshake: None,
                            request: None,
                            response: Some(mesh_client::proto::node::OwnerControlResponse {
                                request_id: request.request_id,
                                get_config: None,
                                watch_config: Some(watch_response),
                                apply_config: None,
                                refresh_inventory: None,
                            }),
                            error: None,
                        },
                    )
                    .await;
                    let _ = read_len_prefixed(&mut recv).await;
                    if let Some(tx) = state.watch_closed_tx.lock().await.take() {
                        let _ = tx.send(());
                    }
                }
            });
        }
    });
    (endpoint, token, state, watch_closed_rx)
}

async fn spawn_auth_failure_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let _ = read_control_envelope(&mut recv).await;
        write_control_envelope(
            &mut send,
            OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(mesh_client::proto::node::OwnerControlError {
                    code: OwnerControlErrorCode::Unauthorized as i32,
                    message: "owner attestation rejected".to_string(),
                    request_id: None,
                    current_revision: None,
                }),
            },
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = send.finish();
    });
    (endpoint, token)
}

async fn spawn_control_unsupported_server() -> (Endpoint, String) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    let server_endpoint = endpoint.clone();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
        let _ = read_control_envelope(&mut recv).await;
        let request = read_control_envelope(&mut recv).await;
        let request_id = request.request.map(|request| request.request_id);
        write_control_envelope(
            &mut send,
            OwnerControlEnvelope {
                r#gen: NODE_PROTOCOL_GENERATION,
                handshake: None,
                request: None,
                response: None,
                error: Some(mesh_client::proto::node::OwnerControlError {
                    code: OwnerControlErrorCode::ControlUnsupported as i32,
                    message: "remote endpoint did not negotiate mesh-llm-control/1".to_string(),
                    request_id,
                    current_revision: None,
                }),
            },
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let _ = send.finish();
    });
    (endpoint, token)
}

async fn read_control_envelope(recv: &mut iroh::endpoint::RecvStream) -> OwnerControlEnvelope {
    let bytes = read_len_prefixed(recv)
        .await
        .expect("frame should be len-prefixed");
    decode_owner_control_envelope(&bytes).expect("owner-control envelope should decode")
}

async fn write_control_envelope(
    send: &mut iroh::endpoint::SendStream,
    envelope: OwnerControlEnvelope,
) {
    write_len_prefixed(send, &envelope.encode_to_vec())
        .await
        .expect("owner-control envelope should write");
}

fn owner_control_client(connection: ControlPlaneConnection) -> OwnerControlClient {
    match connection {
        ControlPlaneConnection::OwnerControl(client) => *client,
    }
}

#[tokio::test]
async fn control_plane_client_apply_config_get_watch_refresh_and_close() {
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let client = make_client().await;
    let (_server, token, state, watch_closed_rx) = spawn_success_server(&owner_keypair).await;

    let connection = client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
        .await
        .expect("control session should connect");
    let control = owner_control_client(connection);

    let get_snapshot = control
        .get_config()
        .await
        .expect("get_config should succeed");
    assert_eq!(get_snapshot.revision, 3);
    assert_eq!(get_snapshot.hostname.as_deref(), Some("control.test"));

    let apply_response = control
        .apply_config(
            3,
            NodeConfigSnapshot {
                version: 1,
                gpu: Some(NodeGpuConfig {
                    assignment: mesh_client::proto::node::GpuAssignment::Auto as i32,
                }),
                models: vec![NodeModelEntry {
                    model: "applied-model.gguf".to_string(),
                    mmproj: None,
                    ctx_size: Some(8192),
                    gpu_id: None,
                    model_ref: None,
                    mmproj_ref: None,
                }],
                plugins: vec![],
            },
        )
        .await
        .expect("apply_config should succeed");
    assert!(apply_response.success);
    assert_eq!(apply_response.current_revision, 4);

    let refresh_snapshot = control
        .refresh_inventory()
        .await
        .expect("refresh_inventory should succeed");
    assert_eq!(refresh_snapshot.revision, 5);

    let mut watch = control
        .watch_config(true)
        .await
        .expect("watch_config should open");
    match watch.next().await.expect("watch snapshot should arrive") {
        OwnerControlWatchEvent::Snapshot(snapshot) => assert_eq!(snapshot.revision, 6),
        _ => panic!("expected initial watch snapshot"),
    }
    watch.close().await.expect("watch close should succeed");
    tokio::time::timeout(std::time::Duration::from_secs(5), watch_closed_rx)
        .await
        .expect("server should observe watch close")
        .expect("watch close signal should succeed");

    let apply_requests = state.received_apply.lock().await;
    assert_eq!(apply_requests.len(), 1);
    assert_eq!(apply_requests[0].expected_revision, 3);
    assert_eq!(
        apply_requests[0].config.as_ref().unwrap().models[0].model,
        "applied-model.gguf"
    );
}

#[tokio::test]
async fn control_plane_client_watch_without_snapshot_returns_accepted() {
    let client = make_client().await;
    let owner_keypair = test_owner_keypair(0x11, 0x12);
    let (_server, token, _state, watch_closed_rx) = spawn_success_server(&owner_keypair).await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should use owner-control ALPN"),
    );

    let mut watch = control
        .watch_config(false)
        .await
        .expect("watch_config should open");
    match watch.next().await.expect("watch accepted should arrive") {
        OwnerControlWatchEvent::Accepted(accepted) => {
            assert_eq!(accepted.target_node_id.len(), 32);
        }
        _ => panic!("expected initial watch accepted event"),
    }
    watch.close().await.expect("watch close should succeed");
    tokio::time::timeout(std::time::Duration::from_secs(5), watch_closed_rx)
        .await
        .expect("server should observe watch close")
        .expect("watch close signal should succeed");
}

#[tokio::test]
async fn control_plane_client_auth_failure_surfaces_structured_error() {
    let client = make_client().await;
    let (_server, token) = spawn_auth_failure_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("connection should bootstrap before first request"),
    );

    let err = control
        .get_config()
        .await
        .expect_err("auth failure should bubble up");
    match err {
        ControlPlaneClientError::Remote(OwnerControlRemoteError { code, message, .. }) => {
            assert_eq!(code, OwnerControlErrorCode::Unauthorized);
            assert!(message.contains("rejected"));
        }
        other => panic!("expected structured remote auth error, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_client_rejects_alpn_mismatch() {
    let client = make_client().await;
    let (_server, token) = spawn_control_unsupported_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("control session should bootstrap before request"),
    );
    let err = control
        .get_config()
        .await
        .expect_err("unsupported configured endpoint should surface a structured error");

    match err {
        ControlPlaneClientError::Remote(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
        }
        other => panic!("expected structured unsupported error, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_client_unreachable_listener_returns_structured_negotiation_error() {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let token = control_endpoint_token(&endpoint.addr());
    drop(endpoint);

    let client = make_client().await;
    let err = match client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
        .await
    {
        Ok(_) => panic!("unreachable listener should fail"),
        Err(err) => err,
    };

    match err {
        ControlPlaneClientError::Negotiation(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnavailable);
            assert!(
                err.message
                    .contains("remote owner-control endpoint is unavailable or unreachable"),
                "message: {}",
                err.message
            );
            assert!(!err.legacy_retry_allowed);
        }
        other => panic!("expected negotiation error, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_client_does_not_silently_fallback_when_endpoint_fails() {
    let client = make_client().await;
    let (_server, token) = spawn_control_unsupported_server().await;
    let control = owner_control_client(
        client
            .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
            .await
            .expect("configured endpoint should stay on the owner-control lane"),
    );
    let err = control
        .get_config()
        .await
        .expect_err("configured endpoint failure must not silently fall back");

    match err {
        ControlPlaneClientError::Remote(err) => {
            assert_eq!(err.code, OwnerControlErrorCode::ControlUnsupported);
        }
        other => panic!("expected structured unsupported error, got {other:?}"),
    }
}
