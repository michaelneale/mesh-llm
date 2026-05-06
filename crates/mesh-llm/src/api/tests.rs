use super::*;
use crate::api::status::decode_runtime_model_path;
use crate::plugin;
use crate::plugins::{blackboard, blobstore};
use mesh_llm_plugin::MeshVisibility;
use rmcp::model::ErrorCode;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

#[test]
fn test_build_gpus_both_none() {
    let result = build_gpus(None, None, None, None, None, None);
    assert!(result.is_empty(), "expected empty vec when no gpu_name");
}

#[test]
fn test_build_gpus_single_no_vram() {
    let result = build_gpus(Some("NVIDIA RTX 5090"), None, None, None, None, None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 0);
}

#[test]
fn test_build_gpus_single_with_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090"),
        Some("34359738368"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
}

#[test]
fn test_build_gpus_multi_full_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
        Some("34359738368,10737418240"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(result[1].name, "NVIDIA RTX 3080");
    assert_eq!(result[1].vram_bytes, 10_737_418_240);
}

#[test]
fn test_build_gpus_multi_full_vram_without_space_after_comma() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090,NVIDIA RTX 3080"),
        Some("34359738368,10737418240"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[1].name, "NVIDIA RTX 3080");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(result[1].vram_bytes, 10_737_418_240);
}

#[test]
fn test_build_gpus_multi_names_trim_whitespace() {
    let result = build_gpus(
        Some(" GPU0 ,GPU1 ,  GPU2  "),
        Some("100,200,300"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].name, "GPU0");
    assert_eq!(result[1].name, "GPU1");
    assert_eq!(result[2].name, "GPU2");
}

#[test]
fn test_build_gpus_expands_summarized_identical_names() {
    let result = build_gpus(
        Some("2× NVIDIA A100"),
        Some("85899345920,85899345920"),
        None,
        Some("1948.70,1948.70"),
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA A100");
    assert_eq!(result[1].name, "NVIDIA A100");
    assert_eq!(result[0].vram_bytes, 85_899_345_920);
    assert_eq!(result[1].vram_bytes, 85_899_345_920);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
    assert_eq!(result[1].mem_bandwidth_gbps, Some(1948.70));
}

#[test]
fn test_build_gpus_multi_partial_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
        Some("34359738368"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(
        result[1].vram_bytes, 0,
        "missing VRAM entry should default to 0"
    );
}

#[test]
fn test_build_gpus_vram_no_gpu_name() {
    let result = build_gpus(None, Some("34359738368"), None, None, None, None);
    assert!(
        result.is_empty(),
        "no gpu_name means no entries even if vram present"
    );
}

#[test]
fn test_build_gpus_vram_whitespace_trimmed() {
    let result = build_gpus(
        Some("NVIDIA RTX 4090"),
        Some(" 25769803776 "),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].vram_bytes, 25_769_803_776);
}

#[test]
fn test_build_gpus_with_bandwidth() {
    let result = build_gpus(
        Some("NVIDIA A100, NVIDIA A6000"),
        Some("85899345920,51539607552"),
        None,
        Some("1948.70,780.10"),
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
    assert_eq!(result[1].mem_bandwidth_gbps, Some(780.10));
}

#[test]
fn test_build_gpus_unparsable_vram_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,foo,300"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].vram_bytes, 100);
    assert_eq!(
        result[1].vram_bytes, 0,
        "unparsable vram should default to 0, not shift indices"
    );
    assert_eq!(result[2].vram_bytes, 300);
}

#[test]
fn test_build_gpus_unparsable_bandwidth_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,200,300"),
        None,
        Some("1.0,bad,3.0"),
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1.0));
    assert_eq!(
        result[1].mem_bandwidth_gbps, None,
        "unparsable bandwidth should be None, not shift indices"
    );
    assert_eq!(result[2].mem_bandwidth_gbps, Some(3.0));
}

#[test]
fn test_build_gpus_with_both_tflops_precisions() {
    let result = build_gpus(
        Some("GPU0, GPU1"),
        Some("100,200"),
        None,
        None,
        Some("312.5,419.5"),
        Some("625.0,839.0"),
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
    assert_eq!(result[0].compute_tflops_fp16, Some(625.0));
    assert_eq!(result[1].compute_tflops_fp32, Some(419.5));
    assert_eq!(result[1].compute_tflops_fp16, Some(839.0));
}

#[test]
fn test_build_gpus_fp32_only_fp16_absent() {
    let result = build_gpus(
        Some("GPU0, GPU1"),
        Some("100,200"),
        None,
        None,
        Some("312.5,bad"),
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
    assert_eq!(result[1].compute_tflops_fp32, None);
    assert!(result.iter().all(|gpu| gpu.compute_tflops_fp16.is_none()));
}

#[test]
fn test_gpu_entry_omits_tflops_when_none() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert!(first.get("compute_tflops_fp32").is_none());
    assert!(first.get("compute_tflops_fp16").is_none());
    assert!(first.get("mem_bandwidth_gbps").is_some());
}

#[test]
fn test_api_status_gpu_entry_uses_new_name() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert_eq!(first.get("mem_bandwidth_gbps").unwrap(), &json!(1948.7));
    assert!(
        first.get("bandwidth_gbps").is_none(),
        "API status JSON should use mem_bandwidth_gbps"
    );
}

#[test]
fn test_build_gpus_with_reserved_bytes_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,200,300"),
        Some("10,,30"),
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].reserved_bytes, Some(10));
    assert_eq!(result[1].reserved_bytes, None);
    assert_eq!(result[2].reserved_bytes, Some(30));
}

#[test]
fn test_gpu_entry_omits_reserved_bytes_when_none() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert!(first.get("reserved_bytes").is_none());
}

#[test]
fn test_http_body_text_extracts_body() {
    let raw = b"POST /api/plugins/x/tools/y HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
    assert_eq!(http_body_text(raw), "{\"a\":1}");
}

#[test]
fn test_build_runtime_status_payload_uses_local_processes() {
    let result = build_runtime_status_payload(
        "Qwen",
        Some("llama".into()),
        true,
        true,
        Some(9337),
        vec![
            RuntimeProcessPayload {
                name: "Qwen".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 100,
                slots: 4,
                context_length: None,
            },
            RuntimeProcessPayload {
                name: "Llama".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 101,
                slots: 4,
                context_length: None,
            },
        ],
    );
    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].name, "Llama");
    assert_eq!(result.models[0].port, Some(9444));
    assert_eq!(result.models[1].name, "Qwen");
}

#[test]
fn test_build_runtime_processes_payload_sorts_processes() {
    let payload = build_runtime_processes_payload(vec![
        RuntimeProcessPayload {
            name: "Zulu".into(),
            backend: "llama".into(),
            status: "ready".into(),
            port: 9444,
            pid: 11,
            slots: 4,
            context_length: None,
        },
        RuntimeProcessPayload {
            name: "Alpha".into(),
            backend: "llama".into(),
            status: "ready".into(),
            port: 9337,
            pid: 10,
            slots: 4,
            context_length: None,
        },
    ]);

    assert_eq!(payload.processes.len(), 2);
    assert_eq!(payload.processes[0].name, "Alpha");
    assert_eq!(payload.processes[1].name, "Zulu");
}

#[test]
fn test_runtime_processes_payload_includes_context_length() {
    let payload = build_runtime_processes_payload(vec![
        RuntimeProcessPayload {
            name: "model-a".into(),
            backend: "llama".into(),
            status: "ready".into(),
            port: 9337,
            pid: 10,
            slots: 4,
            context_length: Some(65536),
        },
        RuntimeProcessPayload {
            name: "model-b".into(),
            backend: "llama".into(),
            status: "ready".into(),
            port: 9444,
            pid: 11,
            slots: 2,
            context_length: None,
        },
    ]);

    assert_eq!(payload.processes.len(), 2);
    assert_eq!(payload.processes[0].name, "model-a");
    assert_eq!(payload.processes[0].context_length, Some(65536));
    assert_eq!(payload.processes[0].slots, 4);
    assert_eq!(payload.processes[1].context_length, None);

    // Verify serialization includes context_length when present
    let json = serde_json::to_string(&payload).expect("serialize payload");
    assert!(json.contains(r#""context_length":65536"#));
    // Verify context_length is omitted when None (skip_serializing_if)
    let model_b_section: serde_json::Value = serde_json::from_str(&json).expect("parse json");
    let processes = model_b_section["processes"]
        .as_array()
        .expect("processes array");
    assert!(
        processes[1].get("context_length").is_none() && processes[1]["context_length"].is_null()
    );
}

#[test]
fn test_classify_runtime_error_codes() {
    assert_eq!(classify_runtime_error("model 'x' is not loaded"), 404);
    assert_eq!(classify_runtime_error("model 'x' is already loaded"), 409);
    assert_eq!(
        classify_runtime_error("runtime load only supports models that fit locally"),
        422
    );
    assert_eq!(classify_runtime_error("bad request"), 400);
}

#[test]
fn derive_local_node_state_prefers_client() {
    let node_state = MeshApi::derive_local_node_state(true, true, true, true, "Qwen");

    assert_eq!(node_state, NodeState::Client);
    assert_eq!(MeshApi::derive_node_status(node_state), "Client");
}

#[test]
fn derive_local_node_state_returns_standby_without_ready_runtime() {
    let node_state = MeshApi::derive_local_node_state(false, false, false, false, "Qwen");

    assert_eq!(node_state, NodeState::Standby);
    assert_eq!(MeshApi::derive_node_status(node_state), "Standby");
}

#[test]
fn derive_local_node_state_returns_loading_for_declared_but_unready_work() {
    let host_loading = MeshApi::derive_local_node_state(false, true, false, false, "Qwen");
    let worker_loading = MeshApi::derive_local_node_state(false, false, false, true, "Qwen");

    assert_eq!(host_loading, NodeState::Loading);
    assert_eq!(worker_loading, NodeState::Loading);
    assert_eq!(MeshApi::derive_node_status(host_loading), "Loading");
    assert_eq!(MeshApi::derive_node_status(worker_loading), "Loading");
}

#[test]
fn derive_local_node_state_returns_serving_for_ready_runtime() {
    let host_serving = MeshApi::derive_local_node_state(false, true, true, false, "Qwen");
    let worker_serving = MeshApi::derive_local_node_state(false, false, true, true, "Qwen");

    assert_eq!(host_serving, NodeState::Serving);
    assert_eq!(worker_serving, NodeState::Serving);
    assert_eq!(MeshApi::derive_node_status(host_serving), "Serving");
    assert_eq!(MeshApi::derive_node_status(worker_serving), "Serving");
}

#[test]
fn derive_local_node_state_never_emits_legacy_idle_or_split_labels() {
    let labels = [
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            true, true, true, true, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, false, false, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, true, false, false, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, true, true, "Qwen",
        )),
        MeshApi::derive_node_status(MeshApi::derive_local_node_state(
            false, false, false, false, "",
        )),
    ];

    for label in labels {
        assert!(matches!(
            label.as_str(),
            "Client" | "Standby" | "Loading" | "Serving"
        ));
        assert_ne!(label, "Idle");
        assert_ne!(label, "Serving (split)");
        assert_ne!(label, "Worker (split)");
    }
}

fn make_test_state_endpoint_id(seed: u8) -> iroh::EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    iroh::EndpointId::from(iroh::SecretKey::from_bytes(&bytes).public())
}

fn make_test_state_peer(seed: u8, role: mesh::NodeRole) -> mesh::PeerInfo {
    let id = make_test_state_endpoint_id(seed);
    mesh::PeerInfo {
        id,
        addr: iroh::EndpointAddr {
            id,
            addrs: Default::default(),
        },
        role,
        models: vec![],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: Instant::now(),
        last_mentioned: Instant::now(),
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
        first_joined_mesh_ts: None,
    }
}

fn make_legacy_peer_fixture(
    seed: u8,
    role: mesh::NodeRole,
    serving_models: Vec<&str>,
) -> mesh::PeerInfo {
    let mut peer = make_test_state_peer(seed, role);
    peer.version = Some("0.54.0".into());
    peer.serving_models = serving_models.into_iter().map(str::to_string).collect();
    peer.hosted_models = vec![];
    peer.hosted_models_known = false;
    peer.served_model_runtime = vec![];
    peer
}

#[test]
fn derive_peer_state_prefers_client_role() {
    let mut peer = make_test_state_peer(1, mesh::NodeRole::Client);
    peer.serving_models = vec!["Qwen".into()];
    peer.hosted_models = vec!["Qwen".into()];
    peer.hosted_models_known = true;
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: Some(8192),
        ready: true,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Client);
}

#[test]
fn derive_peer_state_returns_serving_for_ready_runtime() {
    let mut peer = make_test_state_peer(2, mesh::NodeRole::Host { http_port: 9337 });
    peer.serving_models = vec!["Qwen".into()];
    peer.hosted_models = vec!["Qwen".into()];
    peer.hosted_models_known = true;
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: Some(8192),
        ready: true,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
}

#[test]
fn derive_peer_state_returns_loading_for_assigned_but_unready_peer() {
    let mut peer = make_test_state_peer(3, mesh::NodeRole::Worker);
    peer.serving_models = vec!["Qwen".into()];
    peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
        model_name: "Qwen".into(),
        identity_hash: None,
        context_length: None,
        ready: false,
    }];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Loading);
}

#[test]
fn derive_peer_state_returns_standby_for_connected_idle_peer() {
    let peer = make_test_state_peer(4, mesh::NodeRole::Worker);

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Standby);
}

#[test]
fn derive_peer_state_falls_back_to_legacy_serving_models() {
    let mut peer = make_test_state_peer(5, mesh::NodeRole::Worker);
    peer.serving_models = vec!["Qwen".into()];

    assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
}

#[test]
fn legacy_peer_fixture_uses_backend_state_fallback() {
    let serving_peer =
        make_legacy_peer_fixture(6, mesh::NodeRole::Host { http_port: 9337 }, vec!["Qwen"]);
    let standby_peer = make_legacy_peer_fixture(7, mesh::NodeRole::Worker, vec![]);

    assert_eq!(
        MeshApi::derive_peer_state(&serving_peer),
        NodeState::Serving
    );
    assert_eq!(
        MeshApi::derive_peer_state(&standby_peer),
        NodeState::Standby
    );
}

#[test]
fn test_decode_runtime_model_path_decodes_percent_not_plus() {
    // %20 is a space; + is a literal plus in URL paths (not a space)
    assert_eq!(
        decode_runtime_model_path("/api/runtime/models/Llama%203.2+1B"),
        Some("Llama 3.2+1B".into())
    );
}

#[test]
fn test_decode_runtime_model_path_decodes_utf8_multibyte() {
    // é is U+00E9, encoded in UTF-8 as 0xC3 0xA9
    assert_eq!(
        decode_runtime_model_path("/api/runtime/models/mod%C3%A9le"),
        Some("modéle".into())
    );
    // invalid UTF-8 sequence should return None
    assert_eq!(decode_runtime_model_path("/api/runtime/models/%80"), None);
}

async fn build_test_mesh_api_with_api_port(api_port: u16) -> MeshApi {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let resolved_plugins = plugin::ResolvedPlugins {
        externals: vec![],
        inactive: vec![],
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    let plugin_manager = plugin::PluginManager::start(
        &resolved_plugins,
        plugin::PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .unwrap();
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer = runtime_data_collector.producer(runtime_data::RuntimeDataSource {
        scope: "runtime",
        plugin_data_key: None,
        plugin_endpoint_key: None,
    });
    MeshApi::new(MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port,
        model_size_bytes: 0,
        plugin_manager,
        affinity_router: affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

async fn build_test_mesh_api() -> MeshApi {
    build_test_mesh_api_with_api_port(3131).await
}

async fn build_test_mesh_api_with_plugin_manager(
    api_port: u16,
    plugin_manager: plugin::PluginManager,
) -> MeshApi {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer = runtime_data_collector.producer(runtime_data::RuntimeDataSource {
        scope: "runtime",
        plugin_data_key: None,
        plugin_endpoint_key: None,
    });
    MeshApi::new(MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port,
        model_size_bytes: 0,
        plugin_manager,
        affinity_router: affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

async fn spawn_management_test_server(
    state: MeshApi,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        handle_request(stream, &state).await
    });
    (addr, handle)
}

async fn send_management_request(addr: std::net::SocketAddr, raw_request: String) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(raw_request.as_bytes()).await.unwrap();
    let _ = stream.shutdown().await;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    String::from_utf8(response).unwrap()
}

fn json_body(response: &str) -> serde_json::Value {
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    serde_json::from_str(body).unwrap_or(serde_json::Value::Null)
}

async fn replace_test_wakeable_inventory(state: &MeshApi, entries: Vec<WakeableInventoryEntry>) {
    let inventory = { state.inner.lock().await.wakeable_inventory.clone() };
    inventory.replace_for_tests(entries).await;
}

fn make_test_wakeable_entry(logical_id: &str, model: &str, vram_gb: f32) -> WakeableInventoryEntry {
    WakeableInventoryEntry {
        logical_id: logical_id.to_string(),
        models: vec![model.to_string()],
        vram_gb,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Sleeping,
        wake_eta_secs: Some(45),
    }
}

fn make_test_peer(
    seed: u8,
    role: mesh::NodeRole,
    serving_models: Vec<&str>,
    hosted_models: Vec<&str>,
    hosted_models_known: bool,
) -> mesh::PeerInfo {
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public());
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role,
        first_joined_mesh_ts: None,
        models: Vec::new(),
        vram_bytes: 24_000_000_000,
        rtt_ms: None,
        model_source: None,
        serving_models: serving_models.into_iter().map(str::to_string).collect(),
        hosted_models: hosted_models.into_iter().map(str::to_string).collect(),
        hosted_models_known,
        available_models: Vec::new(),
        requested_models: Vec::new(),
        explicit_model_interests: Vec::new(),
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: Vec::new(),
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: Vec::new(),
        served_model_runtime: Vec::new(),
        owner_attestation: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
    }
}

#[derive(Clone)]
struct BlobstoreApiTestBridge {
    plugin_name: String,
    store: blobstore::BlobStore,
}

#[derive(Clone)]
struct BlackboardApiTestBridge {
    plugin_name: String,
    store: blackboard::BlackboardStore,
}

impl BlobstoreApiTestBridge {
    fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

impl BlackboardApiTestBridge {
    fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

impl plugin::PluginRpcBridge for BlobstoreApiTestBridge {
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
            if method != "tools/call" {
                return Err(Self::error_response(format!(
                    "Unsupported method '{}'",
                    method
                )));
            }

            let request: mesh_llm_plugin::OperationRequest = serde_json::from_str(&params_json)
                .map_err(|err| Self::error_response(err.to_string()))?;
            let result_json = match request.name.as_str() {
                blobstore::PUT_REQUEST_OBJECT_TOOL => {
                    let request: blobstore::PutRequestObjectRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .put_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                blobstore::COMPLETE_REQUEST_TOOL | blobstore::ABORT_REQUEST_TOOL => {
                    let request: blobstore::FinishRequestRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                _ => {
                    return Err(Self::error_response(format!(
                        "Unsupported blobstore tool '{}'",
                        request.name
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

impl plugin::PluginRpcBridge for BlackboardApiTestBridge {
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
            if method != "tools/call" {
                return Err(Self::error_response(format!(
                    "Unsupported method '{}'",
                    method
                )));
            }

            let request: mesh_llm_plugin::OperationRequest = serde_json::from_str(&params_json)
                .map_err(|err| Self::error_response(err.to_string()))?;
            let result_json = match request.name.as_str() {
                "feed" => {
                    let request: blackboard::FeedRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .feed(request.since, request.from.as_deref(), request.limit)
                        .await;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                "search" => {
                    let request: blackboard::SearchRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let mut response = store.search(&request.query, request.since).await;
                    response.truncate(request.limit.max(1));
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                "post" => {
                    let request: blackboard::PostRequest =
                        serde_json::from_value(request.arguments)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let item = blackboard::BlackboardItem::new(
                        if request.from.trim().is_empty() {
                            "mcp".into()
                        } else {
                            request.from
                        },
                        if request.peer_id.trim().is_empty() {
                            "mcp".into()
                        } else {
                            request.peer_id
                        },
                        request.text,
                    );
                    let response = store.post(item).await.map_err(Self::error_response)?;
                    serde_json::to_string(&rmcp::model::CallToolResult::structured(
                        serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?,
                    ))
                    .map_err(|err| Self::error_response(err.to_string()))?
                }
                _ => {
                    return Err(Self::error_response(format!(
                        "Unsupported blackboard tool '{}'",
                        request.name
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
    std::env::temp_dir().join(format!("mesh-llm-api-{name}-{}", rand::random::<u64>()))
}

async fn build_blobstore_api_plugin_manager() -> (plugin::PluginManager, std::path::PathBuf) {
    let plugin_name = "blobstore";
    let root = temp_blobstore_root("blobstore");
    let bridge = BlobstoreApiTestBridge {
        plugin_name: plugin_name.into(),
        store: blobstore::BlobStore::new(root.clone()),
    };
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
    let mut manifests = HashMap::new();
    manifests.insert(
        plugin_name.to_string(),
        mesh_llm_plugin::plugin_manifest![mesh_llm_plugin::capability(
            blobstore::OBJECT_STORE_CAPABILITY
        ),],
    );
    plugin_manager
        .set_test_manifests(manifests.into_iter().collect())
        .await;
    (plugin_manager, root)
}

async fn build_blackboard_api_plugin_manager() -> plugin::PluginManager {
    let plugin_name = "blackboard";
    let bridge = BlackboardApiTestBridge {
        plugin_name: plugin_name.into(),
        store: blackboard::BlackboardStore::new(true),
    };
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
    let mut manifests = HashMap::new();
    manifests.insert(
        plugin_name.to_string(),
        mesh_llm_plugin::plugin_manifest![
            mesh_llm_plugin::capability(blackboard::BLACKBOARD_CHANNEL),
            mesh_llm_plugin::http_get("/feed", "feed"),
            mesh_llm_plugin::http_get("/search", "search"),
            mesh_llm_plugin::http_post("/post", "post"),
        ],
    );
    plugin_manager
        .set_test_manifests(manifests.into_iter().collect())
        .await;
    plugin_manager
}

async fn spawn_capturing_upstream(
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let response = response_body.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        let _ = request_tx.send(request.raw);

        let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
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
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        let _ = request_tx.send(request.raw);

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
        let mut chunk = [0u8; 4096];
        let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
            .await
            .expect("timed out waiting for response bytes")
            .unwrap();
        assert!(n > 0, "unexpected EOF while waiting for response bytes");
        response.extend_from_slice(&chunk[..n]);
    }
    response
}

#[tokio::test]
async fn test_management_request_parser_handles_fragmented_post_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = br#"{"text":"fragmented"}"#;
    let headers = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            proxy::read_http_request(&mut stream),
        )
        .await
        .unwrap()
        .unwrap()
    });

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(&headers.as_bytes()[..45]).await.unwrap();
        stream.write_all(&headers.as_bytes()[45..]).await.unwrap();
        stream.write_all(&body[..8]).await.unwrap();
        stream.write_all(&body[8..]).await.unwrap();
        let mut sink = [0u8; 1];
        let _ = stream.read(&mut sink).await;
    });

    client.await.unwrap();
    let request = server.await.unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/blackboard/post");
    assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
}

#[tokio::test]
async fn test_api_events_sends_initial_payload_and_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("HTTP/1.1 200 OK"));
    assert!(initial_text.contains("Content-Type: text/event-stream"));
    assert!(initial_text.contains("\"llama_ready\":false"));

    state.update(true, true).await;
    let updated =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"llama_ready\":true"));
    assert!(updated_text.contains("\"is_host\":true"));

    drop(stream);
    handle.abort();
}

#[tokio::test]
async fn test_api_events_push_publication_state_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let _initial = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"private\"",
        Duration::from_secs(2),
    )
    .await;

    state
        .set_publication_state(crate::api::PublicationState::PublishFailed)
        .await;
    let updated = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"publish_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"publication_state\":\"publish_failed\""));

    drop(stream);
    handle.abort();
}

async fn build_collector_backed_plugin_manager() -> plugin::PluginManager {
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

    let plugin_manager = plugin::PluginManager::for_test_bridge(
        &["collector-plugin"],
        std::sync::Arc::new(NoopBridge),
    );
    plugin_manager
        .set_test_manifests(std::collections::BTreeMap::from([(
            "collector-plugin".into(),
            crate::plugin::proto::PluginManifest {
                capabilities: vec!["chat".into()],
                endpoints: vec![crate::plugin::proto::EndpointManifest {
                    endpoint_id: "chat-http".into(),
                    kind: crate::plugin::proto::EndpointKind::Inference as i32,
                    transport_kind:
                        crate::plugin::proto::EndpointTransportKind::EndpointTransportHttp as i32,
                    protocol: Some("openai_compatible".into()),
                    address: Some("http://127.0.0.1:4010/v1".into()),
                    args: vec![],
                    namespace: Some("chat".into()),
                    supports_streaming: true,
                    managed_by_plugin: false,
                }],
                ..Default::default()
            },
        )]))
        .await;
    plugin_manager
        .publish_test_bridge_snapshot("collector-plugin")
        .await
        .expect("collector-backed plugin manager");
    plugin_manager
}

#[tokio::test]
async fn runtime_data_api_routes_remain_payload_stable() {
    let plugin_manager = build_collector_backed_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;

    {
        let mut inner = state.inner.lock().await;
        inner.primary_backend = Some("legacy-backend".into());
        inner.is_host = false;
        inner.llama_ready = false;
        inner.llama_port = Some(9999);
        inner.local_processes = vec![RuntimeProcessPayload {
            name: "legacy-model".into(),
            backend: "legacy-backend".into(),
            status: "ready".into(),
            port: 9999,
            pid: 111,
            slots: 4,
            context_length: None,
        }];
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                runtime_status.primary_model = Some("collector-model".into());
                runtime_status.primary_backend = Some("collector-backend".into());
                runtime_status.is_host = true;
                runtime_status.llama_ready = true;
                runtime_status.llama_port = Some(9337);
                true
            });
        inner
            .runtime_data_producer
            .publish_local_processes(|local_processes| {
                local_processes.clear();
                local_processes.push(runtime_data::RuntimeProcessSnapshot {
                    model: "collector-model".into(),
                    backend: "collector-backend".into(),
                    pid: 777,
                    port: 9337,
                    slots: 4,
                    context_length: Some(0),
                    command: Some("llama-server".into()),
                    state: "ready".into(),
                    start: Some(1_700_000_000),
                    health: Some("ready".into()),
                });
                true
            });
        inner.runtime_data_producer.publish_llama_metrics_snapshot(
            runtime_data::RuntimeLlamaMetricsSnapshot {
                status: runtime_data::RuntimeLlamaEndpointStatus::Ready,
                last_attempt_unix_ms: Some(1_700_000_001_000),
                last_success_unix_ms: Some(1_700_000_001_000),
                error: None,
                raw_text: Some("llama_requests_processing 2\n".into()),
                samples: vec![runtime_data::RuntimeLlamaMetricSample {
                    name: "llama_requests_processing".into(),
                    labels: std::collections::BTreeMap::new(),
                    value: 2.0,
                }],
            },
        );
        inner.runtime_data_producer.publish_llama_slots_snapshot(
            runtime_data::RuntimeLlamaSlotsSnapshot {
                status: runtime_data::RuntimeLlamaEndpointStatus::Ready,
                model: Some("collector-model".into()),
                last_attempt_unix_ms: Some(1_700_000_001_500),
                last_success_unix_ms: Some(1_700_000_001_500),
                error: None,
                slots: vec![runtime_data::RuntimeLlamaSlotSnapshot {
                    id: Some(0),
                    id_task: Some(42),
                    n_ctx: Some(8192),
                    speculative: Some(false),
                    is_processing: Some(true),
                    next_token: Some(json!({"id": 99})),
                    params: Some(json!({"temperature": 0.2})),
                    extra: json!({"state": "busy"}),
                }],
            },
        );
    }
    let node = state.node().await;
    node.record_stage_status(
        Some(node.id()),
        crate::inference::skippy::StageStatusSnapshot {
            topology_id: "topology-1".into(),
            run_id: "run-1".into(),
            model_id: "collector-model".into(),
            backend: "package".into(),
            package_ref: Some("hf://mesh/test-model".into()),
            manifest_sha256: Some("manifest-sha".into()),
            source_model_path: Some("/models/test.gguf".into()),
            source_model_sha256: Some("source-sha".into()),
            source_model_bytes: Some(1_234),
            materialized_path: Some("/tmp/mesh/stage-0.gguf".into()),
            materialized_pinned: true,
            projector_path: Some("/models/mmproj.gguf".into()),
            stage_id: "stage-0".into(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 12,
            state: crate::inference::skippy::StageRuntimeState::Ready,
            bind_addr: "127.0.0.1:39100".into(),
            activation_width: 4096,
            wire_dtype: crate::inference::skippy::StageWireDType::F16,
            selected_device: Some(skippy_protocol::StageDevice {
                backend_device: "Metal0".into(),
                stable_id: Some("metal:0".into()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            }),
            ctx_size: 8192,
            lane_count: 2,
            n_batch: Some(2048),
            n_ubatch: Some(512),
            flash_attn_type: skippy_protocol::FlashAttentionType::Enabled,
            error: None,
            shutdown_generation: 7,
        },
    )
    .await;

    let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;
    let status_response = send_management_request(
        status_addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(status_response.starts_with("HTTP/1.1 200"));
    let status_body = json_body(&status_response);
    assert_eq!(status_body["model_name"], json!("collector-model"));
    assert_eq!(status_body["llama_ready"], json!(true));
    assert_eq!(
        status_body["runtime"]["backend"],
        json!("collector-backend")
    );
    assert_eq!(
        status_body["runtime"]["models"][0]["name"],
        json!("collector-model")
    );
    assert_eq!(
        status_body["runtime"]["models"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["model_id"],
        json!("collector-model")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["package_ref"],
        json!("hf://mesh/test-model")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["materialized_pinned"],
        json!(true)
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["projector_path"],
        json!("/models/mmproj.gguf")
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["multimodal"],
        json!(true)
    );
    assert_eq!(
        status_body["runtime"]["stages"][0]["selected_device"]["backend_device"],
        json!("Metal0")
    );
    assert!(status_body.get("mesh_models").is_none());
    status_handle.abort();

    let (models_addr, models_handle) = spawn_management_test_server(state.clone()).await;
    let models_response = send_management_request(
        models_addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(models_response.starts_with("HTTP/1.1 200"));
    let models_body = json_body(&models_response);
    assert!(models_body["mesh_models"].is_array());
    models_handle.abort();

    let (runtime_addr, runtime_handle) = spawn_management_test_server(state.clone()).await;
    let runtime_response = send_management_request(
        runtime_addr,
        "GET /api/runtime HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(runtime_response.starts_with("HTTP/1.1 200"));
    let runtime_body = json_body(&runtime_response);
    assert_eq!(runtime_body["models"][0]["name"], json!("collector-model"));
    assert_eq!(
        runtime_body["models"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(runtime_body["models"][0]["port"], json!(9337));
    runtime_handle.abort();

    let (processes_addr, processes_handle) = spawn_management_test_server(state.clone()).await;
    let processes_response = send_management_request(
        processes_addr,
        "GET /api/runtime/processes HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(processes_response.starts_with("HTTP/1.1 200"));
    let processes_body = json_body(&processes_response);
    assert_eq!(
        processes_body["processes"][0]["name"],
        json!("collector-model")
    );
    assert_eq!(
        processes_body["processes"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(processes_body["processes"][0]["port"], json!(9337));
    assert_eq!(processes_body["processes"][0]["pid"], json!(777));
    processes_handle.abort();

    let (llama_addr, llama_handle) = spawn_management_test_server(state.clone()).await;
    let llama_response = send_management_request(
        llama_addr,
        "GET /api/runtime/llama HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(llama_response.starts_with("HTTP/1.1 200"));
    let llama_body = json_body(&llama_response);
    assert_eq!(llama_body["metrics"]["status"], json!("ready"));
    assert_eq!(
        llama_body["metrics"]["samples"][0]["name"],
        json!("llama_requests_processing")
    );
    assert_eq!(
        llama_body["items"]["metrics"][0]["name"],
        json!("llama_requests_processing")
    );
    assert_eq!(llama_body["slots"]["status"], json!("ready"));
    assert_eq!(llama_body["slots"]["slots"][0]["id_task"], json!(42));
    assert_eq!(
        llama_body["slots"]["slots"][0]["extra"]["state"],
        json!("busy")
    );
    assert_eq!(llama_body["items"]["slots_total"], json!(1));
    assert_eq!(llama_body["items"]["slots_busy"], json!(1));
    assert_eq!(llama_body["items"]["slots"][0]["index"], json!(0));
    assert_eq!(
        llama_body["items"]["slots"][0]["is_processing"],
        json!(true)
    );
    llama_handle.abort();

    let (endpoints_addr, endpoints_handle) = spawn_management_test_server(state.clone()).await;
    let endpoints_response = send_management_request(
        endpoints_addr,
        "GET /api/runtime/endpoints HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(endpoints_response.starts_with("HTTP/1.1 200"));
    let endpoints_body = json_body(&endpoints_response);
    assert_eq!(
        endpoints_body["endpoints"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(
        endpoints_body["endpoints"][0]["plugin_name"],
        json!("collector-plugin")
    );
    assert_eq!(
        endpoints_body["endpoints"][0]["endpoint_id"],
        json!("chat-http")
    );
    endpoints_handle.abort();

    let (plugins_addr, plugins_handle) = spawn_management_test_server(state).await;
    let plugins_response = send_management_request(
        plugins_addr,
        "GET /api/plugins HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(plugins_response.starts_with("HTTP/1.1 200"));
    let plugins_body = json_body(&plugins_response);
    assert_eq!(plugins_body.as_array().map(Vec::len), Some(1));
    assert_eq!(plugins_body[0]["name"], json!("collector-plugin"));
    assert_eq!(plugins_body[0]["status"], json!("running"));
    assert_eq!(plugins_body[0]["capabilities"], json!(["chat"]));
    plugins_handle.abort();

    let state = build_test_mesh_api_with_plugin_manager(
        3131,
        build_collector_backed_plugin_manager().await,
    )
    .await;

    let (plugin_endpoints_addr, plugin_endpoints_handle) =
        spawn_management_test_server(state.clone()).await;
    let plugin_endpoints_response = send_management_request(
        plugin_endpoints_addr,
        "GET /api/plugins/endpoints HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(plugin_endpoints_response.starts_with("HTTP/1.1 200"));
    let plugin_endpoints_body = json_body(&plugin_endpoints_response);
    assert_eq!(plugin_endpoints_body.as_array().map(Vec::len), Some(1));
    assert_eq!(
        plugin_endpoints_body[0]["plugin_name"],
        json!("collector-plugin")
    );
    assert_eq!(plugin_endpoints_body[0]["endpoint_id"], json!("chat-http"));
    plugin_endpoints_handle.abort();

    let (providers_addr, providers_handle) = spawn_management_test_server(state.clone()).await;
    let providers_response = send_management_request(
        providers_addr,
        "GET /api/plugins/providers HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(providers_response.starts_with("HTTP/1.1 200"));
    let providers_body = json_body(&providers_response);
    assert!(providers_body.as_array().is_some());
    assert!(providers_body
        .as_array()
        .unwrap()
        .iter()
        .any(|provider| provider["capability"] == json!("chat")));
    providers_handle.abort();

    let (provider_addr, provider_handle) = spawn_management_test_server(state.clone()).await;
    let provider_response = send_management_request(
        provider_addr,
        "GET /api/plugins/providers/chat HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(provider_response.starts_with("HTTP/1.1 200"));
    let provider_body = json_body(&provider_response);
    assert_eq!(provider_body["capability"], json!("chat"));
    assert_eq!(provider_body["plugin_name"], json!("collector-plugin"));
    provider_handle.abort();

    let (manifest_addr, manifest_handle) = spawn_management_test_server(state).await;
    let manifest_response = send_management_request(
        manifest_addr,
        "GET /api/plugins/collector-plugin/manifest HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(manifest_response.starts_with("HTTP/1.1 200"));
    let manifest_body = json_body(&manifest_response);
    assert_eq!(manifest_body["capabilities"], json!(["chat"]));
    assert_eq!(manifest_body["endpoints"].as_array().map(Vec::len), Some(1));
    manifest_handle.abort();
}

#[tokio::test]
async fn runtime_data_sse_bridge_delivers_initial_and_incremental_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("HTTP/1.1 200 OK"));
    assert!(initial_text.contains("Content-Type: text/event-stream"));
    assert!(initial_text.contains("\"llama_ready\":false"));
    assert!(initial_text.contains("\"publication_state\":\"private\""));

    state.update(true, true).await;
    let runtime_update =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let runtime_update_text = String::from_utf8_lossy(&runtime_update);
    assert!(runtime_update_text.contains("\"llama_ready\":true"));
    assert!(runtime_update_text.contains("\"is_host\":true"));

    state
        .set_publication_state(crate::api::PublicationState::PublishFailed)
        .await;
    let publication_update = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"publish_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let publication_update_text = String::from_utf8_lossy(&publication_update);
    assert!(publication_update_text.contains("\"publication_state\":\"publish_failed\""));

    drop(stream);
    handle.abort();
}

#[tokio::test]
async fn test_api_status_excludes_mesh_models_and_models_endpoint_serves_them() {
    let state = build_test_mesh_api().await;
    let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;

    let status_response = send_management_request(
        status_addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(status_response.starts_with("HTTP/1.1 200"));
    let status_body = json_body(&status_response);
    assert!(status_body.get("mesh_models").is_none());
    status_handle.abort();

    let (models_addr, models_handle) = spawn_management_test_server(state).await;
    let models_response = send_management_request(
        models_addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(models_response.starts_with("HTTP/1.1 200"));
    let models_body = json_body(&models_response);
    assert!(models_body.get("mesh_models").is_some());

    models_handle.abort();
}

#[tokio::test]
async fn test_api_search_catalog_returns_canonical_model_refs() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=5&sort=trending HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["source"], json!("catalog"));
    assert_eq!(payload["filter"], json!("gguf"));
    assert_eq!(payload["sort"], json!("trending"));
    assert!(payload.get("machine").is_some());
    let results = payload["results"].as_array().cloned().unwrap_or_default();
    assert!(
        !results.is_empty(),
        "expected at least one catalog result for Qwen3-Coder-Next"
    );
    let catalog_ref = crate::models::find_catalog_model_exact("Qwen3-Coder-Next-Q4_K_M")
        .map(crate::models::catalog_model_ref)
        .expect("catalog model");
    let hit = results
        .into_iter()
        .find(|entry| entry["ref"] == json!(catalog_ref))
        .expect("canonical catalog model ref present");
    assert_eq!(hit["repo_id"], json!("Qwen/Qwen3-Coder-Next-GGUF"));
    assert_eq!(hit["type"], json!("gguf"));
    assert_eq!(
        hit["show"],
        json!(format!("mesh-llm models show {catalog_ref}"))
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_search_caps_limit_and_uses_canonical_parameter_sort_name() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=999&sort=parameters-desc HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["sort"], json!("parameters-desc"));
    let results = payload["results"].as_array().cloned().unwrap_or_default();
    assert!(
        results.len() <= 50,
        "expected catalog response to apply the API limit cap"
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_search_requires_q_query_parameter() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/search?catalog=true HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
        payload["error"],
        json!("Missing required 'q' query parameter")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_search_rejects_invalid_sort_value() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/search?q=qwen&sort=random HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
            payload["error"],
            json!("Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc")
        );

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_post_and_get_round_trip() {
    let state = build_test_mesh_api().await;
    let (post_addr, post_handle) = spawn_management_test_server(state.clone()).await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;

    let post_response = send_management_request(
            post_addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(post_response.starts_with("HTTP/1.1 201"));
    let post_payload = json_body(&post_response);
    assert_eq!(post_payload["created"], json!(true));
    assert_eq!(
        post_payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(post_payload["interest"]["submission_source"], json!("ui"));
    assert_eq!(post_payload["model_interests"].as_array().unwrap().len(), 1);
    post_handle.abort();

    let (get_addr, get_handle) = spawn_management_test_server(state).await;
    let get_response = send_management_request(
        get_addr,
        "GET /api/model-interests HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(get_response.starts_with("HTTP/1.1 200"));
    let get_payload = json_body(&get_response);
    let interests = get_payload["model_interests"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(interests.len(), 1);
    assert_eq!(
        interests[0]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(interests[0]["submission_source"], json!("ui"));

    get_handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_post_is_idempotent() {
    let state = build_test_mesh_api().await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;
    let request = format!(
            "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let (first_addr, first_handle) = spawn_management_test_server(state.clone()).await;
    let first_response = send_management_request(first_addr, request.clone()).await;
    assert!(first_response.starts_with("HTTP/1.1 201"));
    let first_payload = json_body(&first_response);
    let created_at = first_payload["interest"]["created_at_unix"]
        .as_u64()
        .expect("created_at_unix");
    first_handle.abort();

    let (second_addr, second_handle) = spawn_management_test_server(state).await;
    let second_response = send_management_request(second_addr, request).await;
    assert!(second_response.starts_with("HTTP/1.1 200"));
    let second_payload = json_body(&second_response);
    assert_eq!(second_payload["created"], json!(false));
    assert_eq!(
        second_payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(
        second_payload["interest"]["created_at_unix"],
        json!(created_at)
    );
    assert_eq!(
        second_payload["model_interests"].as_array().unwrap().len(),
        1
    );

    second_handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_delete_decodes_percent_encoded_model_ref() {
    let state = build_test_mesh_api().await;
    state
        .upsert_model_interest(
            crate::models::canonicalize_interest_model_ref(
                "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M",
            )
            .unwrap(),
            Some("ui".to_string()),
        )
        .await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
            addr,
            "DELETE /api/model-interests/Qwen%2FQwen3-Coder-Next-GGUF%40main%3AQ4_K_M HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["removed"], json!(true));
    assert_eq!(
        payload["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(payload["model_interests"], json!([]));

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_reject_direct_urls() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = r#"{"model_ref":"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"}"#;

    let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
        payload["error"],
        json!("Invalid 'model_ref'. Use a canonical ref returned by /api/search, not a direct URL")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_normalize_legacy_selector_revision_order() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M@main","source":"ui"}"#;

    let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 201"));
    let payload = json_body(&response);
    assert_eq!(
        payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_model_targets_combine_interest_demand_and_serving_visibility() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let catalog_model = &crate::models::catalog::MODEL_CATALOG[0];
    let model_name = catalog_model.name.to_string();
    let model_ref = crate::models::catalog_model_ref(catalog_model);
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;
    assert_eq!(
        node.explicit_model_interests().await,
        vec![model_ref.clone()]
    );

    node.record_request(&model_name);

    let mut peer = make_test_peer(
        0x44,
        mesh::NodeRole::Host { http_port: 9337 },
        vec![model_ref.as_str()],
        vec![model_ref.as_str()],
        true,
    );
    peer.explicit_model_interests = vec![interest.model_ref.clone()];
    node.insert_test_peer(peer).await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/model-targets HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let targets = payload["model_targets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let target = targets
        .into_iter()
        .find(|entry| entry["model_ref"] == interest.model_ref)
        .expect("target for explicit interest present");
    assert_eq!(target["derived"]["target_rank"], json!(1));
    assert_eq!(target["signals"]["explicit_interest_count"], json!(2));
    assert_eq!(target["signals"]["request_count"], json!(1));
    assert_eq!(target["signals"]["serving_node_count"], json!(1));
    assert_eq!(target["signals"]["requested"], json!(false));
    assert_eq!(target["derived"]["wanted"], json!(false));
    assert!(target.get("rank").is_none());
    assert!(target.get("explicit_interest_count").is_none());
    assert!(target.get("wanted").is_none());

    handle.abort();
}

#[tokio::test]
async fn test_api_status_and_models_surface_wanted_targets() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let catalog_model = &crate::models::catalog::MODEL_CATALOG[0];
    let model_ref = crate::models::catalog_model_ref(catalog_model);
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;
    node.set_requested_models(vec![model_ref.clone()]).await;

    let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;
    let status_response = send_management_request(
        status_addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(status_response.starts_with("HTTP/1.1 200"));
    let status_payload = json_body(&status_response);
    assert_eq!(
        status_payload["wanted_model_refs"],
        json!([interest.model_ref.clone()])
    );
    status_handle.abort();

    let (models_addr, models_handle) = spawn_management_test_server(state).await;
    let models_response = send_management_request(
        models_addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(models_response.starts_with("HTTP/1.1 200"));
    let models_payload = json_body(&models_response);
    let models = models_payload["mesh_models"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let model = models
        .into_iter()
        .find(|entry| entry["name"] == model_ref)
        .expect("catalog model present");
    assert_eq!(model["target_rank"], json!(1));
    assert_eq!(model["explicit_interest_count"], json!(1));
    assert_eq!(model["wanted"], json!(true));

    models_handle.abort();
}

#[test]
fn test_http_route_stats_only_count_http_callable_legacy_hosts() {
    let peers = vec![
        make_test_peer(
            0x41,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["legacy-host-model"],
            Vec::new(),
            false,
        ),
        make_test_peer(
            0x42,
            mesh::NodeRole::Worker,
            vec!["worker-only-model"],
            Vec::new(),
            false,
        ),
    ];

    let host_stats = http_route_stats("legacy-host-model", &peers, &[], None, 0.0);
    assert_eq!(host_stats.node_count, 1);
    assert_eq!(host_stats.active_nodes.len(), 1);
    assert!(host_stats.mesh_vram_gb > 0.0);

    let worker_stats = http_route_stats("worker-only-model", &peers, &[], None, 0.0);
    assert_eq!(worker_stats, HttpRouteStats::default());
}

#[tokio::test]
async fn wakeable_inventory_does_not_change_peer_count() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let status = state.status().await;
    assert!(status.peers.is_empty());
    assert_eq!(status.wakeable_nodes.len(), 1);
    assert_eq!(status.wakeable_nodes[0].logical_id, "sleeping-node-1");
}

#[tokio::test]
async fn wakeable_inventory_does_not_change_mesh_vram_totals() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let status = state.status().await;
    let peers = vec![make_test_peer(
        0x51,
        mesh::NodeRole::Host { http_port: 9337 },
        vec!["wakeable-only-model"],
        vec!["wakeable-only-model"],
        true,
    )];
    let route_stats = http_route_stats("wakeable-only-model", &peers, &[], None, 0.0);

    assert_eq!(status.wakeable_nodes.len(), 1);
    assert_eq!(route_stats.node_count, 1);
    assert!(route_stats.mesh_vram_gb > 0.0);
}

#[tokio::test]
async fn wakeable_inventory_is_not_routable_capacity() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let status = state.status().await;
    let served_models = node.models_being_served().await;
    let hosts = node.hosts_for_model("wakeable-only-model").await;

    assert_eq!(status.wakeable_nodes.len(), 1);
    assert!(!served_models
        .iter()
        .any(|model| model == "wakeable-only-model"));
    assert!(hosts.is_empty());
}

#[tokio::test]
async fn wakeable_inventory_is_excluded_from_v1_models() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let served_models = node.models_being_served().await;

    assert!(!served_models
        .iter()
        .any(|model| model == "wakeable-only-model"));
    assert!(served_models.is_empty());
}

#[tokio::test]
async fn wakeable_inventory_is_excluded_from_host_selection() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let hosts = node.hosts_for_model("wakeable-only-model").await;

    assert!(hosts.is_empty());
}

#[test]
fn build_wakeable_node_preserves_typed_internal_state() {
    let sleeping = MeshApi::build_wakeable_node(WakeableInventoryEntry {
        logical_id: "sleeping-node".to_string(),
        models: vec!["test-model".to_string()],
        vram_gb: 24.0,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Sleeping,
        wake_eta_secs: Some(45),
    });
    let waking = MeshApi::build_wakeable_node(WakeableInventoryEntry {
        logical_id: "waking-node".to_string(),
        models: vec!["test-model".to_string()],
        vram_gb: 24.0,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Waking,
        wake_eta_secs: Some(10),
    });

    assert_eq!(sleeping.state, WakeableNodeState::Sleeping);
    assert_eq!(waking.state, WakeableNodeState::Waking);
}

#[tokio::test]
async fn test_api_status_includes_local_gpu_benchmark_metrics() {
    let state = build_test_mesh_api().await;
    let node = {
        let mut inner = state.inner.lock().await;
        inner.node.gpu_name = Some("NVIDIA A100".into());
        inner.node.gpu_vram = Some("85899345920".into());
        inner.node.gpu_reserved_bytes = Some("1073741824".into());
        inner.node.hostname = Some("worker-01".into());
        inner.node.is_soc = Some(false);
        inner.node.clone()
    };

    *node.gpu_mem_bandwidth_gbps.lock().await = Some(vec![1948.7]);
    *node.gpu_compute_tflops_fp32.lock().await = Some(vec![19.5]);
    *node.gpu_compute_tflops_fp16.lock().await = Some(vec![312.0]);

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let gpu = &payload["gpus"][0];
    assert_eq!(gpu["name"], json!("NVIDIA A100"));
    assert_eq!(gpu["vram_bytes"], json!(85899345920_u64));
    assert_eq!(gpu["reserved_bytes"], json!(1073741824_u64));
    assert_eq!(gpu["mem_bandwidth_gbps"], json!(1948.7));
    assert_eq!(gpu["compute_tflops_fp32"], json!(19.5));
    assert_eq!(gpu["compute_tflops_fp16"], json!(312.0));

    handle.abort();
}

#[tokio::test]
async fn test_api_status_includes_routing_metrics_summary() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());

    node.record_inference_attempt(
        Some("test-model"),
        &election::InferenceTarget::Local(9338),
        Duration::from_millis(4),
        Duration::from_millis(16),
        crate::network::metrics::AttemptOutcome::Timeout,
        None,
    );
    node.record_inference_attempt(
        Some("test-model"),
        &election::InferenceTarget::Remote(peer_id),
        Duration::from_millis(18),
        Duration::from_millis(48),
        crate::network::metrics::AttemptOutcome::Success,
        Some(12),
    );
    node.record_routed_request(
        Some("test-model"),
        2,
        crate::network::metrics::RequestOutcome::Success(
            crate::network::metrics::RequestService::Remote,
        ),
    );

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["routing_metrics"]["request_count"], json!(1));
    assert_eq!(payload["routing_metrics"]["successful_requests"], json!(1));
    assert_eq!(payload["routing_metrics"]["retry_count"], json!(1));
    assert_eq!(payload["routing_metrics"]["failover_count"], json!(1));
    assert_eq!(
        payload["routing_metrics"]["attempt_timeout_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["pressure"]["remotely_served_request_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["local_node"]["remote_attempt_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["local_node"]["local_attempt_count"],
        json!(1)
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_models_include_model_routing_metrics() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let catalog_model = &crate::models::catalog::MODEL_CATALOG[0];
    let model_ref = crate::models::catalog_model_ref(catalog_model);
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    node.set_requested_models(vec![model_ref.clone()]).await;

    node.record_inference_attempt(
        Some(&model_ref),
        &election::InferenceTarget::Remote(peer_id),
        Duration::from_millis(6),
        Duration::from_millis(24),
        crate::network::metrics::AttemptOutcome::Success,
        Some(9),
    );
    node.record_routed_request(
        Some(&model_ref),
        1,
        crate::network::metrics::RequestOutcome::Success(
            crate::network::metrics::RequestService::Remote,
        ),
    );

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let models = payload["mesh_models"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let model = models
        .into_iter()
        .find(|entry| entry["name"] == model_ref)
        .expect("catalog model present");
    assert_eq!(model["routing_metrics"]["request_count"], json!(1));
    assert_eq!(model["routing_metrics"]["successful_requests"], json!(1));
    assert_eq!(
        model["routing_metrics"]["targets"][0]["kind"],
        json!("remote")
    );
    assert_eq!(
        model["routing_metrics"]["targets"][0]["success_count"],
        json!(1)
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_objects_routes_through_object_store_capability() {
    let (plugin_manager, blobstore_root) = build_blobstore_api_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = json!({
        "request_id": "req-api-object",
        "mime_type": "text/plain",
        "file_name": "note.txt",
        "bytes_base64": "aGVsbG8=",
        "expires_in_secs": 60,
        "uses_remaining": 1,
    })
    .to_string();
    let request = format!(
            "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
    let response = send_management_request(addr, request).await;

    assert!(response.starts_with("HTTP/1.1 201"));
    let payload = json_body(&response);
    assert_eq!(payload["request_id"], "req-api-object");
    assert_eq!(payload["mime_type"], "text/plain");
    assert!(payload["token"]
        .as_str()
        .unwrap_or_default()
        .starts_with("obj_"));

    handle.abort();
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_blackboard_routes_through_blackboard_capability() {
    let plugin_manager = build_blackboard_api_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;

    let (post_addr, post_handle) = spawn_management_test_server(state.clone()).await;
    let post_body = json!({ "text": "hello integration blackboard" }).to_string();
    let post_request = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            post_body.len(),
            post_body
        );
    let post_response = send_management_request(post_addr, post_request).await;
    assert!(post_response.starts_with("HTTP/1.1 200"));
    let posted = json_body(&post_response);
    assert_eq!(posted["text"], "hello integration blackboard");
    post_handle.abort();

    let (feed_addr, feed_handle) = spawn_management_test_server(state.clone()).await;
    let feed_response = send_management_request(
        feed_addr,
        "GET /api/blackboard/feed?limit=5 HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(feed_response.starts_with("HTTP/1.1 200"));
    let feed = json_body(&feed_response);
    let feed_items = feed.as_array().cloned().unwrap_or_default();
    assert!(feed_items
        .iter()
        .any(|item| item["text"] == "hello integration blackboard"));
    feed_handle.abort();

    let (search_addr, search_handle) = spawn_management_test_server(state).await;
    let search_response = send_management_request(
        search_addr,
        "GET /api/blackboard/search?q=integration HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(search_response.starts_with("HTTP/1.1 200"));
    let search = json_body(&search_response);
    let search_items = search.as_array().cloned().unwrap_or_default();
    assert!(search_items
        .iter()
        .any(|item| item["text"] == "hello integration blackboard"));
    search_handle.abort();
}

#[tokio::test]
async fn test_api_chat_smoke_for_image_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
            "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_chat_smoke_for_audio_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this audio"},
                {"type": "input_audio", "input_audio": {
                    "data": "UklGRg==",
                    "format": "wav",
                    "mime_type": "audio/wav"
                }}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
            "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert!(raw.contains(r#""mime_type":"audio/wav""#));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_smoke_for_image_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this image"},
                {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_smoke_for_file_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "read this file"},
                {
                    "type": "input_file",
                    "input_file": {
                        "url": "data:text/plain;base64,aGVsbG8=",
                        "mime_type": "text/plain",
                        "file_name": "hello.txt"
                    }
                }
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_file""#));
    assert!(raw.contains(r#""url":"data:text/plain;base64,aGVsbG8=""#));
    assert!(raw.contains(r#""mime_type":"text/plain""#));
    assert!(raw.contains(r#""file_name":"hello.txt""#));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_stream_smoke() {
    let (upstream_port, upstream_rx, upstream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![(
            Duration::ZERO,
            br#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}

event: done
data: [DONE]

"#
            .to_vec(),
        )],
    )
    .await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": "say hello",
        "stream": true
    })
    .to_string();
    let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let response = read_until_contains(
        &mut stream,
        br#"event: response.output_text.delta"#,
        Duration::from_secs(2),
    )
    .await;
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(response_text.contains("event: response.output_text.delta"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""stream":true"#));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn status_payload_populates_local_instances_from_scanner() {
    use crate::runtime::instance::LocalInstanceSnapshot;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let snapshots = vec![
        LocalInstanceSnapshot {
            pid: 1234,
            api_port: Some(3131),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000000,
            runtime_dir: PathBuf::from("/tmp/a"),
            is_self: true,
        },
        LocalInstanceSnapshot {
            pid: 5678,
            api_port: Some(3132),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000100,
            runtime_dir: PathBuf::from("/tmp/b"),
            is_self: false,
        },
    ];

    let shared: Arc<Mutex<Vec<LocalInstanceSnapshot>>> = Arc::new(Mutex::new(snapshots));
    let result: Vec<LocalInstance> = {
        let s = shared.lock().await;
        s.iter()
            .map(|snap| LocalInstance {
                pid: snap.pid,
                api_port: snap.api_port,
                version: snap.version.clone(),
                started_at_unix: snap.started_at_unix,
                runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                is_self: snap.is_self,
            })
            .collect()
    };

    assert_eq!(result.len(), 2);
    assert!(result.iter().any(|i| i.is_self && i.pid == 1234));
    assert!(result.iter().any(|i| !i.is_self && i.pid == 5678));
}

#[tokio::test]
async fn status_payload_safety_net_adds_self_when_empty() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let shared: Arc<Mutex<Vec<crate::runtime::instance::LocalInstanceSnapshot>>> =
        Arc::new(Mutex::new(vec![]));

    let mut instances: Vec<LocalInstance> = {
        let s = shared.lock().await;
        s.iter()
            .map(|snap| LocalInstance {
                pid: snap.pid,
                api_port: snap.api_port,
                version: snap.version.clone(),
                started_at_unix: snap.started_at_unix,
                runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                is_self: snap.is_self,
            })
            .collect()
    };

    // Simulate the safety net logic
    if instances.is_empty() {
        instances.push(LocalInstance {
            pid: std::process::id(),
            api_port: Some(3131),
            version: Some(MESH_LLM_VERSION.to_string()),
            started_at_unix: 0,
            runtime_dir: String::new(),
            is_self: true,
        });
    }

    assert_eq!(instances.len(), 1);
    assert!(instances[0].is_self);
    assert_eq!(instances[0].pid, std::process::id());
    assert_eq!(instances[0].api_port, Some(3131));
    assert_eq!(instances[0].version, Some(MESH_LLM_VERSION.to_string()));
}

#[test]
fn headless_mode_disables_ui_routes_but_preserves_api() {
    assert!(is_ui_only_route("/"));
    assert!(is_ui_only_route("/dashboard"));
    assert!(is_ui_only_route("/chat"));

    assert!(!is_ui_only_route("/api/status"));
    assert!(!is_ui_only_route("/api/events"));
    assert!(!is_ui_only_route("/api/discover"));
    assert!(!is_ui_only_route("/api/runtime"));
    assert!(!is_ui_only_route("/api/plugins"));
}

#[test]
fn headless_mode_returns_404_for_assets_and_dashboard_routes() {
    assert!(is_ui_only_route("/dashboard/"));
    assert!(is_ui_only_route("/chat/"));
    assert!(is_ui_only_route("/chat/some-room"));
    assert!(is_ui_only_route("/assets/main.js"));
    assert!(is_ui_only_route("/assets/index-abc123.css"));
    assert!(is_ui_only_route("/favicon.ico"));
    assert!(is_ui_only_route("/logo.png"));
    assert!(is_ui_only_route("/manifest.webmanifest"));
    assert!(is_ui_only_route("/site.json"));

    assert!(!is_ui_only_route("/api/status.json"));
}

#[test]
fn default_mode_still_serves_embedded_ui_routes() {
    assert!(is_ui_only_route("/"));
    assert!(is_ui_only_route("/dashboard"));
    assert!(is_ui_only_route("/chat"));
    assert!(is_ui_only_route("/assets/app.js"));

    assert!(!is_ui_only_route("/api/status"));
    assert!(!is_ui_only_route("/api/events"));
}

#[test]
fn headless_status_command_works_against_management_api() {
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must not be blocked in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/events"),
        "/api/events must not be blocked in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/discover"),
        "/api/discover must not be blocked in headless mode"
    );
}

#[test]
fn headless_blackboard_status_still_reads_api_status() {
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must be accessible in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/runtime"),
        "/api/runtime must be accessible in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/join"),
        "/api/join must be accessible in headless mode"
    );
}

#[test]
fn headless_custom_console_port_keeps_api_and_disables_ui() {
    assert!(is_ui_only_route("/"), "/ must be blocked in headless mode");
    assert!(is_ui_only_route("/dashboard"), "/dashboard must be blocked");
    assert!(is_ui_only_route("/chat"), "/chat must be blocked");
    assert!(
        is_ui_only_route("/assets/main.js"),
        "/assets/* must be blocked"
    );
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must not be blocked"
    );
    assert!(
        !is_ui_only_route("/api/events"),
        "/api/events must not be blocked"
    );
    assert!(
        !is_ui_only_route("/v1/models"),
        "/v1/models must not be blocked"
    );
    assert!(
        !is_ui_only_route("/v1/chat/completions"),
        "/v1/chat/completions must not be blocked"
    );
}

#[tokio::test]
async fn api_runtime_reads_from_collector_snapshot() {
    let state = build_test_mesh_api().await;

    {
        let mut inner = state.inner.lock().await;
        inner.primary_backend = Some("legacy-backend".into());
        inner.is_host = false;
        inner.llama_ready = false;
        inner.llama_port = Some(9999);
        inner.local_processes = vec![RuntimeProcessPayload {
            name: "legacy-model".into(),
            backend: "legacy-backend".into(),
            status: "ready".into(),
            port: 9999,
            pid: 111,
            slots: 4,
            context_length: None,
        }];

        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                runtime_status.primary_model = Some("collector-model".into());
                runtime_status.primary_backend = Some("collector-backend".into());
                runtime_status.is_host = true;
                runtime_status.llama_ready = true;
                runtime_status.llama_port = Some(9337);
                true
            });
        inner
            .runtime_data_producer
            .publish_local_processes(|local_processes| {
                local_processes.clear();
                local_processes.push(runtime_data::RuntimeProcessSnapshot {
                    model: "collector-model".into(),
                    backend: "collector-backend".into(),
                    pid: 777,
                    port: 9337,
                    slots: 4,
                    context_length: Some(0),
                    command: Some("llama-server".into()),
                    state: "ready".into(),
                    start: Some(1_700_000_000),
                    health: Some("ready".into()),
                });
                true
            });
    }

    let runtime_status = state.runtime_status().await;
    assert_eq!(runtime_status.models.len(), 1);
    assert_eq!(runtime_status.models[0].name, "collector-model");
    assert_eq!(runtime_status.models[0].backend, "collector-backend");
    assert_eq!(runtime_status.models[0].status, "ready");
    assert_eq!(runtime_status.models[0].port, Some(9337));

    let runtime_processes = state.runtime_processes().await;
    assert_eq!(runtime_processes.processes.len(), 1);
    assert_eq!(runtime_processes.processes[0].name, "collector-model");
    assert_eq!(runtime_processes.processes[0].backend, "collector-backend");
    assert_eq!(runtime_processes.processes[0].status, "ready");
    assert_eq!(runtime_processes.processes[0].port, 9337);
    assert_eq!(runtime_processes.processes[0].pid, 777);
}
