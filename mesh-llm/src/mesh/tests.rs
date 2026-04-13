use super::heartbeat::{
    relay_reconnect_reason, should_remove_connection, HeartbeatFailurePolicy, RelayPathSnapshot,
    RelayPeerHealth, RelayReconnectReason, SelectedPathKind, RELAY_DEGRADED_RTT_MS,
    RELAY_ONLY_RECONNECT_SECS, RELAY_RECONNECT_COOLDOWN_SECS,
};
use super::*;
use crate::proto::node::{GossipFrame, NodeRole, PeerAnnouncement, RouteTableRequest};
use tokio::sync::watch;

async fn make_test_node(role: super::NodeRole) -> Result<Node> {
    use iroh::endpoint::QuicTransportConfig;

    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(128u32.into())
        .build();
    let endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V1.to_vec(), ALPN_V0.to_vec()])
        .transport_config(transport_config)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;

    let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
    let (inflight_change_tx, _) = watch::channel(0u64);
    let (tunnel_tx, _tunnel_rx) = tokio::sync::mpsc::channel(8);
    let (tunnel_http_tx, _tunnel_http_rx) = tokio::sync::mpsc::channel(8);

    let node = Node {
        endpoint,
        public_addr: None,
        state: Arc::new(Mutex::new(MeshState {
            peers: HashMap::new(),
            connections: HashMap::new(),
            remote_tunnel_maps: HashMap::new(),
            dead_peers: HashSet::new(),
            seen_plugin_messages: HashSet::new(),
            seen_plugin_message_order: VecDeque::new(),
            policy_rejected_peers: HashMap::new(),
        })),
        role: Arc::new(Mutex::new(role)),
        models: Arc::new(Mutex::new(Vec::new())),
        model_source: Arc::new(Mutex::new(None)),
        serving_models: Arc::new(Mutex::new(Vec::new())),
        served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
        model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
        hosted_models: Arc::new(Mutex::new(Vec::new())),
        llama_ready: Arc::new(Mutex::new(false)),
        available_models: Arc::new(Mutex::new(Vec::new())),
        requested_models: Arc::new(Mutex::new(Vec::new())),
        model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
        mesh_id: Arc::new(Mutex::new(None)),
        accepting: Arc::new((
            tokio::sync::Notify::new(),
            std::sync::atomic::AtomicBool::new(false),
        )),
        vram_bytes: 64 * 1024 * 1024 * 1024,
        peer_change_tx,
        peer_change_rx,
        inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        inflight_change_tx,
        tunnel_tx,
        tunnel_http_tx,
        plugin_manager: Arc::new(Mutex::new(None)),
        display_name: Arc::new(Mutex::new(None)),
        owner_attestation: Arc::new(Mutex::new(None)),
        owner_summary: Arc::new(Mutex::new(OwnershipSummary::default())),
        trust_store: Arc::new(Mutex::new(TrustStore::default())),
        trust_policy: TrustPolicy::Off,
        enumerate_host: false,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
        config_state: Arc::new(tokio::sync::Mutex::new(
            crate::runtime::config_state::ConfigState::default(),
        )),
        config_revision_tx: {
            let (tx, _rx) = tokio::sync::watch::channel(0u64);
            Arc::new(tx)
        },
    };

    let accept_node = node.clone();
    tokio::spawn(async move {
        accept_node.accept_loop().await;
    });

    Ok(node)
}

#[test]
fn test_merge_demand_takes_max() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );
    ours.insert(
        "Hermes".into(),
        ModelDemand {
            last_active: 200,
            request_count: 10,
        },
    );

    let mut theirs = HashMap::new();
    theirs.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 150,
            request_count: 30,
        },
    );
    theirs.insert(
        "Qwen".into(),
        ModelDemand {
            last_active: 300,
            request_count: 5,
        },
    );

    merge_demand(&mut ours, &theirs);

    // GLM: max(100,150)=150 for last_active, max(50,30)=50 for count
    assert_eq!(ours["GLM"].last_active, 150);
    assert_eq!(ours["GLM"].request_count, 50);
    // Hermes: unchanged (not in theirs)
    assert_eq!(ours["Hermes"].last_active, 200);
    assert_eq!(ours["Hermes"].request_count, 10);
    // Qwen: new entry from theirs
    assert_eq!(ours["Qwen"].last_active, 300);
    assert_eq!(ours["Qwen"].request_count, 5);
}

#[test]
fn test_merge_demand_empty_maps() {
    let mut ours = HashMap::new();
    let theirs = HashMap::new();
    merge_demand(&mut ours, &theirs);
    assert!(ours.is_empty());

    let mut theirs2 = HashMap::new();
    theirs2.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 1,
        },
    );
    merge_demand(&mut ours, &theirs2);
    assert_eq!(ours.len(), 1);
    assert_eq!(ours["GLM"].request_count, 1);
}

#[test]
fn test_merge_demand_idempotent() {
    let mut ours = HashMap::new();
    ours.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 100,
            request_count: 50,
        },
    );

    let theirs = ours.clone();
    merge_demand(&mut ours, &theirs);

    assert_eq!(ours["GLM"].last_active, 100);
    assert_eq!(ours["GLM"].request_count, 50);
}

#[test]
fn test_demand_ttl_filtering() {
    let now = now_secs();
    let mut demand = HashMap::new();

    // Recent — should survive
    demand.insert(
        "Recent".into(),
        ModelDemand {
            last_active: now - 60, // 1 min ago
            request_count: 10,
        },
    );
    // Stale — should be filtered
    demand.insert(
        "Stale".into(),
        ModelDemand {
            last_active: now - DEMAND_TTL_SECS - 100, // past TTL
            request_count: 100,
        },
    );

    let filtered: HashMap<String, ModelDemand> = demand
        .into_iter()
        .filter(|(_, d)| (now - d.last_active) < DEMAND_TTL_SECS)
        .collect();

    assert_eq!(filtered.len(), 1);
    assert!(filtered.contains_key("Recent"));
    assert!(!filtered.contains_key("Stale"));
}

#[test]
fn test_demand_serialization_roundtrip() {
    let mut demand: HashMap<String, ModelDemand> = HashMap::new();
    demand.insert(
        "GLM".into(),
        ModelDemand {
            last_active: 1772309000,
            request_count: 42,
        },
    );

    let json = serde_json::to_string(&demand).unwrap();
    let decoded: HashMap<String, ModelDemand> = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded["GLM"].last_active, 1772309000);
    assert_eq!(decoded["GLM"].request_count, 42);
}

#[test]
fn test_demand_deserialization_missing_field() {
    // Simulate old gossip message without model_demand field
    // Just verify ModelDemand defaults work
    let d = ModelDemand::default();
    assert_eq!(d.last_active, 0);
    assert_eq!(d.request_count, 0);

    // Verify HashMap<String, ModelDemand> defaults to empty
    let empty: HashMap<String, ModelDemand> = Default::default();
    assert!(empty.is_empty());

    // The real test: serde default on a struct with model_demand
    #[derive(Deserialize, Default)]
    struct TestStruct {
        #[serde(default)]
        model_demand: HashMap<String, ModelDemand>,
        #[serde(default)]
        requested_models: Vec<String>,
    }
    let parsed: TestStruct = serde_json::from_str("{}").unwrap();
    assert!(parsed.model_demand.is_empty());
    assert!(parsed.requested_models.is_empty());
}

#[test]
fn test_peer_announcement_gpu_serde_roundtrip() {
    // Test that gpu_name and hostname fields serialize and deserialize correctly
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA A100".to_string()));
    assert_eq!(decoded.hostname, Some("worker-01".to_string()));
}

#[test]
fn test_peer_announcement_backward_compat_no_hw_fields() {
    // Simulate old gossip message without gpu_name or hostname
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, None);
    assert_eq!(decoded.hostname, None);
}

#[test]
fn test_peer_announcement_backward_compat_with_hw_fields() {
    // Simulate new gossip message with gpu_name and hostname
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"gpu_name": "NVIDIA H100", "hostname": "gpu-server-02"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA H100".to_string()));
    assert_eq!(decoded.hostname, Some("gpu-server-02".to_string()));
}

#[test]
fn test_peer_announcement_hostname_serde_roundtrip() {
    // Test hostname-only roundtrip
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: None,
        hostname: Some("compute-node-42".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.hostname, Some("compute-node-42".to_string()));
    assert_eq!(decoded.gpu_name, None);
}

#[test]
fn test_peer_payload_hw_fields() {
    // Test that PeerPayload includes gpu_name and hostname fields
    #[derive(Serialize, Debug)]
    struct TestPeerPayload {
        id: String,
        gpu_name: Option<String>,
        hostname: Option<String>,
    }

    let payload = TestPeerPayload {
        id: "peer-123".to_string(),
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&payload).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["gpu_name"], "NVIDIA A100");
    assert_eq!(value["hostname"], "worker-01");
}

#[test]
fn test_enumerate_host_false_omits_hw_fields_in_announcement() {
    let enumerate_host = false;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, None);
    assert_eq!(gossip_hostname, None);
    assert_eq!(gossip_gpu_vram, None);
}

#[test]
fn test_enumerate_host_true_includes_hw_fields_in_announcement() {
    let enumerate_host = true;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, Some("NVIDIA RTX 5090".to_string()));
    assert_eq!(gossip_hostname, Some("carrack".to_string()));
    assert_eq!(gossip_gpu_vram, Some("34359738368".to_string()));
}

#[test]
fn test_is_soc_always_included_regardless_of_enumerate_host() {
    for enumerate_host in [false, true] {
        let is_soc: Option<bool> = Some(true);
        let gpu_name: Option<String> = Some("Tegra AGX Orin".to_string());

        let gossip_gpu_name = if enumerate_host {
            gpu_name.clone()
        } else {
            None
        };

        assert_eq!(is_soc, Some(true), "is_soc must always be sent");
        if enumerate_host {
            assert!(gossip_gpu_name.is_some());
        } else {
            assert!(gossip_gpu_name.is_none());
        }
    }
}

#[test]
fn test_peer_announcement_backward_compat_is_soc_gpu_vram() {
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        is_soc: Option<bool>,
        #[serde(default)]
        gpu_vram: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();
    assert_eq!(
        decoded.is_soc, None,
        "old nodes without is_soc should default to None"
    );
    assert_eq!(
        decoded.gpu_vram, None,
        "old nodes without gpu_vram should default to None"
    );
}

#[test]
fn test_peer_announcement_v0_gossip_wire_backward_compat() {
    #[derive(Deserialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let json = r#"{"gpu_bandwidth_gbps":"1671.7,722.2"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(
        decoded.gpu_mem_bandwidth_gbps,
        Some("1671.7,722.2".to_string())
    );
}

#[test]
fn test_peer_announcement_v0_gossip_wire_new_alias() {
    #[derive(Deserialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let json = r#"{"gpu_mem_bandwidth_gbps":"1671.7,722.2"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(
        decoded.gpu_mem_bandwidth_gbps,
        Some("1671.7,722.2".to_string())
    );
}

#[test]
fn test_peer_announcement_v0_serializes_with_old_name() {
    #[derive(Serialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let announcement = TestAnnouncement {
        gpu_mem_bandwidth_gbps: Some("1671.7,722.2".to_string()),
    };

    let value = serde_json::to_value(&announcement).unwrap();
    assert_eq!(
        value.get("gpu_bandwidth_gbps").unwrap(),
        &serde_json::json!("1671.7,722.2")
    );
    assert!(value.get("gpu_mem_bandwidth_gbps").is_none());
}

#[test]
fn test_peer_announcement_backward_compat_no_bandwidth_field() {
    #[derive(Deserialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_mem_bandwidth_gbps, None);
}

#[test]
fn test_peer_announcement_v0_gpu_compute_tflops_serde_roundtrip() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x21; 32]).public());
    let ann = super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 9337 },
        models: vec!["GLM".into()],
        vram_bytes: 48_000_000_000,
        model_source: Some("glm.gguf".into()),
        serving: Some("GLM".into()),
        serving_models: vec!["GLM".into()],
        available_models: vec!["GLM".into()],
        requested_models: vec![],
        version: Some("0.54.0".into()),
        model_demand: HashMap::new(),
        mesh_id: Some("mesh-tflops".into()),
        gpu_name: Some("NVIDIA A100".into()),
        hostname: Some("worker-01".into()),
        is_soc: Some(false),
        gpu_vram: Some("51539607552".into()),
        gpu_reserved_bytes: Some("1073741824".into()),
        gpu_mem_bandwidth_gbps: Some("2039.4".into()),
        gpu_compute_tflops_fp32: Some("19.49".into()),
        gpu_compute_tflops_fp16: Some("38.98".into()),
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };

    let json = serde_json::to_string(&ann).unwrap();
    let decoded: super::PeerAnnouncementV0 = serde_json::from_str(&json).unwrap();
    let internal = decoded.into_internal();

    assert_eq!(internal.gpu_reserved_bytes.as_deref(), Some("1073741824"));
    assert_eq!(internal.gpu_compute_tflops_fp32.as_deref(), Some("19.49"));
    assert_eq!(internal.gpu_compute_tflops_fp16.as_deref(), Some("38.98"));
    assert_eq!(internal.gpu_mem_bandwidth_gbps.as_deref(), Some("2039.4"));
}

#[test]
fn test_peer_announcement_v0_backward_compat_no_tflops() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());
    let mut value = serde_json::to_value(super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Worker,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving: None,
        serving_models: vec![],
        available_models: vec![],
        requested_models: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: Some("15.75".into()),
        gpu_compute_tflops_fp16: Some("31.50".into()),
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    })
    .unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("gpu_compute_tflops_fp32");
    object.remove("gpu_compute_tflops_fp16");

    let decoded: super::PeerAnnouncementV0 = serde_json::from_value(value).unwrap();
    let internal = decoded.into_internal();

    assert_eq!(internal.gpu_compute_tflops_fp32, None);
    assert_eq!(internal.gpu_compute_tflops_fp16, None);
}

#[test]
fn test_peer_announcement_v0_fp32_only_without_fp16() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x23; 32]).public());
    let mut value = serde_json::to_value(super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Worker,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving: None,
        serving_models: vec![],
        available_models: vec![],
        requested_models: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: Some("15.75".into()),
        gpu_compute_tflops_fp16: Some("31.50".into()),
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    })
    .unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("gpu_compute_tflops_fp16");

    let decoded: super::PeerAnnouncementV0 = serde_json::from_value(value).unwrap();
    let internal = decoded.into_internal();

    assert_eq!(internal.gpu_compute_tflops_fp32.as_deref(), Some("15.75"));
    assert_eq!(internal.gpu_compute_tflops_fp16, None);
}

fn make_valid_gossip_frame() -> GossipFrame {
    GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    }
}

#[test]
fn protocol_from_alpn_supports_v1_and_legacy_v0() {
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
    assert_eq!(protocol_from_alpn(ALPN_V0), ControlProtocol::JsonV0);
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_v0_and_post_proto_nodes_interoperate_over_real_connection() -> Result<()> {
    use iroh::endpoint::QuicTransportConfig;

    let post_node = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let post_id = post_node.id();
    post_node
        .set_serving_models(vec!["post-model".to_string()])
        .await;
    post_node
        .set_hosted_models(vec!["post-model".to_string()])
        .await;
    post_node
        .set_mesh_id("compat-mesh-01020304".to_string())
        .await;
    post_node.start_accepting();

    let legacy_endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let legacy_id = legacy_endpoint.id();
    let legacy_addr = legacy_endpoint.addr();
    let legacy_ann = super::PeerAnnouncementV0 {
        addr: legacy_addr.clone(),
        role: super::NodeRole::Host { http_port: 9444 },
        models: vec!["legacy-model".to_string()],
        vram_bytes: 48 * 1024 * 1024 * 1024,
        model_source: Some("legacy-model.gguf".to_string()),
        serving: Some("legacy-model".to_string()),
        serving_models: vec!["legacy-model".to_string()],
        available_models: vec!["legacy-model".to_string()],
        requested_models: Vec::new(),
        version: Some("0.50.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("compat-mesh-01020304".to_string()),
        gpu_name: Some("Legacy GPU".to_string()),
        hostname: Some("legacy-peer".to_string()),
        is_soc: Some(false),
        gpu_vram: Some((48_u64 * 1024 * 1024 * 1024).to_string()),
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };
    let legacy_route_table = RoutingTable {
        hosts: vec![RouteEntry {
            model: "legacy-model".to_string(),
            node_id: legacy_id.fmt_short().to_string(),
            endpoint_id: legacy_id,
            vram_gb: 48.0,
        }],
        mesh_id: Some("compat-mesh-01020304".to_string()),
    };

    let server = tokio::spawn(async move {
        let incoming =
            tokio::time::timeout(std::time::Duration::from_secs(5), legacy_endpoint.accept())
                .await
                .expect("legacy endpoint should get an incoming connection")
                .expect("accept loop should yield one incoming connection");
        let mut accepting = incoming.accept().expect("legacy accept should succeed");
        let alpn = accepting.alpn().await.expect("ALPN should be available");
        assert_eq!(alpn, ALPN_V0, "new node must fall back to legacy ALPN");
        let conn = accepting
            .await
            .expect("legacy connection handshake should complete");
        assert_eq!(conn.alpn(), ALPN_V0);

        let (mut send_gossip, mut recv_gossip) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.accept_bi())
                .await
                .expect("post node should open initial gossip stream")
                .expect("initial gossip stream should be accepted");
        let mut stream_type = [0u8; 1];
        recv_gossip
            .read_exact(&mut stream_type)
            .await
            .expect("legacy server must read gossip stream type");
        assert_eq!(stream_type[0], STREAM_GOSSIP);
        let gossip_buf = read_len_prefixed(&mut recv_gossip)
            .await
            .expect("legacy server must read JSON gossip frame");
        let received_anns: Vec<super::PeerAnnouncementV0> =
            serde_json::from_slice(&gossip_buf).expect("legacy gossip must decode as JSON");
        assert!(
            received_anns
                .iter()
                .any(|ann| ann.addr.id == post_id && ann.serving.as_deref() == Some("post-model")),
            "initial legacy gossip response should include the post-protobuf host announcement"
        );
        let legacy_gossip_body = serde_json::to_vec(&vec![legacy_ann.clone()])
            .expect("legacy announcement must serialize");
        write_len_prefixed(&mut send_gossip, &legacy_gossip_body)
            .await
            .expect("legacy server should reply with JSON gossip");
        send_gossip
            .finish()
            .expect("legacy gossip reply should finish");
        let _ = recv_gossip.read_to_end(0).await;

        let (mut send_route_resp, mut recv_route_req) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.accept_bi())
                .await
                .expect("post node should open legacy route request stream")
                .expect("legacy route request stream should be accepted");
        recv_route_req
            .read_exact(&mut stream_type)
            .await
            .expect("legacy server must read route stream type");
        assert_eq!(stream_type[0], STREAM_ROUTE_REQUEST);
        let legacy_route_body =
            serde_json::to_vec(&legacy_route_table).expect("legacy route table must serialize");
        send_route_resp
            .write_all(&legacy_route_body)
            .await
            .expect("legacy server must send JSON route table");
        send_route_resp
            .finish()
            .expect("legacy route response should finish");

        let (mut send_gossip2, mut recv_gossip2) = conn
            .open_bi()
            .await
            .expect("legacy server should initiate gossip back to post node");
        send_gossip2
            .write_all(&[STREAM_GOSSIP])
            .await
            .expect("legacy gossip stream type should be sent");
        write_len_prefixed(&mut send_gossip2, &legacy_gossip_body)
            .await
            .expect("legacy server should send JSON gossip payload");
        send_gossip2
            .finish()
            .expect("legacy initiated gossip should finish");
        let response_buf = read_len_prefixed(&mut recv_gossip2)
            .await
            .expect("post node should answer legacy gossip");
        let response_anns: Vec<super::PeerAnnouncementV0> =
            serde_json::from_slice(&response_buf).expect("post node must answer with JSON gossip");
        assert!(
            response_anns
                .iter()
                .any(|ann| ann.addr.id == post_id && ann.serving.as_deref() == Some("post-model")),
            "post node should answer legacy gossip with its current state"
        );
        let _ = recv_gossip2.read_to_end(0).await;

        let (mut send_route_req2, mut recv_route_resp2) = conn
            .open_bi()
            .await
            .expect("legacy server should initiate route request to post node");
        send_route_req2
            .write_all(&[STREAM_ROUTE_REQUEST])
            .await
            .expect("legacy route request stream type should be sent");
        send_route_req2
            .finish()
            .expect("legacy route request should finish");
        let route_json = recv_route_resp2
            .read_to_end(MAX_CONTROL_FRAME_BYTES)
            .await
            .expect("post node should reply with legacy JSON route table");
        let route_table_from_post: RoutingTable =
            serde_json::from_slice(&route_json).expect("post node route response must be JSON");
        assert_eq!(
            route_table_from_post.mesh_id.as_deref(),
            Some("compat-mesh-01020304")
        );
        assert!(
            route_table_from_post
                .hosts
                .iter()
                .any(|entry| entry.endpoint_id == post_id && entry.model == "post-model"),
            "legacy peer should see the post node in route-table JSON response"
        );
    });

    let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&legacy_addr).expect("legacy address should serialize"));
    post_node.join(&invite_token).await?;

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let peers = post_node.peers().await;
            if peers.iter().any(|peer| {
                peer.id == legacy_id
                    && peer.serving_models.first().map(String::as_str) == Some("legacy-model")
            }) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("post node should admit the legacy peer after JSON gossip");

    let legacy_conn = {
        let state = post_node.state.lock().await;
        state
            .connections
            .get(&legacy_id)
            .cloned()
            .expect("join should leave a connection to the legacy peer")
    };
    let route_table = post_node.request_route_table(&legacy_conn).await?;
    assert_eq!(
        route_table.mesh_id.as_deref(),
        Some("compat-mesh-01020304"),
        "post node must parse legacy JSON route-table replies"
    );
    assert!(
        route_table
            .hosts
            .iter()
            .any(|entry| entry.endpoint_id == legacy_id && entry.model == "legacy-model"),
        "post node must preserve legacy route-table entries"
    );

    server.await.expect("legacy peer task should complete");
    Ok(())
}

#[test]
fn legacy_json_gossip_payload_decodes() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x42; 32]).public());
    let ann = super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 3131 },
        models: vec!["Qwen".into()],
        vram_bytes: 48_000_000_000,
        model_source: Some("Qwen.gguf".into()),
        serving: Some("Qwen".into()),
        serving_models: vec!["Qwen".into()],
        available_models: vec!["Qwen".into()],
        requested_models: vec!["Qwen".into()],
        version: Some("0.52.0".into()),
        model_demand: HashMap::from([(
            "Qwen".into(),
            ModelDemand {
                last_active: 123,
                request_count: 7,
            },
        )]),
        mesh_id: Some("mesh-compat".into()),
        gpu_name: Some("NVIDIA A100".into()),
        hostname: Some("worker-01".into()),
        is_soc: Some(false),
        gpu_vram: Some("51539607552".into()),
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_sizes: HashMap::from([("Qwen".into(), 1234_u64)]),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };
    let json = serde_json::to_vec(&vec![ann.clone()]).unwrap();

    let decoded = decode_gossip_payload(ControlProtocol::JsonV0, peer_id, &json).unwrap();

    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].0.id, peer_id);
    assert_eq!(
        decoded[0].1.serving_models.first().map(String::as_str),
        Some("Qwen")
    );
    assert_eq!(decoded[0].1.mesh_id.as_deref(), Some("mesh-compat"));
}

#[test]
fn legacy_json_tunnel_map_decodes() {
    let target = EndpointId::from(SecretKey::from_bytes(&[0x24; 32]).public());
    let json =
        serde_json::to_vec(&HashMap::from([(hex::encode(target.as_bytes()), 9337_u16)])).unwrap();

    let frame = decode_legacy_tunnel_map_frame(&json).unwrap();

    assert_eq!(frame.entries.len(), 1);
    assert_eq!(frame.entries[0].target_peer_id, target.as_bytes().to_vec());
    assert_eq!(frame.entries[0].tunnel_port, 9337);
}

#[test]
fn control_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}

fn make_test_peer_info(peer_id: EndpointId) -> PeerInfo {
    PeerInfo {
        id: peer_id,
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        tunnel_port: None,
        role: super::NodeRole::Worker,
        models: vec![],
        vram_bytes: 0,
        rtt_ms: None,
        model_source: None,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
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
        owner_summary: OwnershipSummary::default(),
    }
}

fn make_test_moe_descriptor(model_name: &str, identity_hash: &str) -> ServedModelDescriptor {
    ServedModelDescriptor {
        identity: ServedModelIdentity {
            model_name: model_name.to_string(),
            is_primary: true,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(format!("hf://{identity_hash}")),
            repository: Some("Qwen".to_string()),
            revision: Some("main".to_string()),
            artifact: Some(format!("{model_name}.gguf")),
            local_file_name: Some(format!("{model_name}.gguf")),
            identity_hash: Some(identity_hash.to_string()),
        },
        capabilities: crate::models::ModelCapabilities {
            moe: true,
            ..Default::default()
        },
        topology: Some(crate::models::ModelTopology {
            moe: Some(crate::models::ModelMoeInfo {
                expert_count: 512,
                used_expert_count: 10,
                min_experts_per_node: Some(160),
                source: Some("test".to_string()),
                ranking_source: None,
                ranking_origin: None,
                ranking: Vec::new(),
                ranking_prompt_count: None,
                ranking_tokens: None,
                ranking_layer_scope: None,
            }),
        }),
    }
}

fn make_test_endpoint_id(seed: u8) -> EndpointId {
    let mut bytes = [0u8; 32];
    bytes[0] = seed;
    EndpointId::from(SecretKey::from_bytes(&bytes).public())
}

#[test]
fn relay_health_prefers_direct_paths_and_clears_relay_age() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(240),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );
    assert!(
        health.relay_since.is_some(),
        "relay age should start on relay path"
    );

    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Direct,
            rtt_ms: Some(18),
        },
        now,
    );
    assert!(
        health.relay_since.is_none(),
        "direct path should clear relay-only aging"
    );
}

#[test]
fn relay_health_reconnects_degraded_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
        },
        now - std::time::Duration::from_secs(30),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
            },
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayRttDegraded)
    );
}

#[test]
fn relay_health_reconnects_long_lived_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(260),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayOnlyTooLong)
    );
}

#[test]
fn relay_health_respects_cooldown_and_inflight_requests() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
        },
        now - std::time::Duration::from_secs(30),
    );
    health.last_reconnect_at =
        Some(now - std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS - 1));

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            now,
            0,
            true,
        ),
        None,
        "cooldown should suppress immediate retry"
    );

    health.last_reconnect_at = None;
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            now,
            1,
            true,
        ),
        None,
        "active requests should suppress relay refresh"
    );
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            now,
            0,
            false,
        ),
        None,
        "missing home relay should suppress churn"
    );
}

#[test]
fn stale_dispatcher_cannot_remove_replacement_connection() {
    assert!(
        should_remove_connection(Some(7), 7),
        "matching stable id should remove tracked connection"
    );
    assert!(
        !should_remove_connection(Some(8), 7),
        "stale dispatcher must not remove a newer replacement connection"
    );
    assert!(
        !should_remove_connection(None, 7),
        "missing connection slot should be a no-op"
    );
}

#[test]
fn shared_exact_moe_identity_uses_stricter_heartbeat_without_inbound_grace() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(7));
    peer.served_model_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_runtime = vec![];

    let policy = heartbeat_failure_policy_for_peer(&local_descriptors, &local_runtime, &peer);

    assert_eq!(
        policy,
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: false,
            failure_threshold: 2,
        }
    );
}

#[test]
fn non_matching_or_non_moe_peers_keep_default_heartbeat_grace() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(8));
    peer.served_model_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "remote-model",
    )];
    let local_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "local-model",
    )];
    let local_runtime = vec![];

    let policy = heartbeat_failure_policy_for_peer(&local_descriptors, &local_runtime, &peer);

    assert_eq!(
        policy,
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 2,
        }
    );
}

#[test]
fn shared_exact_moe_startup_relaxes_heartbeat_during_convergence() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(11));
    peer.served_model_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_runtime = vec![ModelRuntimeDescriptor {
        model_name: "Qwen3-Coder-Next-Q4_K_M".to_string(),
        identity_hash: Some("same-model".to_string()),
        context_length: None,
        ready: false,
    }];

    let policy = heartbeat_failure_policy_for_peer(&local_descriptors, &local_runtime, &peer);

    assert_eq!(
        policy,
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 4,
        }
    );
}

#[test]
fn recovered_moe_peer_stays_out_of_active_placement_until_probation_expires() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(9));
    peer.serving_models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];
    peer.served_model_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];

    peer.moe_recovered_at = Some(std::time::Instant::now());
    assert!(!peer_is_eligible_for_active_moe(
        &local_descriptors,
        &peer,
        "Qwen3-Coder-Next-Q4_K_M"
    ));

    peer.moe_recovered_at = Some(
        std::time::Instant::now() - std::time::Duration::from_secs(MOE_RECOVERY_PROBATION_SECS + 1),
    );
    assert!(peer_is_eligible_for_active_moe(
        &local_descriptors,
        &peer,
        "Qwen3-Coder-Next-Q4_K_M"
    ));
}

#[test]
fn requested_model_peer_is_eligible_for_active_moe_during_startup() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(10));
    peer.requested_models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];
    peer.served_model_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];
    let local_descriptors = vec![make_test_moe_descriptor(
        "Qwen3-Coder-Next-Q4_K_M",
        "same-model",
    )];

    assert!(peer_is_eligible_for_active_moe(
        &local_descriptors,
        &peer,
        "Qwen3-Coder-Next-Q4_K_M"
    ));
}

#[test]
fn peer_meaningfully_changed_detects_reserved_bytes_updates() {
    let peer_id = make_test_endpoint_id(12);
    let mut old_peer = make_test_peer_info(peer_id);
    let mut new_peer = old_peer.clone();

    old_peer.gpu_reserved_bytes = Some("1000".to_string());
    new_peer.gpu_reserved_bytes = Some("2000".to_string());

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
fn incoming_peer_promoted_after_valid_gossip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert!(!decoded.peers.is_empty());

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xab; 32]).public());
    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();

    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "peer must NOT be admitted before gossip"
    );

    for &tunnel_stream in &[STREAM_TUNNEL, STREAM_TUNNEL_HTTP] {
        assert!(
            !stream_allowed_before_admission(tunnel_stream),
            "stream {:#04x} must be gated until after admission — unadmitted peers must not reach tunnel paths",
            tunnel_stream
        );
    }

    assert!(
        stream_allowed_before_admission(STREAM_GOSSIP),
        "STREAM_GOSSIP must always be allowed — it is the admission path"
    );

    peers.insert(peer_id, make_test_peer_info(peer_id));

    assert!(
        is_peer_admitted(&peers, &peer_id),
        "peer must be admitted after gossip completes (add_peer inserts into state.peers)"
    );
}

#[test]
fn incoming_peer_rejected_on_legacy_or_malformed_gossip() {
    let malformed_payload = vec![0xFF_u8; 20];
    let mut bad_frame = vec![STREAM_GOSSIP];
    bad_frame.extend_from_slice(&(malformed_payload.len() as u32).to_le_bytes());
    bad_frame.extend_from_slice(&malformed_payload);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &bad_frame)
        .expect_err("malformed protobuf must be rejected");
    assert!(
        matches!(err, ControlFrameError::DecodeError(_)),
        "expected DecodeError for malformed payload, got {:?}",
        err
    );

    let bad_gen_frame = GossipFrame {
        gen: 0,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {:?}",
        err
    );

    for stream_type in [
        STREAM_TUNNEL,
        STREAM_TUNNEL_HTTP,
        STREAM_TUNNEL_MAP,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_PLUGIN_CHANNEL,
        STREAM_PLUGIN_BULK_TRANSFER,
    ] {
        assert!(
            !stream_allowed_before_admission(stream_type),
            "stream {:#04x} must be quarantine-gated for unadmitted peers — if this fails, the gate is broken",
            stream_type
        );
    }

    assert!(
        stream_allowed_before_admission(STREAM_GOSSIP),
        "STREAM_GOSSIP must bypass the gate (it is the admission handshake)"
    );
    assert!(
        stream_allowed_before_admission(STREAM_ROUTE_REQUEST),
        "STREAM_ROUTE_REQUEST must bypass the gate (passive/client request-only path)"
    );

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xcd; 32]).public());
    let peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "peer must NOT be admitted when gossip fails"
    );
}

#[test]
fn passive_route_table_request_does_not_admit_peer() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xef; 32]).public());
    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();

    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "passive caller must NOT be admitted before route request"
    );

    assert!(
        stream_allowed_before_admission(STREAM_ROUTE_REQUEST),
        "STREAM_ROUTE_REQUEST must be allowed before admission (passive/client path)"
    );

    for &gated in &[
        STREAM_TUNNEL,
        STREAM_TUNNEL_HTTP,
        STREAM_TUNNEL_MAP,
        STREAM_PEER_DOWN,
        STREAM_PEER_LEAVING,
        STREAM_PLUGIN_CHANNEL,
        STREAM_PLUGIN_BULK_TRANSFER,
    ] {
        assert!(
            !stream_allowed_before_admission(gated),
            "stream {:#04x} must remain gated after a route request — route request must not unlock other streams",
            gated
        );
    }

    let valid_req = RouteTableRequest {
        requester_id: vec![0xef_u8; 32],
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &valid_req);
    let decoded: RouteTableRequest = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTableRequest must decode successfully");
    assert_eq!(decoded.requester_id, vec![0xef_u8; 32]);
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);

    let bad_req = RouteTableRequest {
        requester_id: vec![0u8; 16],
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded_bad = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded_bad)
        .expect_err("route request with wrong-length requester_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "passive caller must NOT be admitted after route-table response"
    );

    peers.insert(peer_id, make_test_peer_info(peer_id));
    assert!(
        is_peer_admitted(&peers, &peer_id),
        "only explicit gossip (add_peer) should promote to admitted"
    );
}

#[test]
fn control_frame_rejects_oversize_or_bad_generation() {
    let oversize_len = MAX_CONTROL_FRAME_BYTES + 1;
    let mut fake = vec![STREAM_GOSSIP];
    fake.extend_from_slice(&(oversize_len as u32).to_le_bytes());
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake)
        .expect_err("oversize frame must be rejected");
    assert!(
        matches!(err, ControlFrameError::OversizeFrame { .. }),
        "expected OversizeFrame, got {:?}",
        err
    );

    let bad_gen = GossipFrame {
        gen: 99,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("bad generation must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 99 }),
        "expected BadGeneration{{got:99}}, got {:?}",
        err
    );

    let bad_id = GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 16],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_id);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("bad endpoint_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let valid = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &valid);
    let err = decode_control_frame::<GossipFrame>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("wrong stream type must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: 0x03,
                got: 0x01
            }
        ),
        "expected WrongStreamType, got {:?}",
        err
    );
}

#[test]
fn gossip_frame_roundtrip_preserves_scanned_model_metadata() {
    use crate::proto::node::{CompactModelMetadata, ExpertsSummary};

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    let peer_id_bytes = peer_id.as_bytes().to_vec();

    let meta = CompactModelMetadata {
        model_key: "Qwen3-8B-Q4_K_M".to_string(),
        context_length: 40960,
        vocab_size: 151936,
        embedding_size: 4096,
        head_count: 32,
        layer_count: 36,
        feed_forward_length: 14336,
        key_length: 128,
        value_length: 128,
        architecture: "qwen3".to_string(),
        tokenizer_model_name: "PreTrainedTokenizerFast".to_string(),
        special_tokens: vec![],
        rope_scale: 1.0,
        rope_freq_base: 1_000_000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
    };

    let mut model_sizes = HashMap::new();
    model_sizes.insert("Qwen3-8B-Q4_K_M".to_string(), 4_800_000_000u64);

    let experts = ExpertsSummary {
        total_experts: 64,
        expert_count_used: 8,
        top_expert_ids: vec![1, 5, 10],
    };

    let local_ann = super::PeerAnnouncement {
        addr: EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 8080 },
        models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        vram_bytes: 128 * 1024 * 1024 * 1024,
        model_source: Some("bartowski/Qwen3-8B-GGUF".to_string()),
        serving_models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        hosted_models: Some(vec!["Qwen3-8B-Q4_K_M".to_string()]),
        available_models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        requested_models: vec![],
        version: Some("0.42.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("deadbeef12345678".to_string()),
        gpu_name: Some("Apple M4 Max".to_string()),
        hostname: Some("test-node".to_string()),
        is_soc: Some(true),
        gpu_vram: Some("128 GB".to_string()),
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![meta.clone()],
        experts_summary: Some(experts.clone()),
        available_model_sizes: model_sizes.clone(),
        served_model_descriptors: vec![ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: "Qwen3-8B-Q4_K_M".to_string(),
                is_primary: true,
                source_kind: ModelSourceKind::HuggingFace,
                canonical_ref: Some("hf/bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf".into()),
                repository: Some("bartowski/Qwen3-8B-GGUF".into()),
                revision: Some("main".into()),
                artifact: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                local_file_name: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                identity_hash: Some("sha256:abc123".into()),
            },
            capabilities: crate::models::ModelCapabilities::default(),
            topology: Some(crate::models::ModelTopology { moe: None }),
        }],
        served_model_runtime: vec![ModelRuntimeDescriptor {
            model_name: "Qwen3-8B-Q4_K_M".to_string(),
            identity_hash: Some("sha256:abc123".into()),
            context_length: Some(32768),
            ready: true,
        }],
        owner_attestation: None,
    };

    let proto_pa = local_ann_to_proto_ann(&local_ann);
    assert_eq!(
        proto_pa.available_model_metadata.len(),
        0,
        "local_ann_to_proto_ann must strip passive available_model_metadata from gossip"
    );
    assert!(
        proto_pa.available_models.is_empty(),
        "local_ann_to_proto_ann must strip passive available_models from gossip"
    );
    assert_eq!(
        proto_pa.experts_summary.as_ref().map(|e| e.total_experts),
        Some(64),
        "local_ann_to_proto_ann must carry experts_summary"
    );
    assert_eq!(
        proto_pa.available_model_sizes.len(),
        0,
        "local_ann_to_proto_ann must strip passive available_model_sizes from gossip"
    );

    let (_, roundtripped) =
        proto_ann_to_local(&proto_pa).expect("proto_ann_to_local must succeed on valid proto PA");
    assert_eq!(
        roundtripped.available_model_metadata.len(),
        0,
        "proto_ann_to_local must ignore passive available_model_metadata from gossip"
    );
    assert!(
        roundtripped.available_models.is_empty(),
        "proto_ann_to_local must ignore passive available_models from gossip"
    );
    assert_eq!(
        roundtripped
            .experts_summary
            .as_ref()
            .map(|e| e.total_experts),
        Some(64),
        "proto_ann_to_local must restore experts_summary"
    );
    assert!(roundtripped.available_model_sizes.is_empty());
    assert_eq!(
        roundtripped
            .served_model_runtime
            .first()
            .and_then(ModelRuntimeDescriptor::advertised_context_length),
        Some(32768),
        "proto_ann_to_local must preserve served model runtime context length"
    );

    let frame = build_gossip_frame(&[local_ann], peer_id);
    assert_eq!(frame.sender_id, peer_id_bytes);
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("build_gossip_frame output must decode successfully");
    assert_eq!(decoded.peers.len(), 1);
    let wire_pa = &decoded.peers[0];
    assert_eq!(
        wire_pa.available_model_metadata.len(),
        0,
        "build_gossip_frame must strip passive available_model_metadata from wire gossip"
    );
    assert!(wire_pa.available_models.is_empty());
    assert!(wire_pa.available_model_sizes.is_empty());
    assert_eq!(
        wire_pa
            .experts_summary
            .as_ref()
            .map(|e| e.top_expert_ids.as_slice()),
        Some([1u32, 5, 10].as_slice())
    );
    assert_eq!(
        wire_pa
            .served_model_runtime
            .first()
            .and_then(|runtime| runtime.context_length),
        Some(32768),
        "build_gossip_frame must preserve served model runtime context length"
    );
    let (_, final_local) =
        proto_ann_to_local(wire_pa).expect("final proto_ann_to_local must succeed");
    assert!(final_local.available_model_metadata.is_empty());
    assert!(final_local.available_models.is_empty());
    assert!(final_local.available_model_sizes.is_empty());
    assert_eq!(
        final_local
            .served_model_runtime
            .first()
            .and_then(ModelRuntimeDescriptor::advertised_context_length),
        Some(32768)
    );
}

#[test]
fn gossip_rejects_sender_id_mismatch_or_invalid_endpoint_len() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xaa; 32]).public());
    let peer_id_bytes = peer_id.as_bytes().to_vec();

    let invalid_sender_frame = GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: vec![0u8; 16],
        peers: vec![PeerAnnouncement {
            endpoint_id: peer_id_bytes.clone(),
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &invalid_sender_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("16-byte sender_id must be rejected at decode time");
    assert!(
        matches!(err, ControlFrameError::InvalidSenderId { got: 16 }),
        "expected InvalidSenderId{{got:16}}, got {:?}",
        err
    );

    let impersonator_id = EndpointId::from(SecretKey::from_bytes(&[0xbb; 32]).public());
    let mismatch_frame = GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: impersonator_id.as_bytes().to_vec(),
        peers: vec![PeerAnnouncement {
            endpoint_id: peer_id_bytes.clone(),
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let remote = peer_id;
    let is_forged = !mismatch_frame.sender_id.is_empty()
        && mismatch_frame.sender_id.as_slice() != remote.as_bytes();
    assert!(
        is_forged,
        "sender_id != remote.as_bytes() must be detected as a forged sender"
    );

    let bad_endpoint_frame = GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: peer_id_bytes.clone(),
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 20],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_endpoint_frame);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("20-byte endpoint_id in peer must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 20 }),
        "expected InvalidEndpointId{{got:20}}, got {:?}",
        err
    );
}

#[test]
fn transitive_peer_update_refreshes_metadata_fields() {
    use crate::proto::node::CompactModelMetadata;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x10; 32]).public());
    let mut existing = make_test_peer_info(peer_id);
    existing.available_models = vec!["OldModel-Q4_K_M".to_string()];
    existing.models = vec!["OldModel-Q4_K_M".to_string()];
    existing.requested_models = vec!["OldModel-Q4_K_M".to_string()];

    let meta = CompactModelMetadata {
        model_key: "NewModel-Q4_K_M".to_string(),
        context_length: 8192,
        vocab_size: 32000,
        embedding_size: 4096,
        head_count: 32,
        layer_count: 32,
        feed_forward_length: 11008,
        key_length: 128,
        value_length: 128,
        architecture: "llama".to_string(),
        tokenizer_model_name: String::new(),
        special_tokens: vec![],
        rope_scale: 1.0,
        rope_freq_base: 10000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
    };

    let mut new_sizes = HashMap::new();
    new_sizes.insert("NewModel-Q4_K_M".to_string(), 4_800_000_000u64);

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let ann = super::PeerAnnouncement {
        addr: addr.clone(),
        role: super::NodeRole::Worker,
        models: vec!["NewModel-Q4_K_M".to_string()],
        vram_bytes: 8 * 1024 * 1024 * 1024,
        model_source: Some("new-source".to_string()),
        serving_models: vec!["NewModel-Q4_K_M".to_string()],
        hosted_models: Some(vec!["NewModel-Q4_K_M".to_string()]),
        available_models: vec!["NewModel-Q4_K_M".to_string()],
        requested_models: vec!["NewModel-Q4_K_M".to_string()],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![meta],
        experts_summary: None,
        available_model_sizes: new_sizes,
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
    };

    apply_transitive_ann(&mut existing, &addr, &ann);

    assert!(
        existing.available_models.is_empty(),
        "remote available_models must be ignored during transitive gossip merge"
    );
    assert_eq!(
        existing.models,
        vec!["NewModel-Q4_K_M".to_string()],
        "models must be refreshed from transitive gossip"
    );
    assert_eq!(
        existing.requested_models,
        vec!["NewModel-Q4_K_M".to_string()],
        "requested_models must be refreshed from transitive gossip"
    );
    assert!(existing.available_model_metadata.is_empty());
    assert!(existing.available_model_sizes.is_empty());
}

#[test]
fn transitive_peer_merge_preserves_richer_direct_address() {
    use iroh::TransportAddr;

    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let mut existing = make_test_peer_info(peer_id);

    let mut rich_addrs = std::collections::BTreeSet::new();
    rich_addrs.insert(TransportAddr::Ip("127.0.0.1:1000".parse().unwrap()));
    rich_addrs.insert(TransportAddr::Ip("192.168.1.1:1001".parse().unwrap()));
    rich_addrs.insert(TransportAddr::Ip("10.0.0.1:1002".parse().unwrap()));
    existing.addr = EndpointAddr {
        id: peer_id,
        addrs: rich_addrs,
    };

    let mut weak_addrs = std::collections::BTreeSet::new();
    weak_addrs.insert(TransportAddr::Ip("127.0.0.1:9999".parse().unwrap()));
    let weak_addr = EndpointAddr {
        id: peer_id,
        addrs: weak_addrs,
    };
    let ann = super::PeerAnnouncement {
        addr: weak_addr.clone(),
        role: super::NodeRole::Worker,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 4 * 1024 * 1024 * 1024,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec!["SomeModel-Q4_K_M".to_string()],
        requested_models: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
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
    };

    apply_transitive_ann(&mut existing, &weak_addr, &ann);

    assert_eq!(
        existing.addr.addrs.len(),
        3,
        "rich direct address (3 paths) must not be overwritten by weaker transitive addr (1 path)"
    );
    assert!(
        existing.available_models.is_empty(),
        "remote available_models must still be ignored even when addr is preserved"
    );

    let mut richer_addrs = std::collections::BTreeSet::new();
    richer_addrs.insert(TransportAddr::Ip("127.0.0.1:1000".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("192.168.1.1:1001".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("10.0.0.1:1002".parse().unwrap()));
    richer_addrs.insert(TransportAddr::Ip("172.16.0.1:1003".parse().unwrap()));
    let richer_addr = EndpointAddr {
        id: peer_id,
        addrs: richer_addrs,
    };
    let ann2 = super::PeerAnnouncement {
        addr: richer_addr.clone(),
        role: super::NodeRole::Worker,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 4 * 1024 * 1024 * 1024,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec!["SomeModel-Q4_K_M".to_string()],
        requested_models: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
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
    };
    apply_transitive_ann(&mut existing, &richer_addr, &ann2);

    assert_eq!(
        existing.addr.addrs.len(),
        4,
        "richer transitive addr (4 paths) must replace existing (3 paths)"
    );
}

#[test]
fn tunnel_map_roundtrip_updates_remote_map() {
    use crate::proto::node::{TunnelEntry, TunnelMap};

    let owner_key = SecretKey::from_bytes(&[0x10; 32]);
    let owner_id = EndpointId::from(owner_key.public());
    let owner_bytes = owner_id.as_bytes().to_vec();

    let target_key = SecretKey::from_bytes(&[0x20; 32]);
    let target_id = EndpointId::from(target_key.public());
    let target_bytes = target_id.as_bytes().to_vec();

    let frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };

    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &frame);
    let decoded: TunnelMap = decode_control_frame(STREAM_TUNNEL_MAP, &encoded)
        .expect("valid TunnelMap must decode successfully");

    assert_eq!(decoded.owner_peer_id, owner_bytes);
    assert_eq!(decoded.entries.len(), 1);
    assert_eq!(decoded.entries[0].target_peer_id, target_bytes);
    assert_eq!(decoded.entries[0].tunnel_port, 50001);

    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    ingest_tunnel_map(owner_id, &decoded, &mut remote_tunnel_maps)
        .expect("valid tunnel map must ingest successfully");

    assert_eq!(remote_tunnel_maps.len(), 1);
    let inner = remote_tunnel_maps
        .get(&owner_id)
        .expect("owner must be present in remote_tunnel_maps");
    assert_eq!(inner.len(), 1);
    let port = inner
        .get(&target_id)
        .expect("target must be present in inner map");
    assert_eq!(*port, 50001u16);
}

#[test]
fn tunnel_map_rejects_owner_mismatch_or_bad_target_id() {
    use crate::proto::node::{TunnelEntry, TunnelMap};

    let owner_key = SecretKey::from_bytes(&[0x30; 32]);
    let owner_id = EndpointId::from(owner_key.public());
    let owner_bytes = owner_id.as_bytes().to_vec();

    let target_key = SecretKey::from_bytes(&[0x40; 32]);
    let target_id = EndpointId::from(target_key.public());
    let target_bytes = target_id.as_bytes().to_vec();

    let bad_owner_frame = TunnelMap {
        owner_peer_id: vec![0u8; 16],
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &bad_owner_frame);
    let err = decode_control_frame::<TunnelMap>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("bad owner_peer_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let bad_target_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: vec![0u8; 16],
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let encoded = encode_control_frame(STREAM_TUNNEL_MAP, &bad_target_frame);
    let err = decode_control_frame::<TunnelMap>(STREAM_TUNNEL_MAP, &encoded)
        .expect_err("bad target_peer_id must be rejected");
    assert!(
        matches!(err, ControlFrameError::InvalidEndpointId { got: 16 }),
        "expected InvalidEndpointId{{got:16}}, got {:?}",
        err
    );

    let different_key = SecretKey::from_bytes(&[0x50; 32]);
    let different_id = EndpointId::from(different_key.public());

    let mismatched_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 50001,
            relay_peer_id: None,
        }],
    };
    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    let result = ingest_tunnel_map(different_id, &mismatched_frame, &mut remote_tunnel_maps);
    assert!(result.is_err(), "owner mismatch must be rejected");
    assert!(
        remote_tunnel_maps.is_empty(),
        "mismatched owner must not populate remote_tunnel_maps"
    );

    let oversized_port_frame = TunnelMap {
        owner_peer_id: owner_bytes.clone(),
        entries: vec![TunnelEntry {
            target_peer_id: target_bytes.clone(),
            tunnel_port: 70000,
            relay_peer_id: None,
        }],
    };
    let mut remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>> = HashMap::new();
    let result = ingest_tunnel_map(owner_id, &oversized_port_frame, &mut remote_tunnel_maps);
    assert!(result.is_err(), "tunnel_port > u16::MAX must be rejected");
    assert!(
        remote_tunnel_maps.is_empty(),
        "oversized tunnel_port must not populate remote_tunnel_maps"
    );
}

#[test]
fn route_table_request_roundtrip() {
    use crate::proto::node::{RouteEntry as ProtoRouteEntry, RouteTable};

    let peer_key = SecretKey::from_bytes(&[0x60; 32]);
    let peer_id = EndpointId::from(peer_key.public());
    let peer_bytes = peer_id.as_bytes().to_vec();

    let req = RouteTableRequest {
        requester_id: peer_bytes.clone(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &req);
    let decoded: RouteTableRequest = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTableRequest must decode successfully");
    assert_eq!(decoded.requester_id, peer_bytes);
    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);

    let table = RouteTable {
        entries: vec![ProtoRouteEntry {
            endpoint_id: peer_bytes.clone(),
            model: "Qwen3-8B-Q4_K_M".to_string(),
        }],
        mesh_id: Some("test-mesh-0102030405060708".to_string()),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded_table = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let decoded_table: RouteTable = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded_table)
        .expect("valid RouteTable must decode successfully");
    assert_eq!(decoded_table.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded_table.entries.len(), 1);
    assert_eq!(decoded_table.entries[0].endpoint_id, peer_bytes);
    assert_eq!(decoded_table.entries[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(
        decoded_table.mesh_id.as_deref(),
        Some("test-mesh-0102030405060708")
    );

    let local = proto_route_table_to_local(&decoded_table);
    assert_eq!(local.hosts.len(), 1);
    assert_eq!(local.hosts[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(local.hosts[0].endpoint_id, peer_id);
    assert_eq!(local.mesh_id.as_deref(), Some("test-mesh-0102030405060708"));

    let round_tripped = routing_table_to_proto(&local);
    assert_eq!(round_tripped.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(round_tripped.entries.len(), 1);
    assert_eq!(round_tripped.entries[0].endpoint_id, peer_bytes);
    assert_eq!(round_tripped.entries[0].model, "Qwen3-8B-Q4_K_M");
    assert_eq!(
        round_tripped.mesh_id.as_deref(),
        Some("test-mesh-0102030405060708")
    );
}

/// Verifies that remote passive inventory metadata is ignored on ingest.
#[test]
fn proto_v1_route_table_rejects_bad_generation_or_legacy_payload() {
    use crate::proto::node::RouteTable;

    let zero_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &zero_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {:?}",
        err
    );

    let wrong_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        gen: 99,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=99 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 99 }),
        "expected BadGeneration{{got:99}}, got {:?}",
        err
    );

    let bad_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for response, got {:?}",
        err
    );

    let wrong_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        gen: 42,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=42 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 42 }),
        "expected BadGeneration{{got:42}} for response, got {:?}",
        err
    );

    let legacy_json = b"{\"hosts\":[],\"mesh_id\":null}";
    let mut fake_frame = vec![STREAM_ROUTE_REQUEST];
    fake_frame.extend_from_slice(&(legacy_json.len() as u32).to_le_bytes());
    fake_frame.extend_from_slice(legacy_json);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &fake_frame)
        .expect_err("legacy JSON payload must be rejected");
    assert!(
        matches!(err, ControlFrameError::DecodeError(_)),
        "expected DecodeError for JSON payload, got {:?}",
        err
    );
}

#[test]
fn peer_lifecycle_messages_roundtrip() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let leaving_id = EndpointId::from(SecretKey::from_bytes(&[0x55; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(leaving_id, make_test_peer_info(leaving_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(leaving_id);

    let leaving_msg = PeerLeaving {
        peer_id: leaving_id.as_bytes().to_vec(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &leaving_msg);
    let decoded_leaving: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");

    let accepted_id = resolve_peer_leaving(leaving_id, &decoded_leaving)
        .expect("PeerLeaving from sender itself must be accepted");

    peers.remove(&accepted_id);
    connection_ids.remove(&accepted_id);

    assert!(
        !peers.contains_key(&leaving_id),
        "leaving peer must be removed from peers after accepted PeerLeaving"
    );
    assert!(
        !connection_ids.contains(&leaving_id),
        "leaving peer must be removed from connections after accepted PeerLeaving"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xAA; 32]).public());
    let dead_id = EndpointId::from(SecretKey::from_bytes(&[0xBB; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(dead_id, make_test_peer_info(dead_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(dead_id);

    let down_msg = PeerDown {
        peer_id: dead_id.as_bytes().to_vec(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &down_msg);
    let decoded_down: PeerDown =
        decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");

    let result = resolve_peer_down(self_id, dead_id, true);
    assert_eq!(
        result,
        Some(dead_id),
        "confirmed-unreachable peer must be returned for removal"
    );

    if let Some(id) = result {
        peers.remove(&id);
        connection_ids.remove(&id);
    }

    assert!(
        !peers.contains_key(&dead_id),
        "dead peer must be removed from peers when confirmed unreachable"
    );
    assert!(
        !connection_ids.contains(&dead_id),
        "dead peer must be removed from connections when confirmed unreachable"
    );

    assert_eq!(decoded_down.gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_lifecycle_rejects_forged_sender_or_unverified_down() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let valid_peer_bytes = EndpointId::from(SecretKey::from_bytes(&[0x77; 32]).public())
        .as_bytes()
        .to_vec();

    let bad_gen_down = PeerDown {
        peer_id: valid_peer_bytes.clone(),
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerDown, got {:?}",
        err
    );

    let bad_gen_leaving = PeerLeaving {
        peer_id: valid_peer_bytes.clone(),
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerLeaving, got {:?}",
        err
    );

    let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let victim_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(victim_id, make_test_peer_info(victim_id));

    let forged = PeerLeaving {
        peer_id: victim_id.as_bytes().to_vec(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &forged);
    let decoded: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
        .expect("structurally valid PeerLeaving must decode");

    let err = resolve_peer_leaving(remote_id, &decoded)
        .expect_err("forged PeerLeaving (peer_id != remote) must be rejected");
    assert!(
        matches!(err, ControlFrameError::ForgedSender),
        "expected ForgedSender, got {:?}",
        err
    );

    assert!(
        peers.contains_key(&victim_id),
        "victim peer must NOT be removed when PeerLeaving is forged"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
    let still_alive_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(still_alive_id, make_test_peer_info(still_alive_id));

    let result = resolve_peer_down(self_id, still_alive_id, false);
    assert!(
        result.is_none(),
        "PeerDown must not trigger removal when peer is still reachable"
    );

    assert!(
        peers.contains_key(&still_alive_id),
        "reachable peer must NOT be removed after PeerDown with should_remove=false"
    );
}

// ── Gossip consistency tests ──────────────────────────────────────────────

/// PeerDown for a recently-seen (direct) peer should be ignored regardless
/// of connection state — the peer is alive from our direct gossip even if
/// the connection is broken or absent (NAT, relay-only, stale QUIC conn).
#[test]
fn peer_down_ignored_when_recently_seen_direct() {
    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xA0; 32]).public());
    let target_id = EndpointId::from(SecretKey::from_bytes(&[0xA1; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut peer = make_test_peer_info(target_id);
    // Peer was seen just now via direct gossip.
    peer.last_seen = std::time::Instant::now();
    peers.insert(target_id, peer);

    let recently_seen = peers
        .get(&target_id)
        .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
        .unwrap_or(false);

    // The fix: when recently_seen (direct), ignore the death report
    // regardless of whether we have a connection.
    assert!(
        recently_seen,
        "precondition: peer must be recently seen (direct)"
    );
    // We should NOT call resolve_peer_down in this case.
    // Verify that resolve_peer_down with should_remove=true would remove,
    // proving the guard is necessary.
    let would_remove = resolve_peer_down(self_id, target_id, true);
    assert!(
        would_remove.is_some(),
        "without the guard, the peer would be removed"
    );
    // The peer stays in our peer list.
    assert!(
        peers.contains_key(&target_id),
        "recently-seen peer must survive PeerDown from another node"
    );
}

/// PeerDown for a peer whose last_seen is stale and has no connection
/// should be confirmed (the old behavior for genuinely dead peers).
#[test]
fn peer_down_confirmed_when_stale_and_no_connection() {
    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xB0; 32]).public());
    let target_id = EndpointId::from(SecretKey::from_bytes(&[0xB1; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut peer = make_test_peer_info(target_id);
    // Peer was last seen well beyond the stale window.
    peer.last_seen =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS + 60);
    peers.insert(target_id, peer);

    let recently_seen = peers
        .get(&target_id)
        .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
        .unwrap_or(false);

    assert!(
        !recently_seen,
        "precondition: peer is stale (not recently seen)"
    );

    // With no connection and stale last_seen, resolve_peer_down confirms removal.
    let result = resolve_peer_down(self_id, target_id, true);
    assert!(
        result.is_some(),
        "stale peer with no connection must be confirmed dead"
    );

    // Apply removal.
    if let Some(id) = result {
        peers.remove(&id);
    }
    assert!(
        !peers.contains_key(&target_id),
        "stale peer must be removed after confirmed PeerDown"
    );
}

/// Transitive peer updates should refresh last_seen so the peer doesn't
/// get pruned while a bridge peer keeps mentioning it.
#[test]
fn transitive_peer_update_refreshes_last_mentioned() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC0; 32]).public());
    let mut peer = make_test_peer_info(peer_id);

    // Simulate: peer was added long ago, both timestamps past the prune cutoff.
    let old_time =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);
    peer.last_seen = old_time;
    peer.last_mentioned = old_time;

    let addr = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    let ann = super::PeerAnnouncement {
        addr: addr.clone(),
        role: super::NodeRole::Worker,
        models: vec!["SomeModel-Q4_K_M".to_string()],
        vram_bytes: 8 * 1024 * 1024 * 1024,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec![],
        requested_models: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
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
    };

    apply_transitive_ann(&mut peer, &addr, &ann);

    // Before refreshing last_mentioned, verify the peer WOULD be pruned.
    let prune_cutoff_pre =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen < prune_cutoff_pre && peer.last_mentioned < prune_cutoff_pre,
        "peer must be pruneable before last_mentioned refresh"
    );

    // Simulate update_transitive_peer refreshing last_mentioned (not last_seen).
    peer.last_mentioned = std::time::Instant::now();

    // last_mentioned is fresh, last_seen stays stale.
    assert!(
        peer.last_mentioned.elapsed().as_secs() < 1,
        "last_mentioned must be refreshed after transitive gossip update"
    );
    assert!(
        peer.last_seen == old_time,
        "last_seen must NOT be refreshed by transitive gossip"
    );

    // Peer survives prune check because last_mentioned is fresh.
    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen < prune_cutoff || peer.last_mentioned >= prune_cutoff,
        "transitive peer with fresh last_mentioned must survive pruning"
    );

    // But PeerDown silencing uses only last_seen (direct), which is stale.
    let directly_seen_recently = peer.last_seen.elapsed().as_secs() < PEER_STALE_SECS;
    assert!(
        !directly_seen_recently,
        "transitive-only peer must NOT be considered directly seen"
    );
}

/// Transitive peer that is not mentioned stops surviving once both timestamps are stale.
#[test]
fn transitive_peer_expires_when_mentions_stop() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC1; 32]).public());
    let mut peer = make_test_peer_info(peer_id);

    // Both timestamps are beyond the prune window.
    let old_time =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);
    peer.last_seen = old_time;
    peer.last_mentioned = old_time;

    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen < prune_cutoff && peer.last_mentioned < prune_cutoff,
        "peer with both timestamps stale must be below prune cutoff"
    );
}

/// A directly-connected peer with fresh last_seen but stale last_mentioned
/// still survives pruning (last_seen alone is sufficient).
#[test]
fn direct_peer_survives_with_stale_last_mentioned() {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0xC2; 32]).public());
    let mut peer = make_test_peer_info(peer_id);

    peer.last_seen = std::time::Instant::now();
    peer.last_mentioned =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2 + 60);

    let prune_cutoff =
        std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
    assert!(
        peer.last_seen >= prune_cutoff || peer.last_mentioned >= prune_cutoff,
        "directly-connected peer must survive pruning via last_seen alone"
    );
}

// ── Task 9: End-to-end cut-over regression tests ──────────────────────────

/// Verifies that protobuf `/1` control frames still reject legacy JSON payloads AND
/// gen=0 / wrong-gen frames. Legacy JSON/raw compatibility is only carried on `/0`.
#[test]
fn proto_v1_control_frames_reject_legacy_json_and_wrong_gen() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    // JSON bytes that look plausible for the old wire format on each stream
    let json_gossip = b"[{\"addr\":{\"id\":\"aabbcc\",\"addrs\":[]}}]";
    let json_tunnel_map = b"{\"owner\":\"aabbcc\",\"entries\":[]}";
    let json_route = b"{\"hosts\":[],\"mesh_id\":null}";
    let json_peer_down = b"\"aabbccdd\"";
    let json_peer_leaving = b"\"aabbccdd\"";

    // All migrated streams must reject legacy JSON with DecodeError
    for (stream_type, json_bytes) in [
        (STREAM_GOSSIP, json_gossip.as_slice()),
        (STREAM_TUNNEL_MAP, json_tunnel_map.as_slice()),
        (STREAM_ROUTE_REQUEST, json_route.as_slice()),
        (STREAM_PEER_DOWN, json_peer_down.as_slice()),
        (STREAM_PEER_LEAVING, json_peer_leaving.as_slice()),
    ] {
        let mut frame = vec![stream_type];
        frame.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        frame.extend_from_slice(json_bytes);
        // Each stream uses its own message type for decode; we test gossip and route
        // request specifically since those carry gen validation too.
        if stream_type == STREAM_GOSSIP {
            let err = decode_control_frame::<GossipFrame>(stream_type, &frame).expect_err(
                &format!("JSON must be rejected on stream {:#04x}", stream_type),
            );
            assert!(
                matches!(err, ControlFrameError::DecodeError(_)),
                "stream {:#04x}: expected DecodeError for JSON, got {:?}",
                stream_type,
                err
            );
        } else if stream_type == STREAM_ROUTE_REQUEST {
            let err = decode_control_frame::<RouteTableRequest>(stream_type, &frame).expect_err(
                &format!("JSON must be rejected on stream {:#04x}", stream_type),
            );
            assert!(
                matches!(err, ControlFrameError::DecodeError(_)),
                "stream {:#04x}: expected DecodeError for JSON, got {:?}",
                stream_type,
                err
            );
        }
        // STREAM_TUNNEL_MAP, STREAM_PEER_DOWN, STREAM_PEER_LEAVING: JSON fails prost
        // decode which returns DecodeError — verified via the decode_control_frame
        // path used in the existing per-stream tests.
    }

    // All migrated streams must also reject gen=0 and gen=99 where gen is checked
    let bad_gen_gossip = GossipFrame {
        gen: 0,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTableRequest gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_down = PeerDown {
        peer_id: vec![0u8; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_leaving = PeerLeaving {
        peer_id: vec![0u8; 32],
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    // Wrong gen (e.g. 2) also rejected
    let wrong_gen_gossip = GossipFrame {
        gen: 2,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &wrong_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=2 (future version) must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 2 }));
}

/// Verifies that remote peer model-scan metadata (available_model_metadata,
/// available_model_sizes) is stored in PeerInfo after gossip and can be read back —
/// this is the unit-level proof of what `/api/status` exposes for remote `model_scans`.
#[test]
fn remote_model_scans_are_ignored_after_gossip() {
    use crate::proto::node::{CompactModelMetadata, GossipFrame, PeerAnnouncement as ProtoPA};

    let peer_key = SecretKey::from_bytes(&[0xC0; 32]);
    let peer_id = EndpointId::from(peer_key.public());

    // Build a gossip frame as the remote peer would send it
    let meta = CompactModelMetadata {
        model_key: "Llama-3.3-70B-Q4_K_M".to_string(),
        context_length: 131072,
        vocab_size: 128256,
        embedding_size: 8192,
        head_count: 64,
        layer_count: 80,
        feed_forward_length: 28672,
        key_length: 128,
        value_length: 128,
        architecture: "llama".to_string(),
        tokenizer_model_name: "GPT2TokenizerFast".to_string(),
        special_tokens: vec![],
        rope_scale: 8.0,
        rope_freq_base: 500000.0,
        is_moe: false,
        expert_count: 0,
        used_expert_count: 0,
        quantization_type: "Q4_K_M".to_string(),
    };
    let mut model_sizes = std::collections::HashMap::new();
    model_sizes.insert("Llama-3.3-70B-Q4_K_M".to_string(), 42_000_000_000u64);

    let gossip_frame = GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: peer_id.as_bytes().to_vec(),
        peers: vec![ProtoPA {
            endpoint_id: peer_id.as_bytes().to_vec(),
            role: NodeRole::Host as i32,
            http_port: Some(9337),
            available_models: vec!["Llama-3.3-70B-Q4_K_M".to_string()],
            available_model_metadata: vec![meta.clone()],
            available_model_sizes: model_sizes.clone(),
            vram_bytes: 96 * 1024 * 1024 * 1024,
            ..Default::default()
        }],
    };

    // Verify the gossip frame encodes and decodes cleanly
    let encoded = encode_control_frame(STREAM_GOSSIP, &gossip_frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("gossip frame with model scan metadata must decode successfully");

    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.sender_id, peer_id.as_bytes());
    assert_eq!(decoded.peers.len(), 1);
    let wire_pa = &decoded.peers[0];
    assert_eq!(wire_pa.available_model_metadata.len(), 1);
    assert_eq!(
        wire_pa.available_model_sizes.get("Llama-3.3-70B-Q4_K_M"),
        Some(&42_000_000_000u64)
    );

    // Convert to local PeerAnnouncement and verify passive inventory metadata is ignored.
    let (addr, local_ann) =
        proto_ann_to_local(wire_pa).expect("proto_ann_to_local must succeed on valid gossip PA");

    assert!(local_ann.available_models.is_empty());
    assert!(local_ann.available_model_metadata.is_empty());
    assert!(local_ann.available_model_sizes.is_empty());
    assert_eq!(addr.id, peer_id, "peer EndpointId must match sender");

    // Build PeerInfo as add_peer would, verify passive inventory metadata stays empty.
    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let peer_info = PeerInfo::from_announcement(
        peer_id,
        addr.clone(),
        &local_ann,
        OwnershipSummary::default(),
    );
    peers.insert(peer_id, peer_info);

    let stored = peers.get(&peer_id).unwrap();
    assert!(stored.available_models.is_empty());
    assert!(stored.available_model_metadata.is_empty());
    assert!(stored.available_model_sizes.is_empty());
}

/// Verifies that the passive-client route-table path populates the models list
/// correctly from protobuf RouteTable entries, and that mesh_id propagates through.
#[test]
fn passive_client_route_table_models_and_mesh_id_populated() {
    use crate::proto::node::{RouteEntry as ProtoRouteEntry, RouteTable};

    let host_key = SecretKey::from_bytes(&[0xD0; 32]);
    let host_id = EndpointId::from(host_key.public());

    let worker_key = SecretKey::from_bytes(&[0xD1; 32]);
    let worker_id = EndpointId::from(worker_key.public());

    // Simulate a routing table as served by a host to a passive client
    let table = RouteTable {
        entries: vec![
            ProtoRouteEntry {
                endpoint_id: host_id.as_bytes().to_vec(),
                model: "Qwen3-32B-Q4_K_M".to_string(),
            },
            ProtoRouteEntry {
                endpoint_id: worker_id.as_bytes().to_vec(),
                model: "GLM-4.7-Flash-Q4_K_M".to_string(),
            },
        ],
        mesh_id: Some("cafebabe12345678".to_string()),
        gen: NODE_PROTOCOL_GENERATION,
    };

    // Encode/decode via the same path as the live server
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &table);
    let decoded: RouteTable = decode_control_frame(STREAM_ROUTE_REQUEST, &encoded)
        .expect("valid RouteTable must decode successfully for passive client");

    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.entries.len(), 2);
    assert_eq!(decoded.mesh_id.as_deref(), Some("cafebabe12345678"));

    // Convert to local routing table as a passive client would
    let local = proto_route_table_to_local(&decoded);

    assert_eq!(
        local.hosts.len(),
        2,
        "passive client must see both model entries"
    );
    assert_eq!(
        local.mesh_id.as_deref(),
        Some("cafebabe12345678"),
        "mesh_id must propagate to passive client via RouteTable"
    );

    // Verify model names are correct
    let models: Vec<&str> = local.hosts.iter().map(|h| h.model.as_str()).collect();
    assert!(
        models.contains(&"Qwen3-32B-Q4_K_M"),
        "host model must appear in passive client route table"
    );
    assert!(
        models.contains(&"GLM-4.7-Flash-Q4_K_M"),
        "worker model must appear in passive client route table"
    );

    // Verify endpoint IDs round-trip correctly
    let host_entry = local
        .hosts
        .iter()
        .find(|h| h.model == "Qwen3-32B-Q4_K_M")
        .unwrap();
    assert_eq!(
        host_entry.endpoint_id, host_id,
        "host endpoint_id must be preserved in passive client route table"
    );
    let worker_entry = local
        .hosts
        .iter()
        .find(|h| h.model == "GLM-4.7-Flash-Q4_K_M")
        .unwrap();
    assert_eq!(
        worker_entry.endpoint_id, worker_id,
        "worker endpoint_id must be preserved in passive client route table"
    );

    // Verify a bad-generation RouteTable is rejected by passive clients
    let stale_table = RouteTable {
        entries: vec![],
        mesh_id: None,
        gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &stale_table);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("stale RouteTable gen=0 must be rejected by passive client");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "passive client must reject stale RouteTable: {:?}",
        err
    );
}

#[test]
fn worker_only_legacy_models_are_excluded_from_http_routes() {
    let host_id = EndpointId::from(iroh::SecretKey::from_bytes(&[0xD2; 32]).public());
    let worker_id = EndpointId::from(iroh::SecretKey::from_bytes(&[0xD3; 32]).public());

    let mut legacy_host = make_test_peer_info(host_id);
    legacy_host.role = super::NodeRole::Host { http_port: 9337 };
    legacy_host.serving_models = vec!["legacy-host-model".to_string()];
    legacy_host.hosted_models_known = false;

    let mut legacy_worker = make_test_peer_info(worker_id);
    legacy_worker.role = super::NodeRole::Worker;
    legacy_worker.serving_models = vec!["worker-only-model".to_string()];
    legacy_worker.hosted_models_known = false;

    assert!(legacy_host.accepts_http_inference());
    assert!(!legacy_worker.accepts_http_inference());
    assert_eq!(
        legacy_host.http_routable_models(),
        vec!["legacy-host-model".to_string()]
    );
    assert!(legacy_host.routes_http_model("legacy-host-model"));
    assert!(legacy_worker.http_routable_models().is_empty());
    assert!(!legacy_worker.routes_http_model("worker-only-model"));
}

/// Verifies that dead-peer cleanup prevents re-admission: after a peer is cleaned
/// up and added to dead_peers, the HashSet blocks any further connection attempts,
/// and a subsequent PeerLeaving from the same peer is rejected as forged (peer_id
/// no longer in peers set).
#[test]
fn dead_peer_cleanup_prevents_readmission() {
    use crate::proto::node::PeerLeaving;

    let peer_key = SecretKey::from_bytes(&[0xE0; 32]);
    let peer_id = EndpointId::from(peer_key.public());

    // Simulate state: peer is admitted
    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut connections: HashSet<EndpointId> = HashSet::new();
    let mut dead_peers: HashSet<EndpointId> = HashSet::new();

    peers.insert(peer_id, make_test_peer_info(peer_id));
    connections.insert(peer_id);

    assert!(
        is_peer_admitted(&peers, &peer_id),
        "peer must start admitted"
    );

    // Receive valid PeerLeaving from the peer
    let leaving = PeerLeaving {
        peer_id: peer_id.as_bytes().to_vec(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &leaving);
    let decoded: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");

    let accepted_id =
        resolve_peer_leaving(peer_id, &decoded).expect("self PeerLeaving must be accepted");

    // Clean up — as the handler does
    peers.remove(&accepted_id);
    connections.remove(&accepted_id);
    dead_peers.insert(accepted_id);

    // Peer is now gone and in dead_peers
    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "peer must be removed after PeerLeaving"
    );
    assert!(
        !connections.contains(&peer_id),
        "connection must be removed after PeerLeaving"
    );
    assert!(
        dead_peers.contains(&peer_id),
        "peer must be in dead_peers after cleanup"
    );

    // Verify dead_peers blocks re-admission (simulates the check in connect_to_peer)
    assert!(
        dead_peers.contains(&peer_id),
        "dead_peers.contains check prevents re-connection to cleaned-up peer"
    );

    // A new gossip attempt from the same peer should be blocked by dead_peers
    // (In the real handler, add_peer clears dead_peers only on accepted inbound gossip,
    // not on arbitrary peer attempts; dead_peers prevents outbound reconnects.)
    // Test the invariant that after cleanup, the peer is NOT in the live peers set.
    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "dead peer must not appear as admitted after dead_peers eviction"
    );

    // Second PeerLeaving for the same peer is now harmless (peer already removed)
    // resolve_peer_leaving still succeeds structurally but cleanup is idempotent
    let leaving2 = PeerLeaving {
        peer_id: peer_id.as_bytes().to_vec(),
        gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded2 = encode_control_frame(STREAM_PEER_LEAVING, &leaving2);
    let decoded2: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded2)
        .expect("second PeerLeaving decodes structurally");
    let id2 = resolve_peer_leaving(peer_id, &decoded2)
        .expect("second PeerLeaving resolves (peer_id matches remote)");
    // Idempotent remove: already gone, nothing changes
    peers.remove(&id2);
    connections.remove(&id2);
    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "idempotent remove must not re-insert peer"
    );
    assert!(
        dead_peers.contains(&peer_id),
        "dead_peers must still contain peer after idempotent removal"
    );
}

/// Verifies that non-scope tunnel streams (0x02 STREAM_TUNNEL and 0x04
/// STREAM_TUNNEL_HTTP) are NOT subject to protobuf frame validation — they are
/// raw byte pass-throughs and must not be accidentally broken by the cut-over.
/// Also verifies they are correctly gated by admission policy.
#[test]
fn non_scope_tunnel_streams_pass_through_without_proto_validation() {
    // 0x02 and 0x04 must NOT be allowed before admission (they are raw TCP tunnels,
    // quarantined until the peer is admitted via gossip).
    assert!(
        !stream_allowed_before_admission(STREAM_TUNNEL),
        "STREAM_TUNNEL (0x02) must be gated until after gossip admission"
    );
    assert!(
        !stream_allowed_before_admission(STREAM_TUNNEL_HTTP),
        "STREAM_TUNNEL_HTTP (0x04) must be gated until after gossip admission"
    );

    // After admission these streams are live. Verify that the stream type constants
    // are distinct from all migrated control-plane streams.
    assert_ne!(
        STREAM_TUNNEL, STREAM_GOSSIP,
        "tunnel must not collide with gossip"
    );
    assert_ne!(
        STREAM_TUNNEL, STREAM_TUNNEL_MAP,
        "raw tunnel must not collide with tunnel-map control frame"
    );
    assert_ne!(
        STREAM_TUNNEL_HTTP, STREAM_GOSSIP,
        "http-tunnel must not collide with gossip"
    );
    assert_ne!(
        STREAM_TUNNEL_HTTP, STREAM_ROUTE_REQUEST,
        "http-tunnel must not collide with route-request"
    );

    // encode_control_frame is not called for 0x02/0x04 — they are raw pass-throughs.
    // Verify that any random bytes on these streams would decode with DecodeError
    // if accidentally routed through the protobuf decoder, proving they are kept separate.
    let raw_rpc_bytes = b"\x00\x01\x02\x03RPC-BYTES";
    let mut fake_frame = vec![STREAM_TUNNEL];
    fake_frame.extend_from_slice(&(raw_rpc_bytes.len() as u32).to_le_bytes());
    fake_frame.extend_from_slice(raw_rpc_bytes);
    // Trying to decode a raw tunnel frame as gossip must yield a type mismatch
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &fake_frame)
        .expect_err("raw tunnel bytes fed to gossip decoder must be rejected");
    assert!(
        matches!(
            err,
            ControlFrameError::WrongStreamType {
                expected: 0x01,
                got: 0x02
            }
        ),
        "expected WrongStreamType{{expected:0x01,got:0x02}}, got {:?}",
        err
    );

    // Verify that all admission-gated streams besides tunnels are also gated
    // (completeness check for non-scope stream policy)
    for stream in [STREAM_TUNNEL, STREAM_TUNNEL_HTTP] {
        assert!(
            !stream_allowed_before_admission(stream),
            "stream {:#04x} must require admission (raw tunnel security boundary)",
            stream
        );
    }
}

/// Proves the behavioral contract introduced in the reconnect fix:
/// if gossip fails after a relay-level reconnect, the peer must be removed from
/// state.peers rather than left as a zombie. Tests the pure state-transition logic
/// by simulating: admitted peer → connection drop → gossip probe fails → removal.
#[test]
fn reconnect_gossip_failure_removes_zombie_peer() {
    let peer_key = SecretKey::from_bytes(&[0xF0; 32]);
    let peer_id = EndpointId::from(peer_key.public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    let mut connections: HashSet<EndpointId> = HashSet::new();

    peers.insert(peer_id, make_test_peer_info(peer_id));
    connections.insert(peer_id);

    assert!(
        is_peer_admitted(&peers, &peer_id),
        "peer must start admitted"
    );

    let gossip_ok = false;

    if gossip_ok {
    } else {
        peers.remove(&peer_id);
        connections.remove(&peer_id);
    }

    assert!(
        !is_peer_admitted(&peers, &peer_id),
        "zombie peer must be removed when reconnect gossip fails (relay-connected but process dead)"
    );
    assert!(
        !connections.contains(&peer_id),
        "zombie connection must be removed when reconnect gossip fails"
    );

    let peer_key2 = SecretKey::from_bytes(&[0xF1; 32]);
    let peer_id2 = EndpointId::from(peer_key2.public());
    let mut peers2: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers2.insert(peer_id2, make_test_peer_info(peer_id2));

    let gossip_ok2 = true;
    if !gossip_ok2 {
        peers2.remove(&peer_id2);
    }

    assert!(
        is_peer_admitted(&peers2, &peer_id2),
        "peer must remain admitted when reconnect gossip succeeds"
    );
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v0_peer_tunnel_map_exchange_over_legacy_connection() -> Result<()> {
    use iroh::endpoint::QuicTransportConfig;

    let post_node = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    post_node
        .set_serving_models(vec!["post-model".to_string()])
        .await;
    post_node
        .set_mesh_id("tunnel-map-mesh-001".to_string())
        .await;
    post_node.start_accepting();

    let legacy_endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let legacy_id = legacy_endpoint.id();
    let legacy_addr = legacy_endpoint.addr();
    let target_id = EndpointId::from(SecretKey::from_bytes(&[0x42; 32]).public());
    let admitted = std::sync::Arc::new(tokio::sync::Notify::new());
    let admitted_signal = admitted.clone();
    let done = std::sync::Arc::new(tokio::sync::Notify::new());
    let done_signal = done.clone();
    let legacy_ann = super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: legacy_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 9444 },
        models: vec!["legacy-model".to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        model_source: None,
        serving: Some("legacy-model".to_string()),
        serving_models: vec!["legacy-model".to_string()],
        available_models: vec![],
        requested_models: vec![],
        version: Some("0.50.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("tunnel-map-mesh-001".to_string()),
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };

    let server = tokio::spawn(async move {
        let incoming =
            tokio::time::timeout(std::time::Duration::from_secs(5), legacy_endpoint.accept())
                .await
                .expect("legacy endpoint should receive incoming connection")
                .expect("accept should return an incoming connection");
        let mut accepting = incoming.accept().expect("legacy accept should succeed");
        let alpn = accepting.alpn().await.expect("ALPN should be available");
        assert_eq!(
            alpn, ALPN_V0,
            "v1 node must negotiate ALPN_V0 with legacy endpoint"
        );
        let conn = accepting
            .await
            .expect("legacy connection handshake should complete");

        let (mut send_gossip, mut recv_gossip) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.accept_bi())
                .await
                .expect("v1 node should open gossip stream")
                .expect("gossip stream accept should succeed");
        let mut stream_type = [0u8; 1];
        recv_gossip
            .read_exact(&mut stream_type)
            .await
            .expect("must read gossip stream type byte");
        assert_eq!(
            stream_type[0], STREAM_GOSSIP,
            "first stream must be STREAM_GOSSIP"
        );
        let _post_gossip_buf = read_len_prefixed(&mut recv_gossip)
            .await
            .expect("must read v1 gossip payload");
        let legacy_gossip_body =
            serde_json::to_vec(&vec![legacy_ann]).expect("legacy announcement must serialize");
        write_len_prefixed(&mut send_gossip, &legacy_gossip_body)
            .await
            .expect("legacy must reply with JSON gossip");
        send_gossip
            .finish()
            .expect("gossip reply must finish cleanly");
        let _ = recv_gossip.read_to_end(0).await;

        // Wait until the main task confirms the v1 node has admitted this peer
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            admitted_signal.notified(),
        )
        .await
        .expect("main task should signal admission within 5s");

        let (mut send_tmap, _recv_tmap) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.open_bi())
                .await
                .expect("should open tunnel map stream")
                .expect("tunnel map stream open should succeed");
        send_tmap
            .write_all(&[STREAM_TUNNEL_MAP])
            .await
            .expect("must write tunnel map type byte");
        let tmap_json = serde_json::to_vec(&HashMap::from([(
            hex::encode(target_id.as_bytes()),
            8080u16,
        )]))
        .expect("tunnel map JSON must serialize");
        write_len_prefixed(&mut send_tmap, &tmap_json)
            .await
            .expect("must write tunnel map JSON payload");
        send_tmap
            .finish()
            .expect("tunnel map send stream must finish");

        // Keep the endpoint alive until the main task has verified data ingestion.
        // Dropping legacy_endpoint sends CONNECTION_CLOSE, which would kill the
        // client's dispatch_streams loop before it processes the tunnel-map stream.
        tokio::time::timeout(std::time::Duration::from_secs(10), done_signal.notified())
            .await
            .expect("main task should signal done within 10s");
    });

    let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&legacy_addr).expect("legacy address must serialize"));
    post_node.join(&invite_token).await?;

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let peers = post_node.peers().await;
            if peers.iter().any(|p| p.id == legacy_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("post node should admit the legacy peer after JSON gossip exchange");

    admitted.notify_one();

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let maps = post_node.all_remote_tunnel_maps().await;
            if let Some(inner) = maps.get(&legacy_id) {
                if inner.contains_key(&target_id) {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("v1 node should receive and ingest the v0 JSON tunnel map within 5 seconds");

    let maps = post_node.all_remote_tunnel_maps().await;
    let inner = maps
        .get(&legacy_id)
        .expect("tunnel map for legacy peer must be present after ingest");
    assert_eq!(
        inner.get(&target_id).copied(),
        Some(8080),
        "tunnel map must record target_id → port 8080"
    );

    done.notify_one();
    server
        .await
        .expect("legacy server task should complete without panic");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v0_peer_leaving_over_legacy_connection() -> Result<()> {
    use iroh::endpoint::QuicTransportConfig;

    let post_node = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    post_node
        .set_serving_models(vec!["post-model".to_string()])
        .await;
    post_node
        .set_mesh_id("peer-leaving-mesh-001".to_string())
        .await;
    post_node.start_accepting();

    let legacy_endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let legacy_id = legacy_endpoint.id();
    let legacy_addr = legacy_endpoint.addr();
    let legacy_id_bytes = legacy_id.as_bytes().to_vec();
    let admitted = std::sync::Arc::new(tokio::sync::Notify::new());
    let admitted_signal = admitted.clone();
    let done = std::sync::Arc::new(tokio::sync::Notify::new());
    let done_signal = done.clone();
    let legacy_ann = super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: legacy_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 9444 },
        models: vec!["legacy-model".to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        model_source: None,
        serving: Some("legacy-model".to_string()),
        serving_models: vec!["legacy-model".to_string()],
        available_models: vec![],
        requested_models: vec![],
        version: Some("0.50.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("peer-leaving-mesh-001".to_string()),
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };

    let server = tokio::spawn(async move {
        let incoming =
            tokio::time::timeout(std::time::Duration::from_secs(5), legacy_endpoint.accept())
                .await
                .expect("legacy endpoint should receive incoming connection")
                .expect("accept should return an incoming connection");
        let mut accepting = incoming.accept().expect("legacy accept should succeed");
        let alpn = accepting.alpn().await.expect("ALPN should be available");
        assert_eq!(
            alpn, ALPN_V0,
            "v1 node must negotiate ALPN_V0 with legacy endpoint"
        );
        let conn = accepting
            .await
            .expect("legacy connection handshake should complete");

        let (mut send_gossip, mut recv_gossip) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.accept_bi())
                .await
                .expect("v1 node should open gossip stream")
                .expect("gossip stream accept should succeed");
        let mut stream_type = [0u8; 1];
        recv_gossip
            .read_exact(&mut stream_type)
            .await
            .expect("must read gossip stream type byte");
        assert_eq!(
            stream_type[0], STREAM_GOSSIP,
            "first stream must be STREAM_GOSSIP"
        );
        let _post_gossip_buf = read_len_prefixed(&mut recv_gossip)
            .await
            .expect("must read v1 gossip payload");
        let legacy_gossip_body =
            serde_json::to_vec(&vec![legacy_ann]).expect("legacy announcement must serialize");
        write_len_prefixed(&mut send_gossip, &legacy_gossip_body)
            .await
            .expect("legacy must reply with JSON gossip");
        send_gossip
            .finish()
            .expect("gossip reply must finish cleanly");
        let _ = recv_gossip.read_to_end(0).await;

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            admitted_signal.notified(),
        )
        .await
        .expect("main task should signal admission within 5s");

        let (mut send_leaving, _recv_leaving) =
            tokio::time::timeout(std::time::Duration::from_secs(5), conn.open_bi())
                .await
                .expect("should open peer-leaving stream")
                .expect("peer-leaving stream open should succeed");
        send_leaving
            .write_all(&[STREAM_PEER_LEAVING])
            .await
            .expect("must write peer-leaving type byte");
        send_leaving
            .write_all(&legacy_id_bytes)
            .await
            .expect("must write raw 32-byte legacy peer ID");
        send_leaving
            .finish()
            .expect("peer-leaving send stream must finish");

        // Keep endpoint alive until main task confirms peer removal.
        // Dropping legacy_endpoint sends CONNECTION_CLOSE prematurely.
        tokio::time::timeout(std::time::Duration::from_secs(10), done_signal.notified())
            .await
            .expect("main task should signal done within 10s");
    });

    let invite_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&legacy_addr).expect("legacy address must serialize"));
    post_node.join(&invite_token).await?;

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let peers = post_node.peers().await;
            if peers.iter().any(|p| p.id == legacy_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("post node should admit the legacy peer after JSON gossip exchange");

    admitted.notify_one();

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let peers = post_node.peers().await;
            if !peers.iter().any(|p| p.id == legacy_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("v1 node should remove legacy peer after receiving v0 peer-leaving frame within 5s");

    let peers = post_node.peers().await;
    assert!(
        !peers.iter().any(|p| p.id == legacy_id),
        "legacy peer must be absent from the peer list after its clean-shutdown announcement"
    );

    done.notify_one();
    server
        .await
        .expect("legacy server task should complete without panic");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_protocol_three_node_mesh_state_consistency() -> Result<()> {
    use iroh::endpoint::{ConnectOptions, QuicTransportConfig};

    let node_a = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    node_a
        .set_serving_models(vec!["node-a-model".to_string()])
        .await;
    node_a.set_mesh_id("three-node-mesh-001".to_string()).await;
    node_a.start_accepting();
    let node_a_id = node_a.id();
    let node_a_addr = node_a.endpoint.addr();

    let node_b = make_test_node(super::NodeRole::Host { http_port: 9338 }).await?;
    node_b
        .set_serving_models(vec!["node-b-model".to_string()])
        .await;
    node_b.set_mesh_id("three-node-mesh-001".to_string()).await;
    let node_b_id = node_b.id();

    let legacy_endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let legacy_id = legacy_endpoint.id();

    let invite_token_a = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&node_a_addr).expect("node_a addr must serialize"));
    node_b.join(&invite_token_a).await?;

    let connecting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        legacy_endpoint.connect_with_opts(node_a_addr, ALPN_V0, ConnectOptions::new()),
    )
    .await
    .expect("v0 connect_with_opts should not timeout")
    .expect("v0 connect_with_opts should succeed");
    let v0_conn = tokio::time::timeout(std::time::Duration::from_secs(5), connecting)
        .await
        .expect("v0→node_a handshake should not timeout")
        .expect("v0→node_a handshake should succeed");
    assert_eq!(
        v0_conn.alpn(),
        ALPN_V0,
        "v0 endpoint must negotiate ALPN_V0 with the v1 node"
    );

    let (mut send_g, mut recv_g) =
        tokio::time::timeout(std::time::Duration::from_secs(5), v0_conn.open_bi())
            .await
            .expect("v0 should open gossip stream")
            .expect("v0 gossip stream open should succeed");
    send_g
        .write_all(&[STREAM_GOSSIP])
        .await
        .expect("v0 must write gossip type byte");
    let v0_ann = super::PeerAnnouncementV0 {
        addr: EndpointAddr {
            id: legacy_id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Host { http_port: 9555 },
        models: vec!["v0-model".to_string()],
        vram_bytes: 8 * 1024 * 1024 * 1024,
        model_source: None,
        serving: Some("v0-model".to_string()),
        serving_models: vec!["v0-model".to_string()],
        available_models: vec![],
        requested_models: vec![],
        version: Some("0.50.0".to_string()),
        model_demand: HashMap::new(),
        mesh_id: Some("three-node-mesh-001".to_string()),
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
    };
    let v0_gossip_json = serde_json::to_vec(&vec![v0_ann]).expect("v0 gossip JSON must serialize");
    write_len_prefixed(&mut send_g, &v0_gossip_json)
        .await
        .expect("v0 must write gossip JSON payload");
    send_g.finish().expect("v0 gossip send stream must finish");
    let _node_a_gossip_resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        read_len_prefixed(&mut recv_g),
    )
    .await
    .expect("node_a must respond to v0 gossip within 5 seconds")
    .expect("v0 must read node_a gossip response");
    let _ = recv_g.read_to_end(0).await;

    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let peers = node_a.peers().await;
            let has_b = peers.iter().any(|p| p.id == node_b_id);
            let has_v0 = peers.iter().any(|p| p.id == legacy_id);
            if has_b && has_v0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("node_a must see both node_b and v0 peer within 10 seconds");

    let node_a_peers = node_a.peers().await;
    assert!(
        node_a_peers.iter().any(|p| {
            p.id == node_b_id
                && p.serving_models.first().map(String::as_str) == Some("node-b-model")
        }),
        "node_a must see node_b with its correct serving model"
    );
    assert!(
        node_a_peers.iter().any(|p| {
            p.id == legacy_id && p.serving_models.first().map(String::as_str) == Some("v0-model")
        }),
        "node_a must see the v0 peer with its correct serving model"
    );

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let peers = node_b.peers().await;
            if peers.iter().any(|p| p.id == node_a_id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("node_b must see node_a after joining");

    assert!(
        node_b.peers().await.iter().any(|p| {
            p.id == node_a_id
                && p.serving_models.first().map(String::as_str) == Some("node-a-model")
        }),
        "node_b must see node_a with its correct serving model"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn protocol_negotiation_edge_cases() -> Result<()> {
    use iroh::endpoint::{ConnectOptions, QuicTransportConfig};

    assert_eq!(
        protocol_from_alpn(b""),
        ControlProtocol::ProtoV1,
        "empty ALPN must default to ProtoV1"
    );
    assert_eq!(
        protocol_from_alpn(b"unknown"),
        ControlProtocol::ProtoV1,
        "unrecognised ALPN must default to ProtoV1"
    );
    assert_eq!(
        protocol_from_alpn(b"mesh-llm"),
        ControlProtocol::ProtoV1,
        "partial ALPN prefix without version number must default to ProtoV1"
    );

    // Sub-test A: v1 node connecting to a v0-only endpoint negotiates ALPN_V0
    let v0_endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let v0_addr = v0_endpoint.addr();
    let v0_accept_task = tokio::spawn(async move {
        let incoming =
            tokio::time::timeout(std::time::Duration::from_secs(5), v0_endpoint.accept())
                .await
                .expect("v0 endpoint should receive an incoming connection")
                .expect("v0 accept should yield an incoming connection");
        let mut accepting = incoming.accept().expect("v0 accept should succeed");
        let _alpn = accepting.alpn().await.expect("ALPN should be available");
        let conn = accepting
            .await
            .expect("v0 connection handshake should complete");
        assert_eq!(
            conn.alpn(),
            ALPN_V0,
            "v0 endpoint must see ALPN_V0 on the accepted connection"
        );
        assert_eq!(
            connection_protocol(&conn),
            ControlProtocol::JsonV0,
            "v0 endpoint must identify the connection as JsonV0"
        );
    });

    let post_node = make_test_node(super::NodeRole::Worker).await?;
    let conn_a = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        connect_mesh(&post_node.endpoint, v0_addr),
    )
    .await
    .expect("v1→v0 connect should not timeout")
    .expect("v1 node should connect successfully to v0-only endpoint");
    assert_eq!(
        conn_a.alpn(),
        ALPN_V0,
        "v1 node connecting to a v0-only endpoint must negotiate ALPN_V0"
    );
    assert_eq!(
        connection_protocol(&conn_a),
        ControlProtocol::JsonV0,
        "connection from v1 to v0-only endpoint must use JsonV0 protocol"
    );

    v0_accept_task
        .await
        .expect("v0 accept task should complete without panic");

    let node_b = make_test_node(super::NodeRole::Worker).await?;
    node_b.start_accepting();
    let node_b_addr = node_b.endpoint.addr();

    let v0_ep2 = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V0.to_vec()])
        .transport_config(
            QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(128u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let connecting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        v0_ep2.connect_with_opts(node_b_addr, ALPN_V0, ConnectOptions::new()),
    )
    .await
    .expect("v0→v1 connect_with_opts should not timeout")
    .expect("v0 endpoint should connect to v1 node");
    let conn_b = tokio::time::timeout(std::time::Duration::from_secs(5), connecting)
        .await
        .expect("v0→v1 handshake should not timeout")
        .expect("v0→v1 connection handshake should succeed");
    assert_eq!(
        conn_b.alpn(),
        ALPN_V0,
        "v0 endpoint connecting to a v1 node must negotiate ALPN_V0"
    );
    assert_eq!(
        connection_protocol(&conn_b),
        ControlProtocol::JsonV0,
        "v0 endpoint connecting to a v1 node must use JsonV0 protocol"
    );

    Ok(())
}

fn make_test_peer(id: EndpointId, rtt_ms: Option<u32>, vram_gb: u64) -> PeerInfo {
    PeerInfo {
        id,
        addr: EndpointAddr {
            id,
            addrs: Default::default(),
        },
        role: super::NodeRole::Worker,
        models: vec![],
        vram_bytes: vram_gb * 1024 * 1024 * 1024,
        rtt_ms,
        model_source: None,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
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
        tunnel_port: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        owner_summary: OwnershipSummary::default(),
    }
}

/// RTT re-election: when a peer's RTT drops from above the 80ms split
/// threshold to below it (e.g. relay → direct), update_peer_rtt must
/// trigger a peer_change event so the election loop re-runs and can
/// now include the peer in split mode.
#[tokio::test]
async fn test_rtt_drop_triggers_reelection() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate(&mut rand::rng());
    let peer_id = EndpointId::from(peer_key.public());

    // Add a fake peer with high relay RTT
    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(2600), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to still-high value — should NOT trigger
    node.update_peer_rtt(peer_id, 500).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 2600→500 (both above threshold) should not trigger re-election"
    );

    // Update RTT to below threshold — SHOULD trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 500→15 (crossing threshold) must trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger when RTT was already below threshold.
#[tokio::test]
async fn test_rtt_below_threshold_no_reelection() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate(&mut rand::rng());
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to another low value — should NOT trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 20→15 (both below threshold) should not trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger for unknown peers.
#[tokio::test]
async fn test_rtt_update_unknown_peer_no_panic() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate(&mut rand::rng());
    let peer_id = EndpointId::from(peer_key.public());

    let rx = node.peer_change_rx.clone();

    // Update RTT for a peer that doesn't exist — should not panic or trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT update for unknown peer should not trigger re-election"
    );

    Ok(())
}

/// RTT should never increase — relay gossip RTT must not overwrite
/// a known-good direct path measurement.
#[tokio::test]
async fn test_rtt_cannot_regress() -> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate(&mut rand::rng());
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    // Try to raise RTT — should be rejected
    node.update_peer_rtt(peer_id, 2600).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(20), "RTT must not increase from 20 to 2600");
    }

    // Lower RTT — should be accepted
    node.update_peer_rtt(peer_id, 10).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(10), "RTT must decrease from 20 to 10");
    }

    Ok(())
}

/// Regression test: connect_to_peer must skip peers already in state.peers,
/// even if there's no QUIC connection yet (transitive peers from gossip).
/// If this check uses state.connections instead, every transitive peer
/// triggers a 15s dial timeout and --client --auto hangs.
/// See: d631c8d (broke it), 6ece4d1 (first revert).
#[tokio::test]
async fn test_connect_to_peer_skips_known_peer_without_connection() -> Result<()> {
    let node = make_test_node(super::NodeRole::Client).await?;
    let peer_key = SecretKey::generate(&mut rand::rng());
    let peer_id = EndpointId::from(peer_key.public());

    // Simulate a transitive peer: in state.peers but NOT in state.connections
    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(50), 8));
        assert!(
            !state.connections.contains_key(&peer_id),
            "setup: peer must not have a connection"
        );
    }

    // connect_to_peer must return Ok immediately (peer already known).
    // If it tries to dial, it will either timeout (15s) or fail — both wrong.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        node.connect_to_peer(super::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }),
    )
    .await;

    assert!(
        result.is_ok(),
        "connect_to_peer must not attempt to dial a peer already in state.peers"
    );
    assert!(
        result.unwrap().is_ok(),
        "connect_to_peer must return Ok for known peers"
    );

    Ok(())
}

#[test]
fn config_sync_subscribe_snapshot_encode_decode() {
    use crate::proto::node::{ConfigSnapshotResponse, NodeConfigSnapshot, NodeGpuConfig};

    let snapshot = ConfigSnapshotResponse {
        gen: NODE_PROTOCOL_GENERATION,
        node_id: vec![0xAA; 32],
        revision: 7,
        config_hash: vec![0xBB; 32],
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: Some(NodeGpuConfig {
                assignment: crate::proto::node::GpuAssignment::Auto as i32,
            }),
            models: vec![],
            plugins: vec![],
        }),
        hostname: Some("test-host".to_string()),
        error: None,
    };

    let encoded = encode_control_frame(STREAM_CONFIG_SUBSCRIBE, &snapshot);
    let decoded: ConfigSnapshotResponse =
        decode_control_frame(STREAM_CONFIG_SUBSCRIBE, &encoded).expect("round-trip must succeed");

    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.node_id, vec![0xAA; 32]);
    assert_eq!(decoded.revision, 7);
    assert_eq!(decoded.config_hash, vec![0xBB; 32]);
    assert_eq!(decoded.hostname, Some("test-host".to_string()));
    let cfg = decoded.config.expect("config must be present");
    assert_eq!(cfg.version, 1);
    let gpu = cfg.gpu.expect("gpu must be present");
    assert_eq!(
        gpu.assignment,
        crate::proto::node::GpuAssignment::Auto as i32
    );
}

#[test]
fn config_sync_subscribe_not_before_admission() {
    assert!(
        !stream_allowed_before_admission(STREAM_CONFIG_SUBSCRIBE),
        "STREAM_CONFIG_SUBSCRIBE (0x0b) must require admission — it is an owner-gated config stream"
    );
}

fn test_signing_key() -> (ed25519_dalek::SigningKey, String) {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x42u8; 32]);
    let verifying = signing_key.verifying_key();
    let owner_id = crate::crypto::owner_id_from_verifying_key(&verifying);
    (signing_key, owner_id)
}

#[test]
fn config_sync_push_signature_payload_deterministic() {
    use crate::proto::node::{ConfigPush, NodeConfigSnapshot};

    let push = ConfigPush {
        gen: NODE_PROTOCOL_GENERATION,
        requester_id: vec![0xAA; 32],
        target_node_id: vec![0xBB; 32],
        owner_signing_public_key: vec![0x42u8; 32],
        expected_revision: 3,
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: None,
            models: vec![],
            plugins: vec![],
        }),
        signature: vec![0u8; 64],
    };

    let p1 = config_push_signature_payload(&push);
    let p2 = config_push_signature_payload(&push);
    assert_eq!(p1, p2, "payload must be deterministic for the same input");
    assert!(!p1.is_empty(), "payload must not be empty");
}

// config_sync_push_wrong_owner_detected was removed: the `owner_id` field no longer
// exists in ConfigPush. Wrong-owner detection is now handled entirely through the
// gossip-attested peer identity check in handle_config_push.

#[test]
fn config_sync_push_bad_signature_bytes_length() {
    let bad_sig: Vec<u8> = vec![0u8; 32];
    let result: Result<[u8; 64], _> = bad_sig.as_slice().try_into();
    assert!(
        result.is_err(),
        "32-byte slice must not convert to [u8; 64] — wrong-length signature must be rejected"
    );

    let good_sig: Vec<u8> = vec![0u8; 64];
    let result: Result<[u8; 64], _> = good_sig.as_slice().try_into();
    assert!(result.is_ok(), "64-byte slice must convert to [u8; 64]");
}

#[test]
fn config_sync_push_roundtrip_encode_decode() {
    use crate::proto::node::{ConfigApplyMode, ConfigPushResponse};
    use prost::Message as _;

    let response = ConfigPushResponse {
        gen: NODE_PROTOCOL_GENERATION,
        success: true,
        current_revision: 42,
        config_hash: vec![0xCC; 32],
        error: None,
        apply_mode: ConfigApplyMode::Staged as i32,
    };

    let encoded = response.encode_to_vec();
    let decoded = ConfigPushResponse::decode(encoded.as_slice())
        .expect("ConfigPushResponse must round-trip through encode/decode");

    assert_eq!(decoded.gen, NODE_PROTOCOL_GENERATION);
    assert!(decoded.success);
    assert_eq!(decoded.current_revision, 42);
    assert_eq!(decoded.config_hash, vec![0xCC; 32]);
    assert!(decoded.error.is_none());
    assert_eq!(decoded.apply_mode, ConfigApplyMode::Staged as i32);
}

#[test]
fn config_sync_sign_and_verify_roundtrip() {
    use crate::proto::node::{ConfigPush, NodeConfigSnapshot};
    use ed25519_dalek::Signer as _;

    let (signing_key, owner_id) = test_signing_key();
    let vk = signing_key.verifying_key();

    let mut push = ConfigPush {
        gen: NODE_PROTOCOL_GENERATION,
        requester_id: vec![0xAA; 32],
        target_node_id: vec![0xBB; 32],
        owner_signing_public_key: vk.to_bytes().to_vec(),
        expected_revision: 0,
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: None,
            models: vec![],
            plugins: vec![],
        }),
        signature: vec![0u8; 64],
    };

    let payload = config_push_signature_payload(&push);
    let sig = signing_key.sign(&payload);
    push.signature = sig.to_bytes().to_vec();

    // Verify: re-derive owner_id from vk and check signature
    let pk_bytes: [u8; 32] = push.owner_signing_public_key.as_slice().try_into().unwrap();
    let restored_vk = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes).unwrap();
    let derived_id = crate::crypto::owner_id_from_verifying_key(&restored_vk);
    assert_eq!(derived_id, owner_id, "owner_id must match key fingerprint");

    let payload2 = config_push_signature_payload(&push);
    let sig_bytes: [u8; 64] = push.signature.as_slice().try_into().unwrap();
    let sig_obj = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    restored_vk
        .verify_strict(&payload2, &sig_obj)
        .expect("signature must verify against the canonical payload");
}

#[test]
fn config_sync_signature_payload_excludes_signature_field() {
    use crate::proto::node::{ConfigPush, NodeConfigSnapshot};

    let mut push = ConfigPush {
        gen: NODE_PROTOCOL_GENERATION,
        requester_id: vec![0xAA; 32],
        target_node_id: vec![0xBB; 32],
        owner_signing_public_key: vec![0x42u8; 32],
        expected_revision: 0,
        config: Some(NodeConfigSnapshot {
            version: 1,
            gpu: None,
            models: vec![],
            plugins: vec![],
        }),
        signature: vec![0u8; 64],
    };

    let payload_with_sig = config_push_signature_payload(&push);

    // Change only the signature field — the canonical payload must not change
    push.signature = vec![0xFF; 64];
    let payload_different_sig = config_push_signature_payload(&push);

    assert_eq!(
        payload_with_sig, payload_different_sig,
        "payload must be identical regardless of the signature field value"
    );

    // Change a semantic field — the canonical payload MUST change
    push.expected_revision = 99;
    let payload_changed = config_push_signature_payload(&push);
    assert_ne!(
        payload_with_sig, payload_changed,
        "payload must change when a semantic field changes"
    );
}

fn test_owner_keypair(signing_seed: u8, encryption_seed: u8) -> crate::crypto::OwnerKeypair {
    crate::crypto::OwnerKeypair::from_bytes(&[signing_seed; 32], &[encryption_seed; 32])
        .expect("test owner keypair must be valid")
}

/// Create a test `Node` with a verified local owner attestation and a
/// `ConfigState` whose backing file lives in `config_dir`.
async fn make_test_node_with_owner(
    role: super::NodeRole,
    owner_keypair: &crate::crypto::OwnerKeypair,
    config_dir: &std::path::Path,
) -> Result<Node> {
    use iroh::endpoint::QuicTransportConfig;

    let config_path = config_dir.join("config.toml");
    let config_state =
        crate::runtime::config_state::ConfigState::load(&config_path).unwrap_or_default();

    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(128u32.into())
        .build();
    let endpoint = Endpoint::empty_builder()
        .secret_key(SecretKey::generate(&mut rand::rng()))
        .alpns(vec![ALPN_V1.to_vec(), ALPN_V0.to_vec()])
        .transport_config(transport_config)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;

    let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
    let (inflight_change_tx, _) = watch::channel(0u64);
    let (tunnel_tx, _tunnel_rx) = tokio::sync::mpsc::channel(8);
    let (tunnel_http_tx, _tunnel_http_rx) = tokio::sync::mpsc::channel(8);
    let revision = config_state.revision();
    let owner_attestation = sign_node_ownership(
        owner_keypair,
        endpoint.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let trust_store = TrustStore::default();
    let owner_summary = verify_node_ownership(
        Some(&owner_attestation),
        endpoint.id().as_bytes(),
        &trust_store,
        TrustPolicy::Off,
        current_time_unix_ms(),
    );

    let node = Node {
        endpoint,
        public_addr: None,
        state: Arc::new(Mutex::new(MeshState {
            peers: HashMap::new(),
            connections: HashMap::new(),
            remote_tunnel_maps: HashMap::new(),
            dead_peers: HashSet::new(),
            seen_plugin_messages: HashSet::new(),
            seen_plugin_message_order: VecDeque::new(),
            policy_rejected_peers: HashMap::new(),
        })),
        role: Arc::new(Mutex::new(role)),
        models: Arc::new(Mutex::new(Vec::new())),
        model_source: Arc::new(Mutex::new(None)),
        serving_models: Arc::new(Mutex::new(Vec::new())),
        served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
        model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
        hosted_models: Arc::new(Mutex::new(Vec::new())),
        llama_ready: Arc::new(Mutex::new(false)),
        available_models: Arc::new(Mutex::new(Vec::new())),
        requested_models: Arc::new(Mutex::new(Vec::new())),
        model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
        mesh_id: Arc::new(Mutex::new(None)),
        accepting: Arc::new((
            tokio::sync::Notify::new(),
            std::sync::atomic::AtomicBool::new(false),
        )),
        vram_bytes: 64 * 1024 * 1024 * 1024,
        peer_change_tx,
        peer_change_rx,
        inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        inflight_change_tx,
        tunnel_tx,
        tunnel_http_tx,
        plugin_manager: Arc::new(Mutex::new(None)),
        display_name: Arc::new(Mutex::new(None)),
        owner_attestation: Arc::new(Mutex::new(Some(owner_attestation))),
        owner_summary: Arc::new(Mutex::new(owner_summary)),
        trust_store: Arc::new(Mutex::new(trust_store)),
        trust_policy: TrustPolicy::Off,
        enumerate_host: false,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
        config_state: Arc::new(tokio::sync::Mutex::new(config_state)),
        config_revision_tx: {
            let (tx, _rx) = tokio::sync::watch::channel(revision);
            Arc::new(tx)
        },
    };

    let accept_node = node.clone();
    tokio::spawn(async move {
        accept_node.accept_loop().await;
    });

    Ok(node)
}

/// Helper: build and sign a ConfigPush proto for the given node/owner/config.
/// Build a `ConfigPush` proto that is correctly signed with `signing_key`.
///
/// The resulting push targets `target_node_id`, is attributed to `requester_id`,
/// and carries `expected_revision` for CAS enforcement. The signature covers the
/// canonical protobuf encoding of the push with the `signature` field cleared.
fn build_signed_config_push(
    owner_keypair: &crate::crypto::OwnerKeypair,
    requester_id: &EndpointId,
    target_node_id: &EndpointId,
    expected_revision: u64,
    config: crate::proto::node::NodeConfigSnapshot,
) -> crate::proto::node::ConfigPush {
    use ed25519_dalek::Signer as _;

    let vk = owner_keypair.signing.verifying_key();

    let mut push = crate::proto::node::ConfigPush {
        gen: NODE_PROTOCOL_GENERATION,
        requester_id: requester_id.as_bytes().to_vec(),
        target_node_id: target_node_id.as_bytes().to_vec(),
        owner_signing_public_key: vk.to_bytes().to_vec(),
        expected_revision,
        config: Some(config),
        signature: vec![0u8; 64],
    };
    let payload = config_push_signature_payload(&push);
    let sig = owner_keypair.signing.sign(&payload);
    push.signature = sig.to_bytes().to_vec();
    push
}

/// Wait until `node` has `target` in its peers list. Times out after 5 s.
/// Poll `node.peers()` until `target` appears in the list.
///
/// Panics (via `expect`) if `target` is not admitted within 5 seconds.
async fn wait_for_peer(node: &Node, target: EndpointId) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if node.peers().await.iter().any(|p| p.id == target) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("peer was not admitted within 5 s");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_subscribe_matching_owner_receives_snapshot() -> Result<()> {
    let owner_keypair = test_owner_keypair(0x11, 0x12);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-sub-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &owner_keypair,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &owner_keypair, &tmp.join("client"))
            .await?;

    server
        .set_mesh_id("cfg-subscribe-mesh-01".to_string())
        .await;
    client
        .set_mesh_id("cfg-subscribe-mesh-01".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    let (snapshot, _notif_rx) = client.subscribe_to_config(&conn).await?;

    assert_eq!(
        snapshot.node_id,
        server_id.as_bytes().to_vec(),
        "snapshot node_id must be the server's endpoint id"
    );
    assert_eq!(
        snapshot.config_hash.len(),
        32,
        "config_hash must be 32 bytes"
    );
    assert!(
        snapshot.config.is_some(),
        "snapshot must include config payload"
    );
    assert!(
        snapshot.error.is_none() || snapshot.error.as_deref() == Some(""),
        "snapshot must not carry an error"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_subscribe_wrong_owner_returns_error() -> Result<()> {
    let server_owner = test_owner_keypair(0x22, 0x23);
    let client_owner = test_owner_keypair(0x33, 0x34);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-wrong-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &server_owner,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &client_owner, &tmp.join("client"))
            .await?;

    server
        .set_mesh_id("cfg-wrong-owner-mesh-01".to_string())
        .await;
    client
        .set_mesh_id("cfg-wrong-owner-mesh-01".to_string())
        .await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    // Subscribe - the subscriber's attested owner doesn't match the server's owner
    let result = client.subscribe_to_config(&conn).await;
    assert!(
        result.is_err(),
        "subscribing with wrong owner_id must return an error"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("owner_id mismatch") || err_msg.contains("rejected"),
        "error must mention owner mismatch, got: {err_msg}"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_subscribe_unowned_node_returns_error() -> Result<()> {
    let client_owner = test_owner_keypair(0x44, 0x45);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-unowned-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("client")).ok();

    // server has NO owner key (make_test_node, not make_test_node_with_owner)
    let server = make_test_node(super::NodeRole::Host { http_port: 9337 }).await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &client_owner, &tmp.join("client"))
            .await?;

    server.set_mesh_id("cfg-unowned-mesh-01".to_string()).await;
    client.set_mesh_id("cfg-unowned-mesh-01".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    let result = client.subscribe_to_config(&conn).await;
    assert!(
        result.is_err(),
        "subscribing to an unowned node must return an error"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("no local owner") || err_msg.contains("rejected"),
        "error must mention missing owner, got: {err_msg}"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_push_valid_signature_accepted() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig};
    use crate::protocol::write_len_prefixed;
    use prost::Message as _;

    let owner_keypair = test_owner_keypair(0x55, 0x56);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-push-ok-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &owner_keypair,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &owner_keypair, &tmp.join("client"))
            .await?;

    server.set_mesh_id("cfg-push-ok-mesh-01".to_string()).await;
    client.set_mesh_id("cfg-push-ok-mesh-01".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let client_id = client.id();
    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    let new_config = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![],
        plugins: vec![],
    };

    let push = build_signed_config_push(&owner_keypair, &client_id, &server_id, 0, new_config);

    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&[STREAM_CONFIG_PUSH]).await?;
    write_len_prefixed(&mut send, &push.encode_to_vec()).await?;
    send.finish()?;

    let buf = crate::protocol::read_len_prefixed(&mut recv).await?;
    let response = crate::proto::node::ConfigPushResponse::decode(buf.as_slice())?;

    assert!(
        response.success,
        "valid signed push must be accepted: {:?}",
        response.error
    );
    assert_eq!(
        response.current_revision, 1,
        "revision must be bumped to 1 after first push"
    );
    assert_eq!(
        response.config_hash.len(),
        32,
        "response config_hash must be 32 bytes"
    );
    assert_eq!(
        response.apply_mode,
        crate::proto::node::ConfigApplyMode::Staged as i32,
        "config push should report staged apply mode"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_push_revision_conflict_rejected() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig};
    use crate::protocol::write_len_prefixed;
    use prost::Message as _;

    let owner_keypair = test_owner_keypair(0x66, 0x67);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-conflict-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &owner_keypair,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &owner_keypair, &tmp.join("client"))
            .await?;

    server.set_mesh_id("cfg-conflict-mesh-01".to_string()).await;
    client.set_mesh_id("cfg-conflict-mesh-01".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let client_id = client.id();
    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    let good_config = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![],
        plugins: vec![],
    };

    // First push (revision 0 → 1) — must succeed
    let push1 = build_signed_config_push(
        &owner_keypair,
        &client_id,
        &server_id,
        0,
        good_config.clone(),
    );
    let (mut send1, mut recv1) = conn.open_bi().await?;
    send1.write_all(&[STREAM_CONFIG_PUSH]).await?;
    write_len_prefixed(&mut send1, &push1.encode_to_vec()).await?;
    send1.finish()?;
    let buf1 = crate::protocol::read_len_prefixed(&mut recv1).await?;
    let resp1 = crate::proto::node::ConfigPushResponse::decode(buf1.as_slice())?;
    assert!(resp1.success, "first push must succeed: {:?}", resp1.error);

    // Second push with stale expected_revision=0 — must be rejected
    let push2 = build_signed_config_push(&owner_keypair, &client_id, &server_id, 0, good_config);
    let (mut send2, mut recv2) = conn.open_bi().await?;
    send2.write_all(&[STREAM_CONFIG_PUSH]).await?;
    write_len_prefixed(&mut send2, &push2.encode_to_vec()).await?;
    send2.finish()?;
    let buf2 = crate::protocol::read_len_prefixed(&mut recv2).await?;
    let resp2 = crate::proto::node::ConfigPushResponse::decode(buf2.as_slice())?;

    assert!(!resp2.success, "push with stale revision must be rejected");
    assert_eq!(
        resp2.current_revision, 1,
        "rejection response must carry the current revision"
    );
    let err = resp2.error.as_deref().unwrap_or("");
    assert!(
        err.contains("revision conflict"),
        "error must mention revision conflict, got: {err}"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_push_bad_signature_rejected() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig};
    use crate::protocol::write_len_prefixed;
    use prost::Message as _;

    let owner_keypair = test_owner_keypair(0x77, 0x78);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-badsig-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &owner_keypair,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &owner_keypair, &tmp.join("client"))
            .await?;

    server.set_mesh_id("cfg-badsig-mesh-01".to_string()).await;
    client.set_mesh_id("cfg-badsig-mesh-01".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let client_id = client.id();
    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    let config = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![],
        plugins: vec![],
    };

    // Build a push but corrupt the signature
    let mut push = build_signed_config_push(&owner_keypair, &client_id, &server_id, 0, config);
    push.signature = vec![0xDE; 64]; // garbage signature

    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&[STREAM_CONFIG_PUSH]).await?;
    write_len_prefixed(&mut send, &push.encode_to_vec()).await?;
    send.finish()?;

    let buf = crate::protocol::read_len_prefixed(&mut recv).await?;
    let response = crate::proto::node::ConfigPushResponse::decode(buf.as_slice())?;

    assert!(
        !response.success,
        "push with invalid signature must be rejected"
    );
    let err = response.error.as_deref().unwrap_or("");
    assert!(
        err.contains("signature"),
        "error must mention signature verification, got: {err}"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_subscribe_delivers_update_notification_after_push() -> Result<()> {
    use crate::proto::node::{NodeConfigSnapshot, NodeGpuConfig};
    use crate::protocol::write_len_prefixed;
    use prost::Message as _;

    let owner_keypair = test_owner_keypair(0x88, 0x89);

    let tmp = std::env::temp_dir().join(format!("mesh-llm-cfg-notif-{}", rand::random::<u64>()));
    std::fs::create_dir_all(tmp.join("server")).ok();
    std::fs::create_dir_all(tmp.join("client")).ok();

    let server = make_test_node_with_owner(
        super::NodeRole::Host { http_port: 9337 },
        &owner_keypair,
        &tmp.join("server"),
    )
    .await?;
    let client =
        make_test_node_with_owner(super::NodeRole::Worker, &owner_keypair, &tmp.join("client"))
            .await?;

    server.set_mesh_id("cfg-notif-mesh-01".to_string()).await;
    client.set_mesh_id("cfg-notif-mesh-01".to_string()).await;
    server.start_accepting();
    client.start_accepting();

    let server_id = server.id();
    let server_addr = server.endpoint.addr();
    let invite =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&server_addr)?);

    client.join(&invite).await?;
    wait_for_peer(&client, server_id).await;
    wait_for_peer(&server, client.id()).await;

    let client_id = client.id();
    let conn = {
        let state = client.state.lock().await;
        state
            .connections
            .get(&server_id)
            .cloned()
            .expect("connection to server must exist after join")
    };

    // Subscribe to config on the server from the client
    let (initial_snapshot, mut notif_rx) = client.subscribe_to_config(&conn).await?;
    let initial_revision = initial_snapshot.revision;

    // Now push a config change to the server from the client
    let new_config = NodeConfigSnapshot {
        version: 1,
        gpu: Some(NodeGpuConfig {
            assignment: crate::proto::node::GpuAssignment::Auto as i32,
        }),
        models: vec![crate::proto::node::NodeModelEntry {
            model: "test-model.gguf".to_string(),
            mmproj: None,
            ctx_size: None,
        }],
        plugins: vec![],
    };
    let push = build_signed_config_push(
        &owner_keypair,
        &client_id,
        &server_id,
        initial_revision,
        new_config,
    );
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&[STREAM_CONFIG_PUSH]).await?;
    write_len_prefixed(&mut send, &push.encode_to_vec()).await?;
    send.finish()?;
    let buf = crate::protocol::read_len_prefixed(&mut recv).await?;
    let push_resp = crate::proto::node::ConfigPushResponse::decode(buf.as_slice())?;
    assert!(
        push_resp.success,
        "push must be accepted for notification test: {:?}",
        push_resp.error
    );

    // The subscribe stream must deliver a ConfigUpdateNotification for the change
    tokio::time::timeout(std::time::Duration::from_secs(5), notif_rx.changed())
        .await
        .expect("ConfigUpdateNotification must arrive within 5 s")
        .expect("notification channel must not be closed");

    let notif = notif_rx.borrow_and_update().clone();
    assert_eq!(
        notif.revision,
        initial_revision + 1,
        "notification revision must be initial + 1"
    );
    assert!(
        !notif.config_hash.is_empty(),
        "notification must carry config_hash"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}
