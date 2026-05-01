use iroh::{EndpointId, SecretKey};
use mesh_client::mesh::{
    infer_available_model_descriptors, infer_local_served_model_descriptor,
    infer_served_model_descriptors, merge_demand, should_be_host_for_model, ModelDemand,
    ModelRuntimeDescriptor, ModelSourceKind, NodeRole, PeerInfo, ServedModelDescriptor,
    ServedModelIdentity,
};
use std::collections::HashMap;

#[test]
fn model_demand_default_is_zero() {
    let d = ModelDemand::default();
    assert_eq!(d.last_active, 0);
    assert_eq!(d.request_count, 0);
}

#[test]
fn model_source_kind_default_is_unknown() {
    let k = ModelSourceKind::default();
    assert!(matches!(k, ModelSourceKind::Unknown));
}

#[test]
fn served_model_identity_default() {
    let id = ServedModelIdentity::default();
    assert!(id.model_name.is_empty());
    assert!(!id.is_primary);
}

#[test]
fn model_runtime_descriptor_not_ready_by_default() {
    let d = ModelRuntimeDescriptor {
        model_name: "Qwen3".to_string(),
        identity_hash: None,
        context_length: None,
        ready: false,
    };
    assert!(d.advertised_context_length().is_none());
}

#[test]
fn model_runtime_descriptor_ready_returns_context_length() {
    let d = ModelRuntimeDescriptor {
        model_name: "Qwen3".to_string(),
        identity_hash: None,
        context_length: Some(4096),
        ready: true,
    };
    assert_eq!(d.advertised_context_length(), Some(4096));
}

#[test]
fn node_role_default_is_worker() {
    let r = NodeRole::default();
    assert!(matches!(r, NodeRole::Worker));
}

#[test]
fn served_model_descriptor_default_capabilities() {
    let d = ServedModelDescriptor::default();
    assert!(!d.capabilities.multimodal);
    assert!(!d.capabilities.moe);
    assert!(d.topology.is_none());
}

#[test]
fn merge_demand_takes_max() {
    let mut ours: HashMap<String, ModelDemand> = HashMap::new();
    ours.insert(
        "model-a".to_string(),
        ModelDemand {
            last_active: 100,
            request_count: 5,
        },
    );

    let mut theirs: HashMap<String, ModelDemand> = HashMap::new();
    theirs.insert(
        "model-a".to_string(),
        ModelDemand {
            last_active: 200,
            request_count: 3,
        },
    );
    theirs.insert(
        "model-b".to_string(),
        ModelDemand {
            last_active: 50,
            request_count: 1,
        },
    );

    merge_demand(&mut ours, &theirs);

    let a = ours.get("model-a").unwrap();
    assert_eq!(a.last_active, 200);
    assert_eq!(a.request_count, 5);

    let b = ours.get("model-b").unwrap();
    assert_eq!(b.last_active, 50);
    assert_eq!(b.request_count, 1);
}

#[test]
fn infer_served_model_descriptors_from_hf_source() {
    let descriptors = infer_served_model_descriptors(
        "Qwen3-8B-Q4_K_M",
        &["Qwen3-8B-Q4_K_M".to_string()],
        Some("Qwen/Qwen3-8B-GGUF@main/Qwen3-8B-Q4_K_M.gguf"),
        None,
    );
    assert_eq!(descriptors.len(), 1);
    let d = &descriptors[0];
    assert_eq!(d.identity.model_name, "Qwen3-8B-Q4_K_M");
    assert!(d.identity.is_primary);
    assert!(matches!(
        d.identity.source_kind,
        ModelSourceKind::HuggingFace
    ));
    assert_eq!(d.identity.repository.as_deref(), Some("Qwen/Qwen3-8B-GGUF"));
}

#[test]
fn infer_served_model_descriptors_from_catalog_source() {
    let descriptors = infer_served_model_descriptors(
        "Qwen3-8B-Q4_K_M",
        &["Qwen3-8B-Q4_K_M".to_string()],
        Some("Qwen3-8B-Q4_K_M"),
        None,
    );
    assert_eq!(descriptors.len(), 1);
    let d = &descriptors[0];
    assert!(matches!(d.identity.source_kind, ModelSourceKind::Catalog));
}

#[test]
fn infer_available_model_descriptors_returns_empty_for_sdk() {
    let descriptors = infer_available_model_descriptors(&["Qwen3-8B-Q4_K_M".to_string()]);
    assert!(descriptors.is_empty());
}

#[test]
fn infer_local_served_model_descriptor_returns_none_for_sdk() {
    let result = infer_local_served_model_descriptor("Qwen3-8B-Q4_K_M", true);
    assert!(result.is_none());
}

#[test]
fn should_be_host_for_model_wins_when_no_peers() {
    let my_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    assert!(should_be_host_for_model(my_id, 48_000_000_000, &[]));
}

#[test]
fn should_be_host_for_model_loses_to_higher_vram_peer() {
    let my_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x02; 32]).public());

    let peers = vec![PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        tunnel_port: None,
        role: NodeRole::Worker,
        models: vec![],
        vram_bytes: 96_000_000_000,
        rtt_ms: None,
        model_source: None,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
        available_models: vec![],
        requested_models: vec![],
        last_seen: std::time::Instant::now(),
        moe_recovered_at: None,
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_bandwidth_gbps: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_id: None,
    }];

    assert!(!should_be_host_for_model(my_id, 48_000_000_000, &peers));
}

#[test]
fn should_be_host_for_model_wins_with_lower_vram_peer() {
    let my_id = EndpointId::from(SecretKey::from_bytes(&[0x01; 32]).public());
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x02; 32]).public());

    let peers = vec![PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        tunnel_port: None,
        role: NodeRole::Worker,
        models: vec![],
        vram_bytes: 24_000_000_000,
        rtt_ms: None,
        model_source: None,
        serving_models: vec![],
        hosted_models: vec![],
        hosted_models_known: false,
        available_models: vec![],
        requested_models: vec![],
        last_seen: std::time::Instant::now(),
        moe_recovered_at: None,
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_bandwidth_gbps: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_id: None,
    }];

    assert!(should_be_host_for_model(my_id, 48_000_000_000, &peers));
}
