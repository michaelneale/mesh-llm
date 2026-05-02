use super::collector::RuntimeDataCollector;
use super::snapshots::{
    LocalInstancesSnapshot, ModelViewSnapshot, PluginDataSnapshot, PluginEndpointsSnapshot,
    RuntimeStatusSnapshot, StatusViewSnapshot,
};
use crate::api::status::{MeshModelPayload, RuntimeStatusPayload, StatusPayload};

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeDataApiViews {
    pub runtime_status: RuntimeStatusSnapshot,
    pub local_instances: LocalInstancesSnapshot,
    pub plugin_data: PluginDataSnapshot,
    pub plugin_endpoints: PluginEndpointsSnapshot,
}

pub(crate) fn collect_views(collector: &RuntimeDataCollector) -> RuntimeDataApiViews {
    RuntimeDataApiViews {
        runtime_status: collector.runtime_status_snapshot(),
        local_instances: collector.local_instances_snapshot(),
        plugin_data: collector.plugin_data_snapshot(),
        plugin_endpoints: collector.plugin_endpoints_snapshot(),
    }
}

pub(crate) fn status_payload(snapshot: StatusViewSnapshot) -> StatusPayload {
    StatusPayload {
        version: snapshot.version,
        latest_version: snapshot.latest_version,
        node_id: snapshot.node_id,
        owner: snapshot.owner,
        token: snapshot.token,
        node_state: snapshot.node_state,
        node_status: snapshot.node_status,
        is_host: snapshot.is_host,
        is_client: snapshot.is_client,
        llama_ready: snapshot.llama_ready,
        runtime: RuntimeStatusPayload {
            backend: None,
            models: vec![],
            stages: vec![],
        },
        model_name: snapshot.model_name,
        models: snapshot.models,
        available_models: snapshot.available_models,
        requested_models: snapshot.requested_models,
        wanted_model_refs: vec![],
        serving_models: snapshot.serving_models,
        hosted_models: snapshot.hosted_models,
        draft_name: snapshot.draft_name,
        api_port: snapshot.api_port,
        my_vram_gb: snapshot.hardware.my_vram_gb,
        model_size_gb: snapshot.hardware.model_size_gb,
        peers: snapshot.peers,
        wakeable_nodes: snapshot.wakeable_nodes,
        local_instances: snapshot.local_instances,
        launch_pi: snapshot.launch_pi,
        launch_goose: snapshot.launch_goose,
        inflight_requests: snapshot.inflight_requests,
        mesh_id: snapshot.mesh_id,
        mesh_name: snapshot.mesh_name,
        nostr_discovery: snapshot.nostr_discovery,
        publication_state: snapshot.publication_state,
        my_hostname: snapshot.hardware.my_hostname,
        my_is_soc: snapshot.hardware.my_is_soc,
        gpus: snapshot.hardware.gpus,
        routing_affinity: snapshot.routing_affinity,
        routing_metrics: snapshot.routing_metrics,
        first_joined_mesh_ts: snapshot.hardware.first_joined_mesh_ts,
    }
}

pub(crate) fn mesh_models(snapshot: ModelViewSnapshot) -> Vec<MeshModelPayload> {
    snapshot.models
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::status::{
        build_gpus, build_ownership_payload, LocalInstance, NodeState, StatusPayload,
    };
    use crate::crypto::OwnershipSummary;
    use crate::mesh::MeshCatalogEntry;
    use crate::models::LocalModelInventorySnapshot;
    use crate::runtime::instance::LocalInstanceSnapshot;
    use crate::runtime_data::collector::RuntimeDataCollector;
    use crate::runtime_data::snapshots::{HardwareViewInput, ModelViewInput, StatusViewInput};
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    #[test]
    fn runtime_data_status_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        collector.replace_local_instances_snapshot(vec![LocalInstanceSnapshot {
            pid: 111,
            api_port: Some(3131),
            version: Some("0.65.0-test".into()),
            started_at_unix: 456,
            runtime_dir: PathBuf::from("/tmp/runtime-1"),
            is_self: true,
        }]);

        let hardware = collector.build_hardware_view(HardwareViewInput {
            gpu_name: Some("RTX 4090".into()),
            gpu_vram: Some("25769803776".into()),
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            first_joined_mesh_ts: Some(123),
        });
        let snapshot = collector.build_status_view(StatusViewInput {
            version: "0.65.0-test".into(),
            latest_version: Some("0.66.0".into()),
            node_id: "node-1".into(),
            owner: OwnershipSummary::default(),
            token: "invite-token".into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            nostr_discovery: true,
            publication_state: "public".into(),
            local_processes: vec![],
            peers: vec![],
            wakeable_nodes: vec![],
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            hardware,
        });

        let payload = status_payload(snapshot);
        let expected = StatusPayload {
            version: "0.65.0-test".into(),
            latest_version: Some("0.66.0".into()),
            node_id: "node-1".into(),
            owner: build_ownership_payload(&OwnershipSummary::default()),
            token: "invite-token".into(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            runtime: RuntimeStatusPayload {
                backend: None,
                models: vec![],
                stages: vec![],
            },
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![LocalInstance {
                pid: 111,
                api_port: Some(3131),
                version: Some("0.65.0-test".into()),
                started_at_unix: 456,
                runtime_dir: "/tmp/runtime-1".into(),
                is_self: true,
            }],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            nostr_discovery: true,
            publication_state: "public".into(),
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            gpus: build_gpus(
                Some("RTX 4090"),
                Some("25769803776"),
                None,
                None,
                None,
                None,
            ),
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            routing_metrics: crate::network::metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: Some(123),
        };

        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }

    #[test]
    fn runtime_data_model_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        let local_inventory = LocalModelInventorySnapshot {
            model_names: HashSet::from(["Example-Model".to_string()]),
            size_by_name: HashMap::from([("Example-Model".to_string(), 8_000_000_000)]),
            metadata_by_name: HashMap::new(),
        };
        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: "Example-Model".into(),
                descriptor: None,
            }],
            served_models: vec![],
            active_demand: HashMap::new(),
            my_serving_models: vec![],
            my_hosted_models: vec![],
            local_inventory,
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: "Another-Model".into(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, "Example-Model");
        assert_eq!(payload[0].status, "cold");
        assert_eq!(payload[0].size_gb, 8.0);
        assert_eq!(
            payload[0].download_command,
            "mesh-llm models download Example-Model"
        );
        assert_eq!(
            payload[0].run_command,
            "mesh-llm serve --model Example-Model"
        );
        assert_eq!(
            payload[0].auto_command,
            "mesh-llm serve --auto --model Example-Model"
        );
        assert_eq!(payload[0].fit_label, "Likely comfortable");
    }
}
