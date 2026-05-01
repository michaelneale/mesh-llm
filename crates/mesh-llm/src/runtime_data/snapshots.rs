use super::plugins::PluginDataValue;
use super::processes::RuntimeProcessSnapshot;
use crate::api::status::{
    GpuEntry, LocalInstance, MeshModelPayload, NodeState, OwnershipPayload, PeerPayload,
    WakeableNode,
};
use crate::api::RuntimeProcessPayload;
use crate::crypto::OwnershipSummary;
use crate::mesh::{MeshCatalogEntry, ModelDemand, PeerInfo};
use crate::models::LocalModelInventorySnapshot;
use crate::network::metrics::RoutingCollectorSnapshot;
use crate::network::{affinity, metrics};
use crate::plugin::PluginEndpointSummary;
use crate::runtime::instance::LocalInstanceSnapshot;
use crate::runtime::wakeable::WakeableInventoryEntry;
use std::collections::{BTreeMap, HashMap};

use super::metrics::RuntimeLlamaRuntimeSnapshot;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PluginDataKey {
    pub plugin_name: String,
    pub data_key: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PluginEndpointKey {
    pub plugin_name: String,
    pub endpoint_id: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeStatusSnapshot {
    pub primary_model: Option<String>,
    pub primary_backend: Option<String>,
    pub is_host: bool,
    pub is_client: bool,
    pub llama_ready: bool,
    pub llama_port: Option<u16>,
    pub local_processes: Vec<RuntimeProcessSnapshot>,
    pub llama_runtime: RuntimeLlamaRuntimeSnapshot,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LocalInstancesSnapshot {
    pub instances: Vec<LocalInstanceSnapshot>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PluginDataSnapshot {
    pub entries: BTreeMap<PluginDataKey, PluginDataValue>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PluginEndpointsSnapshot {
    pub entries: BTreeMap<PluginEndpointKey, PluginEndpointSummary>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeDataSnapshots {
    pub runtime_status: RuntimeStatusSnapshot,
    pub routing: RoutingCollectorSnapshot,
    pub local_instances: LocalInstancesSnapshot,
    pub local_inventory: LocalModelInventorySnapshot,
    pub plugin_data: PluginDataSnapshot,
    pub plugin_endpoints: PluginEndpointsSnapshot,
}

#[derive(Clone, Debug)]
pub(crate) struct HardwareViewInput {
    pub gpu_name: Option<String>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    pub gpu_mem_bandwidth_gbps: Option<String>,
    pub gpu_compute_tflops_fp32: Option<String>,
    pub gpu_compute_tflops_fp16: Option<String>,
    pub my_hostname: Option<String>,
    pub my_is_soc: Option<bool>,
    pub my_vram_gb: f64,
    pub model_size_gb: f64,
    pub first_joined_mesh_ts: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct HardwareViewSnapshot {
    pub my_hostname: Option<String>,
    pub my_is_soc: Option<bool>,
    pub my_vram_gb: f64,
    pub model_size_gb: f64,
    pub gpus: Vec<GpuEntry>,
    pub first_joined_mesh_ts: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct StatusViewInput {
    pub version: String,
    pub latest_version: Option<String>,
    pub node_id: String,
    pub owner: OwnershipSummary,
    pub token: String,
    pub is_host: bool,
    pub is_client: bool,
    pub llama_ready: bool,
    pub model_name: String,
    pub models: Vec<String>,
    pub available_models: Vec<String>,
    pub requested_models: Vec<String>,
    pub serving_models: Vec<String>,
    pub hosted_models: Vec<String>,
    pub draft_name: Option<String>,
    pub api_port: u16,
    pub inflight_requests: u64,
    pub mesh_id: Option<String>,
    pub mesh_name: Option<String>,
    pub nostr_discovery: bool,
    pub publication_state: String,
    pub local_processes: Vec<RuntimeProcessPayload>,
    pub peers: Vec<PeerInfo>,
    pub wakeable_nodes: Vec<WakeableInventoryEntry>,
    pub routing_affinity: affinity::AffinityStatsSnapshot,
    pub hardware: HardwareViewSnapshot,
}

#[derive(Clone, Debug)]
pub(crate) struct StatusViewSnapshot {
    pub version: String,
    pub latest_version: Option<String>,
    pub node_id: String,
    pub owner: OwnershipPayload,
    pub token: String,
    pub node_state: NodeState,
    pub node_status: String,
    pub is_host: bool,
    pub is_client: bool,
    pub llama_ready: bool,
    pub model_name: String,
    pub models: Vec<String>,
    pub available_models: Vec<String>,
    pub requested_models: Vec<String>,
    pub serving_models: Vec<String>,
    pub hosted_models: Vec<String>,
    pub draft_name: Option<String>,
    pub api_port: u16,
    pub peers: Vec<PeerPayload>,
    pub wakeable_nodes: Vec<WakeableNode>,
    pub local_instances: Vec<LocalInstance>,
    pub launch_pi: Option<String>,
    pub launch_goose: Option<String>,
    pub inflight_requests: u64,
    pub mesh_id: Option<String>,
    pub mesh_name: Option<String>,
    pub nostr_discovery: bool,
    pub publication_state: String,
    pub routing_affinity: affinity::AffinityStatsSnapshot,
    pub routing_metrics: metrics::RoutingMetricsStatusSnapshot,
    pub hardware: HardwareViewSnapshot,
}

#[derive(Clone, Debug)]
pub(crate) struct ModelViewInput {
    pub peers: Vec<PeerInfo>,
    pub catalog: Vec<MeshCatalogEntry>,
    pub served_models: Vec<String>,
    pub active_demand: HashMap<String, ModelDemand>,
    pub my_serving_models: Vec<String>,
    pub my_hosted_models: Vec<String>,
    pub local_inventory: LocalModelInventorySnapshot,
    pub node_hostname: Option<String>,
    pub my_vram_gb: f64,
    pub model_name: String,
    pub model_size_bytes: u64,
    pub now_unix_secs: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelViewSnapshot {
    pub models: Vec<MeshModelPayload>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeStatusDerivation {
    pub effective_is_host: bool,
    pub effective_llama_ready: bool,
    pub display_model_name: String,
    pub node_state: NodeState,
    pub node_status: String,
    pub launch_pi: Option<String>,
    pub launch_goose: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ModelRouteStats {
    pub node_count: usize,
    pub active_nodes: Vec<String>,
    pub mesh_vram_gb: f64,
}
