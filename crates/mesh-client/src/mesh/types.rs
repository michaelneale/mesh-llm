use iroh::{EndpointAddr, EndpointId};
pub use mesh_llm_types::mesh::{
    infer_available_model_descriptors, infer_local_served_model_descriptor,
    infer_served_model_descriptors, merge_demand, ModelDemand, ModelRuntimeDescriptor,
    ModelSourceKind, ServedModelDescriptor, ServedModelIdentity, DEMAND_TTL_SECS, MAX_SPLIT_RTT_MS,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum NodeRole {
    #[default]
    Worker,
    Host {
        http_port: u16,
    },
    Client,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub tunnel_port: Option<u16>,
    pub role: NodeRole,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub rtt_ms: Option<u32>,
    pub model_source: Option<String>,
    pub serving_models: Vec<String>,
    pub hosted_models: Vec<String>,
    pub hosted_models_known: bool,
    pub available_models: Vec<String>,
    pub requested_models: Vec<String>,
    pub last_seen: std::time::Instant,
    pub version: Option<String>,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_bandwidth_gbps: Option<String>,
    pub available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_id: Option<String>,
}

impl PeerInfo {
    pub fn is_assigned_model(&self, model: &str) -> bool {
        self.serving_models.iter().any(|m| m == model)
    }

    pub fn routable_models(&self) -> Vec<String> {
        if self.hosted_models_known {
            self.hosted_models.clone()
        } else {
            self.serving_models.clone()
        }
    }

    pub fn routes_model(&self, model: &str) -> bool {
        if self.hosted_models_known {
            self.hosted_models.iter().any(|m| m == model)
        } else {
            self.is_assigned_model(model)
        }
    }

    pub fn advertised_context_length(&self, model: &str) -> Option<u32> {
        self.served_model_runtime
            .iter()
            .find(|r| r.model_name == model)
            .and_then(ModelRuntimeDescriptor::advertised_context_length)
    }
}

#[derive(Debug, Clone)]
pub struct PeerAnnouncement {
    pub addr: EndpointAddr,
    pub role: NodeRole,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub model_source: Option<String>,
    pub serving_models: Vec<String>,
    pub hosted_models: Option<Vec<String>>,
    pub available_models: Vec<String>,
    pub requested_models: Vec<String>,
    pub version: Option<String>,
    pub model_demand: HashMap<String, ModelDemand>,
    pub mesh_id: Option<String>,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_bandwidth_gbps: Option<String>,
    pub available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_id: Option<String>,
}

pub fn should_be_host_for_model(my_id: EndpointId, my_vram: u64, model_peers: &[PeerInfo]) -> bool {
    for peer in model_peers {
        if matches!(peer.role, NodeRole::Client) {
            continue;
        }
        if peer.vram_bytes > my_vram {
            return false;
        }
        if peer.vram_bytes == my_vram && peer.id > my_id {
            return false;
        }
    }
    true
}
