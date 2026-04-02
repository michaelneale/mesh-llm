//! Peer types and gossip helpers for the mesh.
//!
//! Contains the core data types that describe mesh peers and their
//! announcements, plus helpers used by the Node implementation when
//! processing gossip frames and plugin messages.

use std::collections::HashMap;

use iroh::{EndpointAddr, EndpointId};
use serde::{Deserialize, Serialize};

/// Demand signal for a model — tracks interest via API requests and --model declarations.
/// Gossiped across the mesh and merged via max(). Decays naturally when last_active gets old.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    /// Unix timestamp of the most recent request or declaration.
    pub last_active: u64,
    /// Total requests seen (merged across peers via max).
    pub request_count: u64,
}

/// How long a demand entry stays relevant without being refreshed.
pub const DEMAND_TTL_SECS: u64 = 86400; // 24 hours

/// Maximum RTT (ms) for a peer to be included in split mode.
/// Peers above this threshold are skipped during election.
/// Used by both the election RTT gate and the RTT-improvement re-election trigger.
pub const MAX_SPLIT_RTT_MS: u32 = 80;

pub(super) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) fn endpoint_id_hex(id: EndpointId) -> String {
    hex::encode(id.as_bytes())
}

pub(super) fn new_plugin_message_id(source_peer_id: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{source_peer_id}:{nanos}:{}", rand::random::<u64>())
}

fn node_role_label(role: &NodeRole) -> String {
    match role {
        NodeRole::Worker => "worker".into(),
        NodeRole::Host { .. } => "host".into(),
        NodeRole::Client => "client".into(),
    }
}

pub(super) fn peer_info_to_mesh_peer(peer: &PeerInfo) -> crate::plugin::proto::MeshPeer {
    crate::plugin::proto::MeshPeer {
        peer_id: endpoint_id_hex(peer.id),
        version: peer.version.clone().unwrap_or_default(),
        capabilities: Vec::new(),
        role: node_role_label(&peer.role),
        vram_bytes: peer.vram_bytes,
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        available_models: peer.available_models.clone(),
        requested_models: peer.requested_models.clone(),
        rtt_ms: peer.rtt_ms,
        model_source: peer.model_source.clone().unwrap_or_default(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: Some(peer.hosted_models_known),
    }
}

pub(super) fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.role != new.role
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.version != new.version
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PeerAnnouncementV0 {
    pub(super) addr: EndpointAddr,
    #[serde(default)]
    role: NodeRole,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    vram_bytes: u64,
    #[serde(default)]
    model_source: Option<String>,
    #[serde(default)]
    pub(super) serving: Option<String>,
    #[serde(default)]
    serving_models: Vec<String>,
    #[serde(default)]
    available_models: Vec<String>,
    #[serde(default)]
    requested_models: Vec<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    model_demand: HashMap<String, ModelDemand>,
    #[serde(default)]
    mesh_id: Option<String>,
    #[serde(default)]
    gpu_name: Option<String>,
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    is_soc: Option<bool>,
    #[serde(default)]
    gpu_vram: Option<String>,
    #[serde(default)]
    gpu_bandwidth_gbps: Option<String>,
    #[serde(default)]
    available_model_sizes: HashMap<String, u64>,
}

impl PeerAnnouncementV0 {
    pub(crate) fn into_internal(self) -> PeerAnnouncement {
        let serving_models = if !self.serving_models.is_empty() {
            self.serving_models.clone()
        } else {
            self.serving.clone().into_iter().collect()
        };
        PeerAnnouncement {
            addr: self.addr,
            role: self.role,
            models: self.models,
            vram_bytes: self.vram_bytes,
            model_source: self.model_source,
            serving_models,
            hosted_models: None,
            available_models: self.available_models,
            requested_models: self.requested_models,
            version: self.version,
            model_demand: self.model_demand,
            mesh_id: self.mesh_id,
            gpu_name: self.gpu_name,
            hostname: self.hostname,
            is_soc: self.is_soc,
            gpu_vram: self.gpu_vram,
            gpu_bandwidth_gbps: self.gpu_bandwidth_gbps,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: self.available_model_sizes,
        }
    }
}

impl From<&PeerAnnouncement> for PeerAnnouncementV0 {
    fn from(ann: &PeerAnnouncement) -> Self {
        Self {
            addr: ann.addr.clone(),
            role: ann.role.clone(),
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            model_source: ann.model_source.clone(),
            serving: ann.serving_models.first().cloned(),
            serving_models: ann.serving_models.clone(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
            version: ann.version.clone(),
            model_demand: ann.model_demand.clone(),
            mesh_id: ann.mesh_id.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_bandwidth_gbps: ann.gpu_bandwidth_gbps.clone(),
            available_model_sizes: ann.available_model_sizes.clone(),
        }
    }
}

pub(super) fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
    existing.vram_bytes = ann.vram_bytes;
    // Only advance addr if the transitive announcement is at least as path-rich,
    // so a direct peer's richer address is not overwritten by a weaker transitive one.
    if !addr.addrs.is_empty() && addr.addrs.len() >= existing.addr.addrs.len() {
        existing.addr = addr.clone();
    }
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    if ann.gpu_name.is_some() {
        existing.gpu_name = ann.gpu_name.clone();
    }
    if ann.hostname.is_some() {
        existing.hostname = ann.hostname.clone();
    }
    if ann.is_soc.is_some() {
        existing.is_soc = ann.is_soc;
    }
    if ann.gpu_vram.is_some() {
        existing.gpu_vram = ann.gpu_vram.clone();
    }
    if ann.gpu_bandwidth_gbps.is_some() {
        existing.gpu_bandwidth_gbps = ann.gpu_bandwidth_gbps.clone();
    }
    existing.models = ann.models.clone();
    existing.available_models = ann.available_models.clone();
    existing.requested_models = ann.requested_models.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    // Guard: only update when non-empty — old nodes omit these proto fields.
    if !ann.available_model_metadata.is_empty() {
        existing.available_model_metadata = ann.available_model_metadata.clone();
    }
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    if !ann.available_model_sizes.is_empty() {
        existing.available_model_sizes = ann.available_model_sizes.clone();
    }
    serving_changed
}

/// Merge two demand maps. For each model, take max of last_active and request_count.
pub fn merge_demand(
    ours: &mut HashMap<String, ModelDemand>,
    theirs: &HashMap<String, ModelDemand>,
) {
    for (model, their_demand) in theirs {
        let entry = ours.entry(model.clone()).or_default();
        entry.last_active = entry.last_active.max(their_demand.last_active);
        entry.request_count = entry.request_count.max(their_demand.request_count);
    }
}

/// Role a node plays in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeRole {
    /// Provides GPU compute via rpc-server for a specific model.
    Worker,
    /// Runs llama-server for a specific model, orchestrates inference, provides HTTP API.
    Host { http_port: u16 },
    /// Lite client — no compute, accesses the API via tunnel.
    Client,
}

impl Default for NodeRole {
    fn default() -> Self {
        NodeRole::Worker
    }
}

/// Gossip payload — extends EndpointAddr with role metadata.
/// Internal mesh gossip model. Legacy JSON v0 is adapted at the boundary.
#[derive(Debug, Clone)]
pub(crate) struct PeerAnnouncement {
    pub(crate) addr: EndpointAddr,
    pub(crate) role: NodeRole,
    pub(crate) models: Vec<String>,
    pub(crate) vram_bytes: u64,
    pub(crate) model_source: Option<String>,
    pub(crate) serving_models: Vec<String>,
    pub(crate) hosted_models: Option<Vec<String>>,
    /// All GGUF filenames on disk in managed or legacy local storage (for mesh catalog)
    pub(crate) available_models: Vec<String>,
    pub(crate) requested_models: Vec<String>,
    pub(crate) version: Option<String>,
    pub(crate) model_demand: HashMap<String, ModelDemand>,
    pub(crate) mesh_id: Option<String>,
    pub(crate) gpu_name: Option<String>,
    pub(crate) hostname: Option<String>,
    pub(crate) is_soc: Option<bool>,
    pub(crate) gpu_vram: Option<String>,
    pub(crate) gpu_bandwidth_gbps: Option<String>,
    pub(crate) available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub(crate) experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub(crate) available_model_sizes: HashMap<String, u64>,
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
    /// All models assigned to this peer, even if not yet healthy.
    pub serving_models: Vec<String>,
    /// Models this node is actively routing inference for.
    pub hosted_models: Vec<String>,
    /// True when this peer explicitly advertised `hosted_models`.
    pub hosted_models_known: bool,
    /// All GGUFs on disk
    pub available_models: Vec<String>,
    /// Models this node has requested the mesh to serve
    pub requested_models: Vec<String>,
    /// Last time we directly communicated with this peer (gossip, heartbeat, tunnel).
    /// Peers not seen in PEER_STALE_SECS are pruned from gossip and eventually removed.
    pub last_seen: std::time::Instant,
    /// mesh-llm version (e.g. "0.23.0")
    pub version: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    pub gpu_name: Option<String>,
    /// Hostname of the node
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_bandwidth_gbps: Option<String>,
    pub available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
}

impl PeerInfo {
    pub(super) fn from_announcement(
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
    ) -> Self {
        Self {
            id,
            addr,
            tunnel_port: None,
            role: ann.role.clone(),
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            rtt_ms: None,
            model_source: ann.model_source.clone(),
            serving_models: ann.serving_models.clone(),
            hosted_models: ann.hosted_models.clone().unwrap_or_default(),
            hosted_models_known: ann.hosted_models.is_some(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
            last_seen: std::time::Instant::now(),
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_bandwidth_gbps: ann.gpu_bandwidth_gbps.clone(),
            available_model_metadata: ann.available_model_metadata.clone(),
            experts_summary: ann.experts_summary.clone(),
            available_model_sizes: ann.available_model_sizes.clone(),
        }
    }

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
}

/// Peers not directly verified within this window are considered stale
/// and excluded from gossip propagation. After 2x this duration they're removed entirely.
pub(super) const PEER_STALE_SECS: u64 = 180; // 3 minutes
