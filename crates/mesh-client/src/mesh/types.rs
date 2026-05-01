use crate::models::capabilities::ModelCapabilities;
use crate::models::topology::ModelTopology;
use iroh::{EndpointAddr, EndpointId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    pub last_active: u64,
    pub request_count: u64,
}

pub const DEMAND_TTL_SECS: u64 = 86400;

pub const MAX_SPLIT_RTT_MS: u32 = 80;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Catalog,
    HuggingFace,
    LocalGguf,
    DirectUrl,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ServedModelIdentity {
    pub model_name: String,
    pub is_primary: bool,
    pub source_kind: ModelSourceKind,
    pub canonical_ref: Option<String>,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub artifact: Option<String>,
    pub local_file_name: Option<String>,
    pub identity_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ServedModelDescriptor {
    pub identity: ServedModelIdentity,
    pub capabilities: ModelCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<ModelTopology>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelRuntimeDescriptor {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    pub ready: bool,
}

impl ModelRuntimeDescriptor {
    pub fn advertised_context_length(&self) -> Option<u32> {
        self.ready.then_some(self.context_length).flatten()
    }
}

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
    pub moe_recovered_at: Option<std::time::Instant>,
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

    pub fn moe_recovery_ready(&self) -> bool {
        self.moe_recovered_at
            .map(|t| t.elapsed().as_secs() >= 30)
            .unwrap_or(true)
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

pub fn infer_served_model_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
    primary_model_path: Option<&std::path::Path>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source
        .and_then(identity_from_model_source)
        .or_else(|| {
            primary_model_path.and_then(|path| identity_from_local_path(primary_model_name, path))
        });
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if idx == 0 || model_name == primary_model_name {
                let mut id = primary.clone().unwrap_or_default();
                id.model_name = model_name.clone();
                id.is_primary = true;
                if id.local_file_name.is_none() {
                    id.local_file_name = Some(format!("{model_name}.gguf"));
                }
                id
            } else {
                ServedModelIdentity {
                    model_name: model_name.clone(),
                    is_primary: false,
                    source_kind: ModelSourceKind::Unknown,
                    local_file_name: Some(format!("{model_name}.gguf")),
                    ..Default::default()
                }
            };
            ServedModelDescriptor {
                identity,
                capabilities: ModelCapabilities::default(),
                topology: None,
            }
        })
        .collect()
}

pub fn infer_available_model_descriptors(
    _available_models: &[String],
) -> Vec<ServedModelDescriptor> {
    Vec::new()
}

pub fn infer_local_served_model_descriptor(
    _model_name: &str,
    _is_primary: bool,
) -> Option<ServedModelDescriptor> {
    None
}

fn identity_from_local_path(
    model_name: &str,
    path: &std::path::Path,
) -> Option<ServedModelIdentity> {
    let local_file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .or_else(|| Some(format!("{model_name}.gguf")));
    Some(ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        local_file_name,
        ..Default::default()
    })
}

fn identity_from_model_source(source: &str) -> Option<ServedModelIdentity> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((repo_id, revision, file)) = parse_hf_resolve_url_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
        });
    }

    if let Some((repo_id, revision, file)) = parse_hf_ref_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
        });
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::DirectUrl,
            canonical_ref: Some(trimmed.to_string()),
            repository: None,
            revision: None,
            artifact: None,
            local_file_name: trimmed.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(trimmed)),
        });
    }

    if trimmed.ends_with(".gguf")
        || trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || (trimmed.contains('/') && !trimmed.ends_with('/') && trimmed.split('/').count() != 2)
    {
        let local_file_name = std::path::Path::new(trimmed)
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_string);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::LocalGguf,
            canonical_ref: None,
            repository: None,
            revision: None,
            artifact: None,
            local_file_name,
            identity_hash: None,
        });
    }

    Some(ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::Catalog,
        canonical_ref: Some(trimmed.to_string()),
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: None,
        identity_hash: Some(identity_hash_for(&format!("catalog:{trimmed}"))),
    })
}

fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, rev)) => (repo, Some(rev.to_string())),
        None => (parts[1], None),
    };
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

fn parse_hf_resolve_url_parts(url: &str) -> Option<(String, Option<String>, String)> {
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = path.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    let canonical = format!("{repo}@{revision}/{file}");
    parse_hf_ref_parts(&canonical)
}

fn format_hf_canonical_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(rev) => format!("{repo}@{rev}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}
