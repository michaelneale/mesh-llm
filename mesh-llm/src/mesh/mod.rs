//! Mesh membership via iroh QUIC connections.
//!
//! Control traffic uses one QUIC connection per peer. Bi-streams are multiplexed by first byte:
//! 0x01 = gossip, 0x02 = tunnel (RPC), 0x03 = tunnel map, 0x04 = tunnel (HTTP).

pub use mesh_client::mesh::{
    infer_available_model_descriptors, infer_local_served_model_descriptor,
    infer_served_model_descriptors, merge_demand, ModelDemand, ModelRuntimeDescriptor,
    ModelSourceKind, ServedModelDescriptor, ServedModelIdentity, DEMAND_TTL_SECS, MAX_SPLIT_RTT_MS,
};

use anyhow::{Context, Result};
use base64::Engine;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::{watch, Mutex};

use crate::crypto::{
    default_node_ownership_path, save_node_ownership, sign_node_ownership, verify_node_ownership,
    OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy, TrustStore,
    DEFAULT_NODE_CERT_LIFETIME_SECS,
};
use crate::inference::moe;
use crate::protocol::*;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const MIN_PINNED_GPU_CONFIG_PEER_VERSION: &str = "0.59.0";
pub(super) const PEER_CONNECT_AND_GOSSIP_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(15);

fn config_uses_pinned_gpu(config: &crate::plugin::MeshConfig) -> bool {
    config.gpu.assignment == crate::plugin::GpuAssignment::Pinned
}

fn peer_supports_pinned_gpu_config(peer_version: Option<&str>) -> bool {
    let Ok(min_version) = semver::Version::parse(MIN_PINNED_GPU_CONFIG_PEER_VERSION) else {
        return false;
    };
    let Some(peer_version) = peer_version else {
        return false;
    };
    let Ok(peer_version) = semver::Version::parse(peer_version) else {
        return false;
    };

    peer_version >= min_version
        || (peer_version.major == min_version.major
            && peer_version.minor == min_version.minor
            && peer_version.patch == min_version.patch)
}

fn pinned_gpu_config_peer_error(peer_version: Option<&str>) -> String {
    let advertised = peer_version.unwrap_or("unknown");
    format!(
        "pinned gpu config sync requires mesh-llm >= {MIN_PINNED_GPU_CONFIG_PEER_VERSION}; subscriber advertised {advertised}"
    )
}

fn preflight_pushed_config_for_current_node(config: &crate::plugin::MeshConfig) -> Result<()> {
    let survey = crate::system::hardware::query(&[
        crate::system::hardware::Metric::GpuName,
        crate::system::hardware::Metric::GpuFacts,
    ]);
    preflight_pushed_config_for_current_node_with_gpus(config, &survey.gpus)
}

fn preflight_pushed_config_for_current_node_with_gpus(
    config: &crate::plugin::MeshConfig,
    gpus: &[crate::system::hardware::GpuFacts],
) -> Result<()> {
    if config.gpu.assignment != crate::plugin::GpuAssignment::Pinned {
        return Ok(());
    }

    for model in &config.models {
        let gpu = crate::system::hardware::resolve_pinned_gpu(model.gpu_id.as_deref(), gpus)
            .map_err(anyhow::Error::new)
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            })?;

        let stable_id = gpu
            .stable_id
            .as_deref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "pushed config model '{}' resolved pinned GPU at index {} without a stable_id",
                    model.model,
                    gpu.index
                )
            })
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            })?;

        if gpu.backend_device.is_none() {
            return Err(anyhow::anyhow!(
                "pushed config model '{}' resolved pinned GPU '{}' at index {} without a backend_device",
                model.model,
                stable_id,
                gpu.index
            ))
            .with_context(|| {
                format!(
                    "pushed config model '{}' failed pinned GPU preflight",
                    model.model
                )
            });
        }
    }

    Ok(())
}

fn endpoint_id_hex(id: EndpointId) -> String {
    hex::encode(id.as_bytes())
}

fn new_plugin_message_id(source_peer_id: &str) -> String {
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

fn infer_remote_served_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source.and_then(identity_from_model_source);
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if idx == 0 || model_name == primary_model_name {
                let mut identity = primary
                    .clone()
                    .unwrap_or_else(|| unknown_identity(model_name));
                identity.model_name = model_name.clone();
                identity.is_primary = true;
                if identity.local_file_name.is_none() {
                    identity.local_file_name = Some(format!("{model_name}.gguf"));
                }
                identity
            } else {
                unknown_identity(model_name)
            };
            ServedModelDescriptor {
                identity,
                capabilities: crate::models::ModelCapabilities::default(),
                topology: None,
            }
        })
        .collect()
}

fn unknown_identity(model_name: &str) -> ServedModelIdentity {
    ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::Unknown,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: Some(format!("{model_name}.gguf")),
        identity_hash: None,
    }
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

fn identity_from_model_path(
    model_name: &str,
    path: &std::path::Path,
) -> Option<ServedModelIdentity> {
    if let Some(identity) = crate::models::huggingface_identity_for_path(path) {
        return Some(ServedModelIdentity {
            model_name: model_name.to_string(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(identity.canonical_ref.clone()),
            repository: Some(identity.repo_id),
            revision: Some(identity.revision),
            artifact: Some(identity.file),
            local_file_name: Some(identity.local_file_name),
            identity_hash: Some(identity_hash_for(&identity.canonical_ref)),
        });
    }

    if path.exists() {
        let local_file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_string)
            .or_else(|| Some(format!("{model_name}.gguf")));
        return Some(ServedModelIdentity {
            model_name: model_name.to_string(),
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

    None
}

#[allow(dead_code)]
fn descriptor_from_model_path(
    model_name: &str,
    path: &std::path::Path,
    is_primary: bool,
) -> Option<ServedModelDescriptor> {
    let mut identity = identity_from_model_path(model_name, path)?;
    identity.is_primary = is_primary;
    Some(descriptor_from_identity(model_name, identity))
}

#[allow(dead_code)]
fn descriptor_from_identity(
    model_name: &str,
    mut identity: ServedModelIdentity,
) -> ServedModelDescriptor {
    identity.model_name = model_name.to_string();
    let path = crate::models::find_model_path(model_name);
    let catalog = crate::models::find_catalog_model_exact(model_name);
    let mut topology = crate::models::infer_local_model_topology(&path, catalog);
    if topology.is_none() {
        if let Some(info) = moe::detect_moe(&path) {
            topology = Some(crate::models::ModelTopology {
                moe: Some(crate::models::ModelMoeInfo {
                    expert_count: info.expert_count,
                    used_expert_count: info.expert_used_count,
                    min_experts_per_node: None,
                    source: Some("gguf_header".to_string()),
                    ranking_source: None,
                    ranking_origin: None,
                    ranking: Vec::new(),
                    ranking_prompt_count: None,
                    ranking_tokens: None,
                    ranking_layer_scope: None,
                }),
            });
        }
    }
    enrich_topology_with_local_shared_ranking(path.as_path(), &mut topology);
    let mut capabilities =
        crate::models::capabilities::infer_local_model_capabilities(model_name, &path, catalog);
    capabilities.moe = capabilities.moe
        || topology
            .as_ref()
            .and_then(|value| value.moe.as_ref())
            .is_some();
    ServedModelDescriptor {
        identity,
        capabilities,
        topology,
    }
}

#[allow(dead_code)]
fn enrich_topology_with_local_shared_ranking(
    path: &std::path::Path,
    topology: &mut Option<crate::models::ModelTopology>,
) {
    let Some(moe_info) = topology.as_mut().and_then(|value| value.moe.as_mut()) else {
        return;
    };
    let Some(artifact) = moe::best_shared_ranking_artifact(path) else {
        return;
    };
    moe_info.ranking_source = Some(artifact.kind.label().to_string());
    moe_info.ranking_origin = Some(artifact.origin.label().to_string());
    moe_info.ranking = artifact.ranking;
    moe_info.ranking_prompt_count = artifact.micro_prompt_count.map(|value| value as u32);
    moe_info.ranking_tokens = artifact.micro_tokens;
    moe_info.ranking_layer_scope = artifact.micro_layer_scope.map(|scope| match scope {
        moe::MoeMicroLayerScope::All => "all".to_string(),
        moe::MoeMicroLayerScope::First => "first".to_string(),
    });
}

fn identities_match_exact(local: &ServedModelIdentity, remote: &ServedModelIdentity) -> bool {
    if let (Some(local_hash), Some(remote_hash)) =
        (local.identity_hash.as_ref(), remote.identity_hash.as_ref())
    {
        return local_hash == remote_hash;
    }
    if let (Some(local_ref), Some(remote_ref)) =
        (local.canonical_ref.as_ref(), remote.canonical_ref.as_ref())
    {
        return local_ref == remote_ref;
    }
    matches!(
        (
            local.repository.as_ref(),
            local.revision.as_ref(),
            local.artifact.as_ref(),
            remote.repository.as_ref(),
            remote.revision.as_ref(),
            remote.artifact.as_ref(),
        ),
        (
            Some(local_repo),
            Some(local_revision),
            Some(local_artifact),
            Some(remote_repo),
            Some(remote_revision),
            Some(remote_artifact),
        ) if local_repo == remote_repo
            && local_revision == remote_revision
            && local_artifact == remote_artifact
    )
}

fn shared_ranking_from_descriptor(
    descriptor: &ServedModelDescriptor,
) -> Option<moe::SharedRankingArtifact> {
    let moe_info = descriptor.topology.as_ref()?.moe.as_ref()?;
    if moe_info.ranking.is_empty() {
        return None;
    }
    let kind = match moe_info.ranking_source.as_deref()? {
        "analyze" => moe::SharedRankingKind::Analyze,
        "micro-analyze" => moe::SharedRankingKind::MicroAnalyze,
        _ => return None,
    };
    let micro_layer_scope = match moe_info.ranking_layer_scope.as_deref() {
        Some("all") => Some(moe::MoeMicroLayerScope::All),
        Some("first") => Some(moe::MoeMicroLayerScope::First),
        _ => None,
    };
    Some(moe::SharedRankingArtifact {
        kind,
        origin: moe_info
            .ranking_origin
            .as_deref()
            .and_then(moe::SharedRankingOrigin::from_label)
            .unwrap_or(moe::SharedRankingOrigin::LegacyCache),
        ranking: moe_info.ranking.clone(),
        micro_prompt_count: moe_info.ranking_prompt_count.map(|value| value as usize),
        micro_tokens: moe_info.ranking_tokens,
        micro_layer_scope,
    })
}

fn import_remote_moe_rankings(descriptors: &[ServedModelDescriptor]) -> bool {
    let mut imported = false;
    for descriptor in descriptors {
        let Some(remote_artifact) = shared_ranking_from_descriptor(descriptor) else {
            continue;
        };
        let path = crate::models::find_model_path(&descriptor.identity.model_name);
        if !path.exists() {
            continue;
        }
        let Some(local_identity) = identity_from_model_path(&descriptor.identity.model_name, &path)
        else {
            continue;
        };
        if !identities_match_exact(&local_identity, &descriptor.identity) {
            continue;
        }
        let imported_artifact = moe::SharedRankingArtifact {
            origin: moe::SharedRankingOrigin::PeerImport,
            ..remote_artifact
        };
        if moe::cache_shared_ranking_if_stronger(path.as_path(), &imported_artifact)
            .unwrap_or(false)
        {
            imported = true;
        }
    }
    imported
}

fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
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
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn peer_info_to_mesh_peer(peer: &PeerInfo) -> crate::plugin::proto::MeshPeer {
    crate::plugin::proto::MeshPeer {
        peer_id: endpoint_id_hex(peer.id),
        version: peer.version.clone().unwrap_or_default(),
        capabilities: Vec::new(),
        role: node_role_label(&peer.role),
        vram_bytes: peer.vram_bytes,
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        available_models: Vec::new(),
        requested_models: peer.requested_models.clone(),
        rtt_ms: peer.rtt_ms,
        model_source: peer.model_source.clone().unwrap_or_default(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: Some(peer.hosted_models_known),
    }
}

fn policy_accepts_peer(policy: TrustPolicy, owner_summary: &OwnershipSummary) -> bool {
    match policy {
        TrustPolicy::Off | TrustPolicy::PreferOwned => true,
        TrustPolicy::RequireOwned | TrustPolicy::Allowlist => {
            owner_summary.status == OwnershipStatus::Verified
        }
    }
}

fn load_or_refresh_owner_attestation(
    owner_keypair: &crate::crypto::OwnerKeypair,
    endpoint_id: EndpointId,
    node_label: Option<String>,
    hostname_hint: Option<String>,
) -> Result<SignedNodeOwnership> {
    // Always sign a fresh attestation on startup when the owner key is available.
    // This ensures that key rotation is always reflected immediately and no stale
    // certificate can persist across restarts.
    let path = default_node_ownership_path()?;
    let ownership = sign_node_ownership(
        owner_keypair,
        endpoint_id.as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        node_label,
        hostname_hint,
    )?;
    save_node_ownership(&path, &ownership)?;
    Ok(ownership)
}

fn model_identity_score(identity: &ServedModelIdentity) -> u8 {
    let kind_score = match identity.source_kind {
        ModelSourceKind::HuggingFace => 4,
        ModelSourceKind::Catalog => 3,
        ModelSourceKind::DirectUrl => 2,
        ModelSourceKind::LocalGguf => 1,
        ModelSourceKind::Unknown => 0,
    };
    let canonical_bonus = if identity.canonical_ref.is_some() {
        2
    } else {
        0
    };
    let revision_bonus = if identity.revision.is_some() { 1 } else { 0 };
    kind_score + canonical_bonus + revision_bonus
}

fn model_descriptor_score(descriptor: &ServedModelDescriptor) -> u8 {
    let identity = &descriptor.identity;
    let capability_bonus = u8::from(descriptor.capabilities.multimodal)
        + u8::from(descriptor.capabilities.audio != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.vision != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.reasoning != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.tool_use != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.moe)
        + u8::from(
            descriptor
                .topology
                .as_ref()
                .and_then(|value| value.moe.as_ref())
                .is_some(),
        );
    model_identity_score(identity) + capability_bonus
}

fn upsert_mesh_catalog_descriptor(
    descriptors: &mut HashMap<String, ServedModelDescriptor>,
    descriptor: ServedModelDescriptor,
) {
    if descriptor.identity.model_name.is_empty() {
        return;
    }
    match descriptors.get(&descriptor.identity.model_name) {
        Some(existing)
            if model_descriptor_score(existing) >= model_descriptor_score(&descriptor) => {}
        _ => {
            descriptors.insert(descriptor.identity.model_name.clone(), descriptor);
        }
    }
}

/// Merge two demand maps. For each model, take max of last_active and request_count.
/// Role a node plays in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum NodeRole {
    /// Provides GPU compute via rpc-server for a specific model.
    #[default]
    Worker,
    /// Runs llama-server for a specific model, orchestrates inference, provides HTTP API.
    Host { http_port: u16 },
    /// Lite client — no compute, accesses the API via tunnel.
    Client,
}

/// Gossip payload — extends EndpointAddr with role metadata.
/// Internal mesh gossip model. Legacy JSON v0 is adapted at the boundary.
#[derive(Debug, Clone)]
pub(crate) struct PeerAnnouncement {
    pub(crate) addr: EndpointAddr,
    pub(crate) role: NodeRole,
    pub(crate) first_joined_mesh_ts: Option<u64>,
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
    pub(crate) gpu_reserved_bytes: Option<String>,
    pub(crate) gpu_mem_bandwidth_gbps: Option<String>,
    pub(crate) gpu_compute_tflops_fp32: Option<String>,
    pub(crate) gpu_compute_tflops_fp16: Option<String>,
    pub(crate) available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub(crate) experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub(crate) available_model_sizes: HashMap<String, u64>,
    pub(crate) served_model_descriptors: Vec<ServedModelDescriptor>,
    pub(crate) served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub(crate) owner_attestation: Option<SignedNodeOwnership>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub tunnel_port: Option<u16>,
    pub role: NodeRole,
    pub first_joined_mesh_ts: Option<u64>,
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
    /// Only updated by direct bi-directional gossip exchanges, heartbeat probes,
    /// and inbound connections — never by transitive mentions.
    /// Used by PeerDown silencing to require independent proof-of-life.
    pub last_seen: std::time::Instant,
    /// Last time a bridge peer mentioned this peer in gossip.
    /// Updated on every transitive gossip update. Used together with `last_seen`
    /// for pruning and `collect_announcements`: a peer is included/kept as long
    /// as either timestamp is fresh.
    pub last_mentioned: std::time::Instant,
    /// When this peer returned after being considered dead. MoE scale-up should
    /// wait briefly before treating the peer as eligible again.
    pub moe_recovered_at: Option<std::time::Instant>,
    /// mesh-llm version (e.g. "0.23.0")
    pub version: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    pub gpu_name: Option<String>,
    /// Hostname of the node
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    pub gpu_mem_bandwidth_gbps: Option<String>,
    pub gpu_compute_tflops_fp32: Option<String>,
    pub gpu_compute_tflops_fp16: Option<String>,
    pub available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_attestation: Option<SignedNodeOwnership>,
    pub owner_summary: OwnershipSummary,
}

#[derive(Debug)]
pub struct OwnerRuntimeConfig {
    pub keypair: Option<crate::crypto::OwnerKeypair>,
    pub node_label: Option<String>,
    pub trust_store: TrustStore,
    pub trust_policy: TrustPolicy,
}
#[derive(Debug, Clone)]
pub struct MeshCatalogEntry {
    pub model_name: String,
    pub descriptor: Option<ServedModelDescriptor>,
}

impl PeerInfo {
    fn from_announcement(
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) -> Self {
        Self {
            id,
            addr,
            tunnel_port: None,
            role: ann.role.clone(),
            first_joined_mesh_ts: ann.first_joined_mesh_ts,
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
            last_mentioned: std::time::Instant::now(),
            moe_recovered_at: None,
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_reserved_bytes: ann.gpu_reserved_bytes.clone(),
            gpu_mem_bandwidth_gbps: ann.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: ann.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: ann.gpu_compute_tflops_fp16.clone(),
            available_model_metadata: ann.available_model_metadata.clone(),
            experts_summary: ann.experts_summary.clone(),
            available_model_sizes: ann.available_model_sizes.clone(),
            served_model_descriptors: ann.served_model_descriptors.clone(),
            served_model_runtime: ann.served_model_runtime.clone(),
            owner_attestation: ann.owner_attestation.clone(),
            owner_summary,
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

    pub fn accepts_http_inference(&self) -> bool {
        matches!(self.role, NodeRole::Host { .. })
    }

    pub fn http_routable_models(&self) -> Vec<String> {
        if self.accepts_http_inference() {
            self.routable_models()
        } else {
            Vec::new()
        }
    }

    pub fn routes_http_model(&self, model: &str) -> bool {
        self.accepts_http_inference() && self.routes_model(model)
    }

    pub fn moe_recovery_ready(&self) -> bool {
        moe_recovery_ready_at(self.moe_recovered_at, std::time::Instant::now())
    }

    pub fn advertised_context_length(&self, model: &str) -> Option<u32> {
        self.served_model_runtime
            .iter()
            .find(|runtime| runtime.model_name == model)
            .and_then(ModelRuntimeDescriptor::advertised_context_length)
    }
}

/// Peers not directly verified within this window are considered stale
/// and excluded from gossip propagation. After 2x this duration they're removed entirely.
const PEER_STALE_SECS: u64 = 180; // 3 minutes
/// Detect available VRAM. On Apple Silicon, uses ~75% of system RAM
/// (the rest is reserved for OS/apps on unified memory).
/// Detect available memory for model loading, capped by max_vram_gb if set.
/// "VRAM" is a misnomer — on macOS unified memory and Linux CPU-only, this
/// is system RAM. On Linux with a GPU, it's actual GPU VRAM.
pub fn detect_vram_bytes_capped(max_vram_gb: Option<f64>) -> u64 {
    let mut detected = crate::system::hardware::survey().vram_bytes;
    if let Some(cap) = max_vram_gb {
        let cap_bytes = (cap * 1e9) as u64;
        if cap_bytes < detected {
            detected = cap_bytes;
        }
    }
    detected
}

/// Lightweight routing table for passive nodes (clients + standby GPU).
/// Contains just enough info to route requests to the right host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub hosts: Vec<RouteEntry>,
    /// Stable mesh identity — shared by all nodes in the same mesh.
    #[serde(default)]
    pub mesh_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEntry {
    pub model: String,
    pub node_id: String,
    pub endpoint_id: EndpointId,
    pub vram_gb: f64,
}

/// Discover our public IP via STUN, then pair it with the given port.
/// We can't send STUN from the bound port (iroh owns it), but we only need
/// the public IP — the port is known from --bind-port + router forwarding.
async fn stun_public_addr(advertised_port: u16) -> Option<std::net::SocketAddr> {
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

    let stun_servers = [
        "stun.l.google.com:19302",
        "stun.cloudflare.com:3478",
        "stun.stunprotocol.org:3478",
    ];

    // Bind to ephemeral port — we only care about the IP, not the mapped port.
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;

    for server in &stun_servers {
        // STUN Binding Request: type=0x0001, len=0, magic=0x2112A442, txn=random
        let mut req = [0u8; 20];
        req[0] = 0x00;
        req[1] = 0x01; // Binding Request
                       // length = 0
        req[4] = 0x21;
        req[5] = 0x12;
        req[6] = 0xA4;
        req[7] = 0x42; // Magic Cookie
        rand::fill(&mut req[8..20]);

        let dest: SocketAddr = match tokio::net::lookup_host(server).await {
            Ok(mut addrs) => match addrs.next() {
                Some(a) => a,
                None => continue,
            },
            Err(_) => continue,
        };

        if sock.send_to(&req, dest).await.is_err() {
            continue;
        }

        let mut buf = [0u8; 256];
        match tokio::time::timeout(std::time::Duration::from_secs(2), sock.recv_from(&mut buf))
            .await
        {
            Ok(Ok((len, _))) if len >= 20 => {
                // Parse STUN response for XOR-MAPPED-ADDRESS (0x0020)
                // or MAPPED-ADDRESS (0x0001)
                let magic = &req[4..8];
                let _txn = &req[8..20];
                let mut i = 20;
                while i + 4 <= len {
                    let attr_type = u16::from_be_bytes([buf[i], buf[i + 1]]);
                    let attr_len = u16::from_be_bytes([buf[i + 2], buf[i + 3]]) as usize;
                    if i + 4 + attr_len > len {
                        break;
                    }
                    let val = &buf[i + 4..i + 4 + attr_len];

                    if attr_type == 0x0020 && attr_len >= 8 && val[1] == 0x01 {
                        // XOR-MAPPED-ADDRESS, IPv4 — extract IP only
                        let ip = Ipv4Addr::new(
                            val[4] ^ magic[0],
                            val[5] ^ magic[1],
                            val[6] ^ magic[2],
                            val[7] ^ magic[3],
                        );
                        let addr = SocketAddr::V4(SocketAddrV4::new(ip, advertised_port));
                        tracing::info!("STUN discovered public address: {addr}");
                        return Some(addr);
                    }
                    if attr_type == 0x0001 && attr_len >= 8 && val[1] == 0x01 {
                        // MAPPED-ADDRESS, IPv4 — extract IP only
                        let ip = Ipv4Addr::new(val[4], val[5], val[6], val[7]);
                        let addr = SocketAddr::V4(SocketAddrV4::new(ip, advertised_port));
                        tracing::info!("STUN discovered public address: {addr}");
                        return Some(addr);
                    }

                    // Attributes are padded to 4-byte boundary
                    i += (4 + (attr_len + 3)) & !3;
                }
            }
            _ => continue,
        }
    }

    tracing::warn!("STUN: could not discover public address");
    None
}

#[derive(Clone)]
pub struct Node {
    endpoint: Endpoint,
    public_addr: Option<std::net::SocketAddr>,
    state: Arc<Mutex<MeshState>>,
    role: Arc<Mutex<NodeRole>>,
    models: Arc<Mutex<Vec<String>>>,
    model_source: Arc<Mutex<Option<String>>>,
    serving_models: Arc<Mutex<Vec<String>>>,
    served_model_descriptors: Arc<Mutex<Vec<ServedModelDescriptor>>>,
    model_runtime_descriptors: Arc<Mutex<Vec<ModelRuntimeDescriptor>>>,
    hosted_models: Arc<Mutex<Vec<String>>>,
    llama_ready: Arc<Mutex<bool>>,
    available_models: Arc<Mutex<Vec<String>>>,
    requested_models: Arc<Mutex<Vec<String>>>,
    /// Mesh-wide demand map — merged from gossip + local API requests.
    /// This is the single source of truth for "what does the mesh want?"
    model_demand: Arc<std::sync::Mutex<HashMap<String, ModelDemand>>>,
    mesh_id: Arc<Mutex<Option<String>>>,
    first_joined_mesh_ts: Arc<Mutex<Option<u64>>>,
    accepting: Arc<(tokio::sync::Notify, std::sync::atomic::AtomicBool)>,
    vram_bytes: u64,
    peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    inflight_change_tx: watch::Sender<u64>,
    routing_metrics: crate::network::metrics::RoutingMetrics,
    tunnel_tx: tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    tunnel_http_tx:
        tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    plugin_manager: Arc<Mutex<Option<crate::plugin::PluginManager>>>,
    display_name: Arc<Mutex<Option<String>>>,
    owner_attestation: Arc<Mutex<Option<SignedNodeOwnership>>>,
    owner_summary: Arc<Mutex<OwnershipSummary>>,
    trust_store: Arc<Mutex<TrustStore>>,
    trust_policy: TrustPolicy,
    pub enumerate_host: bool,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    pub gpu_mem_bandwidth_gbps: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    pub gpu_compute_tflops_fp32: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    pub gpu_compute_tflops_fp16: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    config_state: Arc<tokio::sync::Mutex<crate::runtime::config_state::ConfigState>>,
    config_revision_tx: Arc<tokio::sync::watch::Sender<u64>>,
}

struct MeshState {
    peers: HashMap<EndpointId, PeerInfo>,
    connections: HashMap<EndpointId, Connection>,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    dead_peers: std::collections::HashSet<EndpointId>,
    seen_plugin_messages: HashMap<String, std::time::Instant>,
    seen_plugin_message_order: VecDeque<(std::time::Instant, String)>,
    /// Last policy-rejection status per peer — used to suppress duplicate log lines.
    /// Only logs when the status transitions (first rejection or status change).
    policy_rejected_peers: HashMap<EndpointId, OwnershipStatus>,
}

/// Returns `true` if the given peer has completed gossip validation and is
/// a full mesh member. Unadmitted peers are in `state.connections` but not
/// in `state.peers` — they are quarantined until gossip succeeds.
#[cfg(test)]
pub(crate) fn is_peer_admitted(peers: &HashMap<EndpointId, PeerInfo>, id: &EndpointId) -> bool {
    peers.contains_key(id)
}

/// Returns `true` if the given stream type is permitted before a peer has
/// been admitted through gossip.
///
/// Only two streams bypass the quarantine gate:
/// - `STREAM_GOSSIP (0x01)`: the admission handshake itself.
/// - `STREAM_ROUTE_REQUEST (0x05)`: passive/client request-only path — caller
///   is NEVER promoted to `state.peers`.
///
/// Every other stream — including tunnel (0x02 / 0x04) — requires the
/// remote to have completed gossip first.
pub(crate) fn stream_allowed_before_admission(stream_type: u8) -> bool {
    stream_type == STREAM_GOSSIP || stream_type == STREAM_ROUTE_REQUEST
}

pub(crate) fn ingest_tunnel_map(
    remote: EndpointId,
    frame: &crate::proto::node::TunnelMap,
    remote_tunnel_maps: &mut HashMap<EndpointId, HashMap<EndpointId, u16>>,
) -> Result<()> {
    if frame.owner_peer_id.as_slice() != remote.as_bytes() {
        anyhow::bail!(
            "TunnelMap owner_peer_id mismatch: frame claims owner {}, but connected peer is {}",
            hex::encode(&frame.owner_peer_id),
            remote.fmt_short()
        );
    }

    let mut tunnel_map: HashMap<EndpointId, u16> = HashMap::new();
    for entry in &frame.entries {
        if entry.target_peer_id.len() != 32 {
            anyhow::bail!(
                "TunnelMap entry has invalid target_peer_id length: {} (expected 32)",
                entry.target_peer_id.len()
            );
        }
        if entry.tunnel_port > u16::MAX as u32 {
            anyhow::bail!(
                "TunnelMap entry has out-of-range tunnel_port: {} (max {})",
                entry.tunnel_port,
                u16::MAX
            );
        }
        let arr: [u8; 32] = entry.target_peer_id.as_slice().try_into().unwrap();
        let eid = EndpointId::from(
            iroh::PublicKey::from_bytes(&arr)
                .map_err(|e| anyhow::anyhow!("Invalid target_peer_id bytes: {e}"))?,
        );
        tunnel_map.insert(eid, entry.tunnel_port as u16);
    }

    remote_tunnel_maps.insert(remote, tunnel_map);
    Ok(())
}

/// Validates the sender-identity rule for a validated `PeerLeaving` frame.
/// Returns `Ok(leaving_id)` if `frame.peer_id == remote` (sender is announcing its own departure).
/// Returns `Err(ForgedSender)` if `frame.peer_id != remote` — no peer should be removed.
pub(crate) fn resolve_peer_leaving(
    remote: EndpointId,
    frame: &crate::proto::node::PeerLeaving,
) -> Result<EndpointId, ControlFrameError> {
    if frame.peer_id.as_slice() != remote.as_bytes() {
        return Err(ControlFrameError::ForgedSender);
    }
    let arr: [u8; 32] =
        frame
            .peer_id
            .as_slice()
            .try_into()
            .map_err(|_| ControlFrameError::InvalidEndpointId {
                got: frame.peer_id.len(),
            })?;
    let pk =
        iroh::PublicKey::from_bytes(&arr).map_err(|_| ControlFrameError::InvalidEndpointId {
            got: frame.peer_id.len(),
        })?;
    Ok(EndpointId::from(pk))
}

/// Channels returned by Node::start for inbound tunnel streams.
pub struct TunnelChannels {
    pub rpc: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    pub http: tokio::sync::mpsc::Receiver<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
}

pub struct InflightRequestGuard {
    inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    inflight_change_tx: watch::Sender<u64>,
}

impl Drop for InflightRequestGuard {
    fn drop(&mut self) {
        let _ = self.inflight_requests.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
        let _ = self.inflight_change_tx.send(
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64,
        );
    }
}

impl Node {
    pub fn begin_inflight_request(&self) -> InflightRequestGuard {
        self.inflight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let current = self
            .inflight_requests
            .load(std::sync::atomic::Ordering::Relaxed) as u64;
        let _ = self.inflight_change_tx.send(current);
        self.routing_metrics.observe_inflight(current);
        InflightRequestGuard {
            inflight_requests: self.inflight_requests.clone(),
            inflight_change_tx: self.inflight_change_tx.clone(),
        }
    }

    pub fn inflight_requests(&self) -> u64 {
        self.inflight_requests
            .load(std::sync::atomic::Ordering::Relaxed) as u64
    }

    pub fn inflight_change_rx(&self) -> watch::Receiver<u64> {
        self.inflight_change_tx.subscribe()
    }

    pub fn record_inference_attempt(
        &self,
        model: Option<&str>,
        target: &crate::inference::election::InferenceTarget,
        queue_wait: std::time::Duration,
        attempt_time: std::time::Duration,
        outcome: crate::network::metrics::AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        let attempt_target = match target {
            crate::inference::election::InferenceTarget::Local(port)
            | crate::inference::election::InferenceTarget::MoeLocal(port) => {
                crate::network::metrics::AttemptTarget::Local(format!("127.0.0.1:{port}"))
            }
            crate::inference::election::InferenceTarget::Remote(peer_id)
            | crate::inference::election::InferenceTarget::MoeRemote(peer_id) => {
                crate::network::metrics::AttemptTarget::Remote(peer_id.fmt_short().to_string())
            }
            crate::inference::election::InferenceTarget::None => return,
        };
        self.routing_metrics.record_attempt(
            model,
            attempt_target,
            queue_wait,
            attempt_time,
            outcome,
            completion_tokens,
        );
    }

    pub fn record_endpoint_attempt(
        &self,
        model: Option<&str>,
        endpoint: &str,
        queue_wait: std::time::Duration,
        attempt_time: std::time::Duration,
        outcome: crate::network::metrics::AttemptOutcome,
        completion_tokens: Option<u64>,
    ) {
        self.routing_metrics.record_attempt(
            model,
            crate::network::metrics::AttemptTarget::Endpoint(endpoint.to_string()),
            queue_wait,
            attempt_time,
            outcome,
            completion_tokens,
        );
    }

    pub fn record_routed_request(
        &self,
        model: Option<&str>,
        attempts: usize,
        outcome: crate::network::metrics::RequestOutcome,
    ) {
        self.routing_metrics
            .record_request(model, attempts, outcome);
    }

    pub fn routing_metrics_snapshot(
        &self,
    ) -> crate::network::metrics::RoutingMetricsStatusSnapshot {
        self.routing_metrics
            .status_snapshot(self.inflight_requests())
    }

    pub fn model_routing_metrics(
        &self,
    ) -> HashMap<String, crate::network::metrics::ModelRoutingMetricsSnapshot> {
        self.routing_metrics.model_snapshots()
    }

    pub async fn owner_summary(&self) -> OwnershipSummary {
        self.owner_summary.lock().await.clone()
    }

    pub async fn start(
        role: NodeRole,
        relay_urls: &[String],
        bind_port: Option<u16>,
        max_vram_gb: Option<f64>,
        enumerate_host: bool,
        owner_config: Option<OwnerRuntimeConfig>,
        config_path: Option<&std::path::Path>,
    ) -> Result<(Self, TunnelChannels)> {
        // Clients use an ephemeral key so they get a unique identity even
        // when running on the same machine as a GPU node.
        let secret_key = if matches!(role, NodeRole::Client)
            || std::env::var("MESH_LLM_EPHEMERAL_KEY").is_ok()
        {
            let key = SecretKey::generate();
            tracing::info!("Using ephemeral key (unique identity)");
            key
        } else {
            load_or_create_key().await?
        };
        // Configure QUIC transport for heavy RPC traffic:
        // Use iroh's default transport config — it sets keep_alive, path timeouts,
        // and multipath correctly. Only override the bidi stream limit.
        use iroh::endpoint::QuicTransportConfig;
        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .build();
        let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(secret_key)
            .alpns(vec![ALPN_V1.to_vec()])
            .transport_config(transport_config);

        {
            use iroh::{RelayConfig, RelayMap};
            let urls: Vec<String> = if relay_urls.is_empty() {
                vec![
                    "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./".into(),
                    "https://aps1-1.relay.michaelneale.mesh-llm.iroh.link./".into(),
                ]
            } else {
                relay_urls.to_vec()
            };
            // Two iroh relays: US West (primary) and Asia-Pacific South (fallback).
            let configs: Vec<RelayConfig> = urls
                .iter()
                .map(|url| RelayConfig::new(url.parse().expect("invalid relay URL"), None))
                .collect();
            let relay_map = RelayMap::from_iter(configs);
            tracing::info!("Relay: {:?}", urls);
            builder = builder.relay_mode(iroh::endpoint::RelayMode::Custom(relay_map));
        }
        if let Some(port) = bind_port {
            tracing::info!("Binding QUIC to UDP port {port}");
            builder = builder.bind_addr(std::net::SocketAddr::from(([0, 0, 0, 0], port)))?;
        }
        let endpoint = builder.bind().await?;
        // Wait briefly for relay connection so the invite token includes the relay URL.
        // On sinkholed networks this times out and we proceed without relay (direct UDP only).
        match tokio::time::timeout(std::time::Duration::from_secs(5), endpoint.online()).await {
            Ok(()) => tracing::info!("Relay connected"),
            Err(_) => tracing::warn!("Relay connection timed out (5s) — proceeding without relay"),
        }

        // Discover public IP via STUN so the invite token includes it.
        // With --bind-port, the advertised port is the bound port (for port forwarding).
        // Without --bind-port, we use port 0 — the IP is still useful for hole-punching.
        // Relay STUN may not work on sinkholed networks, so we use raw STUN to Google/Cloudflare.
        let stun_port = bind_port.unwrap_or(0);
        let public_addr = stun_public_addr(stun_port).await;

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);

        let hw = crate::system::hardware::survey();
        let mut vram = hw.vram_bytes;
        let gpu_name = if matches!(role, NodeRole::Client) {
            None
        } else {
            hw.gpu_name
        };
        let hostname = hw.hostname;
        let is_soc = Some(hw.is_soc);
        let gpu_vram = if hw.gpu_vram.is_empty() {
            None
        } else {
            Some(
                hw.gpu_vram
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        };
        let gpu_reserved_bytes = if hw.gpu_reserved.iter().all(Option::is_none) {
            None
        } else {
            Some(
                hw.gpu_reserved
                    .iter()
                    .map(|value| value.map(|v| v.to_string()).unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        };
        if let Some(max_gb) = max_vram_gb {
            let max_bytes = (max_gb * 1e9) as u64;
            if max_bytes < vram {
                tracing::info!(
                    "Detected VRAM: {:.1} GB, capped to {:.1} GB (--max-vram)",
                    vram as f64 / 1e9,
                    max_gb
                );
                vram = max_bytes;
            } else {
                tracing::info!(
                    "Detected VRAM: {:.1} GB (--max-vram {:.1} has no effect)",
                    vram as f64 / 1e9,
                    max_gb
                );
            }
        } else {
            tracing::info!("Detected VRAM: {:.1} GB", vram as f64 / 1e9);
        }

        let trust_store = owner_config
            .as_ref()
            .map(|config| config.trust_store.clone())
            .unwrap_or_default();
        let trust_policy = owner_config
            .as_ref()
            .map(|config| config.trust_policy)
            .unwrap_or_default();
        let owner_attestation = match owner_config
            .as_ref()
            .and_then(|config| config.keypair.as_ref())
        {
            Some(keypair) => Some(load_or_refresh_owner_attestation(
                keypair,
                endpoint.id(),
                owner_config
                    .as_ref()
                    .and_then(|config| config.node_label.clone()),
                hostname.clone(),
            )?),
            None => None,
        };
        let owner_summary = verify_node_ownership(
            owner_attestation.as_ref(),
            endpoint.id().as_bytes(),
            &trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );
        let config_state_init = {
            let path = crate::plugin::config_path(config_path)
                .unwrap_or_else(|_| std::path::PathBuf::from("config.toml"));
            crate::runtime::config_state::ConfigState::load(&path)?
        };
        let config_revision_init = config_state_init.revision();

        let node = Node {
            endpoint,
            public_addr,
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                remote_tunnel_maps: HashMap::new(),
                dead_peers: std::collections::HashSet::new(),
                seen_plugin_messages: HashMap::new(),
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
            first_joined_mesh_ts: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            vram_bytes: vram,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            routing_metrics: crate::network::metrics::RoutingMetrics::default(),
            tunnel_tx,
            tunnel_http_tx,
            plugin_manager: Arc::new(Mutex::new(None)),
            display_name: Arc::new(Mutex::new(None)),
            owner_attestation: Arc::new(Mutex::new(owner_attestation)),
            owner_summary: Arc::new(Mutex::new(owner_summary)),
            trust_store: Arc::new(Mutex::new(trust_store)),
            trust_policy,
            enumerate_host,
            gpu_name,
            hostname,
            is_soc,
            gpu_vram,
            gpu_reserved_bytes,
            gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
            config_state: Arc::new(tokio::sync::Mutex::new(config_state_init)),
            config_revision_tx: {
                let (tx, _rx) = tokio::sync::watch::channel(config_revision_init);
                Arc::new(tx)
            },
        };

        // Accept loop starts but waits for start_accepting() before processing connections.
        // This lets a node exist before it is ready to accept mesh traffic.
        let node2 = node.clone();
        tokio::spawn(async move {
            node2.accept_loop().await;
        });

        Ok((
            node,
            TunnelChannels {
                rpc: tunnel_rx,
                http: tunnel_http_rx,
            },
        ))
    }

    #[cfg(test)]
    pub async fn new_for_tests(role: NodeRole) -> Result<Self> {
        use iroh::endpoint::QuicTransportConfig;

        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .build();
        let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(SecretKey::generate())
            .alpns(vec![ALPN.to_vec()])
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .transport_config(transport_config)
            .bind()
            .await?;

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);

        let _channels = TunnelChannels {
            rpc: tunnel_rx,
            http: tunnel_http_rx,
        };

        Ok(Node {
            endpoint,
            public_addr: None,
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                remote_tunnel_maps: HashMap::new(),
                dead_peers: std::collections::HashSet::new(),
                seen_plugin_messages: HashMap::new(),
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
            first_joined_mesh_ts: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            vram_bytes: 0,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            routing_metrics: crate::network::metrics::RoutingMetrics::default(),
            tunnel_tx,
            tunnel_http_tx,
            plugin_manager: Arc::new(Mutex::new(None)),
            display_name: Arc::new(Mutex::new(None)),
            owner_attestation: Arc::new(Mutex::new(None)),
            owner_summary: Arc::new(Mutex::new(OwnershipSummary::default())),
            trust_store: Arc::new(Mutex::new(TrustStore::default())),
            trust_policy: TrustPolicy::Off,
            enumerate_host: true,
            gpu_name: None,
            hostname: None,
            is_soc: Some(false),
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
        })
    }

    #[cfg(test)]
    pub async fn insert_test_peer(&self, peer: PeerInfo) {
        self.state.lock().await.peers.insert(peer.id, peer);
    }

    #[cfg(test)]
    pub async fn has_test_peer(&self, id: EndpointId) -> bool {
        self.state.lock().await.peers.contains_key(&id)
    }

    pub fn invite_token(&self) -> String {
        let mut addr = self.endpoint.addr();
        // Inject STUN-discovered public address if relay STUN didn't provide one.
        if let Some(pub_addr) = self.public_addr {
            use iroh::TransportAddr;
            let has_public = addr.addrs.iter().any(|a| match a {
                TransportAddr::Ip(sock) => match sock.ip() {
                    std::net::IpAddr::V4(v4) => !v4.is_private() && !v4.is_loopback(),
                    _ => false,
                },
                _ => false,
            });
            if !has_public {
                addr.addrs.insert(TransportAddr::Ip(pub_addr));
            }
        }
        let json = serde_json::to_vec(&addr).expect("serializable");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json)
    }

    /// Decode an invite token into an [`EndpointAddr`] without connecting.
    /// Returns `Err` if the token is not valid base64 or not valid JSON.
    pub fn decode_invite_token(invite_token: &str) -> Result<EndpointAddr> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(invite_token)
            .context("invalid invite token encoding")?;
        serde_json::from_slice(&json).context("invalid invite token JSON")
    }

    #[cfg(test)]
    pub async fn sync_from_peer_for_tests(&self, remote: &Self) {
        let remote_id = remote.endpoint.id();
        let their_announcements = remote.collect_announcements().await;
        for ann in &their_announcements {
            if ann.addr.id == self.endpoint.id() {
                continue;
            }
            if ann.addr.id == remote_id {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote_id, ann.addr.clone(), ann).await;
            } else {
                self.update_transitive_peer(ann.addr.id, &ann.addr, ann)
                    .await;
            }
        }
    }

    async fn build_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<crate::plugin::proto::MeshPeer>,
        detail_json: String,
    ) -> crate::plugin::proto::MeshEvent {
        crate::plugin::proto::MeshEvent {
            kind: kind as i32,
            peer,
            local_peer_id: endpoint_id_hex(self.endpoint.id()),
            mesh_id: self.mesh_id.lock().await.clone().unwrap_or_default(),
            detail_json,
        }
    }

    /// Enable accepting inbound connections. Call before join() or when ready to participate.
    /// Until this is called, the accept loop blocks waiting.
    pub fn start_accepting(&self) {
        self.accepting
            .1
            .store(true, std::sync::atomic::Ordering::Release);
        self.accepting.0.notify_waiters();
        let node = self.clone();
        tokio::spawn(async move {
            let plugin_manager = node.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                let _ = plugin_manager
                    .broadcast_mesh_event(
                        node.build_mesh_event(
                            crate::plugin::proto::mesh_event::Kind::LocalAccepting,
                            None,
                            String::new(),
                        )
                        .await,
                    )
                    .await;
            }
        });
    }

    pub async fn join(&self, invite_token: &str) -> Result<()> {
        let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(invite_token)?;
        let addr: EndpointAddr = serde_json::from_slice(&json)?;
        // Clear dead status — explicit join should always attempt connection
        self.state.lock().await.dead_peers.remove(&addr.id);
        self.connect_to_peer(addr).await
    }

    /// Connect to a peer without gossip exchange — for passive nodes (clients/standby).
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub async fn role(&self) -> NodeRole {
        self.role.lock().await.clone()
    }

    pub async fn set_role(&self, role: NodeRole) {
        *self.role.lock().await = role;
    }

    pub async fn set_models(&self, models: Vec<String>) {
        *self.models.lock().await = models;
    }

    pub async fn models(&self) -> Vec<String> {
        self.models.lock().await.clone()
    }

    pub async fn set_model_source(&self, source: String) {
        *self.model_source.lock().await = Some(source);
        self.refresh_served_model_descriptors().await;
    }

    pub async fn set_serving_models(&self, models: Vec<String>) {
        *self.serving_models.lock().await = models;
        self.refresh_served_model_descriptors().await;
    }

    pub async fn set_served_model_descriptors(&self, descriptors: Vec<ServedModelDescriptor>) {
        let model_names: std::collections::HashSet<_> = descriptors
            .iter()
            .map(|descriptor| descriptor.identity.model_name.clone())
            .collect();
        *self.served_model_descriptors.lock().await = descriptors;
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| model_names.contains(&runtime.model_name));
    }

    pub async fn upsert_served_model_descriptor(&self, descriptor: ServedModelDescriptor) {
        let mut descriptors = self.served_model_descriptors.lock().await;
        if let Some(existing) = descriptors
            .iter_mut()
            .find(|existing| existing.identity.model_name == descriptor.identity.model_name)
        {
            *existing = descriptor;
        } else {
            descriptors.push(descriptor);
        }
    }

    pub async fn remove_served_model_descriptor(&self, model_name: &str) {
        self.served_model_descriptors
            .lock()
            .await
            .retain(|descriptor| descriptor.identity.model_name != model_name);
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| runtime.model_name != model_name);
    }

    pub async fn set_model_runtime_context_length(
        &self,
        model_name: &str,
        context_length: Option<u32>,
    ) {
        let identity_hash = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model_name)
            .and_then(|descriptor| descriptor.identity.identity_hash.clone());
        let mut runtimes = self.model_runtime_descriptors.lock().await;
        if let Some(context_length) = context_length {
            if let Some(runtime) = runtimes
                .iter_mut()
                .find(|runtime| runtime.model_name == model_name)
            {
                runtime.identity_hash = identity_hash.or_else(|| runtime.identity_hash.clone());
                runtime.context_length = Some(context_length);
                runtime.ready = true;
            } else {
                runtimes.push(ModelRuntimeDescriptor {
                    model_name: model_name.to_string(),
                    identity_hash,
                    context_length: Some(context_length),
                    ready: true,
                });
            }
        } else {
            runtimes.retain(|runtime| runtime.model_name != model_name);
        }
    }

    pub async fn set_model_runtime_starting(&self, model_name: &str) {
        let identity_hash = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model_name)
            .and_then(|descriptor| descriptor.identity.identity_hash.clone());
        let mut runtimes = self.model_runtime_descriptors.lock().await;
        if let Some(runtime) = runtimes
            .iter_mut()
            .find(|runtime| runtime.model_name == model_name)
        {
            runtime.identity_hash = identity_hash.or_else(|| runtime.identity_hash.clone());
            runtime.context_length = None;
            runtime.ready = false;
        } else {
            runtimes.push(ModelRuntimeDescriptor {
                model_name: model_name.to_string(),
                identity_hash,
                context_length: None,
                ready: false,
            });
        }
    }

    pub async fn local_model_context_length(&self, model_name: &str) -> Option<u32> {
        self.model_runtime_descriptors
            .lock()
            .await
            .iter()
            .find(|runtime| runtime.model_name == model_name)
            .and_then(ModelRuntimeDescriptor::advertised_context_length)
    }

    pub async fn peer_model_context_length(
        &self,
        peer_id: EndpointId,
        model_name: &str,
    ) -> Option<u32> {
        self.state
            .lock()
            .await
            .peers
            .get(&peer_id)
            .and_then(|peer| peer.advertised_context_length(model_name))
    }

    pub async fn served_model_descriptors(&self) -> Vec<ServedModelDescriptor> {
        self.served_model_descriptors.lock().await.clone()
    }

    pub async fn serving_models(&self) -> Vec<String> {
        self.serving_models.lock().await.clone()
    }

    pub async fn set_hosted_models(&self, models: Vec<String>) {
        *self.hosted_models.lock().await = models;
    }

    pub async fn hosted_models(&self) -> Vec<String> {
        self.hosted_models.lock().await.clone()
    }

    async fn refresh_served_model_descriptors(&self) {
        let serving_models = self.serving_models.lock().await.clone();
        let descriptors = if let Some(primary_model_name) = serving_models.first() {
            let model_source = self.model_source.lock().await.clone();
            let primary_model_path = crate::models::find_model_path(primary_model_name);
            infer_served_model_descriptors(
                primary_model_name,
                &serving_models,
                model_source.as_deref(),
                Some(primary_model_path.as_path()),
            )
        } else {
            Vec::new()
        };
        self.set_served_model_descriptors(descriptors).await;
    }

    /// Set the operator-facing display name for this node.
    pub async fn set_display_name(&self, name: String) {
        *self.display_name.lock().await = Some(name);
    }

    pub async fn set_plugin_manager(&self, plugin_manager: crate::plugin::PluginManager) {
        let peers = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect::<Vec<_>>()
        };
        *self.plugin_manager.lock().await = Some(plugin_manager.clone());
        let local_kind = if self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            crate::plugin::proto::mesh_event::Kind::LocalAccepting
        } else {
            crate::plugin::proto::mesh_event::Kind::LocalStandby
        };
        let _ = plugin_manager
            .broadcast_mesh_event(self.build_mesh_event(local_kind, None, String::new()).await)
            .await;
        if self.mesh_id.lock().await.is_some() {
            let _ = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                        None,
                        String::new(),
                    )
                    .await,
                )
                .await;
        }
        for peer in peers {
            if let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUp,
                        Some(peer_info_to_mesh_peer(&peer)),
                        String::new(),
                    )
                    .await,
                )
                .await
            {
                tracing::debug!(
                    "Failed to send existing peer snapshot to plugins for {}: {err}",
                    peer.id.fmt_short()
                );
            }
        }
    }

    pub async fn plugin_manager(&self) -> Option<crate::plugin::PluginManager> {
        self.plugin_manager.lock().await.clone()
    }

    pub fn start_plugin_channel_forwarder(
        &self,
        mut rx: tokio::sync::mpsc::Receiver<crate::plugin::PluginMeshEvent>,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(err) = node.forward_plugin_event(event).await {
                    tracing::debug!("Plugin mesh forward failed: {err}");
                }
            }
        });
    }

    async fn emit_plugin_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<&PeerInfo>,
        detail_json: String,
    ) {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        if let Some(plugin_manager) = plugin_manager {
            if let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(kind, peer.map(peer_info_to_mesh_peer), detail_json)
                        .await,
                )
                .await
            {
                tracing::debug!(
                    "Failed to deliver plugin mesh event {:?} for {}: {err}",
                    kind,
                    peer.map(|p| p.id.fmt_short().to_string())
                        .unwrap_or_else(|| self.endpoint.id().fmt_short().to_string())
                );
            }
        }
    }

    async fn update_peer_rtt(&self, id: EndpointId, rtt_ms: u32) {
        let (updated_peer, old_rtt) = {
            let mut state = self.state.lock().await;
            if let Some(peer) = state.peers.get_mut(&id) {
                let prev = peer.rtt_ms;
                // Only accept equal-or-lower RTT. Gossip round-trip timing
                // can inflate the value when routed via relay, overwriting a
                // good direct-path measurement. The RTT gate only cares about
                // "fast enough for split", so keeping the best-seen value is
                // correct — if the path truly degrades the peer will be
                // unreachable and removed via the normal liveness path.
                if prev.is_some_and(|p| rtt_ms > p) {
                    return;
                }
                peer.rtt_ms = Some(rtt_ms);
                (Some(peer.clone()), prev)
            } else {
                (None, None)
            }
        };
        if let Some(peer) = updated_peer {
            tracing::info!("Peer {} RTT: {}ms", id.fmt_short(), rtt_ms);
            // If RTT dropped from above the split threshold (80ms) to below it
            // (e.g. relay → direct), trigger a re-election so the peer can now
            // be included in split mode.
            let was_above = old_rtt.is_some_and(|r| r > MAX_SPLIT_RTT_MS);
            if was_above && rtt_ms <= MAX_SPLIT_RTT_MS {
                eprintln!(
                    "📡 Peer {} RTT improved ({}ms → {}ms) — re-electing for split",
                    id.fmt_short(),
                    old_rtt.unwrap_or(0),
                    rtt_ms
                );
                let count = self.state.lock().await.peers.len();
                let _ = self.peer_change_tx.send(count);
            }
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    /// Re-gossip our state to all connected peers.
    /// Call after changing assigned/hosted state, role, or configured models.
    pub async fn regossip(&self) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.initiate_gossip(conn, peer_id).await {
                    tracing::debug!("Regossip to {} failed: {e}", peer_id.fmt_short());
                }
            });
        }
    }

    /// Gossip with one connected peer to update routing table.
    /// Used by: (1) passive nodes' periodic 60s heartbeat, (2) background
    /// refresh on tunnel failure so future requests have fresh routing.
    pub async fn gossip_one_peer(&self) {
        let conn = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .next()
                .map(|(id, c)| (*id, c.clone()))
        };
        if let Some((peer_id, conn)) = conn {
            let _ = self.initiate_gossip_inner(conn, peer_id, false).await;
        }
    }

    pub async fn set_llama_ready(&self, ready: bool) {
        *self.llama_ready.lock().await = ready;
    }

    pub async fn is_llama_ready(&self) -> bool {
        *self.llama_ready.lock().await
    }

    pub async fn mesh_id(&self) -> Option<String> {
        self.mesh_id.lock().await.clone()
    }

    pub async fn first_joined_mesh_ts(&self) -> Option<u64> {
        *self.first_joined_mesh_ts.lock().await
    }

    pub async fn set_first_joined_mesh_ts_if_absent(&self, ts: u64) -> bool {
        let mut current = self.first_joined_mesh_ts.lock().await;
        if current.is_none() {
            *current = Some(ts);
            true
        } else {
            false
        }
    }

    /// Set the mesh identity. If None was set, adopts the given ID (from gossip).
    /// If already set, ignores (originator's ID wins).
    pub async fn set_mesh_id(&self, id: String) {
        let mut current = self.mesh_id.lock().await;
        if current.is_none() {
            *current = Some(id);
            drop(current);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                None,
                String::new(),
            )
            .await;
        }
    }

    /// Set mesh ID unconditionally (for originator).
    pub async fn set_mesh_id_force(&self, id: String) {
        *self.mesh_id.lock().await = Some(id);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
            None,
            String::new(),
        )
        .await;
    }

    pub async fn set_available_models(&self, models: Vec<String>) {
        *self.available_models.lock().await = models;
    }

    pub async fn available_models(&self) -> Vec<String> {
        self.available_models.lock().await.clone()
    }

    /// Record a request for a model — updates the demand map.
    /// Called from API proxy on every request (including misses for unserved models).
    /// Uses std::sync::Mutex (not tokio) so it can be called from sync context too.
    pub fn record_request(&self, model: &str) {
        // "auto" is a routing directive, not a real model — don't pollute demand
        if model == "auto" || model.is_empty() {
            return;
        }
        let mut demand = self.model_demand.lock().unwrap();
        let entry = demand.entry(model.to_string()).or_default();
        entry.last_active = now_secs();
        entry.request_count += 1;
    }

    /// Get the current demand map (for gossip and assignment decisions).
    pub fn get_demand(&self) -> HashMap<String, ModelDemand> {
        self.model_demand.lock().unwrap().clone()
    }

    /// Merge incoming demand from gossip into our local map.
    pub fn merge_remote_demand(&self, remote: &HashMap<String, ModelDemand>) {
        let mut demand = self.model_demand.lock().unwrap();
        merge_demand(&mut demand, remote);
    }

    /// Remove demand entries that have expired (past TTL and not pinned).
    /// Call periodically to prevent unbounded map growth.
    pub async fn gc_demand(&self) {
        let now = now_secs();
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for p in peers.peers.values() {
            for m in &p.requested_models {
                pinned.insert(m.clone());
            }
        }
        drop(peers);
        drop(my_requested);

        let mut demand = self.model_demand.lock().unwrap();
        demand.retain(|model, d| pinned.contains(model) || (now - d.last_active) < DEMAND_TTL_SECS);
    }

    /// Get active demand entries (within TTL or pinned by a live node).
    /// This replaces mesh_wanted_models().
    pub async fn active_demand(&self) -> HashMap<String, ModelDemand> {
        let now = now_secs();
        let demand = self.model_demand.lock().unwrap().clone();

        // Check which models are pinned (declared via --model by self or a live peer)
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for p in peers.peers.values() {
            for m in &p.requested_models {
                pinned.insert(m.clone());
            }
        }
        drop(peers);
        drop(my_requested);

        demand
            .into_iter()
            .filter(|(model, d)| pinned.contains(model) || (now - d.last_active) < DEMAND_TTL_SECS)
            .collect()
    }

    pub async fn set_requested_models(&self, models: Vec<String>) {
        // Seed demand entries for --model declarations
        {
            let mut demand = self.model_demand.lock().unwrap();
            let now = now_secs();
            for m in &models {
                let entry = demand.entry(m.clone()).or_default();
                entry.last_active = entry.last_active.max(now);
            }
        }
        *self.requested_models.lock().await = models;
    }

    pub async fn requested_models(&self) -> Vec<String> {
        self.requested_models.lock().await.clone()
    }

    async fn forward_plugin_event(&self, event: crate::plugin::PluginMeshEvent) -> Result<()> {
        match event {
            crate::plugin::PluginMeshEvent::Channel {
                plugin_id,
                mut message,
            } => {
                let plugin_manager = self.plugin_manager.lock().await.clone();
                if let Some(plugin_manager) = plugin_manager {
                    if !plugin_manager
                        .plugin_declares_mesh_channel(&plugin_id, &message.channel)
                        .await
                    {
                        tracing::debug!(
                            plugin = %plugin_id,
                            channel = %message.channel,
                            "Dropping outbound channel message for undeclared mesh channel"
                        );
                        return Ok(());
                    }
                }
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
                let frame = crate::plugin::proto::MeshChannelFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_channel_frame(&frame, None).await
            }
            crate::plugin::PluginMeshEvent::BulkTransfer {
                plugin_id,
                mut message,
            } => {
                let plugin_manager = self.plugin_manager.lock().await.clone();
                if let Some(plugin_manager) = plugin_manager {
                    if !plugin_manager
                        .plugin_declares_mesh_channel(&plugin_id, &message.channel)
                        .await
                    {
                        tracing::debug!(
                            plugin = %plugin_id,
                            channel = %message.channel,
                            "Dropping outbound bulk transfer for undeclared mesh channel"
                        );
                        return Ok(());
                    }
                }
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
                let frame = crate::plugin::proto::MeshBulkFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_bulk_frame(&frame, None).await
            }
        }
    }

    async fn remember_plugin_message(&self, message_id: String) -> bool {
        /// How long to remember a message ID. Any duplicate arriving within
        /// this window is suppressed. This must be longer than the worst-case
        /// propagation delay across alternate mesh paths — 120s is generous.
        const DEDUP_TTL: std::time::Duration = std::time::Duration::from_secs(120);
        /// Hard cap to bound memory even if message volume is extreme.
        const DEDUP_HARD_CAP: usize = 100_000;

        let now = std::time::Instant::now();
        let mut state = self.state.lock().await;

        // Evict entries older than the TTL
        while let Some((ts, _)) = state.seen_plugin_message_order.front() {
            if now.duration_since(*ts) >= DEDUP_TTL {
                if let Some((_, id)) = state.seen_plugin_message_order.pop_front() {
                    state.seen_plugin_messages.remove(&id);
                }
            } else {
                break;
            }
        }

        // Already seen?
        if state.seen_plugin_messages.contains_key(&message_id) {
            return false;
        }

        // Hard cap: if under extreme load we still accumulate too many,
        // evict the oldest regardless of TTL.
        while state.seen_plugin_message_order.len() >= DEDUP_HARD_CAP {
            if let Some((_, id)) = state.seen_plugin_message_order.pop_front() {
                state.seen_plugin_messages.remove(&id);
            }
        }

        state.seen_plugin_messages.insert(message_id.clone(), now);
        state.seen_plugin_message_order.push_back((now, message_id));
        true
    }

    async fn broadcast_plugin_channel_frame(
        &self,
        frame: &crate::plugin::proto::MeshChannelFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_CHANNEL]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    async fn broadcast_plugin_bulk_frame(
        &self,
        frame: &crate::plugin::proto::MeshBulkFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_BULK_TRANSFER]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin bulk frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    async fn handle_plugin_channel_stream(
        &self,
        _remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            anyhow::bail!("Plugin channel frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshChannelFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_channel_message(crate::plugin::PluginMeshEvent::Channel {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        // Targeted messages: forward only to the specific target peer if we
        // have a direct connection.  Do NOT flood-broadcast targeted messages
        // to all connections — that causes O(N²) amplification across the mesh.
        // Untargeted broadcasts: deliver locally only.  The originator already
        // sent to all their direct connections.
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            // Look up connection to the target peer by hex ID
            let target_conn = {
                let state = self.state.lock().await;
                state
                    .connections
                    .iter()
                    .find(|(id, _)| endpoint_id_hex(**id) == message.target_peer_id)
                    .map(|(id, conn)| (*id, conn.clone()))
            };
            if let Some((_target_id, conn)) = target_conn {
                let data = frame.encode_to_vec();
                tokio::spawn(async move {
                    let result = async {
                        let (mut send, _recv) = conn.open_bi().await?;
                        send.write_all(&[STREAM_PLUGIN_CHANNEL]).await?;
                        send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                        send.write_all(&data).await?;
                        send.finish()?;
                        Ok::<_, anyhow::Error>(())
                    }
                    .await;
                    if let Err(e) = result {
                        tracing::debug!("Failed to forward targeted plugin frame: {e}");
                    }
                });
            }
        }

        Ok(())
    }

    async fn handle_plugin_bulk_stream(
        &self,
        _remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 64_000_000 {
            anyhow::bail!("Plugin bulk frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshBulkFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_bulk_transfer_message(crate::plugin::PluginMeshEvent::BulkTransfer {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        // Same policy as channel frames: targeted → forward to target only,
        // broadcast → deliver locally only (originator already sent to their
        // direct connections).
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            let target_conn = {
                let state = self.state.lock().await;
                state
                    .connections
                    .iter()
                    .find(|(id, _)| endpoint_id_hex(**id) == message.target_peer_id)
                    .map(|(id, conn)| (*id, conn.clone()))
            };
            if let Some((_target_id, conn)) = target_conn {
                let data = frame.encode_to_vec();
                tokio::spawn(async move {
                    let result = async {
                        let (mut send, _recv) = conn.open_bi().await?;
                        send.write_all(&[STREAM_PLUGIN_BULK_TRANSFER]).await?;
                        send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                        send.write_all(&data).await?;
                        send.finish()?;
                        Ok::<_, anyhow::Error>(())
                    }
                    .await;
                    if let Err(e) = result {
                        tracing::debug!("Failed to forward targeted plugin bulk frame: {e}");
                    }
                });
            }
        }

        Ok(())
    }

    /// Get the mesh catalog: local installed models plus mesh served/requested models.
    /// Returns deduplicated list of model names (file stems, no .gguf).
    pub async fn mesh_catalog(&self) -> Vec<String> {
        // Snapshot each lock independently to avoid holding multiple locks.
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| {
                    (
                        p.available_models.clone(),
                        p.requested_models.clone(),
                        p.serving_models.clone(),
                    )
                })
                .collect()
        };
        let mut all = std::collections::HashSet::new();
        for m in &my_available {
            all.insert(m.clone());
        }
        for m in &my_requested {
            all.insert(m.clone());
        }
        for m in &my_serving_models {
            all.insert(m.clone());
        }
        for (avail, req, serving_models) in &peer_data {
            for m in avail {
                all.insert(m.clone());
            }
            for m in req {
                all.insert(m.clone());
            }
            for m in serving_models {
                all.insert(m.clone());
            }
        }
        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    pub async fn mesh_catalog_entries(&self) -> Vec<MeshCatalogEntry> {
        let names = self.mesh_catalog().await;
        let my_available = self.available_models.lock().await.clone();
        let my_served_descriptors = self.served_model_descriptors.lock().await.clone();
        let peer_descriptors: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| p.served_model_descriptors.clone())
                .collect()
        };

        let mut by_name: HashMap<String, ServedModelDescriptor> = HashMap::new();
        for descriptor in infer_available_model_descriptors(&my_available)
            .into_iter()
            .chain(my_served_descriptors.into_iter())
        {
            upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
        }
        for served in peer_descriptors {
            for descriptor in served {
                upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
            }
        }

        names
            .into_iter()
            .map(|model_name| MeshCatalogEntry {
                descriptor: by_name.get(&model_name).cloned(),
                model_name,
            })
            .collect()
    }

    /// Get all models currently reachable via the mesh HTTP/API ingress.
    ///
    /// This is intentionally stricter than "loaded in VRAM somewhere": split
    /// workers may contribute compute for a model but cannot accept chat
    /// requests directly.
    pub async fn models_being_served(&self) -> Vec<String> {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect()
        };
        let mut served = std::collections::HashSet::new();
        for s in &my_hosted_models {
            served.insert(s.clone());
        }
        for peer in &peer_data {
            for m in peer.http_routable_models() {
                served.insert(m.clone());
            }
        }
        let mut result: Vec<String> = served.into_iter().collect();
        result.sort();
        result
    }

    /// Find a host for a specific model, using hash-based selection for load distribution.
    /// When multiple hosts serve the same model, picks one based on our node ID hash.
    /// All host IDs serving a model, with hash-preferred host first.
    /// Used for retry: if the first host fails, try the next.
    pub async fn hosts_for_model(&self, model: &str) -> Vec<EndpointId> {
        let state = self.state.lock().await;
        let mut hosts: Vec<EndpointId> = state
            .peers
            .values()
            .filter(|p| p.routes_http_model(model))
            .map(|p| p.id)
            .collect();
        hosts.sort();
        // Put the hash-preferred host first so normal path tries it first
        if !hosts.is_empty() {
            let my_id = self.endpoint.id();
            let id_bytes = my_id.as_bytes();
            let hash = id_bytes
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % hosts.len();
            hosts.rotate_left(idx);
        }
        hosts
    }

    /// Find ANY host in the mesh (fallback when no model match).
    pub async fn any_host(&self) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .find(|p| !p.http_routable_models().is_empty())
            .cloned()
    }

    /// Build the current routing table from this node's view of the mesh.
    pub async fn routing_table(&self) -> RoutingTable {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_role = self.role.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect()
        };
        let mut hosts = Vec::new();

        // Include self if we're serving through the local API proxy
        if !matches!(my_role, NodeRole::Client) {
            for model in my_hosted_models {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", self.endpoint.id().fmt_short()),
                    endpoint_id: self.endpoint.id(),
                    vram_gb: self.vram_bytes as f64 / 1e9,
                });
            }
        }

        // Include peers that are serving through their local API proxies
        for peer in &peer_data {
            for model in peer.http_routable_models() {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", peer.id.fmt_short()),
                    endpoint_id: peer.id,
                    vram_gb: peer.vram_bytes as f64 / 1e9,
                });
            }
        }

        let mesh_id = self.mesh_id.lock().await.clone();
        RoutingTable { hosts, mesh_id }
    }

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state.lock().await.peers.values().cloned().collect()
    }

    /// Open an HTTP tunnel bi-stream to a peer (tagged STREAM_TUNNEL_HTTP).
    /// If no connection exists, tries to connect on-demand (for passive nodes
    /// that learned about hosts from routing table but aren't directly connected).
    pub async fn open_http_tunnel(
        &self,
        peer_id: EndpointId,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = {
            let state = self.state.lock().await;
            match state.connections.get(&peer_id).cloned() {
                Some(c) => c,
                None => {
                    // Try on-demand connect using peer's addr from peer info
                    let addr = state.peers.get(&peer_id).map(|p| p.addr.clone());
                    drop(state);
                    if let Some(addr) = addr {
                        let c = tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            connect_mesh(&self.endpoint, addr),
                        )
                        .await
                        .map_err(|_| {
                            anyhow::anyhow!("Timeout connecting to {}", peer_id.fmt_short())
                        })?
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to connect to {}: {e}", peer_id.fmt_short())
                        })?;
                        self.state
                            .lock()
                            .await
                            .connections
                            .insert(peer_id, c.clone());
                        c
                    } else {
                        anyhow::bail!("No connection or address for {}", peer_id.fmt_short());
                    }
                }
            }
        };
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (mut send, recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_TUNNEL_HTTP]).await?;
            Ok::<_, anyhow::Error>((send, recv))
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timeout opening tunnel to {}", peer_id.fmt_short()))?;

        if result.is_err() {
            // Connection failed — peer is likely dead, broadcast it
            tracing::info!(
                "Tunnel to {} failed, broadcasting death",
                peer_id.fmt_short()
            );
            self.handle_peer_death(peer_id).await;
        }

        result
    }

    pub async fn set_tunnel_port(&self, id: EndpointId, port: u16) {
        if let Some(peer) = self.state.lock().await.peers.get_mut(&id) {
            peer.tunnel_port = Some(port);
        }
    }

    pub async fn broadcast_tunnel_map(
        &self,
        my_tunnel_map: HashMap<EndpointId, u16>,
    ) -> Result<()> {
        use prost::Message as _;

        let owner_peer_id = self.endpoint.id().as_bytes().to_vec();
        let entries: Vec<crate::proto::node::TunnelEntry> = my_tunnel_map
            .iter()
            .map(|(id, &port)| crate::proto::node::TunnelEntry {
                target_peer_id: id.as_bytes().to_vec(),
                tunnel_port: port as u32,
                relay_peer_id: None,
            })
            .collect();

        let proto_msg = crate::proto::node::TunnelMap {
            owner_peer_id,
            entries,
        };
        let proto_bytes = proto_msg.encode_to_vec();

        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };

        for (peer_id, conn) in conns {
            let proto_bytes = proto_bytes.clone();
            tokio::spawn(async move {
                match conn.open_bi().await {
                    Ok((mut send, _recv)) => {
                        if send.write_all(&[STREAM_TUNNEL_MAP]).await.is_err() {
                            return;
                        }
                        if write_len_prefixed(&mut send, &proto_bytes).await.is_err() {
                            return;
                        }
                        let _ = send.finish();
                        tracing::info!("Sent tunnel map to {}", peer_id.fmt_short());
                    }
                    Err(e) => {
                        tracing::warn!("Failed to send tunnel map to {}: {e}", peer_id.fmt_short());
                    }
                }
            });
        }
        Ok(())
    }

    /// Get all remote tunnel maps: { peer_id → { target_id → tunnel_port } }
    pub async fn all_remote_tunnel_maps(&self) -> HashMap<EndpointId, HashMap<EndpointId, u16>> {
        self.state.lock().await.remote_tunnel_maps.clone()
    }

    /// Wait until we have tunnel maps from at least `n` peers, with timeout.
    pub async fn wait_for_tunnel_maps(&self, n: usize, timeout: std::time::Duration) -> Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let state = self.state.lock().await;
                if state.remote_tunnel_maps.len() >= n {
                    return Ok(());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                let state = self.state.lock().await;
                tracing::warn!(
                    "Timeout waiting for tunnel maps: got {} of {} needed",
                    state.remote_tunnel_maps.len(),
                    n
                );
                return Ok(()); // Don't fail — B2B is optional optimization
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }

    /// Open a tunnel bi-stream to a peer using the stored connection.
    pub async fn open_tunnel_stream(
        &self,
        peer_id: EndpointId,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = {
            self.state
                .lock()
                .await
                .connections
                .get(&peer_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No connection to {}", peer_id.fmt_short()))?
        };
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_TUNNEL]).await?;
        Ok((send, recv))
    }

    // --- Connection handling ---

    async fn accept_loop(&self) {
        // Wait until start_accepting() is called before processing any connections.
        // Check flag first to handle the case where start_accepting() was called before we got here.
        if !self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            self.accepting.0.notified().await;
        }
        tracing::info!("Accept loop: now accepting inbound connections");

        loop {
            let incoming = match self.endpoint.accept().await {
                Some(i) => i,
                None => break,
            };
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.handle_incoming(incoming).await {
                    tracing::warn!("Incoming connection error: {e}");
                }
            });
        }
    }

    async fn handle_incoming(&self, incoming: iroh::endpoint::Incoming) -> Result<()> {
        let mut accepting = incoming.accept()?;
        let _alpn = accepting.alpn().await?;
        let conn = accepting.await?;
        let remote = conn.remote_id();
        tracing::info!("Inbound connection from {}", remote.fmt_short());

        // Store connection for stream dispatch (tunneling, route requests, etc.)
        // Don't add to peer list yet — only gossip exchange promotes to peer.
        let was_dead = {
            let mut state = self.state.lock().await;
            let was_dead = state.dead_peers.remove(&remote);
            if was_dead {
                eprintln!("🔄 Previously dead peer {} reconnected", remote.fmt_short());
            }
            state.connections.insert(remote, conn.clone());
            was_dead
        };

        // If this peer was previously dead, immediately gossip to restore their
        // assigned/routable state in our peer list. Without this, models served by the
        // reconnecting peer stay invisible until the next heartbeat (up to 60s).
        if was_dead {
            let node = self.clone();
            let gossip_conn = conn.clone();
            tokio::spawn(async move {
                if let Err(e) = node.initiate_gossip_inner(gossip_conn, remote, false).await {
                    tracing::debug!("Reconnect gossip with {} failed: {e}", remote.fmt_short());
                }
            });
        }

        self.dispatch_streams(conn, remote).await;
        Ok(())
    }

    /// Dispatch bi-streams on a connection by type byte
    fn dispatch_streams(
        &self,
        conn: Connection,
        remote: EndpointId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(self._dispatch_streams(conn, remote))
    }

    async fn _dispatch_streams(&self, conn: Connection, remote: EndpointId) {
        let protocol = connection_protocol(&conn);
        loop {
            let (send, mut recv) = match conn.accept_bi().await {
                Ok(s) => s,
                Err(e) => {
                    tracing::info!("Connection to {} closed: {e}", remote.fmt_short());
                    // Remove the stale connection
                    {
                        let mut state = self.state.lock().await;
                        state.connections.remove(&remote);
                    }
                    // Try to reconnect — if the peer is still alive, re-learn their role
                    let addr = {
                        let state = self.state.lock().await;
                        state.peers.get(&remote).map(|p| p.addr.clone())
                    };
                    if let Some(addr) = addr {
                        tracing::info!("Attempting reconnect to {}...", remote.fmt_short());
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            connect_mesh(&self.endpoint, addr),
                        )
                        .await
                        {
                            Ok(Ok(new_conn)) => {
                                tracing::info!("Reconnected to {}", remote.fmt_short());
                                {
                                    let mut state = self.state.lock().await;
                                    state.connections.insert(remote, new_conn.clone());
                                }
                                // Verify the peer is actually reachable by waiting for gossip.
                                // A relay-level reconnect can appear to succeed even when the
                                // remote process is dead; fire-and-forget gossip would leave the
                                // peer in state.peers indefinitely. Await the result and remove
                                // the peer immediately if gossip cannot complete.
                                let gossip_ok = tokio::time::timeout(
                                    std::time::Duration::from_secs(10),
                                    self.initiate_gossip(new_conn.clone(), remote),
                                )
                                .await
                                .map(|r| r.is_ok())
                                .unwrap_or(false);

                                if gossip_ok {
                                    let node = self.clone();
                                    tokio::spawn(async move {
                                        node.dispatch_streams(new_conn, remote).await;
                                    });
                                } else {
                                    tracing::info!(
                                        "Reconnect gossip to {} failed — peer is dead, removing",
                                        remote.fmt_short()
                                    );
                                    self.remove_peer(remote).await;
                                }
                            }
                            _ => {
                                tracing::info!(
                                    "Reconnect to {} failed — removing peer",
                                    remote.fmt_short()
                                );
                                self.remove_peer(remote).await;
                            }
                        }
                    } else {
                        // No address on file, can't reconnect
                        self.remove_peer(remote).await;
                    }
                    break;
                }
            };

            let mut type_buf = [0u8; 1];
            if recv.read_exact(&mut type_buf).await.is_err() {
                continue;
            }

            let stream_type = type_buf[0];
            if !stream_allowed_before_admission(stream_type) {
                let admitted = {
                    let state = self.state.lock().await;
                    state.peers.contains_key(&remote)
                };
                if !admitted {
                    tracing::warn!(
                        "Quarantine: stream {:#04x} from unadmitted peer {} rejected — peer must complete gossip first",
                        stream_type,
                        remote.fmt_short()
                    );
                    drop((send, recv));
                    continue;
                }
            }

            match stream_type {
                STREAM_GOSSIP => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node
                            .handle_gossip_stream(remote, protocol, send, recv)
                            .await
                        {
                            tracing::warn!("Gossip stream error from {}: {e}", remote.fmt_short());
                        }
                    });
                }
                STREAM_TUNNEL => {
                    if self.tunnel_tx.send((send, recv)).await.is_err() {
                        tracing::warn!("Tunnel receiver dropped");
                        break;
                    }
                }
                STREAM_TUNNEL_MAP => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_tunnel_map_stream(remote, protocol, recv).await
                        {
                            tracing::warn!(
                                "Tunnel map stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_TUNNEL_HTTP => {
                    if self.tunnel_http_tx.send((send, recv)).await.is_err() {
                        tracing::warn!("HTTP tunnel receiver dropped");
                        break;
                    }
                }
                STREAM_ROUTE_REQUEST => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if protocol == ControlProtocol::ProtoV1 {
                            let proto_buf = match read_len_prefixed(&mut recv).await {
                                Ok(buf) => buf,
                                Err(e) => {
                                    tracing::warn!(
                                        "Route request: failed to read proto body — rejecting: {e}"
                                    );
                                    return;
                                }
                            };
                            let req = match crate::proto::node::RouteTableRequest::decode(
                                proto_buf.as_slice(),
                            ) {
                                Ok(r) => r,
                                Err(e) => {
                                    tracing::warn!(
                                        "Route request: invalid protobuf — rejecting: {e}"
                                    );
                                    return;
                                }
                            };
                            if let Err(e) = req.validate_frame() {
                                tracing::warn!(
                                    "Route request: frame validation failed — rejecting: {e}"
                                );
                                return;
                            }
                        }
                        use prost::Message as _;
                        let mut send = send;
                        let table = node.routing_table().await;
                        let proto_table = routing_table_to_proto(&table);
                        let _ = write_len_prefixed(&mut send, &proto_table.encode_to_vec()).await;
                        let _ = send.finish();
                    });
                }
                STREAM_PEER_DOWN => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        let proto_buf = match read_len_prefixed(&mut recv).await {
                            Ok(buf) => buf,
                            Err(e) => {
                                tracing::warn!(
                                    "PeerDown: failed to read proto body — rejecting: {e}"
                                );
                                return;
                            }
                        };
                        let frame = match crate::proto::node::PeerDown::decode(proto_buf.as_slice())
                        {
                            Ok(f) => f,
                            Err(e) => {
                                tracing::warn!("PeerDown: invalid protobuf — rejecting: {e}");
                                return;
                            }
                        };
                        if let Err(e) = frame.validate_frame() {
                            tracing::warn!("PeerDown: frame validation failed — rejecting: {e}");
                            return;
                        }
                        let peer_id_arr: [u8; 32] = match frame.peer_id.as_slice().try_into() {
                            Ok(b) => b,
                            Err(_) => {
                                tracing::warn!("PeerDown: peer_id is not 32 bytes — rejecting");
                                return;
                            }
                        };
                        let pk = match iroh::PublicKey::from_bytes(&peer_id_arr) {
                            Ok(k) => k,
                            Err(_) => {
                                tracing::warn!(
                                    "PeerDown: peer_id is not a valid public key — rejecting"
                                );
                                return;
                            }
                        };
                        let dead_id = EndpointId::from(pk);

                        // Check existing state before deciding.
                        let (conn_opt, peer_addr, recently_seen) = {
                            let state = node.state.lock().await;
                            let conn = state.connections.get(&dead_id).cloned();
                            let peer = state.peers.get(&dead_id);
                            let addr = peer.map(|p| p.addr.clone());
                            let seen = peer
                                .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
                                .unwrap_or(false);
                            (conn, addr, seen)
                        };

                        // If we've heard from this peer recently via direct gossip,
                        // they're alive from our perspective — ignore the death report
                        // regardless of whether we have a connection (the connection
                        // may be broken/stale while the peer is genuinely alive on
                        // a different path).
                        if recently_seen {
                            eprintln!(
                                "ℹ️  Peer {} reported dead by {} but seen recently (direct alive), ignoring",
                                dead_id.fmt_short(),
                                remote.fmt_short()
                            );
                        } else {
                            let should_remove = if let Some(conn) = conn_opt {
                                // Have a connection — probe it. Treat both
                                // timeout and open_bi() error as unreachable.
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(3),
                                    conn.open_bi(),
                                )
                                .await
                                {
                                    Ok(Ok(_)) => false, // stream opened — peer is alive
                                    _ => true,          // timeout or error — unreachable
                                }
                            } else if let Some(addr) = peer_addr {
                                // No connection but we know the peer — try to reach them
                                // before trusting the reporter's claim.
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(5),
                                    connect_mesh(&node.endpoint, addr),
                                )
                                .await
                                {
                                    Ok(Ok(new_conn)) => {
                                        // Peer is reachable — restore connection.
                                        eprintln!(
                                            "ℹ️  Peer {} reported dead by {} but we reached them, keeping",
                                            dead_id.fmt_short(),
                                            remote.fmt_short()
                                        );
                                        let mut state = node.state.lock().await;
                                        // Only insert if no other task raced and
                                        // established a connection while we were probing.
                                        #[allow(clippy::map_entry)]
                                        // manual drop(state) before async spawn
                                        if !state.connections.contains_key(&dead_id) {
                                            state.connections.insert(dead_id, new_conn.clone());
                                            drop(state);
                                            let n2 = node.clone();
                                            tokio::spawn(async move {
                                                n2.dispatch_streams(new_conn, dead_id).await;
                                            });
                                        } else {
                                            drop(state);
                                        }
                                        false
                                    }
                                    _ => true, // genuinely unreachable
                                }
                            } else {
                                // Unknown peer — trust the reporter.
                                true
                            };
                            if let Some(id) =
                                resolve_peer_down(node.endpoint.id(), dead_id, should_remove)
                            {
                                eprintln!(
                                    "⚠️  Peer {} reported dead by {}, confirmed, removing",
                                    id.fmt_short(),
                                    remote.fmt_short()
                                );
                                let mut state = node.state.lock().await;
                                state.connections.remove(&id);
                                drop(state);
                                node.remove_peer(id).await;
                            } else if dead_id != node.endpoint.id() {
                                eprintln!(
                                    "ℹ️  Peer {} reported dead by {} but still reachable, ignoring",
                                    dead_id.fmt_short(),
                                    remote.fmt_short()
                                );
                            }
                        }
                    });
                }
                STREAM_PEER_LEAVING => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        let proto_buf = match read_len_prefixed(&mut recv).await {
                            Ok(buf) => buf,
                            Err(e) => {
                                tracing::warn!(
                                    "PeerLeaving: failed to read proto body — rejecting: {e}"
                                );
                                return;
                            }
                        };
                        let frame =
                            match crate::proto::node::PeerLeaving::decode(proto_buf.as_slice()) {
                                Ok(f) => f,
                                Err(e) => {
                                    tracing::warn!(
                                        "PeerLeaving: invalid protobuf — rejecting: {e}"
                                    );
                                    return;
                                }
                            };
                        if let Err(e) = frame.validate_frame() {
                            tracing::warn!("PeerLeaving: frame validation failed — rejecting: {e}");
                            return;
                        }
                        let leaving_id = match resolve_peer_leaving(remote, &frame) {
                            Ok(id) => id,
                            Err(e) => {
                                tracing::warn!(
                                    "PeerLeaving from {}: rejected ({})",
                                    remote.fmt_short(),
                                    e
                                );
                                return;
                            }
                        };
                        eprintln!(
                            "👋 Peer {} announced clean shutdown",
                            leaving_id.fmt_short()
                        );
                        let mut state = node.state.lock().await;
                        state.connections.remove(&leaving_id);
                        drop(state);
                        node.remove_peer(leaving_id).await;
                    });
                }
                STREAM_PLUGIN_CHANNEL => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_plugin_channel_stream(remote, send, recv).await
                        {
                            tracing::debug!(
                                "Plugin channel stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_PLUGIN_BULK_TRANSFER => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_plugin_bulk_stream(remote, send, recv).await {
                            tracing::debug!(
                                "Plugin bulk stream error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_CONFIG_SUBSCRIBE => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_config_subscribe(remote, send, recv).await {
                            tracing::warn!(
                                "config subscribe error from {}: {e}",
                                remote.fmt_short()
                            );
                        }
                    });
                }
                STREAM_CONFIG_PUSH => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = node.handle_config_push(remote, send, recv).await {
                            tracing::warn!("config push error from {}: {e}", remote.fmt_short());
                        }
                    });
                }
                other => {
                    tracing::warn!("Unknown stream type {other} from {}", remote.fmt_short());
                }
            }
        }
    }

    // --- Config Subscribe ---

    async fn handle_config_subscribe(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> anyhow::Result<()> {
        use crate::proto::node::{ConfigSnapshotResponse, ConfigUpdateNotification};
        use crate::protocol::convert::mesh_config_to_proto;
        use prost::Message as _;

        let buf = read_len_prefixed(&mut recv).await?;
        let frame = crate::proto::node::ConfigSubscribe::decode(buf.as_slice())
            .map_err(|e| anyhow::anyhow!("ConfigSubscribe decode error: {e}"))?;
        frame
            .validate_frame()
            .map_err(|e| anyhow::anyhow!("ConfigSubscribe validation error: {e}"))?;

        let local_owner_id = match self.local_verified_owner_id().await {
            Some(id) => id,
            None => {
                let error_snapshot = crate::proto::node::ConfigSnapshotResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    node_id: vec![],
                    owner_id: String::new(),
                    revision: 0,
                    config_hash: vec![],
                    config: None,
                    hostname: None,
                    error: Some(self.local_owner_status_error().await),
                };
                write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
                return Ok(());
            }
        };

        if frame.subscriber_id.as_slice() != remote.as_bytes() {
            tracing::warn!(
                "config subscribe from {}: subscriber_id does not match connection identity",
                remote.fmt_short()
            );
            let error_snapshot = crate::proto::node::ConfigSnapshotResponse {
                gen: NODE_PROTOCOL_GENERATION,
                node_id: vec![],
                owner_id: String::new(),
                revision: 0,
                config_hash: vec![],
                config: None,
                hostname: None,
                error: Some("subscriber_id does not match connection identity".to_string()),
            };
            write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
            return Ok(());
        }

        let (subscriber_owner_id, _) = match self.peer_verified_owner(remote).await {
            Some(owner) => owner,
            None => {
                let error_snapshot = crate::proto::node::ConfigSnapshotResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    node_id: vec![],
                    owner_id: String::new(),
                    revision: 0,
                    config_hash: vec![],
                    config: None,
                    hostname: None,
                    error: Some("subscriber is not owner-attested".to_string()),
                };
                write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
                return Ok(());
            }
        };

        if subscriber_owner_id != local_owner_id {
            tracing::warn!(
                "config subscribe from {}: owner_id mismatch (want {}, subscriber {})",
                remote.fmt_short(),
                local_owner_id,
                subscriber_owner_id
            );
            let error_snapshot = crate::proto::node::ConfigSnapshotResponse {
                gen: NODE_PROTOCOL_GENERATION,
                node_id: vec![],
                owner_id: String::new(),
                revision: 0,
                config_hash: vec![],
                config: None,
                hostname: None,
                error: Some("owner_id mismatch".to_string()),
            };
            write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
            return Ok(());
        }

        let subscriber_version = {
            let state = self.state.lock().await;
            state
                .peers
                .get(&remote)
                .and_then(|peer| peer.version.clone())
        };

        let snapshot = {
            let state = self.config_state.lock().await;
            if config_uses_pinned_gpu(state.config())
                && !peer_supports_pinned_gpu_config(subscriber_version.as_deref())
            {
                let error_snapshot = crate::proto::node::ConfigSnapshotResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    node_id: vec![],
                    owner_id: String::new(),
                    revision: 0,
                    config_hash: vec![],
                    config: None,
                    hostname: None,
                    error: Some(pinned_gpu_config_peer_error(subscriber_version.as_deref())),
                };
                write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
                return Ok(());
            }
            let proto_cfg = mesh_config_to_proto(state.config());
            ConfigSnapshotResponse {
                gen: NODE_PROTOCOL_GENERATION,
                node_id: self.endpoint.id().as_bytes().to_vec(),
                owner_id: local_owner_id.clone(),
                revision: state.revision(),
                config_hash: state.config_hash().to_vec(),
                config: Some(proto_cfg),
                hostname: self.hostname.clone(),
                error: None,
            }
        };
        write_len_prefixed(&mut send, &snapshot.encode_to_vec()).await?;

        let mut rev_rx = self.config_revision_tx.subscribe();
        loop {
            tokio::select! {
                changed = rev_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let notification = {
                        let state = self.config_state.lock().await;
                        if config_uses_pinned_gpu(state.config())
                            && !peer_supports_pinned_gpu_config(subscriber_version.as_deref())
                        {
                            tracing::warn!(
                                "closing config subscribe stream to {}: {}",
                                remote.fmt_short(),
                                pinned_gpu_config_peer_error(subscriber_version.as_deref())
                            );
                            break;
                        }
                        let proto_cfg = mesh_config_to_proto(state.config());
                        ConfigUpdateNotification {
                            gen: NODE_PROTOCOL_GENERATION,
                            node_id: self.endpoint.id().as_bytes().to_vec(),
                            owner_id: local_owner_id.clone(),
                            revision: state.revision(),
                            config_hash: state.config_hash().to_vec(),
                            config: Some(proto_cfg),
                        }
                    };
                    if write_len_prefixed(&mut send, &notification.encode_to_vec()).await.is_err() {
                        break;
                    }
                }
                inbound = read_len_prefixed(&mut recv) => {
                    if inbound.is_ok() {
                        tracing::debug!(
                            "config subscribe from {} sent unexpected extra frame; closing stream",
                            remote.fmt_short()
                        );
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    async fn local_verified_owner_id(&self) -> Option<String> {
        let summary = self.owner_summary.lock().await.clone();
        if summary.status == OwnershipStatus::Verified {
            summary.owner_id
        } else {
            None
        }
    }

    async fn local_owner_status_error(&self) -> String {
        let summary = self.owner_summary.lock().await.clone();
        match summary.status {
            OwnershipStatus::Verified => "node owner is verified".to_string(),
            OwnershipStatus::Unsigned => "node has no local owner attestation".to_string(),
            OwnershipStatus::Expired => "node owner attestation is expired".to_string(),
            OwnershipStatus::InvalidSignature => {
                "node owner attestation has invalid signature".to_string()
            }
            OwnershipStatus::MismatchedNodeId => {
                "node owner attestation does not match local node id".to_string()
            }
            OwnershipStatus::RevokedOwner => "node owner is revoked".to_string(),
            OwnershipStatus::RevokedCert => "node owner certificate is revoked".to_string(),
            OwnershipStatus::RevokedNodeId => "node endpoint id is revoked".to_string(),
            OwnershipStatus::UnsupportedProtocol => {
                "node owner attestation uses unsupported protocol version".to_string()
            }
            OwnershipStatus::UntrustedOwner => {
                "node owner is not trusted by local policy".to_string()
            }
        }
    }

    async fn peer_verified_owner(
        &self,
        peer_id: EndpointId,
    ) -> Option<(String, SignedNodeOwnership)> {
        let state = self.state.lock().await;
        let peer = state.peers.get(&peer_id)?;
        if peer.owner_summary.status != OwnershipStatus::Verified {
            return None;
        }
        let owner_id = peer.owner_summary.owner_id.clone()?;
        let attestation = peer.owner_attestation.clone()?;
        Some((owner_id, attestation))
    }

    // --- Config Push ---

    async fn handle_config_push(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> anyhow::Result<()> {
        use crate::protocol::convert::proto_config_to_mesh;
        use prost::Message as _;

        // 1. Read + decode + validate ConfigPush
        let buf = read_len_prefixed(&mut recv).await?;
        let push = crate::proto::node::ConfigPush::decode(buf.as_slice())?;
        push.validate_frame()
            .map_err(|e| anyhow::anyhow!("invalid push frame: {e}"))?;

        if push.target_node_id.as_slice() != self.endpoint.id().as_bytes() {
            send_push_error(&mut send, "target_node_id does not match this node").await?;
            return Ok(());
        }
        if push.requester_id.as_slice() != remote.as_bytes() {
            send_push_error(&mut send, "requester_id does not match connection identity").await?;
            return Ok(());
        }

        let local_id = match self.local_verified_owner_id().await {
            Some(id) => id,
            None => {
                let msg = self.local_owner_status_error().await;
                send_push_error(&mut send, &msg).await?;
                return Ok(());
            }
        };

        let (requester_owner_id, requester_attestation) =
            match self.peer_verified_owner(remote).await {
                Some(owner) => owner,
                None => {
                    send_push_error(&mut send, "requester is not owner-attested").await?;
                    return Ok(());
                }
            };

        if requester_owner_id != local_id {
            send_push_error(&mut send, "not the owner of this node").await?;
            return Ok(());
        }

        let expected_public_key =
            match hex::decode(&requester_attestation.claim.owner_sign_public_key) {
                Ok(bytes) => bytes,
                Err(_) => {
                    send_push_error(&mut send, "requester attestation has invalid public key")
                        .await?;
                    return Ok(());
                }
            };
        if push.owner_signing_public_key != expected_public_key {
            send_push_error(
                &mut send,
                "push signing key does not match requester attestation",
            )
            .await?;
            return Ok(());
        }

        let pk_bytes: [u8; 32] = match expected_public_key.as_slice().try_into() {
            Ok(bytes) => bytes,
            Err(_) => {
                send_push_error(&mut send, "invalid public key length").await?;
                return Ok(());
            }
        };
        let vk = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes)?;

        let payload = config_push_signature_payload(&push);
        let sig_bytes: [u8; 64] = match push.signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                send_push_error(&mut send, "invalid signature length").await?;
                return Ok(());
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        if vk.verify_strict(&payload, &sig).is_err() {
            send_push_error(&mut send, "signature verification failed").await?;
            return Ok(());
        }

        // 5. Convert NodeConfigSnapshot → MeshConfig
        let Some(ref config_snapshot) = push.config else {
            send_push_error(&mut send, "missing config payload").await?;
            return Ok(());
        };
        let mesh_config = proto_config_to_mesh(config_snapshot);

        // 6. Preflight + apply via CAS — use spawn_blocking so blocking hardware
        //    probes and synchronous disk I/O do not run on the Tokio async runtime.
        let config_state = Arc::clone(&self.config_state);
        let expected_revision = push.expected_revision;
        let apply_result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            preflight_pushed_config_for_current_node(&mesh_config)?;
            let mut state = config_state.blocking_lock();
            let result = state.apply(mesh_config, expected_revision);
            let current_revision = state.revision();
            let current_hash = *state.config_hash();
            Ok((result, current_revision, current_hash))
        })
        .await
        .map_err(|e| anyhow::anyhow!("config apply task panicked: {e}"))?;
        let (result, current_revision, current_hash) = match apply_result {
            Ok(values) => values,
            Err(err) => {
                send_push_error(&mut send, &err.to_string()).await?;
                return Ok(());
            }
        };

        // 7. Build + send response
        use crate::proto::node::ConfigApplyMode as ProtoApplyMode;
        use crate::runtime::config_state::{ApplyResult, ConfigApplyMode};
        let response = match result {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
            } => {
                if apply_mode == ConfigApplyMode::Staged {
                    let _ = self.config_revision_tx.send(revision);
                }
                crate::proto::node::ConfigPushResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    success: true,
                    current_revision: revision,
                    config_hash: hash.to_vec(),
                    error: None,
                    apply_mode: match apply_mode {
                        ConfigApplyMode::Staged => ProtoApplyMode::Staged as i32,
                        ConfigApplyMode::Noop => ProtoApplyMode::Noop as i32,
                    },
                }
            }
            ApplyResult::RevisionConflict { current_revision } => {
                crate::proto::node::ConfigPushResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    success: false,
                    current_revision,
                    config_hash: vec![],
                    error: Some(
                        "revision conflict: expected_revision does not match current".to_string(),
                    ),
                    apply_mode: ProtoApplyMode::Unspecified as i32,
                }
            }
            ApplyResult::PersistedWithRevisionTrackingError {
                revision,
                hash,
                error,
            } => {
                let _ = self.config_revision_tx.send(revision);
                crate::proto::node::ConfigPushResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    success: false,
                    current_revision: revision,
                    config_hash: hash.to_vec(),
                    error: Some(error),
                    apply_mode: ProtoApplyMode::Staged as i32,
                }
            }
            ApplyResult::ValidationError(msg) | ApplyResult::PersistError(msg) => {
                crate::proto::node::ConfigPushResponse {
                    gen: NODE_PROTOCOL_GENERATION,
                    success: false,
                    current_revision,
                    config_hash: current_hash.to_vec(),
                    error: Some(msg),
                    apply_mode: ProtoApplyMode::Unspecified as i32,
                }
            }
        };
        write_len_prefixed(&mut send, &response.encode_to_vec()).await?;
        Ok(())
    }

    /// Outbound config subscription helper — opens a bi-stream to the target peer,
    /// sends a `ConfigSubscribe` message, and reads back the initial snapshot.
    ///
    /// This is an intentional API stub for the future UI/API layer that will
    /// materialize a mesh-wide config view from per-node subscriptions.
    /// Not yet called from production code.
    #[allow(dead_code)]
    pub(crate) async fn subscribe_to_config(
        &self,
        conn: &iroh::endpoint::Connection,
    ) -> anyhow::Result<(
        crate::proto::node::ConfigSnapshotResponse,
        tokio::sync::watch::Receiver<crate::proto::node::ConfigUpdateNotification>,
    )> {
        use crate::proto::node::{
            ConfigSnapshotResponse, ConfigSubscribe, ConfigUpdateNotification,
        };

        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_CONFIG_SUBSCRIBE]).await?;

        let req = ConfigSubscribe {
            gen: NODE_PROTOCOL_GENERATION,
            subscriber_id: self.endpoint.id().as_bytes().to_vec(),
            // Owner-id filtering is an embedded-client concept; mesh-llm does
            // not currently filter snapshots, so we leave this empty for
            // backward compatibility with older peers.
            owner_id: String::new(),
        };
        write_len_prefixed(&mut send, &req.encode_to_vec()).await?;

        let buf = read_len_prefixed(&mut recv).await?;
        let snapshot = ConfigSnapshotResponse::decode(buf.as_slice())
            .map_err(|e| anyhow::anyhow!("ConfigSnapshotResponse decode error: {e}"))?;
        snapshot
            .validate_frame()
            .map_err(|e| anyhow::anyhow!("ConfigSnapshotResponse validation error: {e}"))?;
        if let Some(err_msg) = snapshot.error.as_deref().filter(|e| !e.is_empty()) {
            return Err(anyhow::anyhow!("config subscribe rejected: {err_msg}"));
        }

        let empty_notif = ConfigUpdateNotification {
            gen: NODE_PROTOCOL_GENERATION,
            node_id: snapshot.node_id.clone(),
            owner_id: snapshot.owner_id.clone(),
            revision: snapshot.revision,
            config_hash: snapshot.config_hash.clone(),
            config: snapshot.config.clone(),
        };
        let (notif_tx, notif_rx) = tokio::sync::watch::channel(empty_notif);
        tokio::spawn(async move {
            // Keep the request stream's send half alive while subscribed so the
            // remote side does not treat immediate EOF as an unsubscribe.
            let _send = send;
            while let Ok(buf) = read_len_prefixed(&mut recv).await {
                match ConfigUpdateNotification::decode(buf.as_slice()) {
                    Ok(notif) => {
                        if let Err(e) = notif.validate_frame() {
                            tracing::warn!("ConfigUpdateNotification validation error: {e}");
                            break;
                        }
                        if notif_tx.send(notif).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("ConfigUpdateNotification decode error: {e}");
                        break;
                    }
                }
            }
        });

        Ok((snapshot, notif_rx))
    }

    // --- Gossip ---

    async fn connect_to_peer(&self, addr: EndpointAddr) -> Result<()> {
        let peer_id = addr.id;
        if peer_id == self.endpoint.id() {
            return Ok(());
        }

        {
            let state = self.state.lock().await;
            if state.peers.contains_key(&peer_id) {
                return Ok(());
            }
            if state.dead_peers.contains(&peer_id) {
                tracing::debug!("Skipping connection to dead peer {}", peer_id.fmt_short());
                return Ok(());
            }
        }

        tracing::info!("Connecting to peer {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            connect_mesh(&self.endpoint, addr.clone()),
        )
        .await
        {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                anyhow::bail!("Failed to connect to {}: {e}", peer_id.fmt_short());
            }
            Err(_) => {
                anyhow::bail!(
                    "Timeout connecting to {} ({}s)",
                    peer_id.fmt_short(),
                    PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
                );
            }
        };

        // Store connection and start dispatcher for inbound streams from this peer
        {
            let mut state = self.state.lock().await;
            state.connections.insert(peer_id, conn.clone());
        }
        let node_for_dispatch = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        // Gossip exchange to learn peer's role/VRAM and announce ourselves
        self.initiate_gossip(conn.clone(), peer_id).await?;

        // Schedule a delayed RTT recheck: the first gossip often goes via relay
        // (high RTT) because direct holepunch hasn't completed yet. After a few
        // seconds the direct path is usually ready, so re-check path info to get
        // the real RTT and potentially trigger a re-election for split mode.
        let node_for_recheck = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let conn = node_for_recheck
                .state
                .lock()
                .await
                .connections
                .get(&peer_id)
                .cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if rtt_ms > 0 {
                            eprintln!(
                                "📡 Peer {} RTT recheck: {}ms ({})",
                                peer_id.fmt_short(),
                                rtt_ms,
                                path_type
                            );
                            node_for_recheck.update_peer_rtt(peer_id, rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        });
        Ok(())
    }

    async fn handle_tunnel_map_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        use prost::Message as _;

        let buf = read_len_prefixed(&mut recv).await?;
        let _ = protocol;
        let frame = crate::proto::node::TunnelMap::decode(buf.as_slice())
            .map_err(|e| anyhow::anyhow!("TunnelMap decode error: {e}"))?;

        frame
            .validate_frame()
            .map_err(|e| anyhow::anyhow!("TunnelMap validation failed: {e}"))?;

        let entry_count = frame.entries.len();
        {
            let mut state = self.state.lock().await;
            ingest_tunnel_map(remote, &frame, &mut state.remote_tunnel_maps)?;
        }

        tracing::info!(
            "Received tunnel map from {} ({} entries)",
            remote.fmt_short(),
            entry_count
        );

        Ok(())
    }
}

pub(crate) fn config_push_signature_payload(push: &crate::proto::node::ConfigPush) -> Vec<u8> {
    use prost::Message as _;
    let mut unsigned = push.clone();
    unsigned.signature.clear();
    unsigned.encode_to_vec()
}

async fn send_push_error(send: &mut iroh::endpoint::SendStream, msg: &str) -> anyhow::Result<()> {
    use crate::protocol::write_len_prefixed;
    use prost::Message as _;
    let response = crate::proto::node::ConfigPushResponse {
        gen: NODE_PROTOCOL_GENERATION,
        success: false,
        current_revision: 0,
        config_hash: vec![],
        error: Some(msg.to_string()),
        apply_mode: crate::proto::node::ConfigApplyMode::Unspecified as i32,
    };
    write_len_prefixed(send, &response.encode_to_vec()).await?;
    Ok(())
}

/// Generate a mesh ID for a new mesh.
/// Named meshes: `sha256("mesh-llm:" + name + ":" + nostr_pubkey)` — deterministic, unique per creator.
/// Unnamed meshes: random UUID, persisted to `~/.mesh-llm/mesh-id`.
pub fn generate_mesh_id(name: Option<&str>, nostr_pubkey: Option<&str>) -> String {
    if let Some(name) = name {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "mesh-llm:".hash(&mut hasher);
        name.hash(&mut hasher);
        if let Some(pk) = nostr_pubkey {
            pk.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    } else {
        // Try to load persisted mesh-id
        let path = mesh_id_path();
        if let Ok(id) = std::fs::read_to_string(&path) {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return id;
            }
        }
        // Generate new random ID and persist
        let id = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, &id);
        id
    }
}

fn mesh_id_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("mesh-id")
}

/// Save the mesh ID of the last mesh we successfully joined.
pub fn save_last_mesh_id(mesh_id: &str) {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, mesh_id);
}

/// Load the mesh ID of the last mesh we successfully joined.
pub fn load_last_mesh_id() -> Option<String> {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Public-to-private identity transition
// ---------------------------------------------------------------------------

fn was_public_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("was-public")
}

/// Record that this node was started in public mode (--auto / --publish / --mesh-name).
/// Called at startup so we can detect a public→private transition next time.
pub fn mark_was_public() {
    let path = was_public_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, "1");
}

/// Returns true if the previous run was public (marker file exists).
pub fn was_previously_public() -> bool {
    was_public_path().exists()
}

/// Clear identity files (key, nostr.nsec, mesh-id, last-mesh, was-public) so the
/// next start gets a completely fresh identity. Called when transitioning from
/// public → private to avoid reusing a publicly-known identity in a private mesh.
pub fn clear_public_identity() {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let dir = home.join(".mesh-llm");
    let mut ok = true;
    for name in &["key", "nostr.nsec", "mesh-id", "last-mesh"] {
        let p = dir.join(name);
        if p.exists() {
            if std::fs::remove_file(&p).is_ok() {
                tracing::info!("Cleared {}", p.display());
            } else {
                tracing::warn!("Failed to clear {}", p.display());
                ok = false;
            }
        }
    }
    // Only remove the marker after identity files are gone, so a failed
    // cleanup is retried on the next private start.
    let marker = dir.join("was-public");
    if ok {
        let _ = std::fs::remove_file(&marker);
    } else {
        tracing::warn!("Keeping was-public marker — will retry cleanup next start");
    }
}

/// Load secret key from ~/.mesh-llm/key, or create a new one and save it.
async fn load_or_create_key() -> Result<SecretKey> {
    let key_path = default_node_key_path()?;
    let dir = key_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid node key path {}", key_path.display()))?;
    ensure_private_node_key_dir(dir)?;

    if key_path.exists() {
        ensure_private_node_key_file(&key_path)?;
        let hex = tokio::fs::read_to_string(&key_path).await?;
        let bytes = hex::decode(hex.trim())?;
        if bytes.len() != 32 {
            anyhow::bail!("Invalid key length in {}", key_path.display());
        }
        let key = SecretKey::from_bytes(&bytes.try_into().unwrap());
        tracing::info!("Loaded key from {}", key_path.display());
        return Ok(key);
    }

    let key = SecretKey::generate();
    save_node_key_to_path(&key_path, &key)?;
    tracing::info!("Generated new key, saved to {}", key_path.display());
    Ok(key)
}

pub fn default_node_key_path() -> Result<std::path::PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".mesh-llm").join("key"))
}

pub fn load_node_key_from_path(path: &std::path::Path) -> Result<SecretKey> {
    let hex = std::fs::read_to_string(path)?;
    let bytes = hex::decode(hex.trim())?;
    if bytes.len() != 32 {
        anyhow::bail!("Invalid key length in {}", path.display());
    }
    Ok(SecretKey::from_bytes(&bytes.try_into().unwrap()))
}

pub fn save_node_key_to_path(path: &std::path::Path, key: &SecretKey) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid node key path {}", path.display()))?;
    ensure_private_node_key_dir(parent)?;
    if path.exists() {
        ensure_private_node_key_file(path)?;
    }
    crate::crypto::write_keystore_bytes_atomically(path, hex::encode(key.to_bytes()).as_bytes())?;
    ensure_private_node_key_file(path)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_node_key_dir(dir: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(dir)?;
    let metadata = std::fs::metadata(dir)?;
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o700);
        std::fs::set_permissions(dir, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_node_key_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_node_key_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("Node key path is not a regular file");
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_node_key_file(path: &std::path::Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("Node key path is not a regular file");
    }
    Ok(())
}

mod gossip;
mod heartbeat;
pub use gossip::backfill_legacy_descriptors;
#[allow(unused_imports)]
use gossip::{apply_transitive_ann, peer_meaningfully_changed};
#[allow(unused_imports)]
use heartbeat::{
    heartbeat_failure_policy_for_peer, HeartbeatFailurePolicy, MOE_RECOVERY_PROBATION_SECS,
};
pub(crate) use heartbeat::{
    moe_recovery_ready_at, peer_is_eligible_for_active_moe, resolve_peer_down,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod public_identity_tests;
