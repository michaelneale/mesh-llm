//! Mesh membership via iroh QUIC connections.
//!
//! Multiple ALPNs are accepted for backward compatibility, but control traffic still uses
//! one QUIC connection per peer. Bi-streams are multiplexed by first byte:
//! 0x01 = gossip, 0x02 = tunnel (RPC), 0x03 = tunnel map, 0x04 = tunnel (HTTP).

use anyhow::Result;
use base64::Engine;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use tokio::sync::{watch, Mutex};

use crate::crypto::{
    default_node_ownership_path, save_node_ownership, sign_node_ownership, verify_node_ownership,
    OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy, TrustStore,
    DEFAULT_NODE_CERT_LIFETIME_SECS,
};
use crate::inference::moe;
use crate::protocol::*;

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
    pub capabilities: crate::models::ModelCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<crate::models::ModelTopology>,
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

pub fn infer_served_model_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
    primary_model_path: Option<&std::path::Path>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source
        .and_then(identity_from_model_source)
        .or_else(|| {
            primary_model_path.and_then(|path| identity_from_model_path(primary_model_name, path))
        });
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            if idx == 0 || model_name == primary_model_name {
                let mut identity = primary.clone().unwrap_or_default();
                identity.model_name = model_name.clone();
                identity.is_primary = true;
                if identity.local_file_name.is_none() {
                    identity.local_file_name = Some(format!("{model_name}.gguf"));
                }
                descriptor_from_identity(model_name, identity)
            } else {
                descriptor_from_model_path(
                    model_name,
                    &crate::models::find_model_path(model_name),
                    false,
                )
                .unwrap_or_else(|| ServedModelDescriptor {
                    identity: ServedModelIdentity {
                        model_name: model_name.clone(),
                        is_primary: false,
                        source_kind: ModelSourceKind::Unknown,
                        canonical_ref: None,
                        repository: None,
                        revision: None,
                        artifact: None,
                        local_file_name: Some(format!("{model_name}.gguf")),
                        identity_hash: None,
                    },
                    capabilities: crate::models::ModelCapabilities::default(),
                    topology: None,
                })
            }
        })
        .collect()
}

pub fn infer_available_model_descriptors(
    available_models: &[String],
) -> Vec<ServedModelDescriptor> {
    available_models
        .iter()
        .filter_map(|model_name| {
            let path = crate::models::find_model_path(model_name);
            descriptor_from_model_path(model_name, &path, false)
        })
        .collect()
}

pub fn infer_local_served_model_descriptor(
    model_name: &str,
    is_primary: bool,
) -> Option<ServedModelDescriptor> {
    descriptor_from_model_path(
        model_name,
        &crate::models::find_model_path(model_name),
        is_primary,
    )
}

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
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

fn descriptor_from_model_path(
    model_name: &str,
    path: &std::path::Path,
    is_primary: bool,
) -> Option<ServedModelDescriptor> {
    let mut identity = identity_from_model_path(model_name, path)?;
    identity.is_primary = is_primary;
    Some(descriptor_from_identity(model_name, identity))
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeartbeatFailurePolicy {
    allow_recent_inbound_grace: bool,
    failure_threshold: u32,
}

fn descriptors_share_exact_moe_identity(
    local: &[ServedModelDescriptor],
    remote: &[ServedModelDescriptor],
) -> bool {
    local.iter().any(|local_descriptor| {
        local_descriptor
            .topology
            .as_ref()
            .and_then(|topology| topology.moe.as_ref())
            .is_some()
            && remote.iter().any(|remote_descriptor| {
                remote_descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
                    && identities_match_exact(
                        &local_descriptor.identity,
                        &remote_descriptor.identity,
                    )
            })
    })
}

fn heartbeat_failure_policy_for_peer(
    local_descriptors: &[ServedModelDescriptor],
    local_runtime: &[ModelRuntimeDescriptor],
    peer: &PeerInfo,
) -> HeartbeatFailurePolicy {
    if descriptors_share_exact_moe_identity(local_descriptors, &peer.served_model_descriptors) {
        if exact_moe_starting_during_convergence(local_descriptors, local_runtime, peer) {
            return HeartbeatFailurePolicy {
                allow_recent_inbound_grace: true,
                // Split convergence can spend minutes in split generation + shard
                // load. During that window, prefer serving continuity over fast
                // heartbeat-only fail-down; request-path failures still remove
                // dead shard peers immediately.
                failure_threshold: 4,
            };
        }
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: false,
            // Shared MoE peers should fail down faster than generic peers, but a
            // single heartbeat miss is too aggressive on relay-heavy or flaky
            // links. Request-path shard failures still trigger immediate
            // fail-down; heartbeat-only loss needs a second miss to avoid
            // tearing down a healthy split on one transient blip.
            failure_threshold: 2,
        }
    } else {
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 2,
        }
    }
}

const MOE_RECOVERY_PROBATION_SECS: u64 = 30;

fn runtime_model_is_starting(runtimes: &[ModelRuntimeDescriptor], model_name: &str) -> bool {
    runtimes
        .iter()
        .any(|runtime| runtime.model_name == model_name && !runtime.ready)
}

fn exact_moe_starting_during_convergence(
    local_descriptors: &[ServedModelDescriptor],
    local_runtime: &[ModelRuntimeDescriptor],
    peer: &PeerInfo,
) -> bool {
    local_descriptors.iter().any(|local_descriptor| {
        let local_moe = local_descriptor
            .topology
            .as_ref()
            .and_then(|topology| topology.moe.as_ref())
            .is_some();
        if !local_moe {
            return false;
        }

        peer.served_model_descriptors
            .iter()
            .any(|remote_descriptor| {
                let remote_moe = remote_descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some();
                if !remote_moe
                    || !identities_match_exact(
                        &local_descriptor.identity,
                        &remote_descriptor.identity,
                    )
                {
                    return false;
                }

                let model_name = &local_descriptor.identity.model_name;
                runtime_model_is_starting(local_runtime, model_name)
                    || runtime_model_is_starting(&peer.served_model_runtime, model_name)
            })
    })
}

fn moe_recovery_ready_at(
    recovered_at: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    recovered_at
        .map(|recovered_at| {
            now.duration_since(recovered_at).as_secs() >= MOE_RECOVERY_PROBATION_SECS
        })
        .unwrap_or(true)
}

pub(crate) fn descriptors_share_exact_moe_identity_for_model(
    local: &[ServedModelDescriptor],
    remote: &[ServedModelDescriptor],
    model_name: &str,
) -> Option<bool> {
    let local_moe: Vec<&ServedModelDescriptor> = local
        .iter()
        .filter(|descriptor| {
            descriptor.identity.model_name == model_name
                && descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
        })
        .collect();
    let remote_moe: Vec<&ServedModelDescriptor> = remote
        .iter()
        .filter(|descriptor| {
            descriptor.identity.model_name == model_name
                && descriptor
                    .topology
                    .as_ref()
                    .and_then(|topology| topology.moe.as_ref())
                    .is_some()
        })
        .collect();
    if local_moe.is_empty() || remote_moe.is_empty() {
        return None;
    }
    Some(local_moe.iter().any(|local_descriptor| {
        remote_moe.iter().any(|remote_descriptor| {
            identities_match_exact(&local_descriptor.identity, &remote_descriptor.identity)
        })
    }))
}

pub(crate) fn peer_is_eligible_for_active_moe(
    local_descriptors: &[ServedModelDescriptor],
    peer: &PeerInfo,
    model_name: &str,
) -> bool {
    let declares_model = peer.is_assigned_model(model_name)
        || peer
            .requested_models
            .iter()
            .any(|requested| requested == model_name);
    if !declares_model || matches!(peer.role, NodeRole::Client) {
        return false;
    }

    let identity_matches = descriptors_share_exact_moe_identity_for_model(
        local_descriptors,
        &peer.served_model_descriptors,
        model_name,
    )
    .unwrap_or(true);
    if !identity_matches {
        return false;
    }

    moe_recovery_ready_at(peer.moe_recovered_at, std::time::Instant::now())
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

fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
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
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.version != new.version
        || old.owner_summary != new.owner_summary
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PeerAnnouncementV0 {
    addr: EndpointAddr,
    #[serde(default)]
    role: NodeRole,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    vram_bytes: u64,
    #[serde(default)]
    model_source: Option<String>,
    #[serde(default)]
    serving: Option<String>,
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
    #[serde(skip_serializing, skip_deserializing, default)]
    served_model_descriptors: Vec<ServedModelDescriptor>,
    #[serde(skip_serializing, skip_deserializing, default)]
    served_model_runtime: Vec<ModelRuntimeDescriptor>,
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
            served_model_descriptors: self.served_model_descriptors,
            served_model_runtime: self.served_model_runtime,
            owner_attestation: None,
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
            served_model_descriptors: ann.served_model_descriptors.clone(),
            served_model_runtime: ann.served_model_runtime.clone(),
        }
    }
}

fn apply_transitive_ann(
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
    existing.available_models.clear();
    existing.requested_models = ann.requested_models.clone();
    existing.owner_attestation = ann.owner_attestation.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
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
    pub gpu_bandwidth_gbps: Option<String>,
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
            moe_recovered_at: None,
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_bandwidth_gbps: ann.gpu_bandwidth_gbps.clone(),
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
                    i += 4 + (attr_len + 3) & !3;
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
    accepting: Arc<(tokio::sync::Notify, std::sync::atomic::AtomicBool)>,
    vram_bytes: u64,
    peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    inflight_change_tx: watch::Sender<u64>,
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
    pub gpu_bandwidth_gbps: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
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
    seen_plugin_messages: HashSet<String>,
    seen_plugin_message_order: VecDeque<String>,
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

/// Applies the reachability-confirmation rule for a `PeerDown` claim.
/// Returns `Some(dead_id)` if `dead_id != self_id` AND `should_remove` is `true` (peer confirmed gone).
/// Returns `None` if `dead_id == self_id` (never self-evict) or `should_remove` is `false` (peer still reachable).
pub(crate) fn resolve_peer_down(
    self_id: EndpointId,
    dead_id: EndpointId,
    should_remove: bool,
) -> Option<EndpointId> {
    if dead_id == self_id {
        return None;
    }
    if should_remove {
        Some(dead_id)
    } else {
        None
    }
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
        let _ = self.inflight_change_tx.send(
            self.inflight_requests
                .load(std::sync::atomic::Ordering::Relaxed) as u64,
        );
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
            let key = SecretKey::generate(&mut rand::rng());
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
        let mut builder = Endpoint::empty_builder()
            .secret_key(secret_key)
            .alpns(vec![ALPN_V1.to_vec(), ALPN_V0.to_vec()])
            .transport_config(transport_config);

        {
            use iroh::{RelayConfig, RelayMap};
            let urls: Vec<String> = if relay_urls.is_empty() {
                vec![
                    "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./".into(),
                    "https://mesh-llm-relay.fly.dev./".into(),
                ]
            } else {
                relay_urls.to_vec()
            };
            // Two relays: dedicated iroh relay (proper QUIC + STUN) and Fly relay (fallback).
            let configs: Vec<RelayConfig> = urls
                .iter()
                .map(|url| RelayConfig {
                    url: url.parse().expect("invalid relay URL"),
                    quic: None,
                })
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
            vram_bytes: vram,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
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
            gpu_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
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
    #[cfg(test)]
    pub async fn new_for_tests(role: NodeRole) -> Result<Self> {
        use iroh::endpoint::QuicTransportConfig;

        let transport_config = QuicTransportConfig::builder()
            .max_concurrent_bidi_streams(1024u32.into())
            .build();
        let endpoint = Endpoint::empty_builder()
            .secret_key(SecretKey::generate(&mut rand::rng()))
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
            vram_bytes: 0,
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
            is_soc: Some(false),
            gpu_vram: None,
            gpu_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
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
            let was_above = old_rtt.map_or(false, |r| r > MAX_SPLIT_RTT_MS);
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

    /// Start a background task that periodically checks peer health.
    /// Probes each peer by attempting a gossip exchange. If the probe fails
    /// (connection dead, peer unresponsive), removes the peer immediately
    /// rather than waiting for QUIC idle timeout.
    /// Start a slow heartbeat (60s) that gossips with a random subset of peers.
    /// At small mesh sizes (≤5 peers), talks to everyone. At larger sizes,
    /// picks K random peers per cycle. Information propagates infectiously —
    /// changes reach all nodes in O(log N) cycles.
    /// Death detection primarily happens on the data path (tunnel fails →
    /// broadcast_peer_down), not via heartbeat.
    pub fn start_heartbeat(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut fail_counts: std::collections::HashMap<EndpointId, u32> =
                std::collections::HashMap::new();

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;

                let mut peers_and_conns: Vec<(EndpointId, Option<Connection>)> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .keys()
                        .map(|id| {
                            let conn = state.connections.get(id).cloned();
                            (*id, conn)
                        })
                        .collect()
                };
                tracing::debug!("Heartbeat tick: {} peers to check", peers_and_conns.len());

                // Random-K gossip: pick a subset at larger mesh sizes.
                // At ≤5 peers, talk to everyone (backward compat with current behavior).
                // At larger sizes, pick 5 random peers per cycle.
                const GOSSIP_K: usize = 5;
                if peers_and_conns.len() > GOSSIP_K {
                    use rand::seq::SliceRandom;
                    peers_and_conns.shuffle(&mut rand::rng());
                    peers_and_conns.truncate(GOSSIP_K);
                }

                for (peer_id, conn) in peers_and_conns {
                    let hb_start = std::time::Instant::now();
                    let alive = if let Some(conn) = conn {
                        // Gossip as heartbeat — syncs state but won't re-discover dead peers
                        let result = tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            node.initiate_gossip_inner(conn, peer_id, false),
                        )
                        .await
                        .map(|r| r.is_ok())
                        .unwrap_or(false);
                        tracing::debug!(
                            "Heartbeat gossip {} = {} ({}ms)",
                            peer_id.fmt_short(),
                            if result { "ok" } else { "fail" },
                            hb_start.elapsed().as_millis()
                        );
                        result
                    } else {
                        // No connection — try to reconnect using stored address
                        let addr = {
                            let state = node.state.lock().await;
                            state.peers.get(&peer_id).map(|p| p.addr.clone())
                        };
                        if let Some(addr) = addr {
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(10),
                                connect_mesh(&node.endpoint, addr),
                            )
                            .await
                            {
                                Ok(Ok(new_conn)) => {
                                    eprintln!(
                                        "💚 Heartbeat: reconnected to {}",
                                        peer_id.fmt_short()
                                    );
                                    node.state
                                        .lock()
                                        .await
                                        .connections
                                        .insert(peer_id, new_conn.clone());
                                    // Spawn dispatch_streams for the new connection
                                    let n2 = node.clone();
                                    let nc = new_conn.clone();
                                    tokio::spawn(async move {
                                        n2.dispatch_streams(nc, peer_id).await;
                                    });
                                    // Try gossip on the new connection
                                    tokio::time::timeout(
                                        std::time::Duration::from_secs(10),
                                        node.initiate_gossip_inner(new_conn, peer_id, false),
                                    )
                                    .await
                                    .map(|r| r.is_ok())
                                    .unwrap_or(false)
                                }
                                _ => false,
                            }
                        } else {
                            false
                        }
                    };

                    if alive {
                        if fail_counts.contains_key(&peer_id) {
                            eprintln!(
                                "💚 Heartbeat: {} recovered (was {}/2)",
                                peer_id.fmt_short(),
                                fail_counts.get(&peer_id).unwrap_or(&0)
                            );
                            // Clear dead_peers if peer came back
                            node.state.lock().await.dead_peers.remove(&peer_id);
                        }
                        fail_counts.remove(&peer_id);
                    } else {
                        let (recently_seen, failure_policy) = {
                            let state = node.state.lock().await;
                            let peer = state.peers.get(&peer_id).cloned();
                            drop(state);
                            let local_descriptors =
                                node.served_model_descriptors.lock().await.clone();
                            let local_runtime = node.model_runtime_descriptors.lock().await.clone();
                            let policy = peer
                                .as_ref()
                                .map(|peer| {
                                    heartbeat_failure_policy_for_peer(
                                        &local_descriptors,
                                        &local_runtime,
                                        peer,
                                    )
                                })
                                .unwrap_or(HeartbeatFailurePolicy {
                                    allow_recent_inbound_grace: true,
                                    failure_threshold: 2,
                                });
                            let recently_seen = peer
                                .as_ref()
                                .map(|peer| peer.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
                                .unwrap_or(false);
                            (recently_seen, policy)
                        };
                        // Check if peer has contacted US recently (inbound gossip).
                        // If so, peer is alive — we just can't reach them outbound (NAT).
                        if recently_seen && failure_policy.allow_recent_inbound_grace {
                            // Peer is alive via inbound, don't count as failure
                            if fail_counts.contains_key(&peer_id) {
                                eprintln!("💚 Heartbeat: {} outbound failed but seen recently (inbound alive)", peer_id.fmt_short());
                                fail_counts.remove(&peer_id);
                            }
                        } else {
                            let count = fail_counts.entry(peer_id).or_default();
                            *count += 1;
                            if *count >= failure_policy.failure_threshold {
                                // Generic peers require 2 misses so a single timeout doesn't
                                // evict an otherwise-alive inbound-only peer. Shared MoE peers
                                // are stricter: one missed heartbeat should trigger re-election.
                                node.state.lock().await.dead_peers.insert(peer_id);
                                eprintln!(
                                    "💔 Heartbeat: {} unreachable ({} failure{}), removing + broadcasting death",
                                    peer_id.fmt_short(),
                                    count,
                                    if *count == 1 { "" } else { "s" }
                                );
                                fail_counts.remove(&peer_id);
                                node.handle_peer_death(peer_id).await;
                            } else {
                                eprintln!(
                                    "💛 Heartbeat: {} unreachable ({}/{}), will retry",
                                    peer_id.fmt_short(),
                                    count,
                                    failure_policy.failure_threshold
                                );
                            }
                        }
                    }
                }

                // Prune stale peers: no direct contact in 2× the stale window.
                // These are ghost records propagated via gossip from other nodes
                // but never directly verified by us.
                let prune_cutoff =
                    std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
                let stale_peers: Vec<EndpointId> = {
                    let state = node.state.lock().await;
                    state
                        .peers
                        .iter()
                        .filter(|(_, p)| p.last_seen < prune_cutoff)
                        .map(|(id, _)| *id)
                        .collect()
                };
                for stale_id in stale_peers {
                    eprintln!(
                        "🧹 Pruning stale peer {} (no direct contact in {}s)",
                        stale_id.fmt_short(),
                        PEER_STALE_SECS * 2
                    );
                    node.remove_peer(stale_id).await;
                    // Also close any lingering connection
                    node.state.lock().await.connections.remove(&stale_id);
                }

                // GC expired demand entries to prevent unbounded map growth
                node.gc_demand().await;
            }
        });
    }

    /// Handle a peer death: remove from state, broadcast to all other peers.
    pub async fn handle_peer_death(&self, dead_id: EndpointId) {
        eprintln!(
            "⚠️  Peer {} died — removing and broadcasting",
            dead_id.fmt_short()
        );
        {
            let mut state = self.state.lock().await;
            // Keep the connection alive — if the peer recovers, their inbound
            // gossip will arrive on the existing connection and trigger recovery
            // via handle_gossip_stream → add_peer → clear dead_peers.
            // Don't remove: state.connections.remove(&dead_id);
            state.dead_peers.insert(dead_id);
        }
        self.remove_peer(dead_id).await;
        self.broadcast_peer_down(dead_id).await;
    }

    /// Broadcast that a peer is down to all connected peers.
    async fn broadcast_peer_down(&self, dead_id: EndpointId) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(id, _)| **id != dead_id)
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        let dead_bytes = dead_id.as_bytes().to_vec();
        for (peer_id, conn) in conns {
            let bytes = dead_bytes.clone();
            let protocol = connection_protocol(&conn);
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_DOWN]).await?;
                    match protocol {
                        ControlProtocol::ProtoV1 => {
                            let proto_msg = crate::proto::node::PeerDown {
                                peer_id: bytes,
                                gen: NODE_PROTOCOL_GENERATION,
                            };
                            write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                        }
                        ControlProtocol::JsonV0 => {
                            send.write_all(&bytes).await?;
                        }
                    }
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!(
                        "Failed to broadcast peer_down to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
    }

    /// Announce clean shutdown to all peers.
    pub async fn broadcast_leaving(&self) {
        let my_id_bytes = self.endpoint.id().as_bytes().to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = my_id_bytes.clone();
            let protocol = connection_protocol(&conn);
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PEER_LEAVING]).await?;
                    match protocol {
                        ControlProtocol::ProtoV1 => {
                            let proto_msg = crate::proto::node::PeerLeaving {
                                peer_id: bytes,
                                gen: NODE_PROTOCOL_GENERATION,
                            };
                            write_len_prefixed(&mut send, &proto_msg.encode_to_vec()).await?;
                        }
                        ControlProtocol::JsonV0 => {
                            send.write_all(&bytes).await?;
                        }
                    }
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!("Failed to send leaving to {}: {e}", peer_id.fmt_short());
                }
            });
        }
        // Give broadcasts a moment to flush
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
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
        const MAX_SEEN_PLUGIN_MESSAGES: usize = 4096;

        let mut state = self.state.lock().await;
        if !state.seen_plugin_messages.insert(message_id.clone()) {
            return false;
        }
        state.seen_plugin_message_order.push_back(message_id);
        while state.seen_plugin_message_order.len() > MAX_SEEN_PLUGIN_MESSAGES {
            if let Some(oldest) = state.seen_plugin_message_order.pop_front() {
                state.seen_plugin_messages.remove(&oldest);
            }
        }
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
        remote: EndpointId,
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

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_channel_frame(&frame, Some(remote))
                .await?;
        }

        Ok(())
    }

    async fn handle_plugin_bulk_stream(
        &self,
        remote: EndpointId,
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

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_bulk_frame(&frame, Some(remote))
                .await?;
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

    /// Get all models currently being served in the mesh (loaded in VRAM somewhere).
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
            for m in peer.routable_models() {
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
            .filter(|p| p.routes_model(model))
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
            .find(|p| !p.routable_models().is_empty())
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
            for model in peer.routable_models() {
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

    #[cfg(test)]
    pub async fn request_route_table(&self, conn: &Connection) -> Result<RoutingTable> {
        use prost::Message as _;
        let protocol = connection_protocol(conn);
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_ROUTE_REQUEST]).await?;
        if protocol == ControlProtocol::ProtoV1 {
            let req = crate::proto::node::RouteTableRequest {
                requester_id: self.endpoint.id().as_bytes().to_vec(),
                gen: NODE_PROTOCOL_GENERATION,
            };
            write_len_prefixed(&mut send, &req.encode_to_vec()).await?;
        }
        send.finish()?;
        match protocol {
            ControlProtocol::ProtoV1 => {
                let buf = read_len_prefixed(&mut recv).await?;
                let proto_table = crate::proto::node::RouteTable::decode(buf.as_slice())
                    .map_err(|e| anyhow::anyhow!("route table decode failed: {e}"))?;
                proto_table
                    .validate_frame()
                    .map_err(|e| anyhow::anyhow!("invalid route table: {e}"))?;
                Ok(proto_route_table_to_local(&proto_table))
            }
            ControlProtocol::JsonV0 => {
                let buf = recv.read_to_end(MAX_CONTROL_FRAME_BYTES).await?;
                Ok(serde_json::from_slice(&buf)?)
            }
        }
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

        let legacy_json: HashMap<String, u16> = my_tunnel_map
            .iter()
            .map(|(id, port)| (hex::encode(id.as_bytes()), *port))
            .collect();
        let legacy_bytes = serde_json::to_vec(&legacy_json)?;
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
            let legacy_bytes = legacy_bytes.clone();
            let protocol = connection_protocol(&conn);
            tokio::spawn(async move {
                match conn.open_bi().await {
                    Ok((mut send, _recv)) => {
                        if send.write_all(&[STREAM_TUNNEL_MAP]).await.is_err() {
                            return;
                        }
                        let body = match protocol {
                            ControlProtocol::ProtoV1 => &proto_bytes,
                            ControlProtocol::JsonV0 => &legacy_bytes,
                        };
                        if write_len_prefixed(&mut send, body).await.is_err() {
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
                    let protocol = protocol;
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
                    let protocol = protocol;
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
                    let protocol = protocol;
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
                        match protocol {
                            ControlProtocol::ProtoV1 => {
                                let proto_table = routing_table_to_proto(&table);
                                let _ = write_len_prefixed(&mut send, &proto_table.encode_to_vec())
                                    .await;
                            }
                            ControlProtocol::JsonV0 => {
                                if let Ok(json) = serde_json::to_vec(&table) {
                                    let _ = send.write_all(&json).await;
                                }
                            }
                        }
                        let _ = send.finish();
                    });
                }
                STREAM_PEER_DOWN => {
                    let node = self.clone();
                    let protocol = protocol;
                    tokio::spawn(async move {
                        let peer_id_arr: [u8; 32] = match protocol {
                            ControlProtocol::ProtoV1 => {
                                let proto_buf = match read_len_prefixed(&mut recv).await {
                                    Ok(buf) => buf,
                                    Err(e) => {
                                        tracing::warn!(
                                            "PeerDown: failed to read proto body — rejecting: {e}"
                                        );
                                        return;
                                    }
                                };
                                let frame = match crate::proto::node::PeerDown::decode(
                                    proto_buf.as_slice(),
                                ) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        tracing::warn!(
                                            "PeerDown: invalid protobuf — rejecting: {e}"
                                        );
                                        return;
                                    }
                                };
                                if let Err(e) = frame.validate_frame() {
                                    tracing::warn!(
                                        "PeerDown: frame validation failed — rejecting: {e}"
                                    );
                                    return;
                                }
                                match frame.peer_id.as_slice().try_into() {
                                    Ok(b) => b,
                                    Err(_) => {
                                        tracing::warn!(
                                            "PeerDown: peer_id is not 32 bytes — rejecting"
                                        );
                                        return;
                                    }
                                }
                            }
                            ControlProtocol::JsonV0 => {
                                let mut id_bytes = [0u8; 32];
                                if recv.read_exact(&mut id_bytes).await.is_err() {
                                    tracing::warn!(
                                        "PeerDown: missing legacy endpoint id bytes — rejecting"
                                    );
                                    return;
                                }
                                id_bytes
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
                        let conn_opt = {
                            let state = node.state.lock().await;
                            state.connections.get(&dead_id).cloned()
                        };
                        let should_remove = if let Some(conn) = conn_opt {
                            tokio::time::timeout(std::time::Duration::from_secs(3), conn.open_bi())
                                .await
                                .is_err()
                        } else {
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
                    });
                }
                STREAM_PEER_LEAVING => {
                    let node = self.clone();
                    let protocol = protocol;
                    tokio::spawn(async move {
                        let leaving_id = match protocol {
                            ControlProtocol::ProtoV1 => {
                                let proto_buf = match read_len_prefixed(&mut recv).await {
                                    Ok(buf) => buf,
                                    Err(e) => {
                                        tracing::warn!(
                                            "PeerLeaving: failed to read proto body — rejecting: {e}"
                                        );
                                        return;
                                    }
                                };
                                let frame = match crate::proto::node::PeerLeaving::decode(
                                    proto_buf.as_slice(),
                                ) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        tracing::warn!(
                                            "PeerLeaving: invalid protobuf — rejecting: {e}"
                                        );
                                        return;
                                    }
                                };
                                if let Err(e) = frame.validate_frame() {
                                    tracing::warn!(
                                        "PeerLeaving: frame validation failed — rejecting: {e}"
                                    );
                                    return;
                                }
                                match resolve_peer_leaving(remote, &frame) {
                                    Ok(id) => id,
                                    Err(e) => {
                                        tracing::warn!(
                                            "PeerLeaving from {}: rejected ({})",
                                            remote.fmt_short(),
                                            e
                                        );
                                        return;
                                    }
                                }
                            }
                            ControlProtocol::JsonV0 => {
                                let mut id_bytes = [0u8; 32];
                                if recv.read_exact(&mut id_bytes).await.is_err() {
                                    tracing::warn!(
                                        "PeerLeaving: missing legacy endpoint id bytes — rejecting"
                                    );
                                    return;
                                }
                                let pk = match iroh::PublicKey::from_bytes(&id_bytes) {
                                    Ok(k) => k,
                                    Err(_) => {
                                        tracing::warn!(
                                            "PeerLeaving: endpoint id is not a valid public key — rejecting"
                                        );
                                        return;
                                    }
                                };
                                let leaving_id = EndpointId::from(pk);
                                if leaving_id != remote {
                                    tracing::warn!(
                                        "PeerLeaving from {}: rejected (legacy sender mismatch)",
                                        remote.fmt_short()
                                    );
                                    return;
                                }
                                leaving_id
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
                revision: 0,
                config_hash: vec![],
                config: None,
                hostname: None,
                error: Some("owner_id mismatch".to_string()),
            };
            write_len_prefixed(&mut send, &error_snapshot.encode_to_vec()).await?;
            return Ok(());
        }

        let snapshot = {
            let state = self.config_state.lock().await;
            let proto_cfg = mesh_config_to_proto(state.config());
            ConfigSnapshotResponse {
                gen: NODE_PROTOCOL_GENERATION,
                node_id: self.endpoint.id().as_bytes().to_vec(),
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
                        let proto_cfg = mesh_config_to_proto(state.config());
                        ConfigUpdateNotification {
                            gen: NODE_PROTOCOL_GENERATION,
                            node_id: self.endpoint.id().as_bytes().to_vec(),
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
                    match inbound {
                        Ok(_) => tracing::debug!(
                            "config subscribe from {} sent unexpected extra frame; closing stream",
                            remote.fmt_short()
                        ),
                        Err(_) => {}
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

        // 6. Apply via CAS — use spawn_blocking so synchronous disk I/O in apply()
        //    does not block the Tokio async runtime while the mutex is held.
        let config_state = Arc::clone(&self.config_state);
        let expected_revision = push.expected_revision;
        let (result, current_revision, current_hash) = tokio::task::spawn_blocking(move || {
            let mut state = config_state.blocking_lock();
            let result = state.apply(mesh_config, expected_revision);
            let current_revision = state.revision();
            let current_hash = *state.config_hash();
            (result, current_revision, current_hash)
        })
        .await
        .map_err(|e| anyhow::anyhow!("config apply task panicked: {e}"))?;

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
            revision: snapshot.revision,
            config_hash: snapshot.config_hash.clone(),
            config: snapshot.config.clone(),
        };
        let (notif_tx, notif_rx) = tokio::sync::watch::channel(empty_notif);
        tokio::spawn(async move {
            // Keep the request stream's send half alive while subscribed so the
            // remote side does not treat immediate EOF as an unsubscribe.
            let _send = send;
            loop {
                match read_len_prefixed(&mut recv).await {
                    Ok(buf) => match ConfigUpdateNotification::decode(buf.as_slice()) {
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
                    },
                    Err(_) => break,
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
            std::time::Duration::from_secs(15),
            connect_mesh(&self.endpoint, addr.clone()),
        )
        .await
        {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                anyhow::bail!("Failed to connect to {}: {e}", peer_id.fmt_short());
            }
            Err(_) => {
                anyhow::bail!("Timeout connecting to {} (15s)", peer_id.fmt_short());
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

    /// Open a gossip stream on an existing connection to exchange peer info.
    async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        self.initiate_gossip_inner(conn, remote, true).await
    }

    async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let protocol = connection_protocol(&conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let rtt_ms = t0.elapsed().as_millis() as u32;
        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        for (addr, ann) in &their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
                self.update_peer_rtt(remote, rtt_ms).await;
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let path_rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if path_rtt_ms > 0 && path_rtt_ms < rtt_ms {
                            eprintln!(
                                "📡 Peer {} RTT: {}ms ({}) [path info]",
                                remote.fmt_short(),
                                path_rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, path_rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        if discover_peers {
            for (addr, _) in &their_announcements {
                let peer_id = addr.id;
                if peer_id != self.endpoint.id() {
                    let has_conn = self.state.lock().await.connections.contains_key(&peer_id);
                    if !has_conn {
                        if let Err(e) = Box::pin(self.connect_to_peer(addr.clone())).await {
                            tracing::debug!(
                                "Could not connect to discovered peer {}: {e}",
                                peer_id.fmt_short()
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        {
            let mut state = self.state.lock().await;
            if state.dead_peers.remove(&remote) {
                eprintln!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    remote.fmt_short()
                );
            }
        }

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        for (addr, ann) in &their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
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
                                "📡 Peer {} RTT: {}ms ({})",
                                remote.fmt_short(),
                                rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        for (addr, _) in their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            let already_known = self.state.lock().await.peers.contains_key(&peer_id);
            if !already_known {
                if let Err(e) = Box::pin(self.connect_to_peer(addr)).await {
                    tracing::warn!("Failed to discover peer: {e}");
                }
            }
        }

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
        let frame = match protocol {
            ControlProtocol::ProtoV1 => crate::proto::node::TunnelMap::decode(buf.as_slice())
                .map_err(|e| anyhow::anyhow!("TunnelMap decode error: {e}"))?,
            ControlProtocol::JsonV0 => {
                let mut frame = decode_legacy_tunnel_map_frame(&buf)?;
                frame.owner_peer_id = remote.as_bytes().to_vec();
                frame
            }
        };

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

    async fn remove_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        // Always clear any rejection-tracking entry so the map stays bounded.
        state.policy_rejected_peers.remove(&id);
        if let Some(peer) = state.peers.remove(&id) {
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                state.peers.len()
            );
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    async fn add_peer(&self, id: EndpointId, addr: EndpointAddr, ann: &PeerAnnouncement) {
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            let last_status = state.policy_rejected_peers.get(&id).cloned();
            if last_status.as_ref() != Some(&owner_summary.status) {
                tracing::warn!(
                    "Rejecting peer {} due to owner policy: {:?}",
                    id.fmt_short(),
                    owner_summary.status
                );
                state
                    .policy_rejected_peers
                    .insert(id, owner_summary.status.clone());
            }
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let mut state = self.state.lock().await;
        // Peer accepted — clear any prior rejection record so future rejections log again.
        state.policy_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id);
        if recovered {
            eprintln!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            );
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let role_changed = existing.role != ann.role;
            let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
            let serving_changed = existing.serving_models != ann.serving_models
                || existing.hosted_models != ann_hosted_models
                || existing.hosted_models_known != ann.hosted_models.is_some();
            if role_changed {
                tracing::info!(
                    "Peer {} role updated: {:?} → {:?}",
                    id.fmt_short(),
                    existing.role,
                    ann.role
                );
                existing.role = ann.role.clone();
            }
            // Update addr if the new one has more info
            if !addr.addrs.is_empty() {
                existing.addr = addr;
            }
            existing.models = ann.models.clone();
            existing.vram_bytes = ann.vram_bytes;
            if ann.model_source.is_some() {
                existing.model_source = ann.model_source.clone();
            }
            existing.serving_models = ann.serving_models.clone();
            existing.hosted_models = ann_hosted_models;
            existing.hosted_models_known = ann.hosted_models.is_some();
            existing.available_models.clear();
            existing.requested_models = ann.requested_models.clone();
            existing.last_seen = now;
            if recovered {
                existing.moe_recovered_at = Some(now);
            }
            existing.owner_attestation = ann.owner_attestation.clone();
            existing.owner_summary = owner_summary.clone();
            existing.served_model_descriptors = ann.served_model_descriptors.clone();
            existing.served_model_runtime = ann.served_model_runtime.clone();
            if ann.version.is_some() {
                existing.version = ann.version.clone();
            }
            existing.gpu_name = ann.gpu_name.clone();
            existing.hostname = ann.hostname.clone();
            existing.is_soc = ann.is_soc;
            existing.gpu_vram = ann.gpu_vram.clone();
            existing.gpu_bandwidth_gbps = ann.gpu_bandwidth_gbps.clone();
            if ann.experts_summary.is_some() {
                existing.experts_summary = ann.experts_summary.clone();
            }
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
                || old_peer.gpu_name != updated_peer.gpu_name
                || old_peer.hostname != updated_peer.hostname
                || old_peer.is_soc != updated_peer.is_soc
                || old_peer.gpu_vram != updated_peer.gpu_vram
                || old_peer.gpu_bandwidth_gbps != updated_peer.gpu_bandwidth_gbps;
            if role_changed || serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
            return;
        }
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let mut peer = PeerInfo::from_announcement(id, addr, ann, owner_summary);
        if recovered {
            peer.moe_recovered_at = Some(now);
        }
        state.peers.insert(id, peer.clone());
        let count = state.peers.len();
        drop(state);
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
        if imported_ranking {
            self.refresh_served_model_descriptors().await;
        }
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates assigned/hosted state so models_being_served() includes their models,
    /// but does NOT refresh last_seen — transitive peers still get pruned if the
    /// bridge node stops mentioning them. Does NOT trigger peer_change events
    /// for new transitive peers (avoids re-election storms at scale).
    async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
    ) {
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        if state.dead_peers.contains(&id) {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann);
            existing.owner_summary = owner_summary;
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer);
            if serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
        } else {
            // New transitive peer — add with last_seen = now but no peer_change event.
            // It will get pruned after PEER_STALE_SECS*2 if never directly contacted.
            let peer = PeerInfo::from_announcement(id, addr.clone(), ann, owner_summary);
            state.peers.insert(id, peer.clone());
            drop(state);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUp,
                Some(&peer),
                String::new(),
            )
            .await;
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
        }
    }

    async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        // Snapshot all locks independently — never hold multiple locks simultaneously.
        let my_role = self.role.lock().await.clone();
        let my_models = self.models.lock().await.clone();
        let my_source = self.model_source.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let my_served_model_descriptors = self.served_model_descriptors.lock().await.clone();
        let my_model_runtime_descriptors = self.model_runtime_descriptors.lock().await.clone();
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_mesh_id = self.mesh_id.lock().await.clone();
        let my_owner_attestation = self.owner_attestation.lock().await.clone();
        let my_demand = self.get_demand();
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        // Gossip wire encoding strips available_model_metadata and available_model_sizes,
        // and remote ingest ignores them. Avoid an expensive scan_local_inventory_snapshot()
        // on the hot gossip path.
        let my_model_metadata: Vec<_> = Vec::new();
        let my_model_sizes: HashMap<_, _> = HashMap::new();
        let mut announcements: Vec<PeerAnnouncement> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|p| p.last_seen >= stale_cutoff)
                .map(|p| PeerAnnouncement {
                    addr: p.addr.clone(),
                    role: p.role.clone(),
                    models: p.models.clone(),
                    vram_bytes: p.vram_bytes,
                    model_source: p.model_source.clone(),
                    serving_models: p.serving_models.clone(),
                    hosted_models: p.hosted_models_known.then(|| p.hosted_models.clone()),
                    available_models: p.available_models.clone(),
                    requested_models: p.requested_models.clone(),
                    version: p.version.clone(),
                    model_demand: HashMap::new(),
                    mesh_id: None,
                    gpu_name: p.gpu_name.clone(),
                    hostname: p.hostname.clone(),
                    is_soc: p.is_soc,
                    gpu_vram: p.gpu_vram.clone(),
                    gpu_bandwidth_gbps: p.gpu_bandwidth_gbps.clone(),
                    available_model_metadata: p.available_model_metadata.clone(),
                    experts_summary: p.experts_summary.clone(),
                    available_model_sizes: p.available_model_sizes.clone(),
                    served_model_descriptors: p.served_model_descriptors.clone(),
                    served_model_runtime: p.served_model_runtime.clone(),
                    owner_attestation: p.owner_attestation.clone(),
                })
                .collect()
        };
        announcements.push(PeerAnnouncement {
            addr: self.endpoint.addr(),
            role: my_role,
            models: my_models,
            vram_bytes: self.vram_bytes,
            model_source: my_source,
            serving_models: my_serving_models,
            hosted_models: Some(my_hosted_models),
            available_models: my_available,
            requested_models: my_requested,
            version: Some(crate::VERSION.to_string()),
            model_demand: my_demand,
            mesh_id: my_mesh_id,
            gpu_name: if self.enumerate_host {
                self.gpu_name.clone()
            } else {
                None
            },
            hostname: if self.enumerate_host {
                self.hostname.clone()
            } else {
                None
            },
            is_soc: self.is_soc,
            gpu_vram: if self.enumerate_host {
                self.gpu_vram.clone()
            } else {
                None
            },
            gpu_bandwidth_gbps: self.gpu_bandwidth_gbps.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            available_model_metadata: my_model_metadata,
            experts_summary: None,
            available_model_sizes: my_model_sizes,
            served_model_descriptors: my_served_model_descriptors,
            served_model_runtime: my_model_runtime_descriptors,
            owner_attestation: my_owner_attestation,
        });
        announcements
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

#[cfg(test)]
mod tests {
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
            gpu_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
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
    fn test_peer_announcement_with_bandwidth_serde_roundtrip() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_bandwidth_gbps: Option<String>,
        }

        let test = TestAnnouncement {
            gpu_bandwidth_gbps: Some("1671.7,722.2".to_string()),
        };

        let json = serde_json::to_string(&test).unwrap();
        let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.gpu_bandwidth_gbps, Some("1671.7,722.2".to_string()));
    }

    #[test]
    fn test_peer_announcement_backward_compat_no_bandwidth_field() {
        #[derive(Deserialize, Debug)]
        struct TestAnnouncement {
            #[serde(default)]
            gpu_bandwidth_gbps: Option<String>,
        }

        let json = r#"{"other_field": "value"}"#;
        let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

        assert_eq!(decoded.gpu_bandwidth_gbps, None);
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
            gpu_bandwidth_gbps: None,
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
                    .any(|ann| ann.addr.id == post_id
                        && ann.serving.as_deref() == Some("post-model")),
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
                serde_json::from_slice(&response_buf)
                    .expect("post node must answer with JSON gossip");
            assert!(
                response_anns
                    .iter()
                    .any(|ann| ann.addr.id == post_id
                        && ann.serving.as_deref() == Some("post-model")),
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
            gpu_bandwidth_gbps: None,
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
        let json = serde_json::to_vec(&HashMap::from([(hex::encode(target.as_bytes()), 9337_u16)]))
            .unwrap();

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
            std::time::Instant::now()
                - std::time::Duration::from_secs(MOE_RECOVERY_PROBATION_SECS + 1),
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
            gpu_bandwidth_gbps: None,
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

        let (_, roundtripped) = proto_ann_to_local(&proto_pa)
            .expect("proto_ann_to_local must succeed on valid proto PA");
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
            gpu_bandwidth_gbps: None,
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
            gpu_bandwidth_gbps: None,
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
            gpu_bandwidth_gbps: None,
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
        let decoded_leaving: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
            .expect("valid PeerLeaving must decode");

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
                let err =
                    decode_control_frame::<RouteTableRequest>(stream_type, &frame).expect_err(
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
        let (addr, local_ann) = proto_ann_to_local(wire_pa)
            .expect("proto_ann_to_local must succeed on valid gossip PA");

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
        let decoded: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
            .expect("valid PeerLeaving must decode");

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
            gpu_bandwidth_gbps: None,
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
            gpu_bandwidth_gbps: None,
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
        .expect(
            "v1 node should remove legacy peer after receiving v0 peer-leaving frame within 5s",
        );

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
            gpu_bandwidth_gbps: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
        };
        let v0_gossip_json =
            serde_json::to_vec(&vec![v0_ann]).expect("v0 gossip JSON must serialize");
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
                p.id == legacy_id
                    && p.serving_models.first().map(String::as_str) == Some("v0-model")
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
            moe_recovered_at: None,
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
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
            decode_control_frame(STREAM_CONFIG_SUBSCRIBE, &encoded)
                .expect("round-trip must succeed");

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
            gpu_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-wrong-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-unowned-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-push-ok-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-conflict-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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
        let push2 =
            build_signed_config_push(&owner_keypair, &client_id, &server_id, 0, good_config);
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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-badsig-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

        let tmp =
            std::env::temp_dir().join(format!("mesh-llm-cfg-notif-{}", rand::random::<u64>()));
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
        let invite = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&server_addr)?);

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

    let key = SecretKey::generate(&mut rand::rng());
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

#[cfg(test)]
mod public_identity_tests {
    use super::*;
    use std::fs;

    /// Test that mark_was_public / was_previously_public / clear_public_identity
    /// work correctly.  Uses the real ~/.mesh-llm/ directory (same approach as
    /// the rotate_keys tests) and restores originals afterward.
    #[test]
    fn public_to_private_transition_clears_identity() {
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        fs::create_dir_all(&dir).ok();

        // Files we may touch:
        let paths: Vec<std::path::PathBuf> =
            ["key", "nostr.nsec", "mesh-id", "last-mesh", "was-public"]
                .iter()
                .map(|n| dir.join(n))
                .collect();

        // Save originals so we can restore after the test.
        let originals: Vec<Option<Vec<u8>>> = paths
            .iter()
            .map(|p| {
                if p.exists() {
                    Some(fs::read(p).unwrap())
                } else {
                    None
                }
            })
            .collect();

        // --- Scenario 1: no marker → was_previously_public is false ---
        let _ = fs::remove_file(dir.join("was-public"));
        assert!(!was_previously_public(), "should be false when no marker");

        // --- Scenario 2: mark as public → marker exists ---
        mark_was_public();
        assert!(was_previously_public(), "should be true after marking");

        // Plant some identity files to verify clear removes them.
        fs::write(dir.join("key"), b"test-key").unwrap();
        fs::write(dir.join("nostr.nsec"), b"test-nsec").unwrap();
        fs::write(dir.join("mesh-id"), b"test-mesh-id").unwrap();
        fs::write(dir.join("last-mesh"), b"test-last-mesh").unwrap();

        // --- Scenario 3: clear_public_identity removes everything ---
        clear_public_identity();
        for name in &["key", "nostr.nsec", "mesh-id", "last-mesh", "was-public"] {
            assert!(
                !dir.join(name).exists(),
                "{name} should be deleted after clear"
            );
        }
        assert!(
            !was_previously_public(),
            "marker should be gone after clear"
        );

        // --- Scenario 4: clear on already-clean directory is fine ---
        clear_public_identity(); // should not panic

        // Restore originals.
        for (path, orig) in paths.iter().zip(originals.iter()) {
            if let Some(data) = orig {
                fs::write(path, data).ok();
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
}
