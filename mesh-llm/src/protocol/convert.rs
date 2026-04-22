#[cfg(test)]
use crate::mesh::RouteEntry;
use crate::mesh::{ModelDemand, NodeRole, PeerAnnouncement, RoutingTable};
use crate::protocol::NODE_PROTOCOL_GENERATION;
use iroh::{EndpointAddr, EndpointId};
use std::collections::HashMap;

fn split_optional_csv(values: Option<&str>) -> Vec<Option<String>> {
    values
        .map(|values| {
            values
                .split(',')
                .map(|value| {
                    let value = value.trim();
                    (!value.is_empty()).then(|| value.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn join_optional_csv(values: &[Option<String>]) -> Option<String> {
    if values.is_empty() {
        return None;
    }

    let has_present_value = values.iter().any(|value| {
        value
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    });

    if !has_present_value {
        return None;
    }

    Some(
        values
            .iter()
            .map(|value| value.clone().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(","),
    )
}

fn local_owner_attestation_to_proto(
    attestation: &crate::crypto::SignedNodeOwnership,
) -> Option<crate::proto::node::SignedNodeOwnership> {
    let owner_sign_public_key = match hex::decode(&attestation.claim.owner_sign_public_key) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                "dropping local owner attestation from gossip: invalid owner_sign_public_key hex: {err}"
            );
            return None;
        }
    };
    let node_endpoint_id = match hex::decode(&attestation.claim.node_endpoint_id) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                "dropping local owner attestation from gossip: invalid node_endpoint_id hex: {err}"
            );
            return None;
        }
    };
    let signature = match hex::decode(&attestation.signature) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                "dropping local owner attestation from gossip: invalid signature hex: {err}"
            );
            return None;
        }
    };
    Some(crate::proto::node::SignedNodeOwnership {
        version: attestation.claim.version,
        cert_id: attestation.claim.cert_id.clone(),
        owner_id: attestation.claim.owner_id.clone(),
        owner_sign_public_key,
        node_endpoint_id,
        issued_at_unix_ms: attestation.claim.issued_at_unix_ms,
        expires_at_unix_ms: attestation.claim.expires_at_unix_ms,
        node_label: attestation.claim.node_label.clone(),
        hostname_hint: attestation.claim.hostname_hint.clone(),
        signature,
    })
}

fn proto_owner_attestation_to_local(
    attestation: &crate::proto::node::SignedNodeOwnership,
) -> crate::crypto::SignedNodeOwnership {
    crate::crypto::SignedNodeOwnership {
        claim: crate::crypto::NodeOwnershipClaim {
            version: attestation.version,
            cert_id: attestation.cert_id.clone(),
            owner_id: attestation.owner_id.clone(),
            owner_sign_public_key: hex::encode(&attestation.owner_sign_public_key),
            node_endpoint_id: hex::encode(&attestation.node_endpoint_id),
            issued_at_unix_ms: attestation.issued_at_unix_ms,
            expires_at_unix_ms: attestation.expires_at_unix_ms,
            node_label: attestation.node_label.clone(),
            hostname_hint: attestation.hostname_hint.clone(),
        },
        signature: hex::encode(&attestation.signature),
    }
}

fn local_source_kind_to_proto(kind: crate::mesh::ModelSourceKind) -> i32 {
    match kind {
        crate::mesh::ModelSourceKind::Catalog => {
            crate::proto::node::ModelSourceKind::Catalog as i32
        }
        crate::mesh::ModelSourceKind::HuggingFace => {
            crate::proto::node::ModelSourceKind::HuggingFace as i32
        }
        crate::mesh::ModelSourceKind::LocalGguf => {
            crate::proto::node::ModelSourceKind::LocalGguf as i32
        }
        crate::mesh::ModelSourceKind::DirectUrl => {
            crate::proto::node::ModelSourceKind::DirectUrl as i32
        }
        crate::mesh::ModelSourceKind::Unknown => {
            crate::proto::node::ModelSourceKind::Unknown as i32
        }
    }
}

fn proto_source_kind_to_local(kind: i32) -> crate::mesh::ModelSourceKind {
    match crate::proto::node::ModelSourceKind::try_from(kind)
        .unwrap_or(crate::proto::node::ModelSourceKind::Unknown)
    {
        crate::proto::node::ModelSourceKind::Catalog => crate::mesh::ModelSourceKind::Catalog,
        crate::proto::node::ModelSourceKind::HuggingFace => {
            crate::mesh::ModelSourceKind::HuggingFace
        }
        crate::proto::node::ModelSourceKind::LocalGguf => crate::mesh::ModelSourceKind::LocalGguf,
        crate::proto::node::ModelSourceKind::DirectUrl => crate::mesh::ModelSourceKind::DirectUrl,
        crate::proto::node::ModelSourceKind::Unknown
        | crate::proto::node::ModelSourceKind::Unspecified => crate::mesh::ModelSourceKind::Unknown,
    }
}

fn local_capability_level_to_proto(level: crate::models::CapabilityLevel) -> i32 {
    match level {
        crate::models::CapabilityLevel::None => crate::proto::node::CapabilityLevel::None as i32,
        crate::models::CapabilityLevel::Likely => {
            crate::proto::node::CapabilityLevel::Likely as i32
        }
        crate::models::CapabilityLevel::Supported => {
            crate::proto::node::CapabilityLevel::Supported as i32
        }
    }
}

fn proto_capability_level_to_local(level: i32) -> crate::models::CapabilityLevel {
    match crate::proto::node::CapabilityLevel::try_from(level)
        .unwrap_or(crate::proto::node::CapabilityLevel::None)
    {
        crate::proto::node::CapabilityLevel::Likely => crate::models::CapabilityLevel::Likely,
        crate::proto::node::CapabilityLevel::Supported => crate::models::CapabilityLevel::Supported,
        crate::proto::node::CapabilityLevel::None
        | crate::proto::node::CapabilityLevel::Unspecified => crate::models::CapabilityLevel::None,
    }
}

fn descriptor_identity_to_proto(
    identity: &crate::mesh::ServedModelIdentity,
) -> crate::proto::node::ServedModelIdentity {
    crate::proto::node::ServedModelIdentity {
        model_name: identity.model_name.clone(),
        is_primary: identity.is_primary,
        source_kind: local_source_kind_to_proto(identity.source_kind),
        canonical_ref: identity.canonical_ref.clone(),
        repository: identity.repository.clone(),
        revision: identity.revision.clone(),
        artifact: identity.artifact.clone(),
        local_file_name: identity.local_file_name.clone(),
        identity_hash: identity.identity_hash.clone(),
    }
}

fn proto_identity_to_local(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelIdentity {
    crate::mesh::ServedModelIdentity {
        model_name: identity.model_name.clone(),
        is_primary: identity.is_primary,
        source_kind: proto_source_kind_to_local(identity.source_kind),
        canonical_ref: identity.canonical_ref.clone(),
        repository: identity.repository.clone(),
        revision: identity.revision.clone(),
        artifact: identity.artifact.clone(),
        local_file_name: identity.local_file_name.clone(),
        identity_hash: identity.identity_hash.clone(),
    }
}

fn legacy_descriptor_from_identity(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelDescriptor {
    crate::mesh::ServedModelDescriptor {
        identity: proto_identity_to_local(identity),
        capabilities: crate::models::ModelCapabilities::default(),
        topology: None,
    }
}

fn runtime_descriptor_to_proto(
    descriptor: &crate::mesh::ModelRuntimeDescriptor,
) -> crate::proto::node::ModelRuntimeDescriptor {
    crate::proto::node::ModelRuntimeDescriptor {
        model_name: descriptor.model_name.clone(),
        identity_hash: descriptor.identity_hash.clone(),
        context_length: descriptor.context_length,
        ready: descriptor.ready,
    }
}

fn proto_runtime_descriptor_to_local(
    descriptor: &crate::proto::node::ModelRuntimeDescriptor,
) -> crate::mesh::ModelRuntimeDescriptor {
    crate::mesh::ModelRuntimeDescriptor {
        model_name: descriptor.model_name.clone(),
        identity_hash: descriptor.identity_hash.clone(),
        context_length: descriptor.context_length,
        ready: descriptor.ready,
    }
}

fn local_gpu_info_to_proto(ann: &PeerAnnouncement) -> Vec<crate::proto::node::GpuInfo> {
    let legacy_field_count = [
        split_optional_csv(ann.gpu_vram.as_deref()).len(),
        split_optional_csv(ann.gpu_reserved_bytes.as_deref()).len(),
        split_optional_csv(ann.gpu_mem_bandwidth_gbps.as_deref()).len(),
        split_optional_csv(ann.gpu_compute_tflops_fp32.as_deref()).len(),
        split_optional_csv(ann.gpu_compute_tflops_fp16.as_deref()).len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let names =
        crate::system::hardware::expand_gpu_names(ann.gpu_name.as_deref(), legacy_field_count);
    let vram = split_optional_csv(ann.gpu_vram.as_deref());
    let reserved = split_optional_csv(ann.gpu_reserved_bytes.as_deref());
    let mem_bandwidth = split_optional_csv(ann.gpu_mem_bandwidth_gbps.as_deref());
    let fp32 = split_optional_csv(ann.gpu_compute_tflops_fp32.as_deref());
    let fp16 = split_optional_csv(ann.gpu_compute_tflops_fp16.as_deref());
    let count = [
        legacy_field_count,
        names.len(),
        vram.len(),
        reserved.len(),
        mem_bandwidth.len(),
        fp32.len(),
        fp16.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    (0..count)
        .map(|index| crate::proto::node::GpuInfo {
            name: names.get(index).cloned(),
            vram_bytes: vram.get(index).cloned().flatten(),
            reserved_bytes: reserved.get(index).cloned().flatten(),
            mem_bandwidth_gbps: mem_bandwidth.get(index).cloned().flatten(),
            compute_tflops_fp32: fp32.get(index).cloned().flatten(),
            compute_tflops_fp16: fp16.get(index).cloned().flatten(),
        })
        .collect()
}

fn local_hardware_info_to_proto(
    ann: &PeerAnnouncement,
) -> Option<crate::proto::node::HardwareInfo> {
    let gpus = local_gpu_info_to_proto(ann);
    if ann.hostname.is_none() && ann.is_soc.is_none() && gpus.is_empty() {
        None
    } else {
        Some(crate::proto::node::HardwareInfo {
            is_soc: ann.is_soc,
            hostname: ann.hostname.clone(),
            gpus,
        })
    }
}

struct LegacyGpuFields {
    gpu_name: Option<String>,
    gpu_vram: Option<String>,
    gpu_reserved_bytes: Option<String>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
}

fn proto_gpu_info_to_legacy_fields(gpus: &[crate::proto::node::GpuInfo]) -> LegacyGpuFields {
    let names: Vec<String> = gpus.iter().filter_map(|gpu| gpu.name.clone()).collect();
    let gpu_name = crate::system::hardware::summarize_gpu_name(&names);
    let gpu_vram = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.vram_bytes.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_reserved_bytes = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.reserved_bytes.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_mem_bandwidth_gbps = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.mem_bandwidth_gbps.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_compute_tflops_fp32 = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.compute_tflops_fp32.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_compute_tflops_fp16 = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.compute_tflops_fp16.clone())
            .collect::<Vec<_>>(),
    );

    LegacyGpuFields {
        gpu_name,
        gpu_vram,
        gpu_reserved_bytes,
        gpu_mem_bandwidth_gbps,
        gpu_compute_tflops_fp32,
        gpu_compute_tflops_fp16,
    }
}

/// Returns `true` when a proto descriptor carries a non-empty model name.
/// Descriptors without a valid identity are discarded so a partial list
/// cannot suppress the legacy-identity backfill fallback.
fn proto_descriptor_has_valid_identity(
    descriptor: &crate::proto::node::ServedModelDescriptor,
) -> bool {
    descriptor
        .identity
        .as_ref()
        .map(|id| !id.model_name.is_empty())
        .unwrap_or(false)
}

pub(crate) fn sanitize_gossip_announcement_for_wire(ann: &PeerAnnouncement) -> PeerAnnouncement {
    let mut sanitized = ann.clone();
    sanitized.available_models.clear();
    sanitized.available_model_metadata.clear();
    sanitized.available_model_sizes.clear();
    sanitized
}

pub(crate) fn local_role_to_proto(role: &NodeRole) -> (i32, Option<u32>) {
    match role {
        NodeRole::Worker => (crate::proto::node::NodeRole::Worker as i32, None),
        NodeRole::Host { http_port } => (
            crate::proto::node::NodeRole::Host as i32,
            Some(*http_port as u32),
        ),
        NodeRole::Client => (crate::proto::node::NodeRole::Client as i32, None),
    }
}

pub(crate) fn proto_role_to_local(role_int: i32, http_port: Option<u32>) -> NodeRole {
    match crate::proto::node::NodeRole::try_from(role_int).unwrap_or_default() {
        crate::proto::node::NodeRole::Host => NodeRole::Host {
            http_port: http_port.unwrap_or(0) as u16,
        },
        crate::proto::node::NodeRole::Client => NodeRole::Client,
        _ => NodeRole::Worker,
    }
}

pub(crate) fn local_ann_to_proto_ann(
    ann: &PeerAnnouncement,
) -> crate::proto::node::PeerAnnouncement {
    let ann = sanitize_gossip_announcement_for_wire(ann);
    let (role_int, http_port) = local_role_to_proto(&ann.role);
    let serialized_addr = serde_json::to_vec(&ann.addr).unwrap_or_default();
    let demand: Vec<crate::proto::node::ModelDemandEntry> = ann
        .model_demand
        .iter()
        .map(
            |(name, d): (&String, &ModelDemand)| crate::proto::node::ModelDemandEntry {
                model_name: name.clone(),
                last_active: d.last_active,
                request_count: d.request_count,
            },
        )
        .collect();
    let served_model_identities = ann
        .served_model_descriptors
        .iter()
        .map(|descriptor| descriptor_identity_to_proto(&descriptor.identity))
        .collect();
    let served_model_descriptors = ann
        .served_model_descriptors
        .iter()
        .map(|descriptor| crate::proto::node::ServedModelDescriptor {
            identity: Some(descriptor_identity_to_proto(&descriptor.identity)),
            capabilities: Some(crate::proto::node::ModelCapabilities {
                vision: local_capability_level_to_proto(descriptor.capabilities.vision),
                reasoning: local_capability_level_to_proto(descriptor.capabilities.reasoning),
                tool_use: local_capability_level_to_proto(descriptor.capabilities.tool_use),
                moe: descriptor.capabilities.moe,
                multimodal: descriptor.capabilities.multimodal,
                audio: local_capability_level_to_proto(descriptor.capabilities.audio),
            }),
            topology: descriptor.topology.as_ref().map(|topology| {
                crate::proto::node::ModelTopology {
                    moe: topology
                        .moe
                        .as_ref()
                        .map(|moe| crate::proto::node::ModelMoeInfo {
                            expert_count: moe.expert_count,
                            used_expert_count: moe.used_expert_count,
                            min_experts_per_node: moe.min_experts_per_node,
                            source: moe.source.clone(),
                            ranking_source: moe.ranking_source.clone(),
                            ranking_origin: moe.ranking_origin.clone(),
                            ranking: moe.ranking.clone(),
                            ranking_prompt_count: moe.ranking_prompt_count,
                            ranking_tokens: moe.ranking_tokens,
                            ranking_layer_scope: moe.ranking_layer_scope.clone(),
                        }),
                }
            }),
        })
        .collect();
    let served_model_runtime = ann
        .served_model_runtime
        .iter()
        .map(runtime_descriptor_to_proto)
        .collect();
    let hardware = local_hardware_info_to_proto(&ann);
    crate::proto::node::PeerAnnouncement {
        endpoint_id: ann.addr.id.as_bytes().to_vec(),
        role: role_int,
        http_port,
        version: ann.version.clone(),
        gpu_name: ann.gpu_name.clone(),
        hostname: ann.hostname.clone(),
        is_soc: ann.is_soc,
        gpu_vram: ann.gpu_vram.clone(),
        available_models: ann.available_models.clone(),
        serving_models: ann.serving_models.clone(),
        requested_models: ann.requested_models.clone(),
        available_model_metadata: ann.available_model_metadata.clone(),
        experts_summary: ann.experts_summary.clone(),
        rtt_ms: None,
        catalog_models: ann.models.clone(),
        vram_bytes: ann.vram_bytes,
        model_source: ann.model_source.clone(),
        primary_serving: ann.serving_models.first().cloned(),
        mesh_id: ann.mesh_id.clone(),
        demand,
        available_model_sizes: ann.available_model_sizes.clone(),
        serialized_addr,
        hosted_models: ann.hosted_models.clone().unwrap_or_default(),
        hosted_models_known: Some(ann.hosted_models.is_some()),
        served_model_identities,
        served_model_descriptors,
        served_model_runtime,
        owner_attestation: ann
            .owner_attestation
            .as_ref()
            .and_then(local_owner_attestation_to_proto),
        // Legacy GPU metric fields (29-32) are populated alongside `hardware` so that
        // pre-v0.60.0 peers that do not decode the new `hardware` block can still read
        // bandwidth/tflops/reserved data from the flat fields they already know.
        gpu_mem_bandwidth_gbps: ann.gpu_mem_bandwidth_gbps.clone(),
        gpu_compute_tflops_fp32: ann.gpu_compute_tflops_fp32.clone(),
        gpu_compute_tflops_fp16: ann.gpu_compute_tflops_fp16.clone(),
        gpu_reserved_bytes: ann.gpu_reserved_bytes.clone(),
        hardware,
        first_joined_mesh_ts: ann.first_joined_mesh_ts,
    }
}

pub(crate) fn build_gossip_frame(
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> crate::proto::node::GossipFrame {
    let peers: Vec<crate::proto::node::PeerAnnouncement> =
        anns.iter().map(local_ann_to_proto_ann).collect();
    crate::proto::node::GossipFrame {
        gen: NODE_PROTOCOL_GENERATION,
        sender_id: sender_id.as_bytes().to_vec(),
        peers,
    }
}

pub(crate) fn proto_ann_to_local(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Option<(EndpointAddr, PeerAnnouncement)> {
    let id_arr: [u8; 32] = pa.endpoint_id.as_slice().try_into().ok()?;
    let pk = iroh::PublicKey::from_bytes(&id_arr).ok()?;
    let peer_id = EndpointId::from(pk);
    let addr: EndpointAddr = if !pa.serialized_addr.is_empty() {
        serde_json::from_slice(&pa.serialized_addr).unwrap_or(EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        })
    } else {
        EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }
    };
    let role = proto_role_to_local(pa.role, pa.http_port);
    let model_demand: HashMap<String, ModelDemand> = pa
        .demand
        .iter()
        .map(|e| {
            (
                e.model_name.clone(),
                ModelDemand {
                    last_active: e.last_active,
                    request_count: e.request_count,
                },
            )
        })
        .collect();
    let hosted_models = pa
        .hosted_models_known
        .unwrap_or(!pa.hosted_models.is_empty())
        .then(|| pa.hosted_models.clone());
    let hardware = pa.hardware.as_ref();
    let legacy_gpu_fields = proto_gpu_info_to_legacy_fields(
        hardware
            .map(|hardware| hardware.gpus.as_slice())
            .unwrap_or(&[]),
    );
    let mut ann = PeerAnnouncement {
        addr: addr.clone(),
        role,
        first_joined_mesh_ts: pa.first_joined_mesh_ts,
        models: pa.catalog_models.clone(),
        vram_bytes: pa.vram_bytes,
        model_source: pa.model_source.clone(),
        serving_models: pa.serving_models.clone(),
        hosted_models,
        available_models: Vec::new(),
        requested_models: pa.requested_models.clone(),
        version: pa.version.clone(),
        model_demand,
        mesh_id: pa.mesh_id.clone(),
        gpu_name: legacy_gpu_fields.gpu_name.or_else(|| pa.gpu_name.clone()),
        hostname: hardware
            .and_then(|hardware| hardware.hostname.clone())
            .or_else(|| pa.hostname.clone()),
        is_soc: hardware.and_then(|hardware| hardware.is_soc).or(pa.is_soc),
        gpu_vram: legacy_gpu_fields.gpu_vram.or_else(|| pa.gpu_vram.clone()),
        gpu_reserved_bytes: legacy_gpu_fields
            .gpu_reserved_bytes
            .or_else(|| pa.gpu_reserved_bytes.clone()),
        gpu_mem_bandwidth_gbps: legacy_gpu_fields
            .gpu_mem_bandwidth_gbps
            .or_else(|| pa.gpu_mem_bandwidth_gbps.clone()),
        gpu_compute_tflops_fp32: legacy_gpu_fields
            .gpu_compute_tflops_fp32
            .or_else(|| pa.gpu_compute_tflops_fp32.clone()),
        gpu_compute_tflops_fp16: legacy_gpu_fields
            .gpu_compute_tflops_fp16
            .or_else(|| pa.gpu_compute_tflops_fp16.clone()),
        available_model_metadata: Vec::new(),
        experts_summary: pa.experts_summary.clone(),
        available_model_sizes: HashMap::new(),
        served_model_runtime: pa
            .served_model_runtime
            .iter()
            .map(proto_runtime_descriptor_to_local)
            .collect(),
        served_model_descriptors: if !pa.served_model_descriptors.is_empty() {
            let descriptors: Vec<_> = pa
                .served_model_descriptors
                .iter()
                .filter(|descriptor| proto_descriptor_has_valid_identity(descriptor))
                .map(|descriptor| crate::mesh::ServedModelDescriptor {
                    identity: descriptor
                        .identity
                        .as_ref()
                        .map(proto_identity_to_local)
                        .unwrap_or_default(),
                    capabilities: descriptor
                        .capabilities
                        .as_ref()
                        .map(|caps| crate::models::ModelCapabilities {
                            multimodal: caps.multimodal,
                            vision: proto_capability_level_to_local(caps.vision),
                            audio: proto_capability_level_to_local(caps.audio),
                            reasoning: proto_capability_level_to_local(caps.reasoning),
                            tool_use: proto_capability_level_to_local(caps.tool_use),
                            moe: caps.moe,
                        })
                        .unwrap_or_default(),
                    topology: descriptor.topology.as_ref().map(|topology| {
                        crate::models::ModelTopology {
                            moe: topology
                                .moe
                                .as_ref()
                                .map(|moe| crate::models::ModelMoeInfo {
                                    expert_count: moe.expert_count,
                                    used_expert_count: moe.used_expert_count,
                                    min_experts_per_node: moe.min_experts_per_node,
                                    source: moe.source.clone(),
                                    ranking_source: moe.ranking_source.clone(),
                                    ranking_origin: moe.ranking_origin.clone(),
                                    ranking: moe.ranking.clone(),
                                    ranking_prompt_count: moe.ranking_prompt_count,
                                    ranking_tokens: moe.ranking_tokens,
                                    ranking_layer_scope: moe.ranking_layer_scope.clone(),
                                }),
                        }
                    }),
                })
                .collect();
            if descriptors.is_empty() {
                // All descriptors were invalid — fall back to legacy identity list.
                pa.served_model_identities
                    .iter()
                    .map(legacy_descriptor_from_identity)
                    .collect()
            } else {
                descriptors
            }
        } else {
            pa.served_model_identities
                .iter()
                .map(legacy_descriptor_from_identity)
                .collect()
        },
        owner_attestation: pa
            .owner_attestation
            .as_ref()
            .map(proto_owner_attestation_to_local),
    };
    crate::mesh::backfill_legacy_descriptors(&mut ann);
    Some((addr, ann))
}

pub(crate) fn routing_table_to_proto(table: &RoutingTable) -> crate::proto::node::RouteTable {
    let entries = table
        .hosts
        .iter()
        .map(|e| crate::proto::node::RouteEntry {
            endpoint_id: e.endpoint_id.as_bytes().to_vec(),
            model: e.model.clone(),
        })
        .collect();
    crate::proto::node::RouteTable {
        entries,
        mesh_id: table.mesh_id.clone(),
        gen: NODE_PROTOCOL_GENERATION,
    }
}

pub(crate) fn mesh_config_to_proto(
    config: &crate::plugin::MeshConfig,
) -> crate::proto::node::NodeConfigSnapshot {
    use crate::plugin::GpuAssignment;
    fn configured_model_ref(declared_ref: &str) -> crate::proto::node::ConfiguredModelRef {
        crate::proto::node::ConfiguredModelRef {
            declared_ref: declared_ref.to_string(),
            source_kind: None,
            revision: None,
        }
    }

    let assignment = match config.gpu.assignment {
        GpuAssignment::Auto => crate::proto::node::GpuAssignment::Auto as i32,
        GpuAssignment::Pinned => crate::proto::node::GpuAssignment::Pinned as i32,
    };
    let models = config
        .models
        .iter()
        .map(|m| crate::proto::node::NodeModelEntry {
            model: m.model.clone(),
            mmproj: m.mmproj.clone(),
            ctx_size: m.ctx_size,
            gpu_id: m.gpu_id.clone(),
            model_ref: Some(configured_model_ref(&m.model)),
            mmproj_ref: m.mmproj.as_deref().map(configured_model_ref),
        })
        .collect();
    let plugins = config
        .plugins
        .iter()
        .map(|p| crate::proto::node::NodePluginEntry {
            name: p.name.clone(),
            enabled: p.enabled,
            command: p.command.clone(),
            args: p.args.clone(),
        })
        .collect();
    crate::proto::node::NodeConfigSnapshot {
        version: config.version.unwrap_or(1),
        gpu: Some(crate::proto::node::NodeGpuConfig { assignment }),
        models,
        plugins,
    }
}

pub(crate) fn proto_config_to_mesh(
    snapshot: &crate::proto::node::NodeConfigSnapshot,
) -> crate::plugin::MeshConfig {
    use crate::plugin::{
        GpuAssignment, GpuConfig, MeshConfig, ModelConfigEntry, PluginConfigEntry,
    };
    fn declared_ref_or_none(
        configured: Option<&crate::proto::node::ConfiguredModelRef>,
    ) -> Option<String> {
        configured.and_then(|configured| {
            let declared_ref = configured.declared_ref.trim();
            if declared_ref.is_empty() {
                None
            } else {
                Some(declared_ref.to_string())
            }
        })
    }

    let assignment = match snapshot.gpu.as_ref().map(|g| g.assignment) {
        Some(v) if v == crate::proto::node::GpuAssignment::Pinned as i32 => GpuAssignment::Pinned,
        _ => GpuAssignment::Auto,
    };
    let models = snapshot
        .models
        .iter()
        .map(|m| ModelConfigEntry {
            model: declared_ref_or_none(m.model_ref.as_ref()).unwrap_or_else(|| m.model.clone()),
            mmproj: declared_ref_or_none(m.mmproj_ref.as_ref()).or_else(|| m.mmproj.clone()),
            ctx_size: m.ctx_size,
            gpu_id: m.gpu_id.clone(),
            parallel: None,
        })
        .collect();
    let plugins = snapshot
        .plugins
        .iter()
        .map(|p| PluginConfigEntry {
            name: p.name.clone(),
            enabled: p.enabled,
            command: p.command.clone(),
            args: p.args.clone(),
        })
        .collect();
    MeshConfig {
        version: Some(snapshot.version),
        gpu: GpuConfig {
            assignment,
            parallel: None,
        },
        models,
        plugins,
    }
}

pub(crate) fn canonical_config_hash(snapshot: &crate::proto::node::NodeConfigSnapshot) -> [u8; 32] {
    use prost::Message as _;
    use sha2::{Digest, Sha256};
    let bytes = snapshot.encode_to_vec();
    let hash = Sha256::digest(&bytes);
    hash.into()
}

#[cfg(test)]
pub(crate) fn proto_route_table_to_local(table: &crate::proto::node::RouteTable) -> RoutingTable {
    let hosts = table
        .entries
        .iter()
        .filter_map(|e| {
            let arr: [u8; 32] = e.endpoint_id.as_slice().try_into().ok()?;
            let pk = iroh::PublicKey::from_bytes(&arr).ok()?;
            let endpoint_id = EndpointId::from(pk);
            Some(RouteEntry {
                model: e.model.clone(),
                node_id: endpoint_id.fmt_short().to_string(),
                endpoint_id,
                vram_gb: 0.0,
            })
        })
        .collect();
    RoutingTable {
        hosts,
        mesh_id: table.mesh_id.clone(),
    }
}
