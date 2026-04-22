use super::{RuntimeModelPayload, RuntimeProcessPayload};
use crate::crypto::{OwnershipStatus, OwnershipSummary};
use crate::network::{affinity, metrics};
use crate::system::hardware::expand_gpu_names;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum NodeState {
    Client,
    Standby,
    Loading,
    Serving,
}

impl NodeState {
    pub(super) const fn node_status_alias(self) -> &'static str {
        match self {
            Self::Client => "Client",
            Self::Standby => "Standby",
            Self::Loading => "Loading",
            Self::Serving => "Serving",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum WakeableNodeState {
    Sleeping,
    Waking,
}

#[derive(Serialize)]
pub(super) struct RuntimeStatusPayload {
    pub(super) models: Vec<RuntimeModelPayload>,
}

#[derive(Serialize)]
pub(super) struct RuntimeProcessesPayload {
    pub(super) processes: Vec<RuntimeProcessPayload>,
}

#[derive(Serialize)]
pub(super) struct GpuEntry {
    pub(super) name: String,
    pub(super) vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mem_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) compute_tflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) compute_tflops_fp16: Option<f64>,
}

fn inferred_gpu_name_count(gpu_name: Option<&str>) -> usize {
    let Some(raw) = gpu_name.map(str::trim) else {
        return 0;
    };
    if raw.is_empty() {
        return 0;
    }

    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.split_once('×')
                .or_else(|| part.split_once('x'))
                .or_else(|| part.split_once('X'))
                .and_then(|(count, _)| count.trim().parse::<usize>().ok())
                .filter(|&count| count > 0)
                .unwrap_or(1)
        })
        .sum()
}

pub(super) fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_reserved_bytes: Option<&str>,
    gpu_mem_bandwidth: Option<&str>,
    gpu_compute_tflops_fp32: Option<&str>,
    gpu_compute_tflops_fp16: Option<&str>,
) -> Vec<GpuEntry> {
    let vrams: Vec<Option<u64>> = gpu_vram
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let reserved: Vec<Option<u64>> = gpu_reserved_bytes
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let bandwidths: Vec<Option<f64>> = gpu_mem_bandwidth
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp32: Vec<Option<f64>> = gpu_compute_tflops_fp32
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp16: Vec<Option<f64>> = gpu_compute_tflops_fp16
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let expected_count = [
        vrams.len(),
        reserved.len(),
        bandwidths.len(),
        compute_fp32.len(),
        compute_fp16.len(),
        inferred_gpu_name_count(gpu_name),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let names = expand_gpu_names(gpu_name, expected_count)
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return vec![];
    }
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name,
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            reserved_bytes: reserved.get(i).copied().flatten(),
            mem_bandwidth_gbps: bandwidths.get(i).copied().flatten(),
            compute_tflops_fp32: compute_fp32.get(i).copied().flatten(),
            compute_tflops_fp16: compute_fp16.get(i).copied().flatten(),
        })
        .collect()
}

#[derive(Serialize)]
pub(super) struct StatusPayload {
    pub(super) version: String,
    pub(super) latest_version: Option<String>,
    pub(super) node_id: String,
    pub(super) owner: OwnershipPayload,
    pub(super) token: String,
    pub(super) node_state: NodeState,
    pub(super) node_status: String,
    pub(super) is_host: bool,
    pub(super) is_client: bool,
    pub(super) llama_ready: bool,
    pub(super) model_name: String,
    pub(super) models: Vec<String>,
    pub(super) available_models: Vec<String>,
    pub(super) requested_models: Vec<String>,
    pub(super) serving_models: Vec<String>,
    pub(super) hosted_models: Vec<String>,
    pub(super) draft_name: Option<String>,
    pub(super) api_port: u16,
    pub(super) my_vram_gb: f64,
    pub(super) model_size_gb: f64,
    pub(super) peers: Vec<PeerPayload>,
    pub(super) wakeable_nodes: Vec<WakeableNode>,
    pub(super) local_instances: Vec<LocalInstance>,
    pub(super) launch_pi: Option<String>,
    pub(super) launch_goose: Option<String>,
    pub(super) inflight_requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_name: Option<String>,
    pub(super) nostr_discovery: bool,
    /// Best-effort publication state per Issue #240: private | public | publish_failed.
    pub(super) publication_state: String,
    pub(super) my_hostname: Option<String>,
    pub(super) my_is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
    pub(super) routing_affinity: affinity::AffinityStatsSnapshot,
    /// Local-only routing outcome and current-node pressure snapshot measured on
    /// this node only; not mesh-wide aggregates.
    pub(super) routing_metrics: metrics::RoutingMetricsStatusSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) first_joined_mesh_ts: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct WakeableNode {
    pub(super) logical_id: String,
    pub(super) models: Vec<String>,
    pub(super) vram_gb: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) provider: Option<String>,
    pub(super) state: WakeableNodeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) wake_eta_secs: Option<u32>,
}

#[derive(Serialize)]
pub(super) struct PeerPayload {
    pub(super) id: String,
    pub(super) owner: OwnershipPayload,
    pub(super) role: String,
    pub(super) state: NodeState,
    pub(super) models: Vec<String>,
    pub(super) available_models: Vec<String>,
    pub(super) requested_models: Vec<String>,
    pub(super) vram_gb: f64,
    pub(super) serving_models: Vec<String>,
    pub(super) hosted_models: Vec<String>,
    pub(super) hosted_models_known: bool,
    pub(super) version: Option<String>,
    pub(super) rtt_ms: Option<u32>,
    pub(super) hostname: Option<String>,
    pub(super) is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) first_joined_mesh_ts: Option<u64>,
}

#[derive(Serialize)]
pub(super) struct OwnershipPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cert_id: Option<String>,
    pub(super) status: String,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) node_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) hostname_hint: Option<String>,
}

pub(super) fn build_ownership_payload(summary: &OwnershipSummary) -> OwnershipPayload {
    OwnershipPayload {
        owner_id: summary.owner_id.clone(),
        cert_id: summary.cert_id.clone(),
        status: match summary.status {
            OwnershipStatus::Verified => "verified",
            OwnershipStatus::Unsigned => "unsigned",
            OwnershipStatus::Expired => "expired",
            OwnershipStatus::InvalidSignature => "invalid_signature",
            OwnershipStatus::MismatchedNodeId => "mismatched_node_id",
            OwnershipStatus::RevokedOwner => "revoked_owner",
            OwnershipStatus::RevokedCert => "revoked_cert",
            OwnershipStatus::RevokedNodeId => "revoked_node_id",
            OwnershipStatus::UnsupportedProtocol => "unsupported_protocol",
            OwnershipStatus::UntrustedOwner => "untrusted_owner",
        }
        .to_string(),
        verified: summary.verified,
        expires_at_unix_ms: summary.expires_at_unix_ms,
        node_label: summary.node_label.clone(),
        hostname_hint: summary.hostname_hint.clone(),
    }
}

#[derive(Serialize)]
pub(super) struct LocalInstance {
    pub(super) pid: u32,
    pub(super) api_port: Option<u16>,
    pub(super) version: Option<String>,
    pub(super) started_at_unix: i64,
    pub(super) runtime_dir: String,
    pub(super) is_self: bool,
}

#[derive(Serialize)]
pub(super) struct MeshModelPayload {
    pub(super) name: String,
    pub(super) display_name: String,
    pub(super) status: String,
    pub(super) node_count: usize,
    pub(super) mesh_vram_gb: f64,
    pub(super) size_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<String>,
    pub(super) multimodal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) multimodal_status: Option<&'static str>,
    pub(super) vision: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) vision_status: Option<&'static str>,
    pub(super) audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) audio_status: Option<&'static str>,
    pub(super) reasoning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning_status: Option<&'static str>,
    pub(super) tool_use: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_use_status: Option<&'static str>,
    pub(super) moe: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) used_expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_prompt_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_layer_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) draft_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) request_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) last_active_secs_ago: Option<u64>,
    /// Local-only per-model routing outcome snapshot measured on the current
    /// node only; not mesh-wide aggregates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) routing_metrics: Option<metrics::ModelRoutingMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_page_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(super) active_nodes: Vec<String>,
    pub(super) fit_label: String,
    pub(super) fit_detail: String,
    pub(super) download_command: String,
    pub(super) run_command: String,
    pub(super) auto_command: String,
}

pub(super) fn build_runtime_status_payload(
    model_name: &str,
    primary_backend: Option<String>,
    is_host: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeStatusPayload {
    local_processes.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    let mut models: Vec<RuntimeModelPayload> = local_processes
        .into_iter()
        .map(|process| RuntimeModelPayload {
            name: process.name,
            backend: process.backend,
            status: process.status,
            port: Some(process.port),
        })
        .collect();

    let has_model_process = models.iter().any(|model| model.name == model_name);
    if is_host && !llama_ready && !has_model_process && !model_name.is_empty() {
        models.insert(
            0,
            RuntimeModelPayload {
                name: model_name.to_string(),
                backend: primary_backend.unwrap_or_else(|| "unknown".into()),
                status: "starting".into(),
                port: llama_port,
            },
        );
    }

    RuntimeStatusPayload { models }
}

pub(super) fn build_runtime_processes_payload(
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeProcessesPayload {
    local_processes.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    RuntimeProcessesPayload {
        processes: local_processes,
    }
}

pub(crate) fn classify_runtime_error(msg: &str) -> u16 {
    if msg.contains("not loaded") {
        404
    } else if msg.contains("already loaded") {
        409
    } else if msg.contains("fit locally") || msg.contains("runtime load only supports") {
        422
    } else {
        400
    }
}

pub(super) fn decode_runtime_model_path(path: &str) -> Option<String> {
    let raw = path.strip_prefix("/api/runtime/models/")?;
    if raw.is_empty() {
        return None;
    }

    let bytes = raw.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = bytes[i + 1] as char;
                let lo = bytes[i + 2] as char;
                let hex = [hi, lo].iter().collect::<String>();
                if let Ok(value) = u8::from_str_radix(&hex, 16) {
                    decoded.push(value);
                    i += 3;
                    continue;
                } else {
                    return None;
                }
            }
            b'+' => decoded.push(b'+'),
            b => decoded.push(b),
        }
        i += 1;
    }
    String::from_utf8(decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_owner_payload() -> OwnershipPayload {
        OwnershipPayload {
            owner_id: None,
            cert_id: None,
            status: "unsigned".to_string(),
            verified: false,
            expires_at_unix_ms: None,
            node_label: None,
            hostname_hint: None,
        }
    }

    #[test]
    fn test_peer_payload_serializes_version_field() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            version: Some("0.56.0".to_string()),
            rtt_ms: None,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":\"0.56.0\""));
    }

    #[test]
    fn test_peer_payload_serializes_null_version() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            version: None,
            rtt_ms: None,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":null"));
    }

    #[test]
    fn test_status_payload_has_local_instances_field() {
        let instances: Vec<LocalInstance> = vec![];
        let json = serde_json::to_string(&instances).expect("serialization failed");
        assert_eq!(json, "[]");
    }

    #[test]
    fn status_payload_serializes_node_state_and_node_status_alias() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Loading,
            node_status: NodeState::Loading.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen".to_string(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"loading\""));
        assert!(json.contains("\"node_status\":\"Loading\""));
    }

    #[test]
    fn status_payload_keeps_node_status_for_compatibility() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Serving,
            node_status: NodeState::Serving.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: true,
            model_name: "Qwen".to_string(),
            models: vec!["Qwen".to_string()],
            available_models: vec!["Qwen".to_string()],
            requested_models: vec!["Qwen".to_string()],
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 24.0,
            model_size_gb: 4.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"serving\""));
        assert!(json.contains("\"node_status\":\"Serving\""));
    }

    #[test]
    fn status_payload_serializes_wakeable_nodes_separately() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![WakeableNode {
                logical_id: "provider-node-1".to_string(),
                models: vec!["Qwen".to_string()],
                vram_gb: 24.0,
                provider: Some("fly".to_string()),
                state: WakeableNodeState::Sleeping,
                wake_eta_secs: Some(90),
            }],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["peers"], serde_json::json!([]));
        assert_eq!(json["wakeable_nodes"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["wakeable_nodes"][0]["state"], "sleeping");
        assert_eq!(json["wakeable_nodes"][0]["logical_id"], "provider-node-1");
    }

    #[test]
    fn status_payload_defaults_to_empty_wakeable_inventory() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["wakeable_nodes"], serde_json::json!([]));
        assert_eq!(json["peers"], serde_json::json!([]));
    }

    #[test]
    fn peer_status_serializes_state_without_mutating_role() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            role: "Host".to_string(),
            state: NodeState::Serving,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            hosted_models_known: true,
            version: Some("0.60.2".to_string()),
            rtt_ms: Some(12),
            hostname: Some("peer.local".to_string()),
            is_soc: Some(false),
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"role\":\"Host\""));
        assert!(json.contains("\"state\":\"serving\""));
    }

    #[test]
    fn test_local_instance_serializes_is_self() {
        let instance = LocalInstance {
            pid: 1234,
            api_port: Some(3131),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000000,
            runtime_dir: "/home/user/.mesh-llm/runtime/1234".to_string(),
            is_self: true,
        };

        let json = serde_json::to_string(&instance).expect("serialization failed");
        assert!(json.contains("\"is_self\":true"));
    }
}
