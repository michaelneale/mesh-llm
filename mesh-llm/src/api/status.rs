use super::{RuntimeModelPayload, RuntimeProcessPayload};
use crate::crypto::{OwnershipStatus, OwnershipSummary};
use crate::network::affinity;
use serde::Serialize;

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
    pub(super) bandwidth_gbps: Option<f64>,
}

pub(super) fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_bandwidth: Option<&str>,
) -> Vec<GpuEntry> {
    let names: Vec<&str> = gpu_name
        .map(|s| s.split(", ").collect())
        .unwrap_or_default();
    if names.is_empty() {
        return vec![];
    }
    let vrams: Vec<Option<u64>> = gpu_vram
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let bandwidths: Vec<Option<f64>> = gpu_bandwidth
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name: name.to_string(),
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            bandwidth_gbps: bandwidths.get(i).copied().flatten(),
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
    pub(super) launch_pi: Option<String>,
    pub(super) launch_goose: Option<String>,
    pub(super) inflight_requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_name: Option<String>,
    pub(super) nostr_discovery: bool,
    pub(super) my_hostname: Option<String>,
    pub(super) my_is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
    pub(super) routing_affinity: affinity::AffinityStatsSnapshot,
}

#[derive(Serialize)]
pub(super) struct PeerPayload {
    pub(super) id: String,
    pub(super) owner: OwnershipPayload,
    pub(super) role: String,
    pub(super) models: Vec<String>,
    pub(super) available_models: Vec<String>,
    pub(super) requested_models: Vec<String>,
    pub(super) vram_gb: f64,
    pub(super) serving_models: Vec<String>,
    pub(super) hosted_models: Vec<String>,
    pub(super) hosted_models_known: bool,
    pub(super) rtt_ms: Option<u32>,
    pub(super) hostname: Option<String>,
    pub(super) is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
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
