//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state (JSON)
//!   GET  /api/runtime   — local model state (JSON)
//!   GET  /api/runtime/processes — local inference process state (JSON)
//!   POST /api/runtime/models — load a local model
//!   DELETE /api/runtime/models/{model} — unload a local model
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to inference API
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is mostly read-only — shows status, topology, and models.
//! Local model load/unload is exposed for operator control.

use crate::config::{
    mesh_config_path, node_config_path, AuthoredMeshConfig, NodeConfig, PlacementMode,
    AUTHORED_CONFIG_VERSION,
};
use crate::{affinity, election, mesh, nostr, plugin, proxy};
use ed25519_dalek::{Signature, Signer, Verifier, VerifyingKey};
use include_dir::{include_dir, Dir};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");
const MESH_LLM_VERSION: &str = crate::VERSION;
const CONFIG_TIMESTAMP_MAX_AGE_SECS: i64 = 60 * 60 * 24 * 365;
const CONFIG_TIMESTAMP_MAX_FUTURE_SKEW_SECS: i64 = 60 * 15;
const CONFIG_SIGNATURE_HEADER: &str = "X-Mesh-Config-Signature";
const CONFIG_OWNER_FINGERPRINT_HEADER: &str = "X-Mesh-Owner-Fingerprint";
const CONFIG_TIMESTAMP_HEADER: &str = "X-Mesh-Config-Timestamp";
const CONFIG_PREV_HASH_HEADER: &str = "X-Mesh-Prev-Config-Hash";

// ── Shared state ──

pub enum RuntimeControlRequest {
    Load {
        spec: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<String>>,
    },
    Unload {
        model: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
}

#[derive(Clone, Serialize)]
pub struct RuntimeModelPayload {
    pub name: String,
    pub backend: String,
    pub status: String,
    pub port: Option<u16>,
}

#[derive(Serialize)]
struct RuntimeStatusPayload {
    models: Vec<RuntimeModelPayload>,
}

#[derive(Clone, Serialize)]
pub struct RuntimeProcessPayload {
    pub name: String,
    pub backend: String,
    pub status: String,
    pub port: u16,
    pub pid: u32,
}

#[derive(Serialize)]
struct RuntimeProcessesPayload {
    processes: Vec<RuntimeProcessPayload>,
}

/// Shared live state — written by the main process, read by API handlers.
#[derive(Clone)]
pub struct MeshApi {
    inner: Arc<Mutex<ApiInner>>,
}

struct ApiInner {
    node: mesh::Node,
    plugin_manager: plugin::PluginManager,
    affinity_router: affinity::AffinityRouter,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    model_name: String,
    primary_backend: Option<String>,
    draft_name: Option<String>,
    api_port: u16,
    model_size_bytes: u64,
    mesh_name: Option<String>,
    latest_version: Option<String>,
    nostr_relays: Vec<String>,
    nostr_discovery: bool,
    runtime_control: Option<tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>>,
    local_processes: Vec<RuntimeProcessPayload>,
    sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
    inventory_scan_running: bool,
    inventory_scan_waiters:
        Vec<tokio::sync::oneshot::Sender<crate::models::LocalModelInventorySnapshot>>,
}

#[derive(Serialize)]
struct GpuEntry {
    name: String,
    vram_bytes: u64,
    reserved_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bandwidth_gbps: Option<f64>,
}

fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_bandwidth: Option<&str>,
    gpu_reserved: Option<&str>,
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
    let reserveds: Vec<Option<u64>> = gpu_reserved
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name: name.to_string(),
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            reserved_bytes: reserveds.get(i).copied().flatten().unwrap_or(0),
            bandwidth_gbps: bandwidths.get(i).copied().flatten(),
        })
        .collect()
}

#[derive(Serialize)]
struct StatusPayload {
    version: String,
    latest_version: Option<String>,
    node_id: String,
    token: String,
    node_status: String,
    is_host: bool,
    is_client: bool,
    llama_ready: bool,
    model_name: String,
    models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    draft_name: Option<String>,
    api_port: u16,
    my_vram_gb: f64,
    model_size_gb: f64,
    peers: Vec<PeerPayload>,
    launch_pi: Option<String>,
    launch_goose: Option<String>,
    inflight_requests: u64,
    /// Mesh identity (for matching against discovered meshes)
    mesh_id: Option<String>,
    /// Human-readable mesh name (from Nostr publishing)
    mesh_name: Option<String>,
    /// true when this node found the mesh via Nostr discovery (community/public mesh)
    nostr_discovery: bool,
    my_hostname: Option<String>,
    my_is_soc: Option<bool>,
    owner_id: Option<String>,
    owner_fingerprint: Option<String>,
    owner_fingerprint_verified: bool,
    owner_fingerprint_transitive: bool,
    gpus: Vec<GpuEntry>,
    routing_affinity: affinity::AffinityStatsSnapshot,
    model_sizes: Vec<(String, u64)>,
    model_scans: Vec<mesh::ScannedModel>,
}

#[derive(Serialize)]
struct ModelsPayload {
    mesh_models: Vec<MeshModelPayload>,
}

#[derive(Serialize)]
struct PeerPayload {
    id: String,
    role: String,
    models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    vram_gb: f64,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    hosted_models_known: bool,
    rtt_ms: Option<u32>,
    hostname: Option<String>,
    is_soc: Option<bool>,
    owner_id: Option<String>,
    owner_fingerprint: Option<String>,
    owner_fingerprint_verified: bool,
    owner_fingerprint_transitive: bool,
    gpus: Vec<GpuEntry>,
    model_sizes: Option<Vec<(String, u64)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_scans: Option<Vec<mesh::ScannedModel>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ConfigValidationError {
    code: String,
    path: String,
    message: String,
}

#[derive(Serialize)]
struct ConfigValidationErrorResponse {
    error: String,
    errors: Vec<ConfigValidationError>,
}

#[derive(Debug, Clone, Default)]
struct SplitValidationCatalog {
    totals_by_model_key: HashMap<String, u32>,
    gpu_counts_by_node_id: HashMap<String, u32>,
}

#[derive(Debug, Clone)]
struct SplitSegmentRef {
    path: String,
    start: u32,
    end: u32,
}

fn model_sizes_to_map(sizes: &[(String, u64)]) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for (name, size) in sizes {
        let entry = map.entry(name.clone()).or_insert(0);
        *entry = (*entry).max(*size);
    }
    map
}

fn model_sizes_sorted_vec_from_map(map: HashMap<String, u64>) -> Vec<(String, u64)> {
    let mut sizes: Vec<(String, u64)> = map.into_iter().collect();
    sizes.sort_by(|a, b| a.0.cmp(&b.0));
    sizes
}

#[derive(Serialize)]
struct MeshModelPayload {
    name: String,
    display_name: String,
    status: String,
    node_count: usize,
    mesh_vram_gb: f64,
    size_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// Whether this model supports vision/image input
    vision: bool,
    /// Display-oriented vision metadata status.
    #[serde(skip_serializing_if = "Option::is_none")]
    vision_status: Option<&'static str>,
    /// Whether this model appears reasoning-oriented.
    reasoning: bool,
    /// Display-oriented reasoning metadata status.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_status: Option<&'static str>,
    tool_use: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_use_status: Option<&'static str>,
    moe: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    used_expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    draft_model: Option<String>,
    /// Total requests seen across the mesh (from demand map)
    #[serde(skip_serializing_if = "Option::is_none")]
    request_count: Option<u64>,
    /// Seconds since last request or declaration (None if no demand data)
    #[serde(skip_serializing_if = "Option::is_none")]
    last_active_secs_ago: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_page_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    active_nodes: Vec<String>,
    fit_label: String,
    fit_detail: String,
    download_command: String,
    run_command: String,
    auto_command: String,
}

fn find_catalog_model(name: &str) -> Option<&'static crate::models::catalog::CatalogModel> {
    crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name == name || m.file.strip_suffix(".gguf").unwrap_or(m.file.as_str()) == name)
}

fn is_huggingface_repository_like(repository: &str) -> bool {
    let trimmed = repository.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with('/')
        && !trimmed.ends_with('/')
        && !trimmed.contains('\\')
        && trimmed.split('/').count() == 2
}

fn huggingface_repository_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    matches!(identity.source_kind, mesh::ModelSourceKind::HuggingFace)
        .then(|| {
            identity
                .repository
                .clone()
                .filter(|repo| is_huggingface_repository_like(repo))
        })
        .flatten()
}

fn source_page_url_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    huggingface_repository_from_identity(identity)
        .map(|repository| format!("https://huggingface.co/{repository}"))
}

fn source_file_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    identity
        .artifact
        .clone()
        .or_else(|| identity.local_file_name.clone())
}

fn likely_reasoning_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["reasoning", "thinking", "deepseek-r1"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

fn likely_vision_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["vision", "-vl", "llava", "omni", "qwen2.5-vl", "mllama"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

fn fit_hint_for_machine(size_gb: f64, my_vram_gb: f64) -> (String, String) {
    if size_gb <= 0.0 || my_vram_gb <= 0.0 {
        return (
            "Unknown".into(),
            "No local VRAM signal is available for this machine yet.".into(),
        );
    }
    if size_gb * 1.2 <= my_vram_gb {
        return (
            "Likely comfortable".into(),
            format!(
                "This machine has {:.1} GB VRAM, which should handle a {:.1} GB model comfortably.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 1.05 <= my_vram_gb {
        return (
            "Likely fits".into(),
            format!(
                "This machine has {:.1} GB VRAM. A {:.1} GB model should fit, but headroom will be tight.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 0.8 <= my_vram_gb {
        return (
            "Possible with tradeoffs".into(),
            format!(
                "This machine has {:.1} GB VRAM. A {:.1} GB model may load, but expect tighter memory pressure.",
                my_vram_gb, size_gb
            ),
        );
    }
    (
        "Likely too large".into(),
        format!(
            "This machine has {:.1} GB VRAM, which is likely not enough for a {:.1} GB model locally.",
            my_vram_gb, size_gb
        ),
    )
}

impl MeshApi {
    pub fn new(
        node: mesh::Node,
        model_name: String,
        api_port: u16,
        model_size_bytes: u64,
        plugin_manager: plugin::PluginManager,
        affinity_router: affinity::AffinityRouter,
    ) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                affinity_router,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                primary_backend: None,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                nostr_discovery: false,
                runtime_control: None,
                local_processes: Vec::new(),
                sse_clients: Vec::new(),
                inventory_scan_running: false,
                inventory_scan_waiters: Vec::new(),
            })),
        }
    }

    pub async fn set_primary_backend(&self, backend: String) {
        self.inner.lock().await.primary_backend = Some(backend);
    }

    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    pub async fn set_client(&self, is_client: bool) {
        self.inner.lock().await.is_client = is_client;
    }

    pub async fn set_mesh_name(&self, name: String) {
        self.inner.lock().await.mesh_name = Some(name);
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn set_nostr_discovery(&self, v: bool) {
        self.inner.lock().await.nostr_discovery = v;
    }

    pub async fn set_runtime_control(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    ) {
        self.inner.lock().await.runtime_control = Some(tx);
    }

    pub async fn upsert_local_process(&self, process: RuntimeProcessPayload) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| p.name != process.name);
            inner.local_processes.push(process);
        }
        self.push_status().await;
    }

    pub async fn remove_local_process(&self, model_name: &str) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| p.name != model_name);
        }
        self.push_status().await;
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
        }
        self.push_status().await;
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        self.inner.lock().await.llama_port = port;
    }

    async fn local_inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        // Always await a oneshot for the result; if no scan is running, start one in
        // a detached task that will perform the scan and notify all waiters.
        let rx = {
            let mut inner = self.inner.lock().await;
            if inner.inventory_scan_running {
                // A scan is already in progress; just register as another waiter.
                let (tx, rx) = tokio::sync::oneshot::channel();
                inner.inventory_scan_waiters.push(tx);
                rx
            } else {
                // No scan running: mark as running, register ourselves as a waiter,
                // and kick off a detached task to perform the scan and cleanup.
                inner.inventory_scan_running = true;
                let (tx, rx) = tokio::sync::oneshot::channel();
                inner.inventory_scan_waiters.push(tx);

                let inner_arc = self.inner.clone();
                tokio::spawn(async move {
                    let snapshot = match tokio::task::spawn_blocking(|| {
                        crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
                    })
                    .await
                    {
                        Ok(snapshot) => snapshot,
                        Err(e) => {
                            tracing::warn!("Local inventory scan failed: {e}");
                            crate::models::LocalModelInventorySnapshot::default()
                        }
                    };

                    let waiters = {
                        let mut inner = inner_arc.lock().await;
                        inner.inventory_scan_running = false;
                        std::mem::take(&mut inner.inventory_scan_waiters)
                    };
                    for tx in waiters {
                        let _ = tx.send(snapshot.clone());
                    }
                });

                rx
            }
        };

        rx.await.unwrap_or_default()
    }

    async fn runtime_status(&self) -> RuntimeStatusPayload {
        let (model_name, primary_backend, is_host, llama_ready, llama_port, local_processes) = {
            let inner = self.inner.lock().await;
            (
                inner.model_name.clone(),
                inner.primary_backend.clone(),
                inner.is_host,
                inner.llama_ready,
                inner.llama_port,
                inner.local_processes.clone(),
            )
        };
        build_runtime_status_payload(
            &model_name,
            primary_backend,
            is_host,
            llama_ready,
            llama_port,
            local_processes,
        )
    }

    async fn runtime_processes(&self) -> RuntimeProcessesPayload {
        let local_processes = self.inner.lock().await.local_processes.clone();
        build_runtime_processes_payload(local_processes)
    }

    async fn mesh_models(&self) -> Vec<MeshModelPayload> {
        let (node, my_vram_gb, model_name, model_size_bytes, local_processes) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.local_processes.clone(),
            )
        };

        let local_scan = self.local_inventory_snapshot().await;
        let all_peers = node.peers().await;
        let catalog = node.mesh_catalog_entries().await;
        let served = node.models_being_served().await;
        let active_demand = node.active_demand().await;
        let my_serving_models = node.serving_models().await;
        let local_model_names = local_scan.model_names;
        let mut metadata_by_name = local_scan.metadata_by_name;
        let mut size_by_name = local_scan.size_by_name;
        for peer in &all_peers {
            for meta in &peer.available_model_metadata {
                metadata_by_name
                    .entry(meta.model_key.clone())
                    .or_insert_with(|| meta.clone());
            }
            for (model_name, size) in &peer.available_model_sizes {
                size_by_name.entry(model_name.clone()).or_insert(*size);
            }
        }
        let my_hosted_models = node.hosted_models().await;
        let _display_model_name = local_processes
            .first()
            .map(|process| process.name.clone())
            .or_else(|| my_hosted_models.first().cloned())
            .or_else(|| my_serving_models.first().cloned())
            .unwrap_or_else(|| model_name.clone());
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        catalog
            .iter()
            .map(|entry| {
                let name = &entry.model_name;
                let descriptor = entry.descriptor.as_ref();
                let identity = descriptor.map(|descriptor| &descriptor.identity);
                let catalog_entry = find_catalog_model(name);
                let is_warm = served.contains(name);
                let local_known = local_model_names.contains(name)
                    || my_hosted_models.iter().any(|s| s == name)
                    || my_serving_models.iter().any(|s| s == name)
                    || name == &model_name;
                let display_name = crate::models::installed_model_display_name(name);
                let node_count = if is_warm {
                    let peer_count = all_peers.iter().filter(|p| p.routes_model(name)).count();
                    let me = if my_hosted_models.iter().any(|s| s == name) {
                        1
                    } else {
                        0
                    };
                    peer_count + me
                } else {
                    0
                };
                let active_nodes: Vec<String> = if is_warm {
                    let mut nodes = Vec::new();
                    if my_hosted_models.iter().any(|s| s == name) {
                        nodes.push(
                            node.hostname
                                .clone()
                                .filter(|hostname| !hostname.trim().is_empty())
                                .unwrap_or_else(|| "This node".to_string()),
                        );
                    }
                    nodes.extend(all_peers.iter().filter_map(|peer| {
                        if !peer.routes_model(name) {
                            return None;
                        }
                        Some(
                            peer.hostname
                                .clone()
                                .filter(|hostname| !hostname.trim().is_empty())
                                .unwrap_or_else(|| peer.id.fmt_short().to_string()),
                        )
                    }));
                    nodes.sort();
                    nodes.dedup();
                    nodes
                } else {
                    Vec::new()
                };
                let mesh_vram_gb = if is_warm {
                    let peer_vram = all_peers
                        .iter()
                        .filter(|p| p.routes_model(name))
                        .map(|p| p.vram_bytes as f64 / 1e9)
                        .sum::<f64>();
                    let my_vram = if my_hosted_models.iter().any(|s| s == name) {
                        my_vram_gb
                    } else {
                        0.0
                    };
                    peer_vram + my_vram
                } else {
                    0.0
                };
                let size_gb = if name == &model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    size_by_name
                        .get(name)
                        .map(|size| *size as f64 / 1e9)
                        .unwrap_or_else(|| {
                            crate::models::catalog::parse_size_gb(
                                catalog_entry.map(|m| m.size.as_str()).unwrap_or("0"),
                            )
                        })
                };
                let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                    Some(d) => (
                        Some(d.request_count),
                        Some(now_ts.saturating_sub(d.last_active)),
                    ),
                    None => (None, None),
                };
                let mut capabilities = descriptor
                    .map(|descriptor| descriptor.capabilities)
                    .unwrap_or_else(|| {
                        if local_known {
                            crate::models::installed_model_capabilities(name)
                        } else {
                            crate::models::ModelCapabilities::default()
                        }
                    });
                if local_known
                    && likely_reasoning_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.reasoning = capabilities
                        .reasoning
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                }
                if local_known
                    && likely_vision_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.vision = capabilities
                        .vision
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                }
                let vision = capabilities.supports_vision_runtime();
                let vision_status = if vision || capabilities.vision_label().is_some() {
                    Some(capabilities.vision_status())
                } else {
                    None
                };
                let reasoning = matches!(
                    capabilities.reasoning,
                    crate::models::capabilities::CapabilityLevel::Supported
                        | crate::models::capabilities::CapabilityLevel::Likely
                );
                let reasoning_status = if reasoning || capabilities.reasoning_label().is_some() {
                    Some(capabilities.reasoning_status())
                } else {
                    None
                };
                let tool_use = capabilities.tool_use_label().is_some();
                let tool_use_status = capabilities
                    .tool_use_label()
                    .map(|_| capabilities.tool_use_status());
                let description = catalog_entry.map(|m| m.description.to_string());
                let metadata = metadata_by_name.get(name);
                let architecture = metadata
                    .map(|m| m.architecture.trim())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let context_length = metadata
                    .map(|m| m.context_length)
                    .filter(|value| *value > 0);
                let quantization = metadata
                    .map(|m| m.quantization_type.trim())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .or_else(|| {
                        catalog_entry.map(|m| m.file.to_string()).and_then(|file| {
                            let quant = file
                                .strip_suffix(".gguf")
                                .map(crate::models::inventory::derive_quantization_type)
                                .filter(|q| !q.is_empty())?;
                            Some(quant)
                        })
                    });
                let topology_moe = descriptor
                    .and_then(|descriptor| descriptor.topology.as_ref())
                    .and_then(|topology| topology.moe.as_ref());
                let moe = capabilities.moe
                    || topology_moe.is_some()
                    || metadata.map(|m| m.is_moe).unwrap_or(false);
                let expert_count = topology_moe
                    .map(|moe| moe.expert_count)
                    .or_else(|| metadata.map(|m| m.expert_count).filter(|count| *count > 0))
                    .or_else(|| {
                        catalog_entry
                            .and_then(|m| m.moe.as_ref())
                            .map(|m| m.n_expert)
                    });
                let used_expert_count = topology_moe
                    .map(|moe| moe.used_expert_count)
                    .or_else(|| {
                        metadata
                            .map(|m| m.used_expert_count)
                            .filter(|count| *count > 0)
                    })
                    .or_else(|| {
                        catalog_entry
                            .and_then(|m| m.moe.as_ref())
                            .map(|m| m.n_expert_used)
                    });
                let draft_model = catalog_entry.and_then(|m| m.draft.clone());
                let source_page_url =
                    identity
                        .and_then(source_page_url_from_identity)
                        .or_else(|| {
                            if local_known {
                                catalog_entry.and_then(|m| {
                                    crate::models::catalog::huggingface_repo_url(&m.url)
                                })
                            } else {
                                None
                            }
                        });
                let source_ref = identity
                    .and_then(huggingface_repository_from_identity)
                    .or_else(|| {
                        source_page_url
                            .as_deref()
                            .map(|url| url.replace("https://huggingface.co/", ""))
                    });
                let source_revision = identity.and_then(|identity| identity.revision.clone());
                let source_file = identity.and_then(source_file_from_identity).or_else(|| {
                    if local_known {
                        catalog_entry.map(|m| m.file.to_string())
                    } else {
                        None
                    }
                });
                let command_ref = identity
                    .and_then(|identity| identity.canonical_ref.clone())
                    .or_else(|| {
                        if local_known {
                            catalog_entry.and_then(|m| {
                                match (m.source_repo(), m.source_revision(), m.source_file()) {
                                    (Some(repo), revision, Some(file)) => Some(match revision {
                                        Some(revision) => format!("{repo}@{revision}/{file}"),
                                        None => format!("{repo}/{file}"),
                                    }),
                                    _ => None,
                                }
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| name.clone());
                let (fit_label, fit_detail) = fit_hint_for_machine(size_gb, my_vram_gb);
                MeshModelPayload {
                    name: name.clone(),
                    display_name,
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    mesh_vram_gb,
                    size_gb,
                    architecture,
                    context_length,
                    quantization,
                    description,
                    vision,
                    vision_status,
                    reasoning,
                    reasoning_status,
                    tool_use,
                    tool_use_status,
                    moe,
                    expert_count,
                    used_expert_count,
                    draft_model,
                    request_count,
                    last_active_secs_ago,
                    source_page_url,
                    source_ref,
                    source_revision,
                    source_file,
                    active_nodes,
                    fit_label,
                    fit_detail,
                    download_command: format!("mesh-llm models download {}", command_ref),
                    run_command: format!("mesh-llm --model {}", command_ref),
                    auto_command: format!("mesh-llm --auto --model {}", command_ref),
                }
            })
            .collect()
    }

    async fn status(&self) -> StatusPayload {
        // Snapshot inner fields and drop the lock before any async node queries.
        // This prevents deadlock: if node.peers() etc. block on node.state.lock(),
        // we don't hold inner.lock() hostage, so other handlers can still proceed.
        let (
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            routing_affinity,
            model_name,
            model_size_bytes,
            llama_ready,
            is_host,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            nostr_discovery,
            local_processes,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.affinity_router.stats_snapshot(),
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.llama_ready,
                inner.is_host,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.nostr_discovery,
                inner.local_processes.clone(),
            )
        }; // inner lock dropped here

        let all_peers = node.peers().await;
        let my_models = node.models().await;
        let my_available_models = node.available_models().await;
        let my_requested_models = node.requested_models().await;
        let owner_fingerprint = node.owner_fingerprint().await;
        let owner_id = owner_fingerprint.clone();
        let owner_fingerprint_verified = owner_fingerprint.is_some();
        let my_model_sizes = model_sizes_sorted_vec_from_map(node.available_model_sizes().await);
        let my_model_scans = node.available_model_metadata().await;
        let peers: Vec<PeerPayload> = all_peers
            .iter()
            .map(|p| PeerPayload {
                id: p.id.fmt_short().to_string(),
                role: match p.role {
                    mesh::NodeRole::Worker => "Worker".into(),
                    mesh::NodeRole::Host { .. } => "Host".into(),
                    mesh::NodeRole::Client => "Client".into(),
                },
                models: p.models.clone(),
                available_models: p.available_models.clone(),
                requested_models: p.requested_models.clone(),
                vram_gb: p.vram_bytes as f64 / 1e9,
                serving_models: p.serving_models.clone(),
                hosted_models: p.hosted_models.clone(),
                hosted_models_known: p.hosted_models_known,
                rtt_ms: p.rtt_ms,
                hostname: p.hostname.clone(),
                is_soc: p.is_soc,
                owner_id: p.owner_id.clone(),
                owner_fingerprint: p.owner_fingerprint.clone(),
                owner_fingerprint_verified: p.owner_fingerprint_verified,
                owner_fingerprint_transitive: p.owner_fingerprint_transitive,
                gpus: build_gpus(
                    p.gpu_name.as_deref(),
                    p.gpu_vram.as_deref(),
                    p.gpu_bandwidth_gbps.as_deref(),
                    p.gpu_reserved.as_deref(),
                ),
                model_sizes: p.model_sizes.clone().or_else(|| {
                    let sizes: Vec<(String, u64)> = p
                        .served_model_descriptors
                        .iter()
                        .filter_map(|d| {
                            d.identity
                                .file_size
                                .map(|s| (d.identity.model_name.clone(), s))
                        })
                        .collect();
                    if sizes.is_empty() {
                        None
                    } else {
                        Some(sizes)
                    }
                }),
                model_scans: p.model_metadata.clone(),
            })
            .collect();

        let my_serving_models = node.serving_models().await;
        let my_hosted_models = node.hosted_models().await;
        let has_local_processes = !local_processes.is_empty();
        let effective_llama_ready = llama_ready || has_local_processes;
        let effective_is_host = is_host || has_local_processes;
        let display_model_name = local_processes
            .first()
            .map(|process| process.name.clone())
            .or_else(|| my_hosted_models.first().cloned())
            .or_else(|| my_serving_models.first().cloned())
            .unwrap_or_else(|| model_name.clone());

        let (launch_pi, launch_goose) = if effective_llama_ready {
            (
                Some(format!("pi --provider mesh --model {display_model_name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={display_model_name} goose session")),
            )
        } else {
            (None, None)
        };

        let mesh_id = node.mesh_id().await;

        // Derive node status for display
        let node_status = if is_client {
            "Client".to_string()
        } else if effective_is_host && effective_llama_ready {
            let has_split_workers = all_peers.iter().any(|p| {
                matches!(p.role, mesh::NodeRole::Worker)
                    && p.is_assigned_model(display_model_name.as_str())
            });
            if has_split_workers {
                "Serving (split)".to_string()
            } else {
                "Serving".to_string()
            }
        } else if !effective_is_host && !display_model_name.is_empty() {
            "Worker (split)".to_string()
        } else if display_model_name.is_empty() {
            if all_peers.is_empty() {
                "Idle".to_string()
            } else {
                "Standby".to_string()
            }
        } else {
            "Standby".to_string()
        };

        StatusPayload {
            version: MESH_LLM_VERSION.to_string(),
            latest_version,
            node_id,
            token,
            node_status,
            is_host: effective_is_host,
            is_client,
            llama_ready: effective_llama_ready,
            model_name: display_model_name,
            models: my_models,
            available_models: my_available_models,
            requested_models: my_requested_models,
            serving_models: my_serving_models,
            hosted_models: my_hosted_models,
            draft_name,
            api_port,
            my_vram_gb,
            model_size_gb: model_size_bytes as f64 / 1e9,
            peers,
            launch_pi,
            launch_goose,
            inflight_requests,
            mesh_id,
            mesh_name,
            nostr_discovery,
            my_hostname: node.hostname.clone(),
            my_is_soc: node.is_soc,
            owner_id,
            owner_fingerprint,
            owner_fingerprint_verified,
            owner_fingerprint_transitive: false,
            gpus: {
                let bw = node.gpu_bandwidth_gbps.lock().await;
                let bw_str = bw.as_ref().map(|v| {
                    v.iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                });
                build_gpus(
                    node.gpu_name.as_deref(),
                    node.gpu_vram.as_deref(),
                    bw_str.as_deref(),
                    node.gpu_reserved.as_deref(),
                )
            },
            routing_affinity,
            model_sizes: my_model_sizes,
            model_scans: my_model_scans,
        }
    }

    async fn push_status(&self) {
        let status = self.status().await;
        if let Ok(json) = serde_json::to_string(&status) {
            let event = format!("data: {json}\n\n");
            let mut inner = self.inner.lock().await;
            inner.sse_clients.retain(|tx| !tx.is_closed());
            for tx in &inner.sse_clients {
                let _ = tx.send(event.clone());
            }
        }
    }
}

// ── Server ──

/// Start the mesh management API server.
pub async fn start(
    port: u16,
    state: MeshApi,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
    listen_all: bool,
) {
    // Watch election target changes
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            if target_rx.changed().await.is_err() {
                break;
            }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port)
                | election::InferenceTarget::MoeLocal(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_) | election::InferenceTarget::MoeRemote(_) => {
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None;
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
            state2.push_status().await;
        }
    });

    // Push status when peers join/leave.
    let mut peer_rx = {
        let inner = state.inner.lock().await;
        inner.node.peer_change_rx.clone()
    };
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            if peer_rx.changed().await.is_err() {
                break;
            }
            state3.push_status().await;
        }
    });

    // Push status when in-flight request count changes.
    let mut inflight_rx = {
        let inner = state.inner.lock().await;
        inner.node.inflight_change_rx()
    };
    let state4 = state.clone();
    tokio::spawn(async move {
        loop {
            if inflight_rx.changed().await.is_err() {
                break;
            }
            state4.push_status().await;
        }
    });

    // One-shot check for newer public release (for UI footer indicator).
    let state5 = state.clone();
    tokio::spawn(async move {
        let Some(latest) = crate::latest_release_version().await else {
            return;
        };
        if !crate::version_newer(&latest, crate::VERSION) {
            return;
        }
        {
            let mut inner = state5.inner.lock().await;
            inner.latest_version = Some(latest);
        }
        state5.push_status().await;
    });

    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Management API: failed to bind :{port}: {e}");
            return;
        }
    };
    tracing::info!("Management API on http://localhost:{port}");

    loop {
        let Ok((stream, remote_addr)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_request(stream, remote_addr.ip(), &state).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

async fn handle_request(
    mut stream: TcpStream,
    remote_ip: std::net::IpAddr,
    state: &MeshApi,
) -> anyhow::Result<()> {
    let request = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        proxy::read_http_request(&mut stream),
    )
    .await
    {
        Ok(Ok(request)) => request,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Ok(()), // read timeout — health check probe, just close
    };
    let req = String::from_utf8_lossy(&request.raw);
    let method = request.method.as_str();
    let path = request.path.as_str();
    let path_only = path.split('?').next().unwrap_or(path);
    let body = http_body_text(&request.raw);

    match (method, path_only) {
        // ── Dashboard UI ──
        ("GET", p) if is_console_index_path(p) => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        // ── Frontend static assets (bundled UI dist) ──
        ("GET", p)
            if p.starts_with("/assets/")
                || matches!(p.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
                || (p.ends_with(".json") && !p.starts_with("/api/")) =>
        {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Discover meshes via Nostr ──
        ("GET", "/api/discover") => {
            let relays = state.inner.lock().await.nostr_relays.clone();
            let filter = nostr::MeshFilter::default();
            match nostr::discover(&relays, &filter, None).await {
                Ok(meshes) => {
                    if let Ok(json) = serde_json::to_string(&meshes) {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(), json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    } else {
                        respond_error(&mut stream, 500, "Failed to serialize").await?;
                    }
                }
                Err(e) => {
                    respond_error(&mut stream, 500, &format!("Discovery failed: {e}")).await?;
                }
            }
        }

        // ── Live status ──
        ("GET", "/api/status") => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
                Ok(status) => {
                    respond_json(&mut stream, 200, &status).await?;
                }
                Err(_) => {
                    respond_error(&mut stream, 503, "Status temporarily unavailable").await?;
                }
            }
        }

        ("GET", "/api/models") => {
            let mesh_models = state.mesh_models().await;
            respond_json(&mut stream, 200, &ModelsPayload { mesh_models }).await?;
        }

        ("GET", "/api/runtime") => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_status())
                .await
            {
                Ok(runtime_status) => {
                    respond_json(&mut stream, 200, &runtime_status).await?;
                }
                Err(_) => {
                    respond_error(&mut stream, 503, "Runtime status temporarily unavailable")
                        .await?;
                }
            }
        }

        ("GET", "/api/runtime/processes") => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_processes())
                .await
            {
                Ok(runtime_processes) => {
                    respond_json(&mut stream, 200, &runtime_processes).await?;
                }
                Err(_) => {
                    respond_error(
                        &mut stream,
                        503,
                        "Runtime process status temporarily unavailable",
                    )
                    .await?;
                }
            }
        }

        ("POST", "/api/runtime/models") => {
            let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
                respond_error(&mut stream, 503, "Runtime control unavailable").await?;
                return Ok(());
            };
            let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);
            match parsed {
                Ok(val) => {
                    let spec = val["model"].as_str().unwrap_or("").to_string();
                    if spec.is_empty() {
                        respond_error(&mut stream, 400, "Missing 'model' field").await?;
                    } else {
                        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                        let _ = control_tx.send(RuntimeControlRequest::Load {
                            spec,
                            resp: resp_tx,
                        });
                        match resp_rx.await {
                            Ok(Ok(loaded)) => {
                                respond_json(
                                    &mut stream,
                                    201,
                                    &serde_json::json!({ "loaded": loaded }),
                                )
                                .await?;
                            }
                            Ok(Err(e)) => {
                                respond_runtime_error(&mut stream, &e.to_string()).await?;
                            }
                            Err(_) => {
                                respond_error(&mut stream, 503, "Runtime control unavailable")
                                    .await?;
                            }
                        }
                    }
                }
                Err(_) => {
                    respond_error(&mut stream, 400, "Invalid JSON body").await?;
                }
            }
        }

        ("DELETE", p) if p.starts_with("/api/runtime/models/") => {
            let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
                respond_error(&mut stream, 503, "Runtime control unavailable").await?;
                return Ok(());
            };
            let Some(model_name) = decode_runtime_model_path(p) else {
                respond_error(&mut stream, 400, "Missing model path").await?;
                return Ok(());
            };
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            let _ = control_tx.send(RuntimeControlRequest::Unload {
                model: model_name.clone(),
                resp: resp_tx,
            });
            match resp_rx.await {
                Ok(Ok(())) => {
                    respond_json(
                        &mut stream,
                        200,
                        &serde_json::json!({ "dropped": model_name }),
                    )
                    .await?;
                }
                Ok(Err(e)) => {
                    respond_runtime_error(&mut stream, &e.to_string()).await?;
                }
                Err(_) => {
                    respond_error(&mut stream, 503, "Runtime control unavailable").await?;
                }
            }
        }
        ("GET", "/api/config") => match read_mesh_config_toml(&mesh_config_path()) {
            Ok(raw) => {
                respond_bytes_cached(
                    &mut stream,
                    200,
                    "OK",
                    "application/toml",
                    "no-store",
                    raw.as_bytes(),
                )
                .await?;
            }
            Err(e) => {
                respond_error(&mut stream, 500, &format!("Failed to read config: {e}")).await?;
            }
        },

        ("POST", "/api/config") => {
            let body = String::from_utf8_lossy(http_body_bytes(&request.raw));
            let owner_fingerprint =
                match require_header(&request.raw, CONFIG_OWNER_FINGERPRINT_HEADER) {
                    Ok(v) => v,
                    Err(code) => {
                        respond_error(&mut stream, code, "Missing owner fingerprint").await?;
                        return Ok(());
                    }
                };
            let timestamp = match require_header(&request.raw, CONFIG_TIMESTAMP_HEADER)
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
            {
                Some(v) => v,
                None => {
                    respond_error(&mut stream, 401, "Missing or invalid timestamp").await?;
                    return Ok(());
                }
            };
            let prev_config_hash = match require_header(&request.raw, CONFIG_PREV_HASH_HEADER) {
                Ok(v) => v,
                Err(code) => {
                    respond_error(&mut stream, code, "Missing prev_config_hash").await?;
                    return Ok(());
                }
            };
            let signature = match require_header(&request.raw, CONFIG_SIGNATURE_HEADER) {
                Ok(v) => v,
                Err(code) => {
                    respond_error(&mut stream, code, "Missing signature").await?;
                    return Ok(());
                }
            };

            match replay_signature_seen_before(&signature) {
                Ok(true) => {
                    respond_error(&mut stream, 409, "replayed signed checkpoint").await?;
                    return Ok(());
                }
                Ok(false) => {}
                Err(e) => {
                    respond_error(
                        &mut stream,
                        500,
                        &format!("Failed to check replay state: {e}"),
                    )
                    .await?;
                    return Ok(());
                }
            }

            let local_owner_fingerprint = {
                let node = state.inner.lock().await.node.clone();
                node.owner_fingerprint().await
            };
            let Some(local_owner_fingerprint) = local_owner_fingerprint else {
                respond_error(&mut stream, 403, "This node has no owner key").await?;
                return Ok(());
            };

            if owner_fingerprint != local_owner_fingerprint {
                respond_error(
                    &mut stream,
                    403,
                    "Owner fingerprint does not match this node",
                )
                .await?;
                return Ok(());
            }

            if is_stale_config_timestamp(timestamp) {
                respond_error(&mut stream, 401, "Stale config timestamp").await?;
                return Ok(());
            }

            let config_path = mesh_config_path();
            let config_file_exists = config_path.exists();
            let current_config_hash = match current_mesh_config_hash(&config_path) {
                Ok(hash) => hash,
                Err(e) => {
                    respond_error(
                        &mut stream,
                        500,
                        &format!("Failed to read current config hash: {e}"),
                    )
                    .await?;
                    return Ok(());
                }
            };
            if config_file_exists && prev_config_hash != current_config_hash {
                respond_error(&mut stream, 409, "stale prev_config_hash").await?;
                return Ok(());
            }

            let canonical_payload = canonical_config_signature_payload(
                method,
                "/api/config",
                body.as_ref(),
                &owner_fingerprint,
                timestamp,
                &prev_config_hash,
            );
            let verify_result =
                verify_signed_config_checkpoint(state, &canonical_payload, &signature).await;
            if !verify_result {
                respond_error(&mut stream, 401, "Invalid signature").await?;
                return Ok(());
            }

            let split_catalog = {
                let node = state.inner.lock().await.node.clone();
                build_split_validation_catalog(&node).await
            };

            let parsed = match parse_and_validate_mesh_config(body.as_ref(), &split_catalog) {
                Ok(cfg) => cfg,
                Err(errors) => {
                    respond_validation_errors(&mut stream, &errors).await?;
                    return Ok(());
                }
            };

            if let Err(e) = parsed.save(&mesh_config_path()) {
                respond_error(&mut stream, 500, &format!("Failed to save config: {e}")).await?;
                return Ok(());
            }

            let local_node_id = {
                let inner = state.inner.lock().await;
                inner.node.id().fmt_short().to_string()
            };
            if let Err(e) = sync_node_runtime_config(&parsed, &local_node_id) {
                respond_error(
                    &mut stream,
                    500,
                    &format!("Failed to save node runtime config: {e}"),
                )
                .await?;
                return Ok(());
            }

            if let Err(e) = remember_applied_signature(&signature) {
                respond_error(
                    &mut stream,
                    500,
                    &format!("Failed to persist replay state: {e}"),
                )
                .await?;
                return Ok(());
            }

            let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\n\r\n{\"status\":\"saved\"}";
            stream.write_all(resp.as_bytes()).await?;
        }

        ("GET", p) if p.starts_with("/api/config/node/") => {
            let node_id = p.trim_start_matches("/api/config/node/");
            if node_id.is_empty() {
                respond_error(&mut stream, 404, "Not found").await?;
                return Ok(());
            }

            let local_node_id = {
                let inner = state.inner.lock().await;
                inner.node.id().fmt_short().to_string()
            };
            if node_id == local_node_id {
                match NodeConfig::load(&node_config_path()) {
                    Ok(Some(node_cfg)) => {
                        let json = serde_json::to_string(&node_cfg)?;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                        return Ok(());
                    }
                    Ok(None) => {}
                    Err(e) => {
                        respond_error(
                            &mut stream,
                            500,
                            &format!("Failed to read node runtime config: {e}"),
                        )
                        .await?;
                        return Ok(());
                    }
                }
            }

            let config = match AuthoredMeshConfig::load(&mesh_config_path()) {
                Ok(cfg) => cfg,
                Err(e) => {
                    respond_error(&mut stream, 500, &format!("Failed to read config: {e}")).await?;
                    return Ok(());
                }
            };

            if let Some(node_cfg) = config.for_node_runtime(node_id) {
                let json = serde_json::to_string(&node_cfg)?;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    json.len(),
                    json
                );
                stream.write_all(resp.as_bytes()).await?;
            } else {
                respond_error(&mut stream, 404, "Node config not found").await?;
            }
        }

        ("POST", "/api/scan") => {
            let node = state.inner.lock().await.node.clone();
            if let Err(e) = mesh::Node::run_local_scan(&node).await {
                tracing::warn!("Local scan failed: {e}");
            }
            state.push_status().await;

            let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── SSE event stream ──
        ("GET", "/api/events") => {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
            stream.write_all(header.as_bytes()).await?;

            let status = state.status().await;
            if let Ok(json) = serde_json::to_string(&status) {
                stream
                    .write_all(format!("data: {json}\n\n").as_bytes())
                    .await?;
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            state.inner.lock().await.sse_clients.push(tx);

            loop {
                tokio::select! {
                    event = rx.recv() => {
                        match event {
                            Some(data) => {
                                if stream.write_all(data.as_bytes()).await.is_err() {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                        // SSE keepalive comment to prevent proxy/browser timeout
                        if stream.write_all(b": keepalive\n\n").await.is_err() {
                            break;
                        }
                    }
                }
            }
        }

        // ── Plugins ──
        ("GET", "/api/plugins") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            let plugins = plugin_manager.list().await;
            let json = serde_json::to_string(&plugins)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            let rest = &p["/api/plugins/".len()..];
            let plugin_name = rest.trim_end_matches("/tools");
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            match plugin_manager.tools(plugin_name).await {
                Ok(tools) => {
                    let json = serde_json::to_string(&tools)?;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json.len(),
                        json
                    );
                    stream.write_all(resp.as_bytes()).await?;
                }
                Err(e) => {
                    respond_error(&mut stream, 404, &e.to_string()).await?;
                }
            }
        }

        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            let rest = &p["/api/plugins/".len()..];
            if let Some((plugin_name, tool_name)) = rest.split_once("/tools/") {
                let payload = if body.trim().is_empty() { "{}" } else { body };
                let plugin_manager = state.inner.lock().await.plugin_manager.clone();
                match plugin_manager
                    .call_tool(plugin_name, tool_name, payload)
                    .await
                {
                    Ok(result) if !result.is_error => {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            result.content_json.len(),
                            result.content_json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => {
                        respond_error(&mut stream, 502, &e.to_string()).await?;
                    }
                }
            } else {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        // ── Blackboard ──
        ("GET", "/api/blackboard/feed") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let query_str = path.split('?').nth(1).unwrap_or("");
                let params: Vec<(&str, &str)> = query_str
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .collect();
                let request = crate::blackboard::FeedRequest {
                    from: params
                        .iter()
                        .find(|(k, _)| *k == "from")
                        .map(|(_, v)| (*v).to_string()),
                    limit: params
                        .iter()
                        .find(|(k, _)| *k == "limit")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(20),
                    since: params
                        .iter()
                        .find(|(k, _)| *k == "since")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0),
                };
                match plugin_manager
                    .call_tool(
                        plugin::BLACKBOARD_PLUGIN_ID,
                        "feed",
                        &serde_json::to_string(&request)?,
                    )
                    .await
                {
                    Ok(result) if !result.is_error => {
                        let items: Vec<crate::blackboard::BlackboardItem> =
                            serde_json::from_str(&result.content_json).unwrap_or_default();
                        let json = serde_json::to_string(&items).unwrap_or_else(|_| "[]".into());
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => respond_error(&mut stream, 502, &e.to_string()).await?,
                }
            }
        }

        ("GET", "/api/blackboard/search") => {
            let plugin_manager = state.inner.lock().await.plugin_manager.clone();
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let query_str = path.split('?').nth(1).unwrap_or("");
                let params: Vec<(&str, &str)> = query_str
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .collect();
                let request = crate::blackboard::SearchRequest {
                    query: params
                        .iter()
                        .find(|(k, _)| *k == "q")
                        .map(|(_, v)| (*v).replace('+', " ").replace("%20", " "))
                        .unwrap_or_default(),
                    limit: params
                        .iter()
                        .find(|(k, _)| *k == "limit")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(20),
                    since: params
                        .iter()
                        .find(|(k, _)| *k == "since")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0),
                };
                match plugin_manager
                    .call_tool(
                        plugin::BLACKBOARD_PLUGIN_ID,
                        "search",
                        &serde_json::to_string(&request)?,
                    )
                    .await
                {
                    Ok(result) if !result.is_error => {
                        let items: Vec<crate::blackboard::BlackboardItem> =
                            serde_json::from_str(&result.content_json).unwrap_or_default();
                        let json = serde_json::to_string(&items).unwrap_or_else(|_| "[]".into());
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            json.len(),
                            json
                        );
                        stream.write_all(resp.as_bytes()).await?;
                    }
                    Ok(result) => {
                        respond_error(&mut stream, 502, &result.content_json).await?;
                    }
                    Err(e) => respond_error(&mut stream, 502, &e.to_string()).await?,
                }
            }
        }

        ("POST", "/api/blackboard/post") => {
            let (node, plugin_manager) = {
                let inner = state.inner.lock().await;
                (inner.node.clone(), inner.plugin_manager.clone())
            };
            if !plugin_manager
                .is_enabled(plugin::BLACKBOARD_PLUGIN_ID)
                .await
            {
                respond_error(&mut stream, 404, "Blackboard is disabled on this node").await?;
            } else {
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(body);
                match parsed {
                    Ok(val) => {
                        let text = val["text"].as_str().unwrap_or("").to_string();
                        if text.is_empty() {
                            respond_error(&mut stream, 400, "Missing 'text' field").await?;
                        } else {
                            let request = crate::blackboard::PostRequest {
                                text,
                                from: node.peer_name().await,
                                peer_id: node.id().fmt_short().to_string(),
                            };
                            match plugin_manager
                                .call_tool(
                                    plugin::BLACKBOARD_PLUGIN_ID,
                                    "post",
                                    &serde_json::to_string(&request)?,
                                )
                                .await
                            {
                                Ok(result) if !result.is_error => {
                                    let json = result.content_json;
                                    let resp = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                        json.len(),
                                        json
                                    );
                                    stream.write_all(resp.as_bytes()).await?;
                                }
                                Ok(result) => {
                                    let status = if result.content_json.contains("Rate limited") {
                                        429
                                    } else {
                                        400
                                    };
                                    respond_error(&mut stream, status, &result.content_json)
                                        .await?;
                                }
                                Err(e) => {
                                    let msg = e.to_string();
                                    let status = if msg.contains("Rate limited") {
                                        429
                                    } else {
                                        400
                                    };
                                    respond_error(&mut stream, status, &msg).await?;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        respond_error(&mut stream, 400, "Invalid JSON body").await?;
                    }
                }
            }
        }

        // ── Chat proxy (routes through inference API port) ──
        (m, p) if m != "POST" && p.starts_with("/api/chat") => {
            respond_error(&mut stream, 405, "Method Not Allowed").await?;
        }
        ("POST", p) if p.starts_with("/api/chat") => {
            let inner = state.inner.lock().await;
            if !inner.llama_ready && !inner.is_client {
                drop(inner);
                return respond_error(&mut stream, 503, "LLM not ready").await;
            }
            let port = inner.api_port;
            drop(inner);
            let target = format!("127.0.0.1:{port}");
            if let Ok(mut upstream) = TcpStream::connect(&target).await {
                let rewritten = req.replacen("/api/chat", "/v1/chat/completions", 1);
                upstream.write_all(rewritten.as_bytes()).await?;
                tokio::io::copy_bidirectional(&mut stream, &mut upstream).await?;
            } else {
                respond_error(&mut stream, 502, "Cannot reach LLM server").await?;
            }
        }

        // ── Broadcast config to all peers ──
        ("POST", "/api/config/broadcast") => {
            if !request_origin_is_loopback(&request.raw, remote_ip) {
                respond_error(
                    &mut stream,
                    403,
                    "Only loopback origin may broadcast config",
                )
                .await?;
                return Ok(());
            }

            let body_str = String::from_utf8_lossy(http_body_bytes(&request.raw));
            let split_catalog = {
                let node = state.inner.lock().await.node.clone();
                build_split_validation_catalog(&node).await
            };

            let parsed = match parse_and_validate_mesh_config(body_str.as_ref(), &split_catalog) {
                Ok(cfg) => cfg,
                Err(errors) => {
                    respond_validation_errors(&mut stream, &errors).await?;
                    return Ok(());
                }
            };

            let prev_config_hash = match current_mesh_config_hash(&mesh_config_path()) {
                Ok(hash) => hash,
                Err(e) => {
                    respond_error(
                        &mut stream,
                        500,
                        &format!("Failed to read current config hash: {e}"),
                    )
                    .await?;
                    return Ok(());
                }
            };

            let owner_fingerprint = {
                let node = state.inner.lock().await.node.clone();
                node.owner_fingerprint().await
            };
            let Some(owner_fingerprint) = owner_fingerprint else {
                respond_error(&mut stream, 403, "This node has no owner key").await?;
                return Ok(());
            };
            let timestamp = now_unix_timestamp();
            let canonical_payload = canonical_config_signature_payload(
                "POST",
                "/api/config",
                body_str.as_ref(),
                &owner_fingerprint,
                timestamp,
                &prev_config_hash,
            );
            let signature = match sign_config_payload(&canonical_payload) {
                Ok(signature) => signature,
                Err(e) => {
                    respond_error(
                        &mut stream,
                        500,
                        &format!("Failed to sign config payload: {e}"),
                    )
                    .await?;
                    return Ok(());
                }
            };

            if let Err(e) = parsed.save(&mesh_config_path()) {
                respond_error(
                    &mut stream,
                    500,
                    &format!("Failed to save config locally: {e}"),
                )
                .await?;
                return Ok(());
            }

            let local_node_id = {
                let inner = state.inner.lock().await;
                inner.node.id().fmt_short().to_string()
            };
            if let Err(e) = sync_node_runtime_config(&parsed, &local_node_id) {
                respond_error(
                    &mut stream,
                    500,
                    &format!("Failed to save node runtime config: {e}"),
                )
                .await?;
                return Ok(());
            }

            let (peers, api_port) = {
                let inner = state.inner.lock().await;
                (inner.node.peers().await, inner.api_port)
            };

            let toml_body = body_str.to_string();
            let relay_peers: Vec<mesh::PeerInfo> = peers
                .into_iter()
                .filter(|peer| {
                    peer.owner_fingerprint_verified
                        && peer.owner_fingerprint.as_deref() == Some(owner_fingerprint.as_str())
                })
                .collect();
            let relay_headers = ConfigRelayHeaders {
                owner_fingerprint,
                timestamp,
                prev_config_hash,
                signature,
            };
            let results =
                relay_config_to_peers(&relay_peers, api_port, &toml_body, &relay_headers).await;

            let saved = 1 + results.iter().filter(|r| r.ok).count();
            let total = 1 + results.len();
            let failed: Vec<String> = results
                .iter()
                .filter(|r| !r.ok)
                .map(|r| r.node.clone())
                .collect();

            let resp_body = serde_json::json!({
                "saved": saved,
                "total": total,
                "failed": failed,
            });
            let resp_str = resp_body.to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp_str.len(),
                resp_str
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        // ── Broadcast scan to all peers ──
        ("POST", "/api/scan/broadcast") => {
            let node = state.inner.lock().await.node.clone();
            if let Err(e) = mesh::Node::run_local_scan(&node).await {
                tracing::warn!("Local scan failed during broadcast: {e}");
            }
            state.push_status().await;

            let results = node.relay_scan_to_peers().await;

            let refreshed = 1 + results.iter().filter(|(_, ok)| *ok).count();
            let total = 1 + results.len();
            let failed: Vec<String> = results
                .into_iter()
                .filter(|(_, ok)| !*ok)
                .map(|(name, _)| name)
                .collect();

            let resp_body = serde_json::json!({
                "refreshed": refreshed,
                "total": total,
                "failed": failed,
            });
            let resp_str = resp_body.to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp_str.len(),
                resp_str
            );
            stream.write_all(resp.as_bytes()).await?;
        }

        _ => {
            respond_error(&mut stream, 404, "Not found").await?;
        }
    }
    Ok(())
}

// ── Peer relay helpers ──

struct RelayResult {
    node: String,
    ok: bool,
}

#[derive(Debug, Clone)]
struct ConfigRelayHeaders {
    owner_fingerprint: String,
    timestamp: i64,
    prev_config_hash: String,
    signature: String,
}

async fn relay_config_to_peers(
    peers: &[mesh::PeerInfo],
    api_port: u16,
    toml_body: &str,
    headers: &ConfigRelayHeaders,
) -> Vec<RelayResult> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    let mut set = tokio::task::JoinSet::new();
    for peer in peers {
        let Some(ref hostname) = peer.hostname else {
            continue;
        };
        let url = format!("http://{}:{}/api/config", hostname, api_port);
        let node_label = hostname.clone();
        let client = client.clone();
        let body = toml_body.to_string();
        let headers = headers.clone();
        set.spawn(async move {
            let ok = client
                .post(&url)
                .header("Content-Type", "application/toml")
                .header(CONFIG_OWNER_FINGERPRINT_HEADER, headers.owner_fingerprint)
                .header(CONFIG_TIMESTAMP_HEADER, headers.timestamp.to_string())
                .header(CONFIG_PREV_HASH_HEADER, headers.prev_config_hash)
                .header(CONFIG_SIGNATURE_HEADER, headers.signature)
                .body(body)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            RelayResult {
                node: node_label,
                ok,
            }
        });
    }

    let mut results = Vec::with_capacity(set.len());
    while let Some(res) = set.join_next().await {
        if let Ok(r) = res {
            results.push(r);
        }
    }
    results
}

fn http_body_text(raw: &[u8]) -> &str {
    let body_start = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(raw.len());
    std::str::from_utf8(&raw[body_start..]).unwrap_or("")
}

fn parse_header_value(raw: &[u8], header_name: &str) -> Option<String> {
    let header_end = find_header_end(raw)?;
    let header_text = String::from_utf8_lossy(&raw[..header_end]);
    for line in header_text.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case(header_name) {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn require_header(raw: &[u8], header_name: &str) -> Result<String, u16> {
    parse_header_value(raw, header_name).ok_or(401)
}

fn now_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn is_stale_config_timestamp(timestamp: i64) -> bool {
    let now = now_unix_timestamp();
    timestamp < now.saturating_sub(CONFIG_TIMESTAMP_MAX_AGE_SECS)
        || timestamp > now.saturating_add(CONFIG_TIMESTAMP_MAX_FUTURE_SKEW_SECS)
}

fn canonical_config_signature_payload(
    method: &str,
    path: &str,
    body: &str,
    owner_fingerprint: &str,
    timestamp: i64,
    prev_config_hash: &str,
) -> String {
    let mut body_hasher = Sha256::new();
    body_hasher.update(body.as_bytes());
    let body_sha256 = format!("{:x}", body_hasher.finalize());
    format!(
        "method={method}\npath={path}\nbody_sha256={body_sha256}\nowner_fingerprint={owner_fingerprint}\ntimestamp={timestamp}\nprev_config_hash={prev_config_hash}"
    )
}

fn current_mesh_config_hash(path: &Path) -> anyhow::Result<String> {
    let toml = read_mesh_config_toml(path)?;
    let mut hasher = Sha256::new();
    hasher.update(toml.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn replay_state_path() -> std::path::PathBuf {
    mesh_config_path()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("config-checkpoint-replay.log")
}

fn replay_signature_seen_before(signature: &str) -> anyhow::Result<bool> {
    let path = replay_state_path();
    if !path.exists() {
        return Ok(false);
    }
    let raw = std::fs::read_to_string(path)?;
    Ok(raw.lines().any(|line| line.trim() == signature))
}

fn remember_applied_signature(signature: &str) -> anyhow::Result<()> {
    let path = replay_state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    use std::io::Write;
    writeln!(file, "{signature}")?;
    Ok(())
}

fn owner_signing_key_path_from_args() -> anyhow::Result<std::path::PathBuf> {
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--owner-key" {
            if let Some(path) = args.next() {
                return Ok(std::path::PathBuf::from(path));
            }
            break;
        }
        if let Some(arg_str) = arg.to_str() {
            if let Some(rest) = arg_str.strip_prefix("--owner-key=") {
                return Ok(std::path::PathBuf::from(rest));
            }
        }
    }
    let mut default_path = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    default_path.push(".mesh-llm");
    default_path.push("owner-key");
    Ok(default_path)
}

fn load_owner_signing_key() -> anyhow::Result<Option<ed25519_dalek::SigningKey>> {
    let path = owner_signing_key_path_from_args()?;
    if !path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&path)?;
    let decoded = hex::decode(contents.trim())
        .map_err(|e| anyhow::anyhow!("owner key at {} is not valid hex: {e}", path.display()))?;
    let key_bytes: [u8; 32] = decoded.try_into().map_err(|_| {
        anyhow::anyhow!(
            "owner key at {} must be exactly 32 bytes (64 hex chars)",
            path.display()
        )
    })?;
    Ok(Some(ed25519_dalek::SigningKey::from_bytes(&key_bytes)))
}

fn sign_config_payload(canonical_payload: &str) -> anyhow::Result<String> {
    let signing_key = load_owner_signing_key()?
        .ok_or_else(|| anyhow::anyhow!("owner signing key not available for config broadcast"))?;
    let sig: Signature = signing_key.sign(canonical_payload.as_bytes());
    Ok(hex::encode(sig.to_bytes()))
}

async fn verify_signed_config_checkpoint(
    state: &MeshApi,
    canonical_payload: &str,
    signature: &str,
) -> bool {
    if let Ok(Some(signing_key)) = load_owner_signing_key() {
        let verifying = VerifyingKey::from(&signing_key);
        if let Ok(sig_bytes) = hex::decode(signature) {
            if let Ok(sig_array) = <[u8; 64]>::try_from(sig_bytes.as_slice()) {
                let sig = Signature::from_bytes(&sig_array);
                if verifying.verify(canonical_payload.as_bytes(), &sig).is_ok() {
                    return true;
                }
            }
        }
    }

    let node = state.inner.lock().await.node.clone();
    let Some(attestation) = node.owner_attestation().await else {
        return false;
    };
    let Ok(pubkey) = <[u8; 32]>::try_from(attestation.owner_public_key.as_slice()) else {
        return false;
    };
    let Ok(verifying) = VerifyingKey::from_bytes(&pubkey) else {
        return false;
    };
    let Ok(sig_bytes) = hex::decode(signature) else {
        return false;
    };
    let Ok(sig_array) = <[u8; 64]>::try_from(sig_bytes.as_slice()) else {
        return false;
    };
    let sig = Signature::from_bytes(&sig_array);
    verifying.verify(canonical_payload.as_bytes(), &sig).is_ok()
}

fn request_origin_is_loopback(raw: &[u8], remote_ip: std::net::IpAddr) -> bool {
    if let Some(forwarded) = parse_header_value(raw, "X-Forwarded-For") {
        return forwarded.split(',').map(|part| part.trim()).all(|part| {
            if part.eq_ignore_ascii_case("localhost") {
                return true;
            }
            part.parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false)
        });
    }
    remote_ip.is_loopback()
}

fn is_console_index_path(path: &str) -> bool {
    matches!(
        path,
        "/" | "/dashboard" | "/chat" | "/config" | "/dashboard/" | "/chat/" | "/config/"
    ) || path.starts_with("/chat/")
}

fn sync_node_runtime_config(config: &AuthoredMeshConfig, node_id: &str) -> anyhow::Result<()> {
    let path = node_config_path();
    sync_node_runtime_config_at(&path, config, node_id)
}

fn sync_node_runtime_config_at(
    path: &Path,
    config: &AuthoredMeshConfig,
    node_id: &str,
) -> anyhow::Result<()> {
    if let Some(node_cfg) = config.for_node_runtime(node_id) {
        node_cfg.save(path)?;
    } else if path.exists() {
        std::fs::remove_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to remove stale node runtime config: {e}"))?;
    }
    Ok(())
}

fn read_mesh_config_toml(path: &Path) -> anyhow::Result<String> {
    let config = AuthoredMeshConfig::load(path)?;
    Ok(toml::to_string_pretty(&config)?)
}

fn parse_and_validate_mesh_config(
    raw: &str,
    split_catalog: &SplitValidationCatalog,
) -> Result<AuthoredMeshConfig, Vec<ConfigValidationError>> {
    let parsed: AuthoredMeshConfig = toml::from_str(raw).map_err(|e| {
        vec![ConfigValidationError {
            code: "invalid_toml".into(),
            path: "$".into(),
            message: format!("Invalid TOML: {e}"),
        }]
    })?;
    validate_mesh_config(&parsed, split_catalog)?;
    Ok(parsed)
}

fn validate_mesh_config(
    config: &AuthoredMeshConfig,
    split_catalog: &SplitValidationCatalog,
) -> Result<(), Vec<ConfigValidationError>> {
    let mut errors = Vec::new();

    if config.version != AUTHORED_CONFIG_VERSION {
        errors.push(ConfigValidationError {
            code: "unsupported_version".into(),
            path: "version".into(),
            message: format!(
                "Unsupported config version {}, expected {}",
                config.version, AUTHORED_CONFIG_VERSION
            ),
        });
    }

    let mut split_groups: HashMap<String, Vec<SplitSegmentRef>> = HashMap::new();
    let mut split_group_totals: HashMap<String, u32> = HashMap::new();

    for (node_idx, node) in config.nodes.iter().enumerate() {
        if node.node_id.trim().is_empty() {
            errors.push(ConfigValidationError {
                code: "empty_node_id".into(),
                path: format!("nodes[{node_idx}].node_id"),
                message: "node_id must not be empty".into(),
            });
        }

        for (model_idx, model) in node.models.iter().enumerate() {
            if model.name.trim().is_empty() {
                errors.push(ConfigValidationError {
                    code: "empty_model_name".into(),
                    path: format!("nodes[{node_idx}].models[{model_idx}].name"),
                    message: "name must not be empty".into(),
                });
            }

            if let Some(model_key) = model.model_key.as_ref() {
                if model_key.trim().is_empty() {
                    errors.push(ConfigValidationError {
                        code: "empty_model_key".into(),
                        path: format!("nodes[{node_idx}].models[{model_idx}].model_key"),
                        message: "model_key must not be empty when present".into(),
                    });
                }
            }

            let Some(split) = model.split.as_ref() else {
                continue;
            };

            let model_path = format!("nodes[{node_idx}].models[{model_idx}]");
            let Some(model_key_raw) = model.model_key.as_deref() else {
                errors.push(ConfigValidationError {
                    code: "missing_model_key".into(),
                    path: format!("{model_path}.model_key"),
                    message: "model_key is required when split is set".into(),
                });
                continue;
            };

            let model_key = model_key_raw.trim();
            if model_key.is_empty() {
                errors.push(ConfigValidationError {
                    code: "empty_model_key".into(),
                    path: format!("{model_path}.model_key"),
                    message: "model_key must not be empty when split is set".into(),
                });
                continue;
            }

            if split.total == 0 {
                errors.push(ConfigValidationError {
                    code: "invalid_split_total".into(),
                    path: format!("{model_path}.split.total"),
                    message: "split.total must be greater than 0".into(),
                });
                continue;
            }

            if split.end <= split.start {
                errors.push(ConfigValidationError {
                    code: "zero_length_range".into(),
                    path: format!("{model_path}.split"),
                    message: "split range must have end > start".into(),
                });
                continue;
            }

            if split.start >= split.total || split.end > split.total {
                errors.push(ConfigValidationError {
                    code: "split_out_of_range".into(),
                    path: format!("{model_path}.split"),
                    message: "split.start must be less than split.total and split.end must not exceed split.total".into(),
                });
                continue;
            }

            if let Some(existing_total) = split_group_totals.get(model_key) {
                if *existing_total != split.total {
                    errors.push(ConfigValidationError {
                        code: "mismatched_split_totals".into(),
                        path: format!("{model_path}.split.total"),
                        message: format!(
                            "split.total {} does not match group total {} for model_key {}",
                            split.total, existing_total, model_key
                        ),
                    });
                }
            } else {
                split_group_totals.insert(model_key.to_string(), split.total);
            }

            split_groups
                .entry(model_key.to_string())
                .or_default()
                .push(SplitSegmentRef {
                    path: model_path,
                    start: split.start,
                    end: split.end,
                });
        }
    }

    for (model_key, segments) in split_groups.iter_mut() {
        let Some(expected_total) = split_group_totals.get(model_key).copied() else {
            continue;
        };

        match split_catalog.totals_by_model_key.get(model_key).copied() {
            Some(catalog_total) if catalog_total != expected_total => {
                errors.push(ConfigValidationError {
                    code: "mismatched_total_offloadable_layers".into(),
                    path: format!("model_key:{model_key}"),
                    message: format!(
                        "split total {} does not match total_offloadable_layers {} for model_key {}",
                        expected_total, catalog_total, model_key
                    ),
                });
            }
            None => {
                errors.push(ConfigValidationError {
                    code: "unknown_model_key".into(),
                    path: format!("model_key:{model_key}"),
                    message: format!(
                        "model_key {} not found in model_scans with total_offloadable_layers",
                        model_key
                    ),
                });
            }
            _ => {}
        }

        segments.sort_by_key(|segment| segment.start);

        let mut expected_start = 0u32;
        for segment in segments.iter() {
            if segment.start < expected_start {
                errors.push(ConfigValidationError {
                    code: "overlapping_split_ranges".into(),
                    path: format!("{}.split", segment.path),
                    message: format!(
                        "split range {}-{} overlaps previous segment for model_key {}",
                        segment.start, segment.end, model_key
                    ),
                });
            } else if segment.start > expected_start {
                errors.push(ConfigValidationError {
                    code: "split_gap".into(),
                    path: format!("{}.split", segment.path),
                    message: format!(
                        "split gap before segment {}-{} for model_key {}",
                        segment.start, segment.end, model_key
                    ),
                });
            }

            expected_start = segment.end;
        }

        if expected_start != expected_total {
            errors.push(ConfigValidationError {
                code: "incomplete_split_coverage".into(),
                path: format!("model_key:{model_key}"),
                message: format!(
                    "split coverage ends at {}, expected contiguous coverage through {} for model_key {}",
                    expected_start,
                    expected_total,
                    model_key
                ),
            });
        }
    }

    for (node_idx, node) in config.nodes.iter().enumerate() {
        for (model_idx, model) in node.models.iter().enumerate() {
            let model_path = format!("nodes[{node_idx}].models[{model_idx}]");
            match node.placement_mode {
                PlacementMode::Pooled => {
                    if model.gpu_index.is_some() {
                        errors.push(ConfigValidationError {
                            code: "gpu_index_in_pooled_mode".into(),
                            path: format!("{model_path}.gpu_index"),
                            message: format!(
                                "gpu_index must not be set when placement_mode is pooled (node {})",
                                node.node_id
                            ),
                        });
                    }
                }
                PlacementMode::Separate => {
                    if model.gpu_index.is_none() {
                        errors.push(ConfigValidationError {
                            code: "missing_gpu_index_in_separate_mode".into(),
                            path: format!("{model_path}.gpu_index"),
                            message: format!(
                                "gpu_index is required when placement_mode is separate (node {})",
                                node.node_id
                            ),
                        });
                    }
                }
            }

            if let (Some(gpu_idx), Some(&gpu_count)) = (
                model.gpu_index,
                split_catalog.gpu_counts_by_node_id.get(&node.node_id),
            ) {
                if gpu_idx >= gpu_count {
                    errors.push(ConfigValidationError {
                        code: "gpu_index_out_of_range".into(),
                        path: format!("{model_path}.gpu_index"),
                        message: format!(
                            "gpu_index {} is out of range (node {} has {} GPU(s))",
                            gpu_idx, node.node_id, gpu_count
                        ),
                    });
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

async fn build_split_validation_catalog(node: &mesh::Node) -> SplitValidationCatalog {
    let mut totals_by_model_key = HashMap::new();

    for model in node.available_model_metadata().await {
        if let Some(total) = model.metadata.total_offloadable_layers {
            totals_by_model_key.insert(model.model_key, total);
        }
    }

    for peer in node.peers().await {
        if let Some(model_metadata) = peer.model_metadata {
            for model in model_metadata {
                if let Some(total) = model.metadata.total_offloadable_layers {
                    totals_by_model_key.entry(model.model_key).or_insert(total);
                }
            }
        }
    }

    SplitValidationCatalog {
        totals_by_model_key,
        ..Default::default()
    }
}

fn find_header_end(req: &[u8]) -> Option<usize> {
    req.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_content_length(header: &str) -> Option<usize> {
    header.lines().find_map(|line| {
        line.split_once(':').and_then(|(k, v)| {
            if k.trim().eq_ignore_ascii_case("content-length") {
                v.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
    })
}

fn http_body_bytes(req: &[u8]) -> &[u8] {
    if let Some(header_end) = find_header_end(req) {
        let header_text = String::from_utf8_lossy(&req[..header_end]);
        let body_start = header_end + 4;
        if let Some(content_len) = parse_content_length(&header_text) {
            let body_end = body_start.saturating_add(content_len).min(req.len());
            return &req[body_start..body_end];
        }
        return &req[body_start..];
    }
    &[]
}

async fn respond_error(stream: &mut TcpStream, code: u16, msg: &str) -> anyhow::Result<()> {
    let body = serde_json::to_string(&serde_json::json!({"error": msg}))
        .unwrap_or_else(|_| r#"{"error":"internal error"}"#.to_string());
    let status = match code {
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_json<T: Serialize>(
    stream: &mut TcpStream,
    code: u16,
    value: &T,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(value)?;
    let status = match code {
        200 => "OK",
        201 => "Created",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn respond_validation_errors(
    stream: &mut TcpStream,
    errors: &[ConfigValidationError],
) -> anyhow::Result<()> {
    let payload = ConfigValidationErrorResponse {
        error: "config_validation_failed".into(),
        errors: errors.to_vec(),
    };
    let body = serde_json::to_string(&payload)?;
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

fn build_runtime_status_payload(
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

fn build_runtime_processes_payload(
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

async fn respond_runtime_error(stream: &mut TcpStream, msg: &str) -> anyhow::Result<()> {
    respond_error(stream, classify_runtime_error(msg), msg).await
}

fn decode_runtime_model_path(path: &str) -> Option<String> {
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

async fn respond_console_index(stream: &mut TcpStream) -> anyhow::Result<bool> {
    if let Some(file) = CONSOLE_DIST.get_file("index.html") {
        respond_bytes_cached(
            stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            "no-cache",
            file.contents(),
        )
        .await?;
        return Ok(true);
    }
    Ok(false)
}

async fn respond_console_asset(stream: &mut TcpStream, path: &str) -> anyhow::Result<bool> {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return Ok(false);
    }
    let Some(file) = CONSOLE_DIST.get_file(rel) else {
        return Ok(false);
    };
    let content_type = match rel.rsplit('.').next().unwrap_or("") {
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    };
    // Hashed asset filenames (Vite output) are immutable — cache forever.
    // Non-hashed assets (favicon, manifest) get short cache.
    let cache_control = if rel.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    };
    respond_bytes_cached(
        stream,
        200,
        "OK",
        content_type,
        cache_control,
        file.contents(),
    )
    .await?;
    Ok(true)
}

async fn respond_bytes_cached(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    cache_control: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: {cache_control}\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AuthoredModelAssignment, AuthoredNodeConfig, MeshConfig, ModelAssignment, ModelSplit,
        NodeConfig, PlacementMode,
    };
    use iroh::SecretKey;
    use iroh::{EndpointAddr, EndpointId};
    use mesh_llm_plugin::MeshVisibility;
    use serial_test::serial;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    fn split_catalog(entries: &[(&str, u32)]) -> SplitValidationCatalog {
        let totals_by_model_key = entries
            .iter()
            .map(|(key, total)| (key.to_string(), *total))
            .collect();
        SplitValidationCatalog {
            totals_by_model_key,
            ..Default::default()
        }
    }

    fn make_temp_dir() -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dir.push(format!("mesh-llm-api-test-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_build_gpus_both_none() {
        let result = build_gpus(None, None, None, None);
        assert!(result.is_empty(), "expected empty vec when no gpu_name");
    }

    #[test]
    fn test_build_gpus_single_no_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), None, None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 0);
    }

    #[test]
    fn test_build_gpus_single_with_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), Some("34359738368"), None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
    }

    #[test]
    fn test_build_gpus_multi_full_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(result[1].name, "NVIDIA RTX 3080");
        assert_eq!(result[1].vram_bytes, 10_737_418_240);
    }

    #[test]
    fn test_build_gpus_multi_partial_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368"),
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(
            result[1].vram_bytes, 0,
            "missing VRAM entry should default to 0"
        );
    }

    #[test]
    fn test_build_gpus_vram_no_gpu_name() {
        let result = build_gpus(None, Some("34359738368"), None, None);
        assert!(
            result.is_empty(),
            "no gpu_name means no entries even if vram present"
        );
    }

    #[test]
    fn test_build_gpus_vram_whitespace_trimmed() {
        let result = build_gpus(Some("NVIDIA RTX 4090"), Some(" 25769803776 "), None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vram_bytes, 25_769_803_776);
    }

    #[test]
    fn test_build_gpus_with_bandwidth() {
        let result = build_gpus(
            Some("NVIDIA A100, NVIDIA A6000"),
            Some("85899345920,51539607552"),
            Some("1948.70,780.10"),
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].bandwidth_gbps, Some(1948.70));
        assert_eq!(result[1].bandwidth_gbps, Some(780.10));
    }

    #[test]
    fn test_build_gpus_unparsable_vram_preserves_index() {
        let result = build_gpus(Some("GPU0, GPU1, GPU2"), Some("100,foo,300"), None, None);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].vram_bytes, 100);
        assert_eq!(
            result[1].vram_bytes, 0,
            "unparsable vram should default to 0, not shift indices"
        );
        assert_eq!(result[2].vram_bytes, 300);
    }

    #[test]
    fn test_build_gpus_unparsable_bandwidth_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,200,300"),
            Some("1.0,bad,3.0"),
            None,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].bandwidth_gbps, Some(1.0));
        assert_eq!(
            result[1].bandwidth_gbps, None,
            "unparsable bandwidth should be None, not shift indices"
        );
        assert_eq!(result[2].bandwidth_gbps, Some(3.0));
    }

    #[test]
    fn test_build_gpus_with_reserved() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
            None,
            Some("555745280,380534784"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].reserved_bytes, 555_745_280);
        assert_eq!(result[1].reserved_bytes, 380_534_784);
    }

    #[test]
    fn test_build_gpus_reserved_defaults_to_zero() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), Some("34359738368"), None, None);
        assert_eq!(result[0].reserved_bytes, 0, "reserved_bytes defaults to 0 when not provided");
    }

    #[test]
    fn test_build_gpus_unparsable_reserved_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,200,300"),
            None,
            Some("1024,bad,3072"),
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].reserved_bytes, 1024);
        assert_eq!(
            result[1].reserved_bytes, 0,
            "unparsable reserved should be 0, not shift indices"
        );
        assert_eq!(result[2].reserved_bytes, 3072);
    }

    #[test]
    fn test_http_body_text_extracts_body() {
        let raw = b"POST /api/plugins/x/tools/y HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        assert_eq!(http_body_text(raw), "{\"a\":1}");
    }

    #[test]
    fn test_build_runtime_status_payload_uses_local_processes() {
        let result = build_runtime_status_payload(
            "Qwen",
            Some("llama".into()),
            true,
            true,
            Some(9337),
            vec![
                RuntimeProcessPayload {
                    name: "Qwen".into(),
                    backend: "llama".into(),
                    status: "ready".into(),
                    port: 9337,
                    pid: 100,
                },
                RuntimeProcessPayload {
                    name: "Llama".into(),
                    backend: "llama".into(),
                    status: "ready".into(),
                    port: 9444,
                    pid: 101,
                },
            ],
        );
        assert_eq!(result.models.len(), 2);
        assert_eq!(result.models[0].name, "Llama");
        assert_eq!(result.models[0].port, Some(9444));
        assert_eq!(result.models[1].name, "Qwen");
    }

    #[test]
    fn test_build_runtime_status_payload_adds_starting_primary() {
        let payload = build_runtime_status_payload(
            "Qwen",
            Some("llama".into()),
            true,
            false,
            Some(9337),
            vec![],
        );

        assert_eq!(payload.models.len(), 1);
        assert_eq!(payload.models[0].status, "starting");
        assert_eq!(payload.models[0].port, Some(9337));
    }

    #[test]
    fn test_build_runtime_processes_payload_sorts_processes() {
        let payload = build_runtime_processes_payload(vec![
            RuntimeProcessPayload {
                name: "Zulu".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
            },
            RuntimeProcessPayload {
                name: "Alpha".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 10,
            },
        ]);

        assert_eq!(payload.processes.len(), 2);
        assert_eq!(payload.processes[0].name, "Alpha");
        assert_eq!(payload.processes[1].name, "Zulu");
    }

    #[test]
    fn test_classify_runtime_error_codes() {
        assert_eq!(classify_runtime_error("model 'x' is not loaded"), 404);
        assert_eq!(classify_runtime_error("model 'x' is already loaded"), 409);
        assert_eq!(
            classify_runtime_error("runtime load only supports models that fit locally"),
            422
        );
        assert_eq!(classify_runtime_error("bad request"), 400);
    }

    #[test]
    fn test_decode_runtime_model_path_decodes_percent_not_plus() {
        // %20 is a space; + is a literal plus in URL paths (not a space)
        assert_eq!(
            decode_runtime_model_path("/api/runtime/models/Llama%203.2+1B"),
            Some("Llama 3.2+1B".into())
        );
    }

    #[test]
    fn test_decode_runtime_model_path_decodes_utf8_multibyte() {
        // é is U+00E9, encoded in UTF-8 as 0xC3 0xA9
        assert_eq!(
            decode_runtime_model_path("/api/runtime/models/mod%C3%A9le"),
            Some("modéle".into())
        );
        // invalid UTF-8 sequence should return None
        assert_eq!(decode_runtime_model_path("/api/runtime/models/%80"), None);
    }

    async fn build_test_mesh_api() -> MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        node.set_owner_key_material(Some([0x42; 32])).await;
        let resolved_plugins = plugin::ResolvedPlugins {
            externals: vec![],
            inactive: vec![],
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let plugin_manager = plugin::PluginManager::start(
            &resolved_plugins,
            plugin::PluginHostMode {
                mesh_visibility: MeshVisibility::Private,
            },
            mesh_tx,
        )
        .await
        .unwrap();
        MeshApi::new(
            node,
            "test-model".to_string(),
            3131,
            0,
            plugin_manager,
            affinity::AffinityRouter::default(),
        )
    }

    async fn spawn_management_test_server(
        state: MeshApi,
    ) -> (
        std::net::SocketAddr,
        tokio::task::JoinHandle<anyhow::Result<()>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (stream, remote) = listener.accept().await.unwrap();
            handle_request(stream, remote.ip(), &state).await
        });
        (addr, handle)
    }

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    async fn read_until_contains(
        stream: &mut TcpStream,
        needle: &[u8],
        timeout: Duration,
    ) -> Vec<u8> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut response = Vec::new();
        while !contains_bytes(&response, needle) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out waiting for {:?} in response: {}",
                String::from_utf8_lossy(needle),
                String::from_utf8_lossy(&response)
            );
            let mut chunk = [0u8; 4096];
            let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
                .await
                .expect("timed out waiting for response bytes")
                .unwrap();
            assert!(n > 0, "unexpected EOF while waiting for response bytes");
            response.extend_from_slice(&chunk[..n]);
        }
        response
    }

    fn hex_signature(secret: &SecretKey, canonical_payload: &str) -> String {
        let signature = secret.sign(canonical_payload.as_bytes());
        hex::encode(signature.to_bytes())
    }

    fn configure_test_owner_key(home_root: &std::path::Path, owner_secret: &SecretKey) {
        std::fs::create_dir_all(home_root.join(".mesh-llm")).unwrap();
        std::env::set_var("HOME", home_root);
        std::fs::write(
            home_root.join(".mesh-llm/owner-key"),
            hex::encode(owner_secret.to_bytes()),
        )
        .unwrap();
    }

    fn peer_info_with_owner(
        seed: u8,
        hostname: &str,
        owner_fingerprint: Option<&str>,
        verified: bool,
    ) -> mesh::PeerInfo {
        let secret = SecretKey::from_bytes(&[seed; 32]);
        let id = EndpointId::from(secret.public());
        mesh::PeerInfo {
            id,
            addr: EndpointAddr {
                id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role: mesh::NodeRole::Worker,
            models: vec![],
            vram_bytes: 0,
            rtt_ms: None,
            model_source: None,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            model_sizes: None,
            model_metadata: None,
            requested_models: vec![],
            last_seen: std::time::Instant::now(),
            version: None,
            gpu_name: None,
            hostname: Some(hostname.to_string()),
            is_soc: None,
            gpu_vram: None,
            gpu_bandwidth_gbps: None,
            gpu_reserved: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            owner_id: owner_fingerprint.map(|v| v.to_string()),
            owner_fingerprint: owner_fingerprint.map(|v| v.to_string()),
            served_model_descriptors: vec![],
            owner_fingerprint_verified: verified,
            owner_fingerprint_transitive: !verified && owner_fingerprint.is_some(),
        }
    }

    fn http_post_with_headers(path: &str, body: &str, headers: &[(&str, String)]) -> Vec<u8> {
        let mut req = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/toml\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (name, value) in headers {
            req.push_str(&format!("{name}: {value}\r\n"));
        }
        req.push_str("\r\n");

        let mut bytes = req.into_bytes();
        bytes.extend_from_slice(body.as_bytes());
        bytes
    }

    async fn send_management_request_once(state: MeshApi, request_bytes: Vec<u8>) -> String {
        let (addr, handle) = spawn_management_test_server(state).await;

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(&request_bytes).await.unwrap();

        let mut response_bytes = Vec::new();
        stream.read_to_end(&mut response_bytes).await.unwrap();
        handle.await.unwrap().unwrap();

        String::from_utf8_lossy(&response_bytes).to_string()
    }

    fn status_code(response: &str) -> u16 {
        response
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn test_management_request_parser_handles_fragmented_post_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = br#"{"text":"fragmented"}"#;
        let headers = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                proxy::read_http_request(&mut stream),
            )
            .await
            .unwrap()
            .unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(&headers.as_bytes()[..45]).await.unwrap();
            stream.write_all(&headers.as_bytes()[45..]).await.unwrap();
            stream.write_all(&body[..8]).await.unwrap();
            stream.write_all(&body[8..]).await.unwrap();
            let mut sink = [0u8; 1];
            let _ = stream.read(&mut sink).await;
        });

        client.await.unwrap();
        let request = server.await.unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/blackboard/post");
        assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
    }

    #[tokio::test]
    async fn test_api_events_sends_initial_payload_and_updates() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state.clone()).await;

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();

        let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
        let initial_text = String::from_utf8_lossy(&initial);
        assert!(initial_text.contains("HTTP/1.1 200 OK"));
        assert!(initial_text.contains("Content-Type: text/event-stream"));
        assert!(initial_text.contains("\"llama_ready\":false"));

        state.update(true, true).await;
        let updated =
            read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
        let updated_text = String::from_utf8_lossy(&updated);
        assert!(updated_text.contains("\"llama_ready\":true"));
        assert!(updated_text.contains("\"is_host\":true"));

        drop(stream);
        handle.abort();
    }

    #[tokio::test]
    #[serial]
    async fn config_requires_valid_owner_signature() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";
        let method = "POST";
        let path = "/api/config";

        let temp = make_temp_dir();
        std::fs::create_dir_all(temp.join(".mesh-llm")).unwrap();
        std::env::set_var("HOME", &temp);

        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        let owner_fingerprint = hex::encode(Sha256::digest(owner_secret.public().as_bytes()));
        std::fs::write(
            temp.join(".mesh-llm/owner-key"),
            hex::encode(owner_secret.to_bytes()),
        )
        .unwrap();

        std::fs::write(temp.join(".mesh-llm/mesh.toml"), "version = 1\n").unwrap();

        let fresh_timestamp = now_unix_timestamp();
        let stale_timestamp = now_unix_timestamp() - (CONFIG_TIMESTAMP_MAX_AGE_SECS + 5);
        let wrong_owner_fingerprint = "owner-fp-other";

        let prev_config_hash = current_mesh_config_hash(&mesh_config_path()).unwrap();
        let stale_prev_config_hash = "cfg-hash-stale";

        let valid_payload = canonical_config_signature_payload(
            method,
            path,
            body,
            &owner_fingerprint,
            fresh_timestamp,
            &prev_config_hash,
        );
        let valid_signature = hex_signature(&owner_secret, &valid_payload);

        let wrong_owner_payload = canonical_config_signature_payload(
            method,
            path,
            body,
            wrong_owner_fingerprint,
            fresh_timestamp,
            &prev_config_hash,
        );
        let wrong_owner_signature = hex_signature(&owner_secret, &wrong_owner_payload);

        let stale_payload = canonical_config_signature_payload(
            method,
            path,
            body,
            &owner_fingerprint,
            stale_timestamp,
            &prev_config_hash,
        );
        let stale_signature = hex_signature(&owner_secret, &stale_payload);

        let stale_prev_payload = canonical_config_signature_payload(
            method,
            path,
            body,
            &owner_fingerprint,
            fresh_timestamp,
            stale_prev_config_hash,
        );
        let stale_prev_signature = hex_signature(&owner_secret, &stale_prev_payload);

        let mut cases: Vec<(&str, Vec<(&str, String)>, u16)> = Vec::new();
        cases.push((
            "missing signature",
            vec![
                (
                    CONFIG_OWNER_FINGERPRINT_HEADER,
                    owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, fresh_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash.to_string()),
            ],
            401,
        ));
        cases.push((
            "invalid signature",
            vec![
                (
                    CONFIG_OWNER_FINGERPRINT_HEADER,
                    owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, fresh_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash.to_string()),
                (CONFIG_SIGNATURE_HEADER, "invalid-signature".to_string()),
            ],
            401,
        ));
        cases.push((
            "wrong owner key",
            vec![
                (
                    "X-Mesh-Owner-Fingerprint",
                    wrong_owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, fresh_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash.to_string()),
                (CONFIG_SIGNATURE_HEADER, wrong_owner_signature),
            ],
            403,
        ));
        cases.push((
            "stale timestamp",
            vec![
                (
                    CONFIG_OWNER_FINGERPRINT_HEADER,
                    owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, stale_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash.to_string()),
                (CONFIG_SIGNATURE_HEADER, stale_signature),
            ],
            401,
        ));
        cases.push((
            "stale prev_config_hash",
            vec![
                (
                    CONFIG_OWNER_FINGERPRINT_HEADER,
                    owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, fresh_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, stale_prev_config_hash.to_string()),
                (CONFIG_SIGNATURE_HEADER, stale_prev_signature),
            ],
            409,
        ));
        cases.push((
            "valid signed apply",
            vec![
                (
                    CONFIG_OWNER_FINGERPRINT_HEADER,
                    owner_fingerprint.to_string(),
                ),
                (CONFIG_TIMESTAMP_HEADER, fresh_timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash.to_string()),
                (CONFIG_SIGNATURE_HEADER, valid_signature),
            ],
            200,
        ));

        for (name, headers, expected_status) in cases {
            let response = send_management_request_once(
                build_test_mesh_api().await,
                http_post_with_headers(path, body, &headers),
            )
            .await;

            assert_eq!(
                status_code(&response),
                expected_status,
                "case {name} failed; response was: {response}"
            );
        }
    }

    #[tokio::test]
    #[serial]
    async fn config_broadcast_requires_loopback_origin() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";

        let temp = make_temp_dir();
        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        configure_test_owner_key(&temp, &owner_secret);

        let non_loopback_response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config/broadcast",
                body,
                &[("X-Forwarded-For", "203.0.113.8".to_string())],
            ),
        )
        .await;

        assert_eq!(
            status_code(&non_loopback_response),
            403,
            "non-loopback origin must be rejected; response was: {non_loopback_response}"
        );

        let mixed_forward_chain_response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config/broadcast",
                body,
                &[("X-Forwarded-For", "203.0.113.8, 127.0.0.1".to_string())],
            ),
        )
        .await;

        assert_eq!(
            status_code(&mixed_forward_chain_response),
            403,
            "forward chain with non-loopback hop must be rejected; response was: {mixed_forward_chain_response}"
        );

        let loopback_response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config/broadcast",
                body,
                &[("X-Forwarded-For", "127.0.0.1".to_string())],
            ),
        )
        .await;

        assert_eq!(
            status_code(&loopback_response),
            200,
            "loopback origin should remain allowed; response was: {loopback_response}"
        );

        let full_loopback_chain_response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config/broadcast",
                body,
                &[("X-Forwarded-For", "127.0.0.1, ::1, localhost".to_string())],
            ),
        )
        .await;

        assert_eq!(
            status_code(&full_loopback_chain_response),
            200,
            "all-loopback forwarded chain should remain allowed; response was: {full_loopback_chain_response}"
        );
    }

    #[tokio::test]
    #[serial]
    async fn stale_prev_config_hash_returns_conflict() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";
        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        let owner_fingerprint = hex::encode(Sha256::digest(owner_secret.public().as_bytes()));

        let temp = make_temp_dir();
        configure_test_owner_key(&temp, &owner_secret);
        std::fs::write(temp.join(".mesh-llm/mesh.toml"), "version = 1\n").unwrap();

        let timestamp = now_unix_timestamp();
        let stale_prev_config_hash = "cfg-hash-stale";
        let payload = canonical_config_signature_payload(
            "POST",
            "/api/config",
            body,
            &owner_fingerprint,
            timestamp,
            stale_prev_config_hash,
        );
        let signature = hex_signature(&owner_secret, &payload);

        let response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config",
                body,
                &[
                    (CONFIG_OWNER_FINGERPRINT_HEADER, owner_fingerprint),
                    (CONFIG_TIMESTAMP_HEADER, timestamp.to_string()),
                    (CONFIG_PREV_HASH_HEADER, stale_prev_config_hash.to_string()),
                    (CONFIG_SIGNATURE_HEADER, signature),
                ],
            ),
        )
        .await;

        assert_eq!(status_code(&response), 409, "response was: {response}");
    }

    #[tokio::test]
    #[serial]
    async fn config_accepted_when_receiver_has_no_config_file() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";
        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        let owner_fingerprint = hex::encode(Sha256::digest(owner_secret.public().as_bytes()));

        let temp = make_temp_dir();
        configure_test_owner_key(&temp, &owner_secret);

        // Verify no config file exists (configure_test_owner_key creates .mesh-llm/
        // but not mesh.toml).
        assert!(
            !mesh_config_path().exists(),
            "precondition: config file must not exist"
        );

        // Use a prev_config_hash that does NOT match the default empty config hash —
        // simulates a broadcaster that already has a saved config.
        let mismatched_prev_hash = "abc123deadbeef";
        let timestamp = now_unix_timestamp();
        let payload = canonical_config_signature_payload(
            "POST",
            "/api/config",
            body,
            &owner_fingerprint,
            timestamp,
            mismatched_prev_hash,
        );
        let signature = hex_signature(&owner_secret, &payload);

        let response = send_management_request_once(
            build_test_mesh_api().await,
            http_post_with_headers(
                "/api/config",
                body,
                &[
                    (CONFIG_OWNER_FINGERPRINT_HEADER, owner_fingerprint),
                    (CONFIG_TIMESTAMP_HEADER, timestamp.to_string()),
                    (CONFIG_PREV_HASH_HEADER, mismatched_prev_hash.to_string()),
                    (CONFIG_SIGNATURE_HEADER, signature),
                ],
            ),
        )
        .await;

        assert_eq!(
            status_code(&response),
            200,
            "config push to node with no config file must succeed regardless of prev_config_hash; response was: {response}"
        );
    }

    #[tokio::test]
    #[serial]
    async fn replayed_signed_checkpoint_is_rejected_after_restart() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";
        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        let owner_fingerprint = hex::encode(Sha256::digest(owner_secret.public().as_bytes()));

        let temp = make_temp_dir();
        configure_test_owner_key(&temp, &owner_secret);

        let timestamp = now_unix_timestamp();
        let prev_config_hash = current_mesh_config_hash(&mesh_config_path()).unwrap();
        let payload = canonical_config_signature_payload(
            "POST",
            "/api/config",
            body,
            &owner_fingerprint,
            timestamp,
            &prev_config_hash,
        );
        let signature = hex_signature(&owner_secret, &payload);

        let signed_request = http_post_with_headers(
            "/api/config",
            body,
            &[
                (CONFIG_OWNER_FINGERPRINT_HEADER, owner_fingerprint.clone()),
                (CONFIG_TIMESTAMP_HEADER, timestamp.to_string()),
                (CONFIG_PREV_HASH_HEADER, prev_config_hash),
                (CONFIG_SIGNATURE_HEADER, signature),
            ],
        );

        let first =
            send_management_request_once(build_test_mesh_api().await, signed_request.clone()).await;
        assert_eq!(status_code(&first), 200, "first apply failed: {first}");
        assert!(
            replay_signature_seen_before(
                &parse_header_value(&signed_request, CONFIG_SIGNATURE_HEADER).unwrap()
            )
            .unwrap(),
            "signature must be persisted after first apply"
        );

        let replay =
            send_management_request_once(build_test_mesh_api().await, signed_request).await;
        assert_eq!(status_code(&replay), 409, "replay must conflict: {replay}");
    }

    #[tokio::test]
    #[serial]
    async fn loopback_broadcast_signs_and_relays_to_owned_peers() {
        let owner_secret = SecretKey::from_bytes(&[0x42; 32]);
        let owner_fingerprint = hex::encode(Sha256::digest(owner_secret.public().as_bytes()));
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";

        let temp = make_temp_dir();
        configure_test_owner_key(&temp, &owner_secret);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay_port = listener.local_addr().unwrap().port();
        let (captured_tx, captured_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = proxy::read_http_request(&mut stream).await.unwrap();
            let _ = captured_tx.send(request.raw.clone());
            let response = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(response).await.unwrap();
        });

        let prev_config_hash = current_mesh_config_hash(&mesh_config_path()).unwrap();
        let timestamp = now_unix_timestamp();
        let canonical_payload = canonical_config_signature_payload(
            "POST",
            "/api/config",
            body,
            &owner_fingerprint,
            timestamp,
            &prev_config_hash,
        );
        let signature = sign_config_payload(&canonical_payload).unwrap();

        let peers = vec![
            peer_info_with_owner(0x11, "127.0.0.1", Some(&owner_fingerprint), true),
            peer_info_with_owner(0x22, "198.51.100.11", Some("owner-fp-other"), true),
            peer_info_with_owner(0x33, "198.51.100.12", Some(&owner_fingerprint), false),
        ];
        let relay_peers: Vec<mesh::PeerInfo> = peers
            .into_iter()
            .filter(|peer| {
                peer.owner_fingerprint_verified
                    && peer.owner_fingerprint.as_deref() == Some(owner_fingerprint.as_str())
            })
            .collect();
        assert_eq!(
            relay_peers.len(),
            1,
            "only same-owner verified peers must be relayed to"
        );

        let headers = ConfigRelayHeaders {
            owner_fingerprint: owner_fingerprint.clone(),
            timestamp,
            prev_config_hash: prev_config_hash.clone(),
            signature: signature.clone(),
        };
        let relay_results = relay_config_to_peers(&relay_peers, relay_port, body, &headers).await;
        assert_eq!(relay_results.len(), 1, "exactly one relay expected");
        assert!(relay_results[0].ok, "owned peer relay should succeed");

        let relayed_raw = captured_rx.await.unwrap();
        let relayed_text = String::from_utf8_lossy(&relayed_raw);
        assert!(relayed_text.starts_with("POST /api/config HTTP/1.1"));
        assert_eq!(
            parse_header_value(&relayed_raw, CONFIG_OWNER_FINGERPRINT_HEADER),
            Some(owner_fingerprint)
        );
        assert_eq!(
            parse_header_value(&relayed_raw, CONFIG_TIMESTAMP_HEADER),
            Some(timestamp.to_string())
        );
        assert_eq!(
            parse_header_value(&relayed_raw, CONFIG_PREV_HASH_HEADER),
            Some(prev_config_hash)
        );
        assert_eq!(
            parse_header_value(&relayed_raw, CONFIG_SIGNATURE_HEADER),
            Some(signature)
        );
    }

    #[tokio::test]
    #[serial]
    async fn unowned_broadcast_does_not_mutate_local_config_or_runtime_files() {
        let body = "version = 1\n\n[[nodes]]\nnode_id = \"node-a\"\n";

        let temp = make_temp_dir();
        std::fs::create_dir_all(temp.join(".mesh-llm")).unwrap();
        std::env::set_var("HOME", &temp);

        let mesh_path = mesh_config_path();
        let node_path = node_config_path();
        let initial_mesh = "version = 1\n";
        let initial_node = "node_id = \"sentinel\"\n";
        std::fs::write(&mesh_path, initial_mesh).unwrap();
        std::fs::write(&node_path, initial_node).unwrap();

        let state = build_test_mesh_api().await;
        {
            let node = state.inner.lock().await.node.clone();
            node.set_owner_key_material(None).await;
        }

        let response = send_management_request_once(
            state,
            http_post_with_headers(
                "/api/config/broadcast",
                body,
                &[("X-Forwarded-For", "127.0.0.1".to_string())],
            ),
        )
        .await;

        assert_eq!(
            status_code(&response),
            403,
            "unowned node must reject broadcast before local mutation; response was: {response}"
        );
        assert_eq!(std::fs::read_to_string(&mesh_path).unwrap(), initial_mesh);
        assert_eq!(std::fs::read_to_string(&node_path).unwrap(), initial_node);
    }

    #[test]
    fn test_peer_payload_serializes_model_scans_when_present() {
        let payload = PeerPayload {
            id: "peer1".into(),
            role: "Worker".into(),
            models: vec!["Qwen3-8B-Q4_K_M".into()],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 24.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            rtt_ms: Some(12),
            hostname: Some("node-a".into()),
            is_soc: Some(false),
            owner_id: None,
            owner_fingerprint: None,
            owner_fingerprint_verified: false,
            owner_fingerprint_transitive: false,
            gpus: vec![],
            model_sizes: Some(vec![("Qwen3-8B-Q4_K_M".into(), 5_000_000_000)]),
            model_scans: Some(vec![mesh::ScannedModel {
                name: "Qwen3-8B-Q4_K_M".into(),
                model_key: "abc123".into(),
                size_bytes: 5_000_000_000,
                metadata: mesh::CompactModelMetadata {
                    architecture: Some("llama".into()),
                    total_layers: Some(32),
                    total_offloadable_layers: Some(33),
                    file_size: 5_000_000_000,
                    dense_split_capable: true,
                    ..Default::default()
                },
            }]),
        };

        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("model_scans").is_some());
        assert_eq!(json["model_scans"][0]["model_key"], "abc123");
        assert_eq!(
            json["model_scans"][0]["metadata"]["total_offloadable_layers"],
            33
        );
    }

    #[test]
    fn test_peer_payload_omits_model_scans_when_missing() {
        let payload = PeerPayload {
            id: "peer1".into(),
            role: "Worker".into(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 0.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            rtt_ms: None,
            hostname: None,
            is_soc: None,
            owner_id: None,
            owner_fingerprint: None,
            owner_fingerprint_verified: false,
            owner_fingerprint_transitive: false,
            gpus: vec![],
            model_sizes: None,
            model_scans: None,
        };

        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("model_scans").is_none());
    }

    #[test]
    fn test_config_round_trip_post_then_get_toml() {
        let temp = make_temp_dir();
        let path = temp.join("mesh.toml");

        let input = r#"
version = 1

[[nodes]]
node_id = "abc123"

[[nodes.models]]
name = "llama-3.3-70b"
model_key = "mk-llama-70b"
split = { start = 0, end = 33, total = 33 }
ctx_size = 4096
"#;

        let parsed =
            parse_and_validate_mesh_config(input, &split_catalog(&[("mk-llama-70b", 33)])).unwrap();
        parsed.save(&path).unwrap();

        let got_toml = read_mesh_config_toml(&path).unwrap();
        let got: AuthoredMeshConfig = toml::from_str(&got_toml).unwrap();

        assert_eq!(got.version, AUTHORED_CONFIG_VERSION);
        assert_eq!(got.nodes.len(), 1);
        assert_eq!(got.nodes[0].node_id, "abc123");
        assert_eq!(got.nodes[0].models.len(), 1);
        assert_eq!(got.nodes[0].models[0].name, "llama-3.3-70b");
        assert_eq!(
            got.nodes[0].models[0].model_key.as_deref(),
            Some("mk-llama-70b")
        );
        assert_eq!(got.nodes[0].models[0].split.as_ref().unwrap().total, 33);
        assert_eq!(got.nodes[0].models[0].ctx_size, Some(4096));
    }

    #[test]
    fn test_invalid_toml_rejected() {
        let err = parse_and_validate_mesh_config(
            "invalid = [broken toml",
            &SplitValidationCatalog::default(),
        )
        .unwrap_err();
        assert!(err.iter().any(|e| e.code == "invalid_toml"));
    }

    #[test]
    fn test_empty_node_id_rejected() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "   ".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![],
            }],
        };
        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(err.iter().any(|e| e.code == "empty_node_id"));
    }

    #[test]
    fn test_empty_model_name_rejected() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "   ".into(),
                    model_key: None,
                    split: None,
                    path: None,
                    ctx_size: None,
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };
        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(err.iter().any(|e| e.code == "empty_model_name"));
    }

    #[test]
    fn test_node_lookup_found_and_missing() {
        let cfg = MeshConfig {
            nodes: vec![NodeConfig {
                node_id: "node-1".into(),
                hostname: Some("alpha.local".into()),
                models: vec![ModelAssignment {
                    name: "Qwen3-8B-Q4_K_M".into(),
                    path: None,
                    ctx_size: Some(4096),
                    moe_experts: None,
                }],
            }],
        };

        let found = cfg.for_node("node-1");
        assert!(found.is_some());
        assert!(cfg.for_node("missing").is_none());
    }

    #[test]
    fn test_sync_node_runtime_config_writes_selected_node() {
        let temp = make_temp_dir();
        let node_path = temp.join("node.toml");
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![
                AuthoredNodeConfig {
                    node_id: "node-a".into(),
                    hostname: Some("alpha.local".into()),
                    placement_mode: PlacementMode::Pooled,
                    models: vec![],
                },
                AuthoredNodeConfig {
                    node_id: "node-b".into(),
                    hostname: Some("beta.local".into()),
                    placement_mode: PlacementMode::Pooled,
                    models: vec![AuthoredModelAssignment {
                        name: "Qwen3-8B-Q4_K_M".into(),
                        model_key: Some("mk-qwen3-8b".into()),
                        split: Some(ModelSplit {
                            start: 0,
                            end: 32,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: Some(8192),
                        moe_experts: None,
                        gpu_index: None,
                    }],
                },
            ],
        };

        sync_node_runtime_config_at(&node_path, &cfg, "node-b").unwrap();
        let saved = NodeConfig::load(&node_path).unwrap().unwrap();

        assert_eq!(saved.node_id, "node-b");
        assert_eq!(saved.hostname.as_deref(), Some("beta.local"));
        assert_eq!(saved.models.len(), 1);
        assert_eq!(saved.models[0].name, "Qwen3-8B-Q4_K_M");
    }

    #[test]
    fn test_sync_node_runtime_config_removes_stale_file_when_node_missing() {
        let temp = make_temp_dir();
        let node_path = temp.join("node.toml");
        let stale = NodeConfig {
            node_id: "node-a".into(),
            hostname: None,
            models: vec![],
        };
        stale.save(&node_path).unwrap();
        assert!(node_path.exists());

        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-b".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![],
            }],
        };

        sync_node_runtime_config_at(&node_path, &cfg, "node-a").unwrap();
        assert!(!node_path.exists());
    }

    #[test]
    fn test_read_mesh_config_missing_file_returns_empty_toml() {
        let temp = make_temp_dir();
        let missing = temp.join("missing-mesh.toml");
        let toml = read_mesh_config_toml(&missing).unwrap();
        let parsed: AuthoredMeshConfig = toml::from_str(&toml).unwrap();
        assert_eq!(parsed.version, AUTHORED_CONFIG_VERSION);
        assert!(parsed.nodes.is_empty());
    }

    #[test]
    fn test_split_validation_rejects_overlap_gap_and_total_mismatch() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![
                    AuthoredModelAssignment {
                        name: "ModelA".into(),
                        model_key: Some("mk-model-a".into()),
                        split: Some(ModelSplit {
                            start: 0,
                            end: 12,
                            total: 33,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: None,
                    },
                    AuthoredModelAssignment {
                        name: "ModelA".into(),
                        model_key: Some("mk-model-a".into()),
                        split: Some(ModelSplit {
                            start: 10,
                            end: 20,
                            total: 34,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: None,
                    },
                ],
            }],
        };

        let err = validate_mesh_config(&cfg, &split_catalog(&[("mk-model-a", 33)])).unwrap_err();
        assert!(err.iter().any(|e| e.code == "mismatched_split_totals"));
        assert!(err.iter().any(|e| e.code == "overlapping_split_ranges"));
        assert!(err
            .iter()
            .any(|e| e.code == "incomplete_split_coverage" || e.code == "split_gap"));
    }

    #[test]
    fn test_split_validation_rejects_unknown_model_key_and_out_of_range() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "ModelA".into(),
                    model_key: Some("mk-missing".into()),
                    split: Some(ModelSplit {
                        start: 0,
                        end: 40,
                        total: 33,
                    }),
                    path: None,
                    ctx_size: None,
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };

        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(err.iter().any(|e| e.code == "split_out_of_range"));
    }

    #[test]
    fn test_split_validation_rejects_unsupported_version() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION + 1,
            nodes: vec![],
        };

        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(err.iter().any(|e| e.code == "unsupported_version"));
    }

    #[test]
    fn placement_validation_requires_gpu_index_in_separate_mode() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Separate,
                models: vec![AuthoredModelAssignment {
                    name: "ModelA".into(),
                    model_key: None,
                    split: None,
                    path: None,
                    ctx_size: None,
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };

        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(
            err.iter()
                .any(|e| e.code == "missing_gpu_index_in_separate_mode"),
            "expected missing_gpu_index_in_separate_mode error, got: {:?}",
            err
        );
    }

    #[test]
    fn placement_validation_forbids_gpu_index_in_pooled_mode() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "ModelA".into(),
                    model_key: None,
                    split: None,
                    path: None,
                    ctx_size: None,
                    moe_experts: None,
                    gpu_index: Some(0),
                }],
            }],
        };

        let err = validate_mesh_config(&cfg, &SplitValidationCatalog::default()).unwrap_err();
        assert!(
            err.iter().any(|e| e.code == "gpu_index_in_pooled_mode"),
            "expected gpu_index_in_pooled_mode error, got: {:?}",
            err
        );
    }

    #[test]
    fn placement_validation_and_split_validation_report_separately() {
        let cfg = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Separate,
                models: vec![
                    AuthoredModelAssignment {
                        name: "ModelA".into(),
                        model_key: Some("mk-model-a".into()),
                        split: Some(ModelSplit {
                            start: 0,
                            end: 10,
                            total: 20,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: Some(0),
                    },
                    AuthoredModelAssignment {
                        name: "ModelA".into(),
                        model_key: Some("mk-model-a".into()),
                        split: Some(ModelSplit {
                            start: 5,
                            end: 19,
                            total: 20,
                        }),
                        path: None,
                        ctx_size: None,
                        moe_experts: None,
                        gpu_index: Some(5),
                    },
                ],
            }],
        };

        let mut catalog = split_catalog(&[("mk-model-a", 20)]);
        catalog.gpu_counts_by_node_id.insert("node-a".into(), 2);

        let err = validate_mesh_config(&cfg, &catalog).unwrap_err();
        assert!(
            err.iter()
                .any(|e| e.code == "overlapping_split_ranges" || e.code == "split_gap"),
            "expected a split error, got: {:?}",
            err
        );
        assert!(
            err.iter().any(|e| e.code == "gpu_index_out_of_range"),
            "expected gpu_index_out_of_range error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_invalid_config_does_not_overwrite_existing_file() {
        let temp = make_temp_dir();
        let path = temp.join("mesh.toml");

        let valid = AuthoredMeshConfig {
            version: AUTHORED_CONFIG_VERSION,
            nodes: vec![AuthoredNodeConfig {
                node_id: "node-a".into(),
                hostname: None,
                placement_mode: PlacementMode::Pooled,
                models: vec![AuthoredModelAssignment {
                    name: "ModelA".into(),
                    model_key: Some("mk-model-a".into()),
                    split: Some(ModelSplit {
                        start: 0,
                        end: 32,
                        total: 33,
                    }),
                    path: None,
                    ctx_size: None,
                    moe_experts: None,
                    gpu_index: None,
                }],
            }],
        };
        valid.save(&path).unwrap();
        let before = std::fs::read_to_string(&path).unwrap();

        let invalid = r#"
version = 2

[[nodes]]
node_id = "node-a"

[[nodes.models]]
name = "ModelA"
model_key = "mk-model-a"
split = { start = 0, end = 31, total = 33 }
"#;
        let result = parse_and_validate_mesh_config(invalid, &split_catalog(&[("mk-model-a", 33)]));
        assert!(result.is_err());

        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn test_console_index_routes_include_config_dashboard_chat() {
        assert!(is_console_index_path("/config"));
        assert!(is_console_index_path("/config/"));
        assert!(is_console_index_path("/dashboard"));
        assert!(is_console_index_path("/chat"));
        assert!(is_console_index_path("/chat/abc"));
        assert!(!is_console_index_path("/api/does-not-exist"));
    }

    #[tokio::test]
    async fn test_respond_error_uses_json_404_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            let mut buf = vec![0u8; 512];
            let n = stream.read(&mut buf).await.unwrap();
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        let (mut server_stream, _) = listener.accept().await.unwrap();
        respond_error(&mut server_stream, 404, "Not found")
            .await
            .unwrap();

        let resp = client.await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 404 Not Found"));
        assert!(resp.contains("Content-Type: application/json"));
        assert!(resp.contains("{\"error\":\"Not found\"}"));
    }

    #[tokio::test]
    async fn test_console_index_response_is_html_200() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            let mut buf = vec![0u8; 2048];
            let n = stream.read(&mut buf).await.unwrap();
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        let (mut server_stream, _) = listener.accept().await.unwrap();
        let served = respond_console_index(&mut server_stream).await.unwrap();
        assert!(served, "embedded UI index.html should be available");

        let resp = client.await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"));
        assert!(resp.contains("Content-Type: text/html; charset=utf-8"));
    }

    #[tokio::test]
    async fn test_respond_validation_errors_returns_machine_readable_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            let mut buf = vec![0u8; 2048];
            let n = stream.read(&mut buf).await.unwrap();
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        let (mut server_stream, _) = listener.accept().await.unwrap();
        respond_validation_errors(
            &mut server_stream,
            &[ConfigValidationError {
                code: "split_gap".into(),
                path: "nodes[0].models[0].split".into(),
                message: "split gap before segment".into(),
            }],
        )
        .await
        .unwrap();

        let resp = client.await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(resp.contains("\"error\":\"config_validation_failed\""));
        assert!(resp.contains("\"code\":\"split_gap\""));
    }
}
