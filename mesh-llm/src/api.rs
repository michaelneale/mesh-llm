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

use crate::{affinity, election, mesh, nostr, plugin, proxy};
use include_dir::{include_dir, Dir};
use serde::Serialize;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

static CONSOLE_DIST: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");
const MESH_LLM_VERSION: &str = crate::VERSION;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    bandwidth_gbps: Option<f64>,
}

fn build_gpus(
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
    gpus: Vec<GpuEntry>,
    routing_affinity: affinity::AffinityStatsSnapshot,
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
    gpus: Vec<GpuEntry>,
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
                gpus: build_gpus(
                    p.gpu_name.as_deref(),
                    p.gpu_vram.as_deref(),
                    p.gpu_bandwidth_gbps.as_deref(),
                ),
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
                )
            },
            routing_affinity,
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
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_request(stream, &state).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

async fn handle_request(mut stream: TcpStream, state: &MeshApi) -> anyhow::Result<()> {
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
        ("GET", "/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", "/dashboard") | ("GET", "/chat") | ("GET", "/dashboard/") | ("GET", "/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", p) if p.starts_with("/chat/") => {
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

        _ => {
            respond_error(&mut stream, 404, "Not found").await?;
        }
    }
    Ok(())
}

fn http_body_text(raw: &[u8]) -> &str {
    let body_start = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(raw.len());
    std::str::from_utf8(&raw[body_start..]).unwrap_or("")
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
    use mesh_llm_plugin::MeshVisibility;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    #[test]
    fn test_build_gpus_both_none() {
        let result = build_gpus(None, None, None);
        assert!(result.is_empty(), "expected empty vec when no gpu_name");
    }

    #[test]
    fn test_build_gpus_single_no_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 0);
    }

    #[test]
    fn test_build_gpus_single_with_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), Some("34359738368"), None);
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
        let result = build_gpus(None, Some("34359738368"), None);
        assert!(
            result.is_empty(),
            "no gpu_name means no entries even if vram present"
        );
    }

    #[test]
    fn test_build_gpus_vram_whitespace_trimmed() {
        let result = build_gpus(Some("NVIDIA RTX 4090"), Some(" 25769803776 "), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vram_bytes, 25_769_803_776);
    }

    #[test]
    fn test_build_gpus_with_bandwidth() {
        let result = build_gpus(
            Some("NVIDIA A100, NVIDIA A6000"),
            Some("85899345920,51539607552"),
            Some("1948.70,780.10"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].bandwidth_gbps, Some(1948.70));
        assert_eq!(result[1].bandwidth_gbps, Some(780.10));
    }

    #[test]
    fn test_build_gpus_unparsable_vram_preserves_index() {
        let result = build_gpus(Some("GPU0, GPU1, GPU2"), Some("100,foo,300"), None);
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
            let (stream, _) = listener.accept().await.unwrap();
            handle_request(stream, &state).await
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
}
