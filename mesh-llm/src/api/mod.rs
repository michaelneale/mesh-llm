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

mod assets;
mod http;
mod routes;
mod state;
mod status;

pub use self::state::{MeshApi, RuntimeControlRequest, RuntimeModelPayload, RuntimeProcessPayload};
pub(crate) use self::status::classify_runtime_error;

use self::assets::{respond_console_asset, respond_console_index};
use self::http::{http_body_text, respond_error};
use self::routes::dispatch_request;
use self::state::ApiInner;
use self::status::{
    build_gpus, build_runtime_processes_payload, build_runtime_status_payload, MeshModelPayload,
    PeerPayload, RuntimeProcessesPayload, RuntimeStatusPayload, StatusPayload,
};
use crate::{affinity, election, mesh, nostr, plugin, proxy};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

const MESH_LLM_VERSION: &str = crate::VERSION;

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

        let catalog = node.mesh_catalog().await;
        let served = node.models_being_served().await;
        let active_demand = node.active_demand().await;
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
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mesh_models: Vec<MeshModelPayload> = catalog
            .iter()
            .map(|name| {
                let is_warm = served.contains(name);
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
                let size_gb = if *name == model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    crate::models::catalog::parse_size_gb(
                        crate::models::catalog::MODEL_CATALOG
                            .iter()
                            .find(|m| {
                                m.file.strip_suffix(".gguf").unwrap_or(&m.file) == name.as_str()
                                    || m.name == name.as_str()
                            })
                            .map(|m| m.size.as_str())
                            .unwrap_or("0"),
                    )
                };
                let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                    Some(d) => (
                        Some(d.request_count),
                        Some(now_ts.saturating_sub(d.last_active)),
                    ),
                    None => (None, None),
                };
                let capabilities = crate::models::installed_model_capabilities(name);
                let vision = capabilities.supports_vision_runtime();
                let vision_status = if vision || capabilities.vision_label().is_some() {
                    Some(capabilities.vision_status())
                } else {
                    None
                };
                let reasoning = matches!(
                    capabilities.reasoning,
                    crate::models::capabilities::CapabilityLevel::Supported
                );
                let reasoning_status = if reasoning || capabilities.reasoning_label().is_some() {
                    Some(capabilities.reasoning_status())
                } else {
                    None
                };
                MeshModelPayload {
                    name: name.clone(),
                    display_name,
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    size_gb,
                    vision,
                    vision_status,
                    reasoning,
                    reasoning_status,
                    request_count,
                    last_active_secs_ago,
                }
            })
            .collect();

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
            mesh_models,
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

        _ => {
            if !dispatch_request(
                &mut stream,
                state,
                method,
                path,
                path_only,
                body,
                req.as_ref(),
            )
            .await?
            {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::status::decode_runtime_model_path;
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
