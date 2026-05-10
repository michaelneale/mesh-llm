//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state plus local-only routing metrics (JSON)
//!   GET  /api/models    — mesh model inventory plus local-only routing metrics (JSON)
//!   GET  /api/search    — catalog or Hugging Face model search with the same JSON payload as `mesh-llm models search --json`
//!   GET  /api/model-interests — local explicit-interest readback (JSON)
//!   POST /api/model-interests — register local explicit interest for a canonical model ref
//!   DELETE /api/model-interests/{model_ref} — clear local explicit interest
//!   GET  /api/model-targets — ranked model targets from explicit interest and demand
//!   GET  /api/runtime   — local model state (JSON)
//!   GET  /api/runtime/llama — local llama.cpp runtime metrics + slots snapshots (JSON)
//!   GET  /api/runtime/events — SSE stream of llama.cpp runtime metrics + slots snapshots
//!   GET  /api/runtime/endpoints — registered plugin endpoint state (JSON)
//!   GET  /api/runtime/processes — local inference process state (JSON)
//!   GET  /api/runtime/stages — backend-neutral staged-serving state (JSON)
//!   POST /api/runtime/models — load a local model
//!   DELETE /api/runtime/models/{model} — unload a local model
//!   DELETE /api/runtime/instances/{instance_id} — unload one local runtime instance
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to chat completions API
//!   POST /api/responses — proxy to responses API
//!   POST /api/objects   — upload a request-scoped media object
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is mostly read-only — shows status, topology, and models.
//! Local model load/unload is exposed for operator control.
//!
//! Broad runtime reads should stay behind `runtime_data` helpers so the API
//! layer keeps using stable collector-backed views instead of fresh fan-in.
//!
//! `routing_metrics`, `routing_metrics.local_node`, `routing_metrics.pressure`,
//! and `/api/models` per-model `routing_metrics.targets` are measured on the
//! current node only; not mesh-wide aggregates.

mod assets;
mod http;
mod model_targets;
mod routes;
mod server;
mod state;
pub(crate) mod status;

pub(crate) use self::server::start_with_listener;
#[cfg(test)]
pub(crate) use self::server::{handle_request, is_ui_only_route};
pub use self::state::{
    LocalModelInterest, MeshApi, PublicationState, RuntimeControlRequest, RuntimeLoadResponse,
    RuntimeModelPayload, RuntimeProcessPayload, RuntimeUnloadResponse,
};
pub(crate) use self::status::classify_runtime_error;

use self::state::ApiInner;
use self::status::{
    build_runtime_processes_payload, build_runtime_stage_payloads, build_runtime_status_payload,
    runtime_stage_state_label, runtime_stage_wire_dtype_label, MeshModelPayload,
    RuntimeLlamaPayload, RuntimeProcessesPayload, RuntimeStatusPayload, StatusPayload,
};
use crate::mesh;
use crate::network::{affinity, nostr};
use crate::plugin;
use crate::runtime_data;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(test)]
use self::http::http_body_text;
#[cfg(test)]
use self::status::{build_gpus, LocalInstance, NodeState, WakeableNode, WakeableNodeState};
#[cfg(test)]
use crate::inference::election;
#[cfg(test)]
use crate::network::proxy;
#[cfg(test)]
use crate::runtime::wakeable::{WakeableInventoryEntry, WakeableState};

const MESH_LLM_VERSION: &str = crate::VERSION;

#[cfg(test)]
#[derive(Debug, Default, PartialEq)]
pub(crate) struct HttpRouteStats {
    node_count: usize,
    active_nodes: Vec<String>,
    mesh_vram_gb: f64,
}

#[cfg(test)]
pub(crate) fn http_route_stats(
    model_name: &str,
    peers: &[mesh::PeerInfo],
    my_hosted_models: &[String],
    my_hostname: Option<&str>,
    my_vram_gb: f64,
) -> HttpRouteStats {
    let mut active_nodes = Vec::new();
    let mut node_count = 0usize;
    let mut mesh_vram_gb = 0.0;

    if my_hosted_models.iter().any(|hosted| hosted == model_name) {
        node_count += 1;
        mesh_vram_gb += my_vram_gb;
        active_nodes.push(
            my_hostname
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or("This node")
                .to_string(),
        );
    }

    for peer in peers {
        if !peer.routes_http_model(model_name) {
            continue;
        }
        node_count += 1;
        mesh_vram_gb += peer.vram_bytes as f64 / 1e9;
        active_nodes.push(
            peer.hostname
                .clone()
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or_else(|| peer.id.fmt_short().to_string()),
        );
    }

    active_nodes.sort();
    active_nodes.dedup();

    HttpRouteStats {
        node_count,
        active_nodes,
        mesh_vram_gb,
    }
}

pub struct MeshApiConfig {
    pub(crate) node: mesh::Node,
    pub(crate) model_name: String,
    pub(crate) api_port: u16,
    pub(crate) model_size_bytes: u64,
    pub(crate) plugin_manager: plugin::PluginManager,
    pub(crate) affinity_router: affinity::AffinityRouter,
    pub(crate) runtime_data_collector: runtime_data::RuntimeDataCollector,
    pub(crate) runtime_data_producer: runtime_data::RuntimeDataProducer,
}

impl MeshApi {
    pub fn new(config: MeshApiConfig) -> Self {
        let MeshApiConfig {
            node,
            model_name,
            api_port,
            model_size_bytes,
            plugin_manager,
            affinity_router,
            runtime_data_collector,
            runtime_data_producer,
        } = config;

        runtime_data_producer.publish_runtime_status(|runtime_status| {
            if runtime_status.primary_model.as_deref() == Some(model_name.as_str()) {
                return false;
            }
            runtime_status.primary_model = Some(model_name.clone());
            true
        });
        let initial_runtime_data_views = runtime_data::collect_views(&runtime_data_collector);
        let _ = (
            initial_runtime_data_views
                .runtime_status
                .primary_model
                .as_ref(),
            initial_runtime_data_views
                .runtime_status
                .primary_backend
                .as_ref(),
            initial_runtime_data_views.runtime_status.is_host,
            initial_runtime_data_views.runtime_status.is_client,
            initial_runtime_data_views.runtime_status.llama_ready,
            initial_runtime_data_views.runtime_status.llama_port,
            initial_runtime_data_views
                .runtime_status
                .local_processes
                .len(),
            initial_runtime_data_views.local_instances.instances.len(),
            initial_runtime_data_views.plugin_data.entries.len(),
            initial_runtime_data_views.plugin_endpoints.entries.len(),
            runtime_data_producer.scope(),
            runtime_data_producer.has_plugin_data_key(),
            runtime_data_producer.has_plugin_endpoint_key(),
            runtime_data_producer.initial_process_count(),
        );
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                affinity_router,
                runtime_data_collector,
                runtime_data_producer,
                headless: false,
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
                mesh_discovery_mode: crate::network::discovery::MeshDiscoveryMode::Nostr,
                nostr_discovery: false,
                publication_state: state::PublicationState::Private,
                runtime_control: None,
                local_processes: Vec::new(),
                sse_clients: Vec::new(),
                model_interests: std::collections::HashMap::new(),
                wakeable_inventory: crate::runtime::wakeable::WakeableInventory::default(),
            })),
        }
    }

    pub async fn node(&self) -> mesh::Node {
        self.inner.lock().await.node.clone()
    }

    pub(super) async fn model_interests(&self) -> Vec<LocalModelInterest> {
        let mut interests = {
            let inner = self.inner.lock().await;
            inner
                .model_interests
                .values()
                .cloned()
                .collect::<Vec<LocalModelInterest>>()
        };
        interests.sort_by(|left, right| {
            right
                .updated_at_unix
                .cmp(&left.updated_at_unix)
                .then_with(|| left.model_ref.cmp(&right.model_ref))
        });
        interests
    }

    pub(super) async fn upsert_model_interest(
        &self,
        model_ref: String,
        submission_source: Option<String>,
    ) -> (LocalModelInterest, bool) {
        let now = current_unix_secs();
        let (interest, created, model_refs) = {
            let mut inner = self.inner.lock().await;
            let (interest, created) = match inner.model_interests.entry(model_ref.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get().clone();
                    let updated = LocalModelInterest {
                        model_ref,
                        submission_source: submission_source.or(existing.submission_source),
                        created_at_unix: existing.created_at_unix,
                        updated_at_unix: now,
                    };
                    entry.insert(updated.clone());
                    (updated, false)
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let created = LocalModelInterest {
                        model_ref,
                        submission_source,
                        created_at_unix: now,
                        updated_at_unix: now,
                    };
                    entry.insert(created.clone());
                    (created, true)
                }
            };
            let mut model_refs = inner.model_interests.keys().cloned().collect::<Vec<_>>();
            model_refs.sort();
            (interest, created, model_refs)
        };
        self.sync_node_model_interests(model_refs).await;
        (interest, created)
    }

    pub(super) async fn remove_model_interest(&self, model_ref: &str) -> bool {
        let (removed, model_refs) = {
            let mut inner = self.inner.lock().await;
            let removed = inner.model_interests.remove(model_ref).is_some();
            let mut model_refs = inner.model_interests.keys().cloned().collect::<Vec<_>>();
            model_refs.sort();
            (removed, model_refs)
        };
        if removed {
            self.sync_node_model_interests(model_refs).await;
        }
        removed
    }

    async fn sync_node_model_interests(&self, model_refs: Vec<String>) {
        let node = { self.inner.lock().await.node.clone() };
        node.set_explicit_model_interests(model_refs).await;
        self.push_status().await;
    }

    pub async fn set_primary_backend(&self, backend: String) {
        let mut inner = self.inner.lock().await;
        inner.primary_backend = Some(backend.clone());
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.primary_backend.as_deref() == Some(backend.as_str()) {
                    return false;
                }
                runtime_status.primary_backend = Some(backend.clone());
                true
            });
    }

    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    pub async fn set_client(&self, is_client: bool) {
        let mut inner = self.inner.lock().await;
        inner.is_client = is_client;
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.is_client == is_client {
                    return false;
                }
                runtime_status.is_client = is_client;
                true
            });
    }

    pub async fn set_mesh_name(&self, name: String) {
        self.inner.lock().await.mesh_name = Some(name);
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn set_mesh_discovery_mode(
        &self,
        mode: crate::network::discovery::MeshDiscoveryMode,
    ) {
        self.inner.lock().await.mesh_discovery_mode = mode;
    }

    pub async fn set_nostr_discovery(&self, v: bool) {
        self.inner.lock().await.nostr_discovery = v;
    }

    pub async fn set_publication_state(&self, state: state::PublicationState) {
        {
            let mut inner = self.inner.lock().await;
            inner.publication_state = state;
        }
        self.push_status().await;
    }

    #[cfg(test)]
    pub(crate) async fn publication_state(&self) -> state::PublicationState {
        self.inner.lock().await.publication_state
    }

    pub(crate) async fn runtime_data_producer(&self) -> runtime_data::RuntimeDataProducer {
        self.inner.lock().await.runtime_data_producer.clone()
    }

    pub async fn set_runtime_control(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    ) {
        self.inner.lock().await.runtime_control = Some(tx);
    }

    pub(crate) async fn status_snapshot_string(&self) -> String {
        let status = self.status().await;
        match serde_json::to_string_pretty(&status) {
            Ok(json) => json,
            Err(err) => {
                tracing::warn!("failed to serialize local status snapshot: {err}");
                format!(
                    "{{\n  \"error\": \"status snapshot unavailable\",\n  \"detail\": {:?}\n}}",
                    err.to_string()
                )
            }
        }
    }

    pub async fn upsert_local_process(&self, process: RuntimeProcessPayload) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| {
                runtime_process_payload_identity(p) != runtime_process_payload_identity(&process)
            });
            inner.local_processes.push(process.clone());
            inner
                .runtime_data_producer
                .publish_local_processes(|local_processes| {
                    runtime_data::upsert_runtime_process_snapshot(
                        local_processes,
                        runtime_data::RuntimeProcessSnapshot::from_payload(&process),
                    )
                });
        }
    }

    pub async fn remove_local_process(&self, target: &str) {
        {
            let mut inner = self.inner.lock().await;
            let has_instance_match = inner
                .local_processes
                .iter()
                .any(|process| process.instance_id.as_deref() == Some(target));
            inner.local_processes.retain(|process| {
                if has_instance_match {
                    process.instance_id.as_deref() != Some(target)
                } else {
                    process.name != target
                }
            });
            inner
                .runtime_data_producer
                .publish_local_processes(|local_processes| {
                    runtime_data::remove_runtime_process_snapshot(local_processes, target)
                });
        }
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
            inner
                .runtime_data_producer
                .publish_runtime_status(|runtime_status| {
                    let mut changed = false;
                    if runtime_status.is_host != is_host {
                        runtime_status.is_host = is_host;
                        changed = true;
                    }
                    if runtime_status.llama_ready != llama_ready {
                        runtime_status.llama_ready = llama_ready;
                        changed = true;
                    }
                    changed
                });
        }
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        let mut inner = self.inner.lock().await;
        inner.llama_port = port;
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.llama_port == port {
                    return false;
                }
                runtime_status.llama_port = port;
                true
            });
    }

    pub async fn set_headless(&self, headless: bool) {
        self.inner.lock().await.headless = headless;
    }

    pub(super) async fn is_headless(&self) -> bool {
        self.inner.lock().await.headless
    }

    async fn runtime_status(&self) -> RuntimeStatusPayload {
        let runtime_status = self
            .inner
            .lock()
            .await
            .runtime_data_collector
            .runtime_status_snapshot();
        build_runtime_status_payload(
            runtime_status.primary_model.as_deref().unwrap_or_default(),
            runtime_status.primary_backend,
            runtime_status.is_host,
            runtime_status.llama_ready,
            runtime_status.llama_port,
            runtime_data::runtime_process_payloads(&runtime_status.local_processes),
        )
    }

    async fn runtime_processes(&self) -> RuntimeProcessesPayload {
        let runtime_processes = self
            .inner
            .lock()
            .await
            .runtime_data_collector
            .runtime_processes_snapshot();
        build_runtime_processes_payload(runtime_data::runtime_process_payloads(&runtime_processes))
    }

    async fn runtime_stages(&self) -> serde_json::Value {
        let node = self.inner.lock().await.node.clone();
        node.refresh_stage_runtime_statuses(std::time::Duration::from_secs(2))
            .await;
        let topologies = node.stage_topologies().await;
        let statuses = node.stage_runtime_statuses().await;
        let stage_statuses = statuses
            .iter()
            .map(|status| {
                serde_json::json!({
                    "topology_id": status.topology_id.clone(),
                    "run_id": status.run_id.clone(),
                    "model_id": status.model_id.clone(),
                    "backend": status.backend.clone(),
                    "package_ref": status.package_ref.clone(),
                    "manifest_sha256": status.manifest_sha256.clone(),
                    "source_model_path": status.source_model_path.clone(),
                    "source_model_sha256": status.source_model_sha256.clone(),
                    "source_model_bytes": status.source_model_bytes,
                    "materialized_path": status.materialized_path.clone(),
                    "materialized_bytes": status
                        .materialized_path
                        .as_deref()
                        .and_then(|path| std::fs::metadata(path).ok())
                        .filter(|metadata| metadata.is_file())
                        .map(|metadata| metadata.len()),
                    "materialized_pinned": status.materialized_pinned,
                    "projector_path": status.projector_path.clone(),
                    "multimodal": status.projector_path.is_some(),
                    "stage_id": status.stage_id.clone(),
                    "stage_index": status.stage_index,
                    "node_id": status.node_id.map(|id| id.to_string()),
                    "layer_start": status.layer_start,
                    "layer_end": status.layer_end,
                    "state": runtime_stage_state_label(status.state),
                    "bind_addr": status.bind_addr.clone(),
                    "activation_width": status.activation_width,
                    "wire_dtype": runtime_stage_wire_dtype_label(status.wire_dtype),
                    "selected_device": status.selected_device.as_ref().map(|device| {
                        serde_json::json!({
                            "backend_device": device.backend_device,
                            "stable_id": device.stable_id,
                            "index": device.index,
                            "vram_bytes": device.vram_bytes,
                        })
                    }),
                    "ctx_size": status.ctx_size,
                    "lane_count": status.lane_count,
                    "error": status.error.clone(),
                    "shutdown_generation": status.shutdown_generation,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "stages": stage_statuses.clone(),
            "topologies": topologies.into_iter().map(|topology| {
                serde_json::json!({
                    "topology_id": topology.topology_id,
                    "run_id": topology.run_id,
                    "model_id": topology.model_id,
                    "package_ref": topology.package_ref,
                    "manifest_sha256": topology.manifest_sha256,
                    "stages": topology.stages.into_iter().map(|stage| {
                        serde_json::json!({
                            "stage_id": stage.stage_id,
                            "stage_index": stage.stage_index,
                            "node_id": stage.node_id.to_string(),
                            "layer_start": stage.layer_start,
                            "layer_end": stage.layer_end,
                            "endpoint": {
                                "bind_addr": stage.endpoint.bind_addr,
                            },
                        })
                    }).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "statuses": stage_statuses,
        })
    }

    async fn runtime_llama(&self) -> RuntimeLlamaPayload {
        let (runtime_llama, runtime_llama_by_instance) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.runtime_llama_snapshot(),
                inner
                    .runtime_data_collector
                    .runtime_llama_snapshots_by_instance(),
            )
        };
        status::build_runtime_llama_payload(runtime_llama, runtime_llama_by_instance)
    }

    async fn runtime_endpoints(&self) -> anyhow::Result<Vec<plugin::PluginEndpointSummary>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.endpoints().await
    }

    async fn plugins(&self) -> Vec<plugin::PluginSummary> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.list().await
    }

    async fn plugin_capability_providers(
        &self,
    ) -> anyhow::Result<Vec<plugin::PluginCapabilityProvider>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.capability_providers().await
    }

    async fn plugin_provider_for_capability(
        &self,
        capability: &str,
    ) -> anyhow::Result<Option<plugin::PluginCapabilityProvider>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.provider_for_capability(capability).await
    }

    async fn local_inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        let runtime_data_collector = self.inner.lock().await.runtime_data_collector.clone();
        runtime_data_collector
            .coalesce_local_inventory_scan(|| {
                crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
            })
            .await
    }

    async fn mesh_models(&self) -> Vec<MeshModelPayload> {
        let (runtime_data_collector, node, my_vram_gb, fallback_model_name, model_size_bytes) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.clone(),
                inner.node.clone(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.model_name.clone(),
                inner.model_size_bytes,
            )
        };

        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let runtime_status = runtime_data_collector.runtime_status_snapshot();
        let model_name = runtime_status.primary_model.unwrap_or(fallback_model_name);

        let target_lookup = self.model_target_lookup().await;
        let mut models = runtime_data::mesh_models(runtime_data_collector.build_model_view(
            runtime_data::ModelViewInput {
                peers: node.peers().await,
                catalog: node.mesh_catalog_entries().await,
                served_models: node.models_being_served().await,
                active_demand: node.active_demand().await,
                my_serving_models: node.serving_models().await,
                my_hosted_models: node.hosted_models().await,
                local_inventory: self.local_inventory_snapshot().await,
                node_hostname: node.hostname.clone(),
                my_vram_gb,
                model_name,
                model_size_bytes,
                now_unix_secs: now_ts,
            },
        ));
        for model in &mut models {
            let target = target_lookup
                .by_model_name
                .get(&model.name)
                .or_else(|| target_lookup.by_model_ref.get(&model.name));
            if let Some(target) = target {
                model.target_rank = Some(target.rank);
                model.explicit_interest_count = Some(target.explicit_interest_count);
                model.wanted = Some(target.wanted);
            }
        }
        models
    }

    #[cfg(test)]
    fn derive_local_node_state(
        is_client: bool,
        effective_is_host: bool,
        effective_llama_ready: bool,
        has_local_worker_activity: bool,
        display_model_name: &str,
    ) -> NodeState {
        let has_declared_local_serving_work = (effective_is_host || has_local_worker_activity)
            && !display_model_name.trim().is_empty();

        if is_client {
            NodeState::Client
        } else if effective_llama_ready && has_declared_local_serving_work {
            NodeState::Serving
        } else if has_declared_local_serving_work {
            NodeState::Loading
        } else {
            NodeState::Standby
        }
    }

    #[cfg(test)]
    fn derive_node_status(node_state: NodeState) -> String {
        node_state.node_status_alias().to_string()
    }

    #[cfg(test)]
    fn derive_peer_state(peer: &mesh::PeerInfo) -> NodeState {
        fn has_nonempty_models(models: &[String]) -> bool {
            models.iter().any(|model| !model.trim().is_empty())
        }

        match peer.role {
            mesh::NodeRole::Client => NodeState::Client,
            mesh::NodeRole::Host { .. } | mesh::NodeRole::Worker => {
                let has_runtime_descriptors = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| !runtime.model_name.trim().is_empty());
                let has_ready_runtime = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| runtime.ready && !runtime.model_name.trim().is_empty());
                let has_assigned_model_work = has_runtime_descriptors
                    || has_nonempty_models(&peer.serving_models)
                    || has_nonempty_models(&peer.hosted_models);
                let has_legacy_serving_signal = has_nonempty_models(&peer.hosted_models)
                    || has_nonempty_models(&peer.serving_models)
                    || peer
                        .routable_models()
                        .iter()
                        .any(|model| !model.trim().is_empty());

                if has_ready_runtime {
                    NodeState::Serving
                } else if has_runtime_descriptors && has_assigned_model_work {
                    NodeState::Loading
                } else if has_legacy_serving_signal {
                    NodeState::Serving
                } else {
                    NodeState::Standby
                }
            }
        }
    }

    #[cfg(test)]
    fn build_wakeable_node(entry: WakeableInventoryEntry) -> WakeableNode {
        WakeableNode {
            logical_id: entry.logical_id,
            models: entry.models,
            vram_gb: entry.vram_gb,
            provider: entry.provider,
            state: match entry.state {
                WakeableState::Sleeping => WakeableNodeState::Sleeping,
                WakeableState::Waking => WakeableNodeState::Waking,
            },
            wake_eta_secs: entry.wake_eta_secs,
        }
    }

    async fn status(&self) -> StatusPayload {
        let (
            runtime_data_collector,
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            routing_affinity,
            model_size_bytes,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            nostr_discovery,
            publication_state,
            wakeable_inventory,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.clone(),
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.affinity_router.stats_snapshot(),
                inner.model_size_bytes,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.nostr_discovery,
                inner.publication_state,
                inner.wakeable_inventory.clone(),
            )
        };
        let runtime_status = runtime_data_collector.runtime_status_snapshot();
        let model_name = runtime_status.primary_model.clone().unwrap_or_default();
        let local_processes =
            runtime_data::runtime_process_payloads(&runtime_status.local_processes);
        let mut runtime = build_runtime_status_payload(
            &model_name,
            runtime_status.primary_backend.clone(),
            runtime_status.is_host,
            runtime_status.llama_ready,
            runtime_status.llama_port,
            local_processes.clone(),
        );
        node.refresh_stage_runtime_statuses(std::time::Duration::from_secs(2))
            .await;
        runtime.stages = build_runtime_stage_payloads(node.stage_runtime_statuses().await);

        let wakeable_nodes = wakeable_inventory.status_snapshot().await;
        let bw_str = {
            let bw = node.gpu_mem_bandwidth_gbps.lock().await;
            bw.as_ref().map(|v| {
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
        };
        let tf32_str = {
            let tf32 = node.gpu_compute_tflops_fp32.lock().await;
            tf32.as_ref().map(|v| {
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
        };
        let tf16_str = {
            let tf16 = node.gpu_compute_tflops_fp16.lock().await;
            tf16.as_ref().map(|v| {
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
        };

        let mut payload = runtime_data::status_payload(runtime_data_collector.build_status_view(
            runtime_data::StatusViewInput {
                version: MESH_LLM_VERSION.to_string(),
                latest_version,
                node_id,
                owner: node.owner_summary().await,
                token,
                is_host: runtime_status.is_host,
                is_client,
                llama_ready: runtime_status.llama_ready,
                model_name,
                models: node.models().await,
                available_models: node.available_models().await,
                requested_models: node.requested_models().await,
                serving_models: node.serving_models().await,
                hosted_models: node.hosted_models().await,
                draft_name,
                api_port,
                inflight_requests,
                mesh_id: node.mesh_id().await,
                mesh_name,
                nostr_discovery,
                publication_state: publication_state.as_str().into(),
                local_processes,
                peers: node.peers().await,
                wakeable_nodes,
                routing_affinity,
                hardware: runtime_data_collector.build_hardware_view(
                    runtime_data::HardwareViewInput {
                        gpu_name: node.gpu_name.clone(),
                        gpu_vram: node.gpu_vram.clone(),
                        gpu_reserved_bytes: node.gpu_reserved_bytes.clone(),
                        gpu_mem_bandwidth_gbps: bw_str,
                        gpu_compute_tflops_fp32: tf32_str,
                        gpu_compute_tflops_fp16: tf16_str,
                        my_hostname: node.hostname.clone(),
                        my_is_soc: node.is_soc,
                        my_vram_gb,
                        model_size_gb: model_size_bytes as f64 / 1e9,
                        first_joined_mesh_ts: node.first_joined_mesh_ts().await,
                    },
                ),
            },
        ));
        payload.runtime = runtime;
        payload.wanted_model_refs = self.wanted_model_refs().await;
        payload
    }

    async fn push_status(&self) {
        let mut inner = self.inner.lock().await;
        inner.runtime_data_producer.mark_status_dirty();
        inner.sse_clients.retain(|tx| !tx.is_closed());
    }
}

fn runtime_process_payload_identity(process: &RuntimeProcessPayload) -> &str {
    process.instance_id.as_deref().unwrap_or(&process.name)
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests;
