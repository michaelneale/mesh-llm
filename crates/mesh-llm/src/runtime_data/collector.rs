//! Collector-backed snapshot storage and synchronous publish helpers.
//!
//! Keep mutation local, drop locks before publish, and let readers observe
//! shared snapshots through this boundary.

use super::inventory::{
    replace_local_instances_snapshot, replace_local_inventory_snapshot, InventoryScanCoordinator,
};
use super::plugins::{
    clear_plugin_data, clear_plugin_endpoints, plugins_snapshot, upsert_plugin_data,
    upsert_plugin_endpoint, PluginDataValue, PluginsSnapshotView,
};
#[cfg(test)]
use super::plugins::{plugin_endpoint_snapshot, plugin_snapshot, PluginScopedSnapshot};
use super::processes::RuntimeProcessSnapshot;
use super::producers::{RuntimeDataProducer, RuntimeDataSource};
use super::snapshots::{
    HardwareViewInput, HardwareViewSnapshot, LocalInstancesSnapshot, ModelRouteStats,
    ModelViewInput, ModelViewSnapshot, PluginDataKey, PluginDataSnapshot, PluginEndpointKey,
    PluginEndpointsSnapshot, RuntimeDataSnapshots, RuntimeStatusDerivation, RuntimeStatusSnapshot,
    StatusViewInput, StatusViewSnapshot,
};
use super::subscriptions::{
    RuntimeDataDirty, RuntimeDataSubscriptionState, RuntimeDataSubscriptions,
};
use super::{
    RuntimeLlamaMetricItem, RuntimeLlamaMetricsSnapshot, RuntimeLlamaRuntimeItems,
    RuntimeLlamaRuntimeSnapshot, RuntimeLlamaSlotItem, RuntimeLlamaSlotsSnapshot,
};
use crate::api::status::{
    build_gpus, build_ownership_payload, LocalInstance, MeshModelPayload, NodeState, PeerPayload,
    WakeableNode, WakeableNodeState,
};
use crate::mesh;
use crate::models::LocalModelInventorySnapshot;
use crate::network::metrics::RoutingCollectorSnapshot;
use crate::plugin::PluginEndpointSummary;
use crate::runtime::instance::LocalInstanceSnapshot;
use crate::runtime::wakeable::{WakeableInventoryEntry, WakeableState};
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::watch;

#[derive(Default)]
struct RuntimeDataSharedState {
    snapshots: RwLock<RuntimeDataSnapshots>,
    subscriptions: RuntimeDataSubscriptions,
    inventory_scan: Mutex<InventoryScanCoordinator>,
}

#[derive(Clone, Default)]
pub(crate) struct RuntimeDataCollector {
    shared: Arc<RuntimeDataSharedState>,
}

impl RuntimeDataCollector {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn producer(&self, source: RuntimeDataSource) -> RuntimeDataProducer {
        RuntimeDataProducer::new(self.clone(), source)
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<RuntimeDataSubscriptionState> {
        self.shared.subscriptions.subscribe()
    }

    #[cfg(test)]
    pub(crate) fn subscription_state(&self) -> RuntimeDataSubscriptionState {
        self.shared.subscriptions.state()
    }

    pub(crate) fn mark_dirty(&self, dirty: RuntimeDataDirty) -> RuntimeDataSubscriptionState {
        self.shared.subscriptions.publish(dirty)
    }

    pub(crate) fn update_runtime_status<F>(&self, dirty: RuntimeDataDirty, update: F) -> bool
    where
        F: FnOnce(&mut RuntimeStatusSnapshot) -> bool,
    {
        self.update_snapshots(dirty, |snapshots| update(&mut snapshots.runtime_status))
    }

    pub(crate) fn snapshots(&self) -> RuntimeDataSnapshots {
        self.shared
            .snapshots
            .read()
            .expect("runtime data snapshots lock poisoned")
            .clone()
    }

    pub(crate) fn runtime_status_snapshot(&self) -> RuntimeStatusSnapshot {
        self.snapshots().runtime_status
    }

    pub(crate) fn runtime_processes_snapshot(&self) -> Vec<RuntimeProcessSnapshot> {
        self.runtime_status_snapshot().local_processes
    }

    pub(crate) fn runtime_llama_snapshot(&self) -> RuntimeLlamaRuntimeSnapshot {
        self.runtime_status_snapshot().llama_runtime
    }

    pub(crate) fn routing_snapshot(&self) -> RoutingCollectorSnapshot {
        self.snapshots().routing
    }

    pub(crate) fn local_instances_snapshot(&self) -> LocalInstancesSnapshot {
        self.snapshots().local_instances
    }

    pub(crate) fn local_inventory_snapshot(&self) -> LocalModelInventorySnapshot {
        self.snapshots().local_inventory
    }

    pub(crate) fn replace_local_instances_snapshot(
        &self,
        instances: Vec<LocalInstanceSnapshot>,
    ) -> bool {
        self.update_snapshots(RuntimeDataDirty::INVENTORY, |snapshots| {
            replace_local_instances_snapshot(&mut snapshots.local_instances, instances)
        })
    }

    pub(crate) fn replace_llama_metrics_snapshot(
        &self,
        snapshot: RuntimeLlamaMetricsSnapshot,
    ) -> bool {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |runtime_status| {
            let next_items =
                build_llama_runtime_items(&snapshot, &runtime_status.llama_runtime.slots);
            if runtime_status.llama_runtime.metrics == snapshot
                && runtime_status.llama_runtime.items == next_items
            {
                return false;
            }
            runtime_status.llama_runtime.metrics = snapshot;
            runtime_status.llama_runtime.items = next_items;
            true
        })
    }

    pub(crate) fn replace_llama_slots_snapshot(&self, snapshot: RuntimeLlamaSlotsSnapshot) -> bool {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |runtime_status| {
            let next_items =
                build_llama_runtime_items(&runtime_status.llama_runtime.metrics, &snapshot);
            if runtime_status.llama_runtime.slots == snapshot
                && runtime_status.llama_runtime.items == next_items
            {
                return false;
            }
            runtime_status.llama_runtime.slots = snapshot;
            runtime_status.llama_runtime.items = next_items;
            true
        })
    }

    pub(crate) async fn coalesce_local_inventory_scan<F>(
        &self,
        load: F,
    ) -> LocalModelInventorySnapshot
    where
        F: FnOnce() -> LocalModelInventorySnapshot + Send + 'static,
    {
        let (rx, start_scan) = {
            let mut inventory_scan = self
                .shared
                .inventory_scan
                .lock()
                .expect("runtime data inventory scan lock poisoned");
            inventory_scan.begin_or_join()
        };

        if start_scan {
            let collector = self.clone();
            tokio::spawn(async move {
                let snapshot = match tokio::task::spawn_blocking(load).await {
                    Ok(snapshot) => snapshot,
                    Err(err) => {
                        tracing::warn!("Local inventory scan failed: {err}");
                        LocalModelInventorySnapshot::default()
                    }
                };

                collector.replace_local_inventory_snapshot(snapshot.clone());
                let waiters = {
                    let mut inventory_scan = collector
                        .shared
                        .inventory_scan
                        .lock()
                        .expect("runtime data inventory scan lock poisoned");
                    inventory_scan.finish()
                };
                for waiter in waiters {
                    let _ = waiter.send(snapshot.clone());
                }
            });
        }

        rx.await.unwrap_or_else(|_| self.local_inventory_snapshot())
    }

    pub(crate) fn plugin_data_snapshot(&self) -> PluginDataSnapshot {
        self.snapshots().plugin_data
    }

    pub(crate) fn plugin_endpoints_snapshot(&self) -> PluginEndpointsSnapshot {
        self.snapshots().plugin_endpoints
    }

    pub(crate) fn plugins_snapshot(&self) -> PluginsSnapshotView {
        let snapshots = self.snapshots();
        plugins_snapshot(&snapshots.plugin_data, &snapshots.plugin_endpoints)
    }

    #[cfg(test)]
    pub(crate) fn plugin_snapshot(&self, plugin_name: &str) -> PluginScopedSnapshot {
        let snapshots = self.snapshots();
        plugin_snapshot(
            &snapshots.plugin_data,
            &snapshots.plugin_endpoints,
            plugin_name,
        )
    }

    #[cfg(test)]
    pub(crate) fn plugin_endpoint_snapshot(
        &self,
        plugin_name: &str,
        endpoint_id: &str,
    ) -> Option<PluginEndpointSummary> {
        plugin_endpoint_snapshot(&self.snapshots().plugin_endpoints, plugin_name, endpoint_id)
    }

    pub(crate) fn publish_plugin_data(&self, key: PluginDataKey, value: PluginDataValue) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            upsert_plugin_data(&mut snapshots.plugin_data, key, value)
        })
    }

    pub(crate) fn publish_plugin_endpoint(
        &self,
        key: PluginEndpointKey,
        value: PluginEndpointSummary,
    ) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            upsert_plugin_endpoint(&mut snapshots.plugin_endpoints, key, value)
        })
    }

    pub(crate) fn clear_plugin_reports(&self, plugin_name: &str) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            let data_changed = clear_plugin_data(&mut snapshots.plugin_data, plugin_name);
            let endpoints_changed =
                clear_plugin_endpoints(&mut snapshots.plugin_endpoints, plugin_name);
            data_changed || endpoints_changed
        })
    }

    pub(crate) fn build_hardware_view(&self, input: HardwareViewInput) -> HardwareViewSnapshot {
        HardwareViewSnapshot {
            my_hostname: input.my_hostname,
            my_is_soc: input.my_is_soc,
            my_vram_gb: input.my_vram_gb,
            model_size_gb: input.model_size_gb,
            gpus: build_gpus(
                input.gpu_name.as_deref(),
                input.gpu_vram.as_deref(),
                input.gpu_reserved_bytes.as_deref(),
                input.gpu_mem_bandwidth_gbps.as_deref(),
                input.gpu_compute_tflops_fp32.as_deref(),
                input.gpu_compute_tflops_fp16.as_deref(),
            ),
            first_joined_mesh_ts: input.first_joined_mesh_ts,
        }
    }

    pub(crate) fn build_status_view(&self, input: StatusViewInput) -> StatusViewSnapshot {
        let derivation = derive_runtime_status(RuntimeStatusDerivationInput {
            is_client: input.is_client,
            is_host: input.is_host,
            llama_ready: input.llama_ready,
            local_processes: &input.local_processes,
            hosted_models: &input.hosted_models,
            serving_models: &input.serving_models,
            model_name: &input.model_name,
            api_port: input.api_port,
        });
        let routing_snapshot = self.routing_snapshot();

        StatusViewSnapshot {
            version: input.version.clone(),
            latest_version: input.latest_version,
            node_id: input.node_id,
            owner: build_ownership_payload(&input.owner),
            token: input.token,
            node_state: derivation.node_state,
            node_status: derivation.node_status,
            is_host: derivation.effective_is_host,
            is_client: input.is_client,
            llama_ready: derivation.effective_llama_ready,
            model_name: derivation.display_model_name,
            models: input.models,
            available_models: input.available_models,
            requested_models: input.requested_models,
            serving_models: input.serving_models,
            hosted_models: input.hosted_models,
            draft_name: input.draft_name,
            api_port: input.api_port,
            peers: input.peers.iter().map(build_peer_payload).collect(),
            wakeable_nodes: input
                .wakeable_nodes
                .into_iter()
                .map(build_wakeable_node)
                .collect(),
            local_instances: build_local_instances(
                self.local_instances_snapshot().instances,
                input.api_port,
                &input.version,
            ),
            launch_pi: derivation.launch_pi,
            launch_goose: derivation.launch_goose,
            inflight_requests: input.inflight_requests,
            mesh_id: input.mesh_id,
            mesh_name: input.mesh_name,
            nostr_discovery: input.nostr_discovery,
            publication_state: input.publication_state,
            routing_affinity: input.routing_affinity,
            routing_metrics: routing_snapshot.status,
            hardware: input.hardware,
        }
    }

    pub(crate) fn build_model_view(&self, input: ModelViewInput) -> ModelViewSnapshot {
        let routing_metrics_by_model = self.routing_snapshot().models;
        let local_model_names = input.local_inventory.model_names;
        let mut metadata_by_name = input.local_inventory.metadata_by_name;
        let mut size_by_name = input.local_inventory.size_by_name;
        for peer in &input.peers {
            for meta in &peer.available_model_metadata {
                metadata_by_name
                    .entry(meta.model_key.clone())
                    .or_insert_with(|| meta.clone());
            }
            for (model_name, size) in &peer.available_model_sizes {
                size_by_name.entry(model_name.clone()).or_insert(*size);
            }
        }

        let models = input
            .catalog
            .iter()
            .map(|entry| {
                let name = &entry.model_name;
                let descriptor = entry.descriptor.as_ref();
                let identity = descriptor.map(|descriptor| &descriptor.identity);
                let catalog_entry = find_catalog_model(name);
                let is_warm = input.served_models.iter().any(|served| served == name);
                let local_known = local_model_names.contains(name)
                    || input.my_hosted_models.iter().any(|s| s == name)
                    || input.my_serving_models.iter().any(|s| s == name)
                    || name == &input.model_name;
                let display_name = crate::models::installed_model_display_name(name);
                let route_stats = is_warm.then(|| {
                    http_route_stats(
                        name,
                        &input.peers,
                        &input.my_hosted_models,
                        input.node_hostname.as_deref(),
                        input.my_vram_gb,
                    )
                });
                let node_count = route_stats
                    .as_ref()
                    .map(|stats| stats.node_count)
                    .unwrap_or(0);
                let active_nodes = route_stats
                    .as_ref()
                    .map(|stats| stats.active_nodes.clone())
                    .unwrap_or_default();
                let mesh_vram_gb = route_stats
                    .as_ref()
                    .map(|stats| stats.mesh_vram_gb)
                    .unwrap_or(0.0);
                let size_gb = if name == &input.model_name && input.model_size_bytes > 0 {
                    input.model_size_bytes as f64 / 1e9
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
                let (request_count, last_active_secs_ago) = match input.active_demand.get(name) {
                    Some(demand) => (
                        Some(demand.request_count),
                        Some(input.now_unix_secs.saturating_sub(demand.last_active)),
                    ),
                    None => (None, None),
                };
                let routing_metrics = routing_metrics_by_model.get(name).cloned();
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
                    capabilities.multimodal = true;
                }
                if local_known
                    && likely_audio_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.audio = capabilities
                        .audio
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                    capabilities.multimodal = true;
                }
                let multimodal = capabilities.supports_multimodal_runtime();
                let multimodal_status = if multimodal || capabilities.multimodal_label().is_some() {
                    Some(capabilities.multimodal_status())
                } else {
                    None
                };
                let vision = capabilities.supports_vision_runtime();
                let vision_status = if vision || capabilities.vision_label().is_some() {
                    Some(capabilities.vision_status())
                } else {
                    None
                };
                let audio = matches!(
                    capabilities.audio,
                    crate::models::capabilities::CapabilityLevel::Supported
                        | crate::models::capabilities::CapabilityLevel::Likely
                );
                let audio_status = if audio || capabilities.audio_label().is_some() {
                    Some(capabilities.audio_status())
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
                let (fit_label, fit_detail) = fit_hint_for_machine(size_gb, input.my_vram_gb);

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
                    multimodal,
                    multimodal_status,
                    vision,
                    vision_status,
                    audio,
                    audio_status,
                    reasoning,
                    reasoning_status,
                    tool_use,
                    tool_use_status,
                    draft_model,
                    request_count,
                    last_active_secs_ago,
                    target_rank: None,
                    explicit_interest_count: None,
                    wanted: None,
                    routing_metrics,
                    source_page_url,
                    source_ref,
                    source_revision,
                    source_file,
                    active_nodes,
                    fit_label,
                    fit_detail,
                    download_command: format!("mesh-llm models download {}", command_ref),
                    run_command: format!("mesh-llm serve --model {}", command_ref),
                    auto_command: format!("mesh-llm serve --auto --model {}", command_ref),
                }
            })
            .collect();

        ModelViewSnapshot { models }
    }

    pub(crate) fn replace_routing_snapshot(&self, snapshot: RoutingCollectorSnapshot) -> bool {
        self.update_snapshots(RuntimeDataDirty::ROUTING, |snapshots| {
            if snapshots.routing == snapshot {
                false
            } else {
                snapshots.routing = snapshot;
                true
            }
        })
    }

    fn replace_local_inventory_snapshot(&self, snapshot: LocalModelInventorySnapshot) -> bool {
        self.update_snapshots(RuntimeDataDirty::INVENTORY, |snapshots| {
            replace_local_inventory_snapshot(&mut snapshots.local_inventory, snapshot)
        })
    }

    fn update_snapshots<F>(&self, dirty: RuntimeDataDirty, update: F) -> bool
    where
        F: FnOnce(&mut RuntimeDataSnapshots) -> bool,
    {
        let changed = {
            let mut snapshots = self
                .shared
                .snapshots
                .write()
                .expect("runtime data snapshots lock poisoned");
            update(&mut snapshots)
        };

        if changed {
            self.shared.subscriptions.publish(dirty);
        }

        changed
    }
}

fn build_llama_runtime_items(
    metrics: &RuntimeLlamaMetricsSnapshot,
    slots: &RuntimeLlamaSlotsSnapshot,
) -> RuntimeLlamaRuntimeItems {
    let slot_items = slots
        .slots
        .iter()
        .enumerate()
        .map(|(index, slot)| RuntimeLlamaSlotItem {
            index,
            id: slot.id,
            id_task: slot.id_task,
            n_ctx: slot.n_ctx,
            is_processing: slot.is_processing.unwrap_or(false),
        })
        .collect::<Vec<_>>();
    RuntimeLlamaRuntimeItems {
        metrics: metrics
            .samples
            .iter()
            .map(|sample| RuntimeLlamaMetricItem {
                name: sample.name.clone(),
                labels: sample.labels.clone(),
                value: sample.value,
            })
            .collect(),
        slots_total: slot_items.len(),
        slots_busy: slot_items.iter().filter(|slot| slot.is_processing).count(),
        slots: slot_items,
    }
}

struct RuntimeStatusDerivationInput<'a> {
    is_client: bool,
    is_host: bool,
    llama_ready: bool,
    local_processes: &'a [crate::api::RuntimeProcessPayload],
    hosted_models: &'a [String],
    serving_models: &'a [String],
    model_name: &'a str,
    api_port: u16,
}

fn derive_runtime_status(input: RuntimeStatusDerivationInput<'_>) -> RuntimeStatusDerivation {
    let has_local_processes = !input.local_processes.is_empty();
    let effective_llama_ready = input.llama_ready || has_local_processes;
    let effective_is_host = input.is_host || has_local_processes;
    let display_model_name = input
        .local_processes
        .first()
        .map(|process| process.name.clone())
        .or_else(|| input.hosted_models.first().cloned())
        .or_else(|| input.serving_models.first().cloned())
        .unwrap_or_else(|| input.model_name.to_string());
    let has_local_worker_activity = has_local_processes || !input.hosted_models.is_empty();
    let node_state = derive_local_node_state(
        input.is_client,
        effective_is_host,
        effective_llama_ready,
        has_local_worker_activity,
        &display_model_name,
    );
    let launch_pi = if effective_llama_ready {
        Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            input.api_port,
            crate::cli::shell::single_quote(&display_model_name)
        ))
    } else {
        None
    };
    let launch_goose = if effective_llama_ready {
        let api_port = input.api_port;
        Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={display_model_name} goose session"
        ))
    } else {
        None
    };

    RuntimeStatusDerivation {
        effective_is_host,
        effective_llama_ready,
        display_model_name,
        node_state,
        node_status: node_state.node_status_alias().to_string(),
        launch_pi,
        launch_goose,
    }
}

fn derive_local_node_state(
    is_client: bool,
    effective_is_host: bool,
    effective_llama_ready: bool,
    has_local_worker_activity: bool,
    display_model_name: &str,
) -> NodeState {
    let has_declared_local_serving_work =
        (effective_is_host || has_local_worker_activity) && !display_model_name.trim().is_empty();

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

fn build_peer_payload(peer: &mesh::PeerInfo) -> PeerPayload {
    PeerPayload {
        id: peer.id.fmt_short().to_string(),
        owner: build_ownership_payload(&peer.owner_summary),
        role: match peer.role {
            mesh::NodeRole::Worker => "Worker".into(),
            mesh::NodeRole::Host { .. } => "Host".into(),
            mesh::NodeRole::Client => "Client".into(),
        },
        state: derive_peer_state(peer),
        models: peer.models.clone(),
        available_models: peer.available_models.clone(),
        requested_models: peer.requested_models.clone(),
        vram_gb: peer.vram_bytes as f64 / 1e9,
        serving_models: peer.serving_models.clone(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: peer.hosted_models_known,
        version: peer.version.clone(),
        rtt_ms: peer.rtt_ms,
        hostname: peer.hostname.clone(),
        is_soc: peer.is_soc,
        gpus: build_gpus(
            peer.gpu_name.as_deref(),
            peer.gpu_vram.as_deref(),
            peer.gpu_reserved_bytes.as_deref(),
            peer.gpu_mem_bandwidth_gbps.as_deref(),
            peer.gpu_compute_tflops_fp32.as_deref(),
            peer.gpu_compute_tflops_fp16.as_deref(),
        ),
        first_joined_mesh_ts: peer.first_joined_mesh_ts,
    }
}

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

fn build_local_instances(
    snapshots: Vec<crate::runtime::instance::LocalInstanceSnapshot>,
    api_port: u16,
    version: &str,
) -> Vec<LocalInstance> {
    let mut instances: Vec<LocalInstance> = snapshots
        .iter()
        .map(|snapshot| LocalInstance {
            pid: snapshot.pid,
            api_port: snapshot.api_port,
            version: snapshot.version.clone(),
            started_at_unix: snapshot.started_at_unix,
            runtime_dir: snapshot.runtime_dir.to_string_lossy().to_string(),
            is_self: snapshot.is_self,
        })
        .collect();

    if instances.is_empty() {
        instances.push(LocalInstance {
            pid: std::process::id(),
            api_port: Some(api_port),
            version: Some(version.to_string()),
            started_at_unix: 0,
            runtime_dir: String::new(),
            is_self: true,
        });
    }

    instances
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

fn likely_audio_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    [
        "audio",
        "speech",
        "voice",
        "omni",
        "ultravox",
        "qwen2-audio",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

fn fit_hint_for_machine(size_gb: f64, my_vram_gb: f64) -> (String, String) {
    if size_gb <= 0.0 || my_vram_gb <= 0.0 {
        return (
            "Unknown".into(),
            "No local capacity signal is available for this machine yet.".into(),
        );
    }
    if size_gb * 1.2 <= my_vram_gb {
        return (
            "Likely comfortable".into(),
            format!(
                "This machine has {:.1} GB capacity, which should handle a {:.1} GB model comfortably.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 1.05 <= my_vram_gb {
        return (
            "Likely fits".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model should fit, but headroom will be tight.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 0.8 <= my_vram_gb {
        return (
            "Possible with tradeoffs".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model may load, but expect tighter memory pressure.",
                my_vram_gb, size_gb
            ),
        );
    }
    (
        "Likely too large".into(),
        format!(
            "This machine has {:.1} GB capacity, which is likely not enough for a {:.1} GB model locally.",
            my_vram_gb, size_gb
        ),
    )
}

fn http_route_stats(
    model_name: &str,
    peers: &[mesh::PeerInfo],
    my_hosted_models: &[String],
    my_hostname: Option<&str>,
    my_vram_gb: f64,
) -> ModelRouteStats {
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

    ModelRouteStats {
        node_count,
        active_nodes,
        mesh_vram_gb,
    }
}
