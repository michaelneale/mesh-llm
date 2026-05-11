use super::context_planning::{
    plan_runtime_resources, RuntimeResourcePlan, RuntimeResourcePlanInput,
};
#[cfg(test)]
use super::split_planning::{format_aggregate_split_capacity_error, validate_split_capacity};
use super::split_planning::{
    format_gb, plan_runtime_slice_topology_with_resources, split_participant_exclusion_labels,
    split_participant_labels, split_participants_for_stages, split_stage_plan_labels,
    RuntimeSliceStagePlan, SplitTopologyResourceInputs,
};
use crate::api;
use crate::cli::output::{emit_event, OutputEvent};
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::models;
use crate::network::router;
use crate::runtime_data::{
    RuntimeLlamaEndpointStatus, RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot,
};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SPLIT_PARTICIPANT_POLL_INTERVAL: Duration = Duration::from_millis(500);
const SPLIT_PARTICIPANT_STABLE_FOR: Duration = Duration::from_secs(2);
pub(super) const SPLIT_DEFAULT_MIN_PARTICIPANTS: usize = 2;
const SPLIT_INITIAL_SHUTDOWN_GENERATION: u64 = 1;
const SPLIT_COORDINATOR_LEASE_SECS: u64 = 4 * 60 * 60;
const RUNTIME_MODEL_FIT_HEADROOM_NUMERATOR: u64 = 11;
const RUNTIME_MODEL_FIT_HEADROOM_DENOMINATOR: u64 = 10;

pub(super) enum RuntimeEvent {
    Exited {
        instance_id: String,
        model: String,
        port: u16,
    },
}

pub(super) enum LocalRuntimeBackendHandle {
    Skippy {
        model: skippy::SkippyModelHandle,
        http: skippy::SkippyHttpHandle,
        _death_tx: tokio::sync::oneshot::Sender<()>,
    },
}

pub(super) struct LocalRuntimeModelHandle {
    pub(super) port: u16,
    pub(super) backend: String,
    pub(super) context_length: u32,
    pub(super) slots: usize,
    inner: LocalRuntimeBackendHandle,
}

impl LocalRuntimeModelHandle {
    pub(super) fn pid(&self) -> u32 {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { .. } => std::process::id(),
        }
    }

    pub(super) fn ctx_used_tokens(&self) -> Option<u64> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => {
                Some(model.status().max_session_tokens)
            }
        }
    }

    pub(super) fn llama_slots_snapshot(
        &self,
        model_name: &str,
        instance_id: Option<&str>,
    ) -> Option<RuntimeLlamaSlotsSnapshot> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => {
                let status = model.status();
                let ctx_size = status.ctx_size as u64;
                let now = current_time_unix_ms();
                Some(RuntimeLlamaSlotsSnapshot {
                    status: RuntimeLlamaEndpointStatus::Ready,
                    model: Some(model_name.to_string()),
                    instance_id: instance_id.map(str::to_string),
                    last_attempt_unix_ms: Some(now),
                    last_success_unix_ms: Some(now),
                    error: None,
                    slots: status
                        .lanes
                        .into_iter()
                        .map(|lane| RuntimeLlamaSlotSnapshot {
                            id: Some(lane.index as u64),
                            id_task: None,
                            n_ctx: Some(ctx_size),
                            speculative: None,
                            is_processing: Some(lane.active),
                            next_token: None,
                            params: None,
                            extra: serde_json::json!({
                                "model": model_name,
                                "lane_index": lane.index,
                                "active": lane.active,
                                "session_id": lane.session_id,
                                "token_count": lane.token_count,
                            }),
                        })
                        .collect(),
                })
            }
        }
    }

    pub(super) async fn shutdown(self) {
        match self.inner {
            LocalRuntimeBackendHandle::Skippy { model, http, .. } => {
                let _ = http.shutdown().await;
                model.shutdown();
            }
        }
    }
}

fn current_time_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn split_coordinator_lease_until_unix_ms() -> u64 {
    current_time_unix_ms().saturating_add(SPLIT_COORDINATOR_LEASE_SECS.saturating_mul(1000))
}

pub(super) struct ManagedModelController {
    pub(super) model_name: String,
    pub(super) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(super) task: tokio::task::JoinHandle<()>,
}

pub(super) struct LocalRuntimeModelStartSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) model_path: &'a Path,
    pub(super) model_bytes: u64,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) capacity_budget_bytes: Option<u64>,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
}

pub(super) enum SplitRuntimeStart {
    Started(Box<SplitRuntimeGenerationHandle>),
    Standby { coordinator: iroh::EndpointId },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StartupRuntimePlan {
    Local,
    Split { reason: SplitRuntimeReason },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitRuntimeReason {
    Forced,
    LocalCapacity,
}

pub(super) struct SplitRuntimeGenerationHandle {
    pub(super) loaded_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) cleanup: Option<SplitGenerationCleanup>,
    pub(super) coordinator_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) coordinator_task: Option<tokio::task::JoinHandle<()>>,
}

pub(super) enum SplitCoordinatorEvent {
    Replace(Box<SplitCoordinatorReplaceEvent>),
    LocalFallback(SplitCoordinatorLocalFallbackEvent),
    Withdraw(SplitCoordinatorWithdrawEvent),
}

pub(super) struct SplitCoordinatorReplaceEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) loaded: SplitRuntimeGenerationHandle,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) struct SplitCoordinatorLocalFallbackEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) topology_id: String,
    pub(super) missing_stage_nodes: Vec<iroh::EndpointId>,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) struct SplitCoordinatorWithdrawEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) topology_id: String,
    pub(super) missing_stage_nodes: Vec<iroh::EndpointId>,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) enum SplitCoordinatorAck {
    Accepted,
}

#[derive(Clone, Debug)]
pub(super) struct SplitGenerationCleanup {
    generation: SplitTopologyGeneration,
}

pub(super) async fn stop_split_generation_cleanup(
    node: &mesh::Node,
    cleanup: SplitGenerationCleanup,
    shutdown_generation: u64,
) {
    stop_split_generation(node, &cleanup.generation, shutdown_generation).await;
}

pub(super) fn resolved_model_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    router::strip_split_suffix_owned(&stem)
}

fn mmproj_path_for_model(model_name: &str) -> Option<PathBuf> {
    let model_path = models::find_model_path(model_name);
    models::find_mmproj_path(model_name, &model_path)
}

fn effective_flash_attention(
    override_value: FlashAttentionType,
    effective_cache_type_v: &str,
) -> FlashAttentionType {
    if override_value != FlashAttentionType::Auto {
        return override_value;
    }

    match effective_cache_type_v {
        "f16" => FlashAttentionType::Auto,
        _ => FlashAttentionType::Enabled,
    }
}

async fn alloc_local_port() -> Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub(super) fn add_runtime_local_target(
    target_tx: &std::sync::Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    port: u16,
) {
    let mut targets = target_tx.borrow().clone();
    let entry = targets.targets.entry(model_name.to_string()).or_default();
    entry.retain(
        |target| !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port),
    );
    entry.insert(0, election::InferenceTarget::Local(port));
    target_tx.send_replace(targets);
}

pub(super) fn remove_runtime_local_target(
    target_tx: &std::sync::Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    port: u16,
) {
    let mut targets = target_tx.borrow().clone();
    let mut should_remove_model = false;
    if let Some(entry) = targets.targets.get_mut(model_name) {
        entry.retain(|target| {
            !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port)
        });
        should_remove_model = entry.is_empty();
    }
    if should_remove_model {
        targets.targets.remove(model_name);
    }
    target_tx.send_replace(targets);
}

pub(super) async fn advertise_model_ready(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
) {
    let mut hosted_models = node.hosted_models().await;
    if hosted_models.iter().any(|m| m == model_name) {
        return;
    }
    hosted_models.push(model_name.to_string());
    hosted_models.sort();
    if let Some(pos) = hosted_models.iter().position(|m| m == primary_model_name) {
        let primary = hosted_models.remove(pos);
        hosted_models.insert(0, primary);
    }
    node.set_hosted_models(hosted_models).await;
    node.regossip().await;
}

pub(super) async fn set_advertised_model_context(
    node: &mesh::Node,
    model_name: &str,
    context_length: Option<u32>,
) {
    node.set_model_runtime_context_length(model_name, context_length)
        .await;
    node.regossip().await;
}

pub(super) async fn withdraw_advertised_model(node: &mesh::Node, model_name: &str) {
    let mut hosted_models = node.hosted_models().await;
    let old_len = hosted_models.len();
    hosted_models.retain(|m| m != model_name);
    if hosted_models.len() == old_len {
        return;
    }
    node.set_hosted_models(hosted_models).await;
    node.regossip().await;
}

pub(super) async fn add_serving_assignment(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
) {
    let mut serving_models = node.serving_models().await;
    if serving_models.iter().any(|m| m == model_name) {
        return;
    }
    serving_models.push(model_name.to_string());
    serving_models.sort();
    if let Some(pos) = serving_models.iter().position(|m| m == primary_model_name) {
        let primary = serving_models.remove(pos);
        serving_models.insert(0, primary);
    }
    node.set_serving_models(serving_models).await;
    if let Some(descriptor) =
        mesh::infer_local_served_model_descriptor(model_name, model_name == primary_model_name)
    {
        node.upsert_served_model_descriptor(descriptor).await;
    }
    node.regossip().await;
}

pub(super) async fn remove_serving_assignment(node: &mesh::Node, model_name: &str) {
    let mut serving_models = node.serving_models().await;
    let old_len = serving_models.len();
    serving_models.retain(|m| m != model_name);
    if serving_models.len() == old_len {
        return;
    }
    node.set_serving_models(serving_models).await;
    node.remove_served_model_descriptor(model_name).await;
    node.regossip().await;
}

pub(super) async fn start_runtime_local_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    runtime_model_name: &str,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let model_name = runtime_model_name.to_string();
    let package_ref = spec.model_path.to_string_lossy().to_string();
    let layer_package = if skippy::is_layer_package_ref(&package_ref) {
        let package_ref_for_identity = package_ref.clone();
        Some(
            tokio::task::spawn_blocking(move || {
                skippy::identity_from_layer_package(&package_ref_for_identity)
            })
            .await
            .context("join identify skippy layer package task")??,
        )
    } else {
        None
    };
    let total_model_bytes = layer_package
        .as_ref()
        .map(|package| package.source_model_bytes)
        .unwrap_or_else(|| election::total_model_bytes(spec.model_path));
    let my_vram = spec
        .capacity_budget_bytes
        .or_else(|| spec.pinned_gpu.map(|gpu| gpu.vram_bytes))
        .unwrap_or_else(|| spec.node.vram_bytes());

    // For split/layer-package models, compute the local share of model weights
    // and the layer fraction so the context planner budgets correctly.
    // At planning time the exact layer assignment is not yet known, so we
    // estimate the local fraction from the VRAM ratio: this node's VRAM
    // divided by total mesh VRAM (local + peers).
    // This is the local (solo) load path — the entire model is loaded on
    // this node.  Fractional scaling only applies in the split path
    // (start_runtime_split_model).
    let local_model_bytes = total_model_bytes;
    let local_layer_fraction: Option<f64> = None;

    let required_bytes = runtime_model_required_bytes(local_model_bytes);
    anyhow::ensure!(
        my_vram >= required_bytes,
        "runtime load only supports models that fit locally on this node; model requires {}, local capacity is {}",
        format_gb(required_bytes),
        format_gb(my_vram)
    );

    let kv_cache = skippy::KvCachePolicy::for_model_size(total_model_bytes);
    let effective_cache_type_k = spec
        .cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k());
    let effective_cache_type_v = spec
        .cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v());
    let kv_cache_quant = models::gguf::GgufKvCacheQuant::from_llama_args(
        effective_cache_type_k,
        effective_cache_type_v,
    )
    .unwrap_or(models::gguf::GgufKvCacheQuant::Q8_0);

    // For layer packages, try to read GGUF metadata from the shared metadata
    // file inside the package.  This carries the model's native context length,
    // head counts, and KV dimensions needed for accurate KV budget planning.
    // Runs on a blocking thread because the underlying calls do filesystem I/O
    // (stat, open, read GGUF headers).
    let compact_meta = {
        let package_clone = layer_package.clone();
        let model_path = spec.model_path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            if let Some(ref package) = package_clone {
                scan_layer_package_metadata(package)
            } else {
                models::gguf::scan_gguf_compact_meta(&model_path)
            }
        })
        .await
        .ok()
        .flatten()
    };
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: spec.ctx_size_override,
        parallel_override: spec.parallel_override,
        model_bytes: local_model_bytes,
        vram_bytes: my_vram,
        metadata: compact_meta.as_ref(),
        kv_cache_quant,
        local_layer_fraction,
    });

    if let Some(package) = layer_package {
        start_runtime_layer_package_model(spec, model_name, package, plan).await
    } else {
        start_runtime_skippy_model(spec, model_name, plan).await
    }
}

/// Try to extract GGUF architecture metadata from a layer package's shared
/// metadata file.  Layer packages store a `shared/metadata.gguf` that carries
/// the model's KV pairs (context_length, head counts, etc.) without any tensor
/// data.  This gives the context planner the information it needs for accurate
/// KV cache budget calculations on split models.
fn scan_layer_package_metadata(
    package: &skippy::SkippyPackageIdentity,
) -> Option<models::gguf::GgufCompactMeta> {
    // The source_model_path in a layer package identity points to the original
    // GGUF.  But for HF layer packages the source model is not downloaded
    // locally.  Instead, look for the shared metadata file in the package dir.
    //
    // The package_ref looks like "hf://meshllm/Qwen3-layers@rev" which resolves
    // to a local cache directory.  Try to find shared/metadata.gguf there.
    let package_ref = &package.package_ref;
    let local_ref = skippy::resolve_hf_package_to_local(package_ref, 0, 0, false, false).ok()?;
    let metadata_path = std::path::Path::new(&local_ref).join("shared/metadata.gguf");
    if metadata_path.is_file() {
        return models::gguf::scan_gguf_compact_meta(&metadata_path);
    }
    // Fallback: try scanning the source model directly (works for local packages).
    if package.source_model_path.is_file() {
        return models::gguf::scan_gguf_compact_meta(&package.source_model_path);
    }
    None
}

pub(super) fn runtime_model_planning_bytes(model_path: &Path) -> Result<u64> {
    let package_ref = model_path.to_string_lossy().to_string();
    if skippy::is_layer_package_ref(&package_ref) {
        return Ok(skippy::identity_from_layer_package(&package_ref)?.source_model_bytes);
    }
    Ok(election::total_model_bytes(model_path))
}

pub(super) fn startup_runtime_plan(
    explicit_split: bool,
    local_vram_bytes: u64,
    model_bytes: u64,
) -> StartupRuntimePlan {
    if explicit_split {
        return StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::Forced,
        };
    }
    if model_fits_runtime_capacity(model_bytes, local_vram_bytes) {
        StartupRuntimePlan::Local
    } else {
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity,
        }
    }
}

pub(super) fn model_fits_runtime_capacity(model_bytes: u64, local_vram_bytes: u64) -> bool {
    local_vram_bytes >= runtime_model_required_bytes(model_bytes)
}

pub(super) fn runtime_model_required_bytes(model_bytes: u64) -> u64 {
    model_bytes
        .saturating_mul(RUNTIME_MODEL_FIT_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_MODEL_FIT_HEADROOM_DENOMINATOR)
}

pub(super) async fn start_runtime_split_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_ref: &str,
) -> Result<SplitRuntimeStart> {
    let model_path_str = spec.model_path.to_string_lossy().to_string();
    let package = if skippy::is_layer_package_ref(&model_path_str) {
        tokio::task::spawn_blocking(move || skippy::identity_from_layer_package(&model_path_str))
            .await
            .context("join identify skippy layer package task")??
    } else {
        skippy::synthetic_direct_gguf_package(model_ref, spec.model_path)?
    };
    let participant_snapshot = wait_for_split_participants(
        spec.node,
        model_ref,
        model_ref,
        &package,
        spec.pinned_gpu.map(|gpu| gpu.vram_bytes),
        Duration::from_secs(30),
    )
    .await?;
    let run_id = format!("mesh-split-{}", now_unix_nanos());
    let topology_id = format!("topology-{run_id}");
    let compact_meta = {
        let pkg = package.clone();
        tokio::task::spawn_blocking(move || scan_layer_package_metadata(&pkg))
            .await
            .ok()
            .flatten()
    }
    .context("split topology planning requires GGUF metadata")?;
    let split_kv_policy = skippy::KvCachePolicy::for_model_size(package.source_model_bytes);
    let kv_cache_quant =
        if spec.cache_type_k_override.is_some() || spec.cache_type_v_override.is_some() {
            let k = spec
                .cache_type_k_override
                .unwrap_or(split_kv_policy.cache_type_k());
            let v = spec
                .cache_type_v_override
                .unwrap_or(split_kv_policy.cache_type_v());
            models::gguf::GgufKvCacheQuant::from_llama_args(k, v).unwrap_or(
                models::gguf::GgufKvCacheQuant::from_llama_args(
                    split_kv_policy.cache_type_k(),
                    split_kv_policy.cache_type_v(),
                )
                .unwrap_or(models::gguf::GgufKvCacheQuant::Q8_0),
            )
        } else {
            models::gguf::GgufKvCacheQuant::from_llama_args(
                split_kv_policy.cache_type_k(),
                split_kv_policy.cache_type_v(),
            )
            .unwrap_or(models::gguf::GgufKvCacheQuant::Q8_0)
        };
    let kv_bytes_per_token = kv_cache_quant
        .kv_cache_bytes_per_token(&compact_meta)
        .context("split topology planning requires KV cache byte metadata")?;
    let planned_topology = plan_runtime_slice_topology_with_resources(
        &topology_id,
        model_ref,
        &package,
        &participant_snapshot.participants,
        &participant_snapshot.excluded,
        SplitTopologyResourceInputs {
            native_context_length: compact_meta.context_length,
            kv_bytes_per_token,
            ctx_size_override: spec.ctx_size_override,
            parallel_override: spec.parallel_override,
        },
    )?;
    let stages = planned_topology.stages;
    let planned_participants =
        split_participants_for_stages(&participant_snapshot.participants, &stages);
    anyhow::ensure!(
        split_stages_meet_minimum(&stages),
        "split runtime needs at least two stage participants"
    );
    let stage0 = stages
        .first()
        .context("split topology did not produce stage 0")?;
    tracing::info!(
        model_ref,
        topology_id,
        run_id,
        context_length = planned_topology.context_length,
        parallel_lanes = planned_topology.slots,
        local_node = %spec.node.id().fmt_short(),
        elected_coordinator = %stage0.node_id.fmt_short(),
        stages = ?split_stage_plan_labels(&stages),
        participants = ?split_participant_labels(&planned_participants),
        excluded = ?split_participant_exclusion_labels(&participant_snapshot.excluded),
        "split topology planned; elected coordinator from stage 0"
    );
    if stage0.node_id != spec.node.id() {
        tracing::info!(
            model_ref,
            topology_id,
            run_id,
            local_node = %spec.node.id().fmt_short(),
            elected_coordinator = %stage0.node_id.fmt_short(),
            "split topology election selected a remote coordinator; local node entering standby"
        );
        return Ok(SplitRuntimeStart::Standby {
            coordinator: stage0.node_id,
        });
    }
    tracing::info!(
        model_ref,
        topology_id,
        run_id,
        local_node = %spec.node.id().fmt_short(),
        context_length = planned_topology.context_length,
        parallel_lanes = planned_topology.slots,
        "split topology election selected local node as coordinator"
    );

    let ctx_size = planned_topology.context_length;
    let slots = planned_topology.slots;
    let projector_path = spec
        .mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&resolved_model_name(spec.model_path)))
        .filter(|path| path.exists())
        .map(|path| path.to_string_lossy().to_string());
    let active = SplitTopologyGeneration::new(
        topology_id.clone(),
        run_id.clone(),
        SPLIT_INITIAL_SHUTDOWN_GENERATION,
        planned_participants,
        stages,
    );
    let mut loaded = load_split_runtime_generation(SplitGenerationLoadSpec {
        node: spec.node,
        model_ref,
        model_path: spec.model_path,
        package: &package,
        generation: &active,
        projector_path: projector_path.clone(),
        ctx_size,
        cache_type_k_override: spec.cache_type_k_override,
        cache_type_v_override: spec.cache_type_v_override,
        n_batch_override: spec.n_batch_override,
        n_ubatch_override: spec.n_ubatch_override,
        flash_attention_override: spec.flash_attention_override,
        pinned_gpu: spec.pinned_gpu,
        slots,
        skippy_telemetry: spec.skippy_telemetry.clone(),
    })
    .await?;
    let (coordinator_tx, coordinator_rx) = tokio::sync::mpsc::channel(1);
    loaded.coordinator_rx = Some(coordinator_rx);
    loaded.coordinator_task = Some(spawn_split_topology_coordinator(SplitTopologyCoordinator {
        node: spec.node.clone(),
        model_name: model_ref.to_string(),
        model_path: spec.model_path.to_path_buf(),
        model_ref: model_ref.to_string(),
        package: package.clone(),
        active,
        projector_path,
        ctx_size,
        topology_resources: SplitTopologyResourceInputs {
            native_context_length: compact_meta.context_length,
            kv_bytes_per_token,
            ctx_size_override: spec.ctx_size_override,
            parallel_override: spec.parallel_override,
        },
        cache_type_k_override: spec.cache_type_k_override.map(str::to_string),
        cache_type_v_override: spec.cache_type_v_override.map(str::to_string),
        n_batch_override: spec.n_batch_override,
        n_ubatch_override: spec.n_ubatch_override,
        flash_attention_override: spec.flash_attention_override,
        pinned_gpu: spec.pinned_gpu.cloned(),
        slots,
        skippy_telemetry: spec.skippy_telemetry.clone(),
        event_tx: coordinator_tx,
    }));

    Ok(SplitRuntimeStart::Started(Box::new(loaded)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipant {
    pub(super) node_id: iroh::EndpointId,
    pub(super) vram_bytes: u64,
    first_joined_mesh_ts: Option<u64>,
    pub(super) cached_slice_bytes: u64,
    pub(super) missing_artifact_bytes: u64,
    pub(super) rtt_ms: Option<u32>,
    pub(super) artifact_transfer_supported: bool,
    availability_score: u32,
}

impl SplitParticipant {
    pub(super) fn new(
        node_id: iroh::EndpointId,
        vram_bytes: u64,
        first_joined_mesh_ts: Option<u64>,
    ) -> Self {
        Self {
            node_id,
            vram_bytes,
            first_joined_mesh_ts,
            cached_slice_bytes: 0,
            missing_artifact_bytes: 0,
            rtt_ms: None,
            artifact_transfer_supported: false,
            availability_score: 0,
        }
    }

    fn local_package(
        node_id: iroh::EndpointId,
        vram_bytes: u64,
        first_joined_mesh_ts: Option<u64>,
        package: &skippy::SkippyPackageIdentity,
    ) -> Self {
        let mut participant = Self::new(node_id, vram_bytes, first_joined_mesh_ts);
        participant.cached_slice_bytes = package.source_model_bytes;
        participant.artifact_transfer_supported = true;
        participant.availability_score = package.layer_count;
        participant
    }

    fn with_package_signals(
        mut self,
        signal: SplitParticipantPackageSignal,
        rtt_ms: Option<u32>,
        artifact_transfer_supported: bool,
    ) -> Self {
        self.cached_slice_bytes = signal.cached_slice_bytes;
        self.missing_artifact_bytes = signal.missing_artifact_bytes;
        self.availability_score = signal.availability_score;
        self.rtt_ms = rtt_ms;
        self.artifact_transfer_supported = artifact_transfer_supported;
        self
    }

    #[cfg(test)]
    fn to_topology_participant(self) -> skippy::StageTopologyParticipant {
        skippy::StageTopologyParticipant {
            node_id: self.node_id,
            vram_bytes: self.vram_bytes,
            cached_slice_bytes: self.cached_slice_bytes,
            missing_artifact_bytes: self.missing_artifact_bytes,
            rtt_ms: self.rtt_ms,
            artifact_transfer_supported: self.artifact_transfer_supported,
            availability_score: self.availability_score,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SplitParticipantPackageSignal {
    cached_slice_bytes: u64,
    missing_artifact_bytes: u64,
    availability_score: u32,
}

impl SplitParticipantPackageSignal {
    fn can_stage_with(
        self,
        package: &skippy::SkippyPackageIdentity,
        artifact_transfer_supported: bool,
    ) -> bool {
        self.missing_artifact_bytes == 0
            || artifact_transfer_supported
            || package_ref_has_independent_prepare_source(&package.package_ref)
    }
}

fn package_ref_has_independent_prepare_source(package_ref: &str) -> bool {
    // HF layer packages can be resolved by the selected worker during prepare;
    // peer artifact transfer is only an optional cache warm path.
    skippy_runtime::package::is_hf_package_ref(package_ref)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitParticipantSnapshot {
    participants: Vec<SplitParticipant>,
    excluded: Vec<SplitParticipantExclusion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipantExclusion {
    pub(super) node_id: iroh::EndpointId,
    pub(super) reason: SplitParticipantExclusionReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitParticipantExclusionReason {
    Client,
    MissingVram,
    MissingModelInterest,
    MissingModelSource,
}

impl SplitParticipantExclusionReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::MissingVram => "missing_vram",
            Self::MissingModelInterest => "missing_model_interest",
            Self::MissingModelSource => "missing_model_source",
        }
    }
}

struct SplitGenerationLoadSpec<'a> {
    node: &'a mesh::Node,
    model_ref: &'a str,
    model_path: &'a Path,
    package: &'a skippy::SkippyPackageIdentity,
    generation: &'a SplitTopologyGeneration,
    projector_path: Option<String>,
    ctx_size: u32,
    pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
    cache_type_k_override: Option<&'a str>,
    cache_type_v_override: Option<&'a str>,
    n_batch_override: Option<u32>,
    n_ubatch_override: Option<u32>,
    flash_attention_override: FlashAttentionType,
    skippy_telemetry: skippy::SkippyTelemetryOptions,
}

async fn load_split_runtime_generation(
    spec: SplitGenerationLoadSpec<'_>,
) -> Result<SplitRuntimeGenerationHandle> {
    let mut cleanup_on_error = false;
    let result = load_split_runtime_generation_inner(&spec, &mut cleanup_on_error).await;
    if let Err(error) = &result {
        if cleanup_on_error {
            tracing::warn!(
                model_ref = spec.model_ref,
                topology_id = %spec.generation.topology_id,
                run_id = %spec.generation.run_id,
                generation = spec.generation.generation,
                error = %error,
                "cleaning up split runtime generation after failed load"
            );
            stop_split_generation(spec.node, spec.generation, spec.generation.generation).await;
        }
    }
    result
}

async fn load_split_runtime_generation_inner(
    spec: &SplitGenerationLoadSpec<'_>,
    cleanup_on_error: &mut bool,
) -> Result<SplitRuntimeGenerationHandle> {
    let stage0 = spec
        .generation
        .stages
        .first()
        .context("split topology did not produce stage 0")?;
    anyhow::ensure!(
        stage0.node_id == spec.node.id(),
        "split topology stage 0 moved to {}; local coordinator is {}",
        stage0.node_id.fmt_short(),
        spec.node.id().fmt_short()
    );

    claim_split_coordinator_lease(spec.node, spec.model_ref, spec.package, spec.generation).await?;

    let mut ready_by_stage: HashMap<String, skippy::StageStatusSnapshot> = HashMap::new();
    let mut downstream: Option<skippy::StagePeerDescriptor> = None;
    let kv_cache = skippy::KvCachePolicy::for_model_size(spec.package.source_model_bytes);
    let family_policy = skippy::family_policy_for_model_path(spec.model_path, Some(spec.model_ref));
    let effective_cache_type_k = spec
        .cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k())
        .to_string();
    let effective_cache_type_v = spec
        .cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v())
        .to_string();
    let resolved_flash_attn_type =
        effective_flash_attention(spec.flash_attention_override, &effective_cache_type_v);
    tracing::info!(model = spec.model_ref, "KV cache: {}", kv_cache.label());

    // Use LayerPackage mode when we have an hf:// distributable package ref,
    // otherwise fall back to RuntimeSlice (requires full GGUF on each node).
    let use_layer_package = skippy::is_layer_package_ref(&spec.package.package_ref);
    let load_mode = if use_layer_package {
        LoadMode::LayerPackage
    } else {
        LoadMode::RuntimeSlice
    };
    let activation_width =
        skippy_stage_activation_width(spec.package.activation_width, spec.model_ref)?;

    if use_layer_package {
        spec.node
            .record_stage_topology(split_stage_topology_instance(
                &spec.generation.topology_id,
                &spec.generation.run_id,
                spec.model_ref,
                spec.package,
                &spec.generation.stages,
                &ready_by_stage,
            ))
            .await;
    }

    for stage in spec.generation.stages.iter().skip(1).rev() {
        *cleanup_on_error = true;
        let load = skippy::StageLoadRequest {
            topology_id: spec.generation.topology_id.clone(),
            run_id: spec.generation.run_id.clone(),
            model_id: spec.model_ref.to_string(),
            backend: "skippy".to_string(),
            package_ref: spec.package.package_ref.clone(),
            manifest_sha256: spec.package.manifest_sha256.clone(),
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            model_path: Some(stage_load_model_path(
                load_mode.clone(),
                &spec.package.package_ref,
                spec.model_path,
            )),
            source_model_bytes: Some(spec.package.source_model_bytes),
            projector_path: spec.projector_path.clone(),
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width,
            wire_dtype: family_policy.activation_wire_dtype,
            ctx_size: spec.ctx_size,
            lane_count: spec.slots as u32,
            n_batch: spec.n_batch_override,
            n_ubatch: spec.n_ubatch_override,
            n_gpu_layers: -1,
            cache_type_k: effective_cache_type_k.clone(),
            cache_type_v: effective_cache_type_v.clone(),
            flash_attn_type: resolved_flash_attn_type,
            shutdown_generation: spec.generation.generation,
            coordinator_term: spec.generation.coordinator_term,
            coordinator_id: Some(spec.node.id()),
            lease_until_unix_ms: spec.generation.lease_until_unix_ms,
            load_mode: load_mode.clone(),
            upstream: None,
            downstream: downstream.clone(),
        };
        prepare_split_stage(spec.node, stage.node_id, load.clone()).await?;
        wait_for_split_stage_source(
            spec.node,
            stage.node_id,
            &load,
            Duration::from_secs(30 * 60),
        )
        .await
        .with_context(|| {
            format!(
                "prepare split stage {} on {}",
                stage.stage_id,
                stage.node_id.fmt_short()
            )
        })?;
        let response = if stage.node_id == spec.node.id() {
            spec.node
                .send_local_stage_control(skippy::StageControlRequest::Load(load))
                .await
        } else {
            spec.node
                .send_stage_control(stage.node_id, skippy::StageControlRequest::Load(load))
                .await
        }
        .with_context(|| {
            format!(
                "load split stage {} on {}",
                stage.stage_id,
                stage.node_id.fmt_short()
            )
        })?;
        let skippy::StageControlResponse::Ready(ready) = response else {
            anyhow::bail!(
                "unexpected status response while loading {}",
                stage.stage_id
            );
        };
        anyhow::ensure!(
            ready.accepted,
            "stage {} rejected load: {}",
            stage.stage_id,
            ready.error.unwrap_or_else(|| "unknown error".to_string())
        );
        downstream = Some(skippy::StagePeerDescriptor {
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index,
            endpoint: ready.status.bind_addr.clone(),
            node_id: Some(stage.node_id),
        });
        ready_by_stage.insert(stage.stage_id.clone(), ready.status);
    }

    let downstream = downstream.context("split topology missing downstream stage")?;
    let downstream_endpoint = if downstream.node_id == Some(spec.node.id()) {
        downstream.endpoint
    } else {
        spec.node
            .ensure_stage_transport_bridge(
                downstream
                    .node_id
                    .context("downstream split stage is missing node id")?,
                spec.generation.topology_id.clone(),
                spec.generation.run_id.clone(),
                downstream.stage_id.clone(),
            )
            .await?
    };
    let config = split_stage0_config(
        &spec.generation.topology_id,
        &spec.generation.run_id,
        spec.model_ref,
        spec.model_path,
        spec.package,
        stage0,
        downstream.stage_id,
        downstream.stage_index,
        downstream_endpoint,
        spec.projector_path.clone(),
        spec.ctx_size,
        spec.slots as u32,
        kv_cache,
        spec.cache_type_k_override,
        spec.cache_type_v_override,
        spec.n_batch_override,
        spec.n_ubatch_override,
        spec.flash_attention_override,
        spec.pinned_gpu,
        load_mode.clone(),
    );
    let slots = spec.slots;
    let node_for_hook = spec.node.clone();
    let model_ref = spec.model_ref.to_string();
    let skippy_telemetry = spec.skippy_telemetry.clone();
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_ref.clone(),
        source: None,
    });
    let handle = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_stage0_config(
            config,
            activation_width,
            slots,
            skippy_server::CONTEXT_BUDGET_MAX_TOKENS,
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
            skippy_telemetry,
        )
    })
    .await
    .context("join load skippy stage0 config task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http(alloc_local_port().await?);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    spec.node
        .activate_stage_topology(split_stage_topology_instance(
            &spec.generation.topology_id,
            &spec.generation.run_id,
            spec.model_ref,
            spec.package,
            &spec.generation.stages,
            &ready_by_stage,
        ))
        .await;

    Ok(SplitRuntimeGenerationHandle {
        loaded_name: spec.model_ref.to_string(),
        handle: LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length: spec.ctx_size,
            slots: spec.slots,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: handle,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
        cleanup: Some(SplitGenerationCleanup {
            generation: spec.generation.clone(),
        }),
        coordinator_rx: None,
        coordinator_task: None,
    })
}

async fn claim_split_coordinator_lease(
    node: &mesh::Node,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> Result<()> {
    let claim = split_coordinator_claim(node.id(), model_ref, package, generation);
    let required_accepts = skippy_coordinator::quorum_requirement(generation.stages.len());
    let mut accepted = 0usize;
    let mut accepted_nodes = Vec::new();
    let mut errors = Vec::new();
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        coordinator = %node.id().fmt_short(),
        planned_stages = generation.stages.len(),
        required_accepts,
        stages = ?split_stage_plan_labels(&generation.stages),
        participants = ?split_participant_labels(&generation.participants),
        "claiming split topology coordinator lease"
    );

    for stage in &generation.stages {
        let request = skippy::StageControlRequest::Claim(claim.clone());
        let response = if stage.node_id == node.id() {
            node.send_local_stage_control(request).await
        } else {
            node.send_stage_control(stage.node_id, request).await
        };
        match response {
            Ok(skippy::StageControlResponse::ClaimAccepted(ack)) if ack.accepted => {
                accepted += 1;
                accepted_nodes.push(stage.node_id);
                tracing::debug!(
                    model_ref,
                    topology_id = generation.topology_id,
                    generation = generation.generation,
                    stage_id = stage.stage_id,
                    stage_node = %stage.node_id.fmt_short(),
                    "split topology coordinator claim accepted by stage"
                );
            }
            Ok(skippy::StageControlResponse::ClaimAccepted(ack)) => {
                let error = ack.error.unwrap_or_else(|| "unknown rejection".to_string());
                tracing::warn!(
                    model_ref,
                    topology_id = generation.topology_id,
                    generation = generation.generation,
                    stage_id = stage.stage_id,
                    stage_node = %stage.node_id.fmt_short(),
                    error = %error,
                    "split topology coordinator claim rejected by stage"
                );
                errors.push(format!(
                    "{} rejected claim: {}",
                    stage.node_id.fmt_short(),
                    error
                ));
            }
            Ok(other) => {
                tracing::warn!(
                    model_ref,
                    topology_id = generation.topology_id,
                    generation = generation.generation,
                    stage_id = stage.stage_id,
                    stage_node = %stage.node_id.fmt_short(),
                    response = ?other,
                    "split topology coordinator claim returned unexpected response"
                );
                errors.push(format!(
                    "{} returned unexpected claim response: {other:?}",
                    stage.node_id.fmt_short()
                ));
            }
            Err(err) => {
                tracing::warn!(
                    model_ref,
                    topology_id = generation.topology_id,
                    generation = generation.generation,
                    stage_id = stage.stage_id,
                    stage_node = %stage.node_id.fmt_short(),
                    error = %err,
                    "split topology coordinator claim failed for stage"
                );
                errors.push(format!(
                    "{} claim failed: {err:#}",
                    stage.node_id.fmt_short()
                ));
            }
        }
    }

    anyhow::ensure!(
        accepted >= required_accepts,
        "coordinator claim for {model_ref} accepted by {accepted}/{} planned stage(s), need {required_accepts}: {}",
        generation.stages.len(),
        errors.join("; ")
    );
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        accepted,
        required_accepts,
        accepted_nodes = ?split_node_labels(&accepted_nodes),
        "split topology coordinator lease quorum reached"
    );
    Ok(())
}

fn split_coordinator_claim(
    coordinator_id: iroh::EndpointId,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> skippy::StageCoordinatorClaim {
    skippy::StageCoordinatorClaim {
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        topology_id: generation.topology_id.clone(),
        run_id: generation.run_id.clone(),
        coordinator_id: coordinator_id.to_string(),
        coordinator_term: generation.coordinator_term,
        participant_set_hash: split_participant_set_hash(&generation.participants),
        topology_hash: split_topology_hash(&generation.stages),
        lease_until_unix_ms: generation.lease_until_unix_ms,
    }
}

fn stage_load_model_path(load_mode: LoadMode, package_ref: &str, model_path: &Path) -> String {
    match load_mode {
        LoadMode::LayerPackage => package_ref.to_string(),
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
            model_path.to_string_lossy().to_string()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitTopologyGeneration {
    topology_id: String,
    run_id: String,
    generation: u64,
    coordinator_term: u64,
    lease_until_unix_ms: u64,
    participants: Vec<SplitParticipant>,
    stages: Vec<RuntimeSliceStagePlan>,
}

impl SplitTopologyGeneration {
    fn new(
        topology_id: String,
        run_id: String,
        generation: u64,
        participants: Vec<SplitParticipant>,
        stages: Vec<RuntimeSliceStagePlan>,
    ) -> Self {
        Self {
            topology_id,
            run_id,
            generation,
            coordinator_term: now_unix_nanos().max(1) as u64,
            lease_until_unix_ms: split_coordinator_lease_until_unix_ms(),
            participants,
            stages,
        }
    }
}

struct SplitTopologyCoordinator {
    node: mesh::Node,
    model_name: String,
    model_path: PathBuf,
    model_ref: String,
    package: skippy::SkippyPackageIdentity,
    active: SplitTopologyGeneration,
    projector_path: Option<String>,
    ctx_size: u32,
    topology_resources: SplitTopologyResourceInputs,
    cache_type_k_override: Option<String>,
    cache_type_v_override: Option<String>,
    n_batch_override: Option<u32>,
    n_ubatch_override: Option<u32>,
    flash_attention_override: FlashAttentionType,
    pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
    skippy_telemetry: skippy::SkippyTelemetryOptions,
    event_tx: tokio::sync::mpsc::Sender<SplitCoordinatorEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitReplanDecision {
    Keep,
    Candidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitLossRecoveryDecision {
    NoActiveStageLoss,
    ReplacementSplit,
    LocalFallback,
    Withdraw,
}

fn spawn_split_topology_coordinator(
    coordinator: SplitTopologyCoordinator,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(Box::pin(coordinator.run()))
}

impl SplitTopologyCoordinator {
    async fn run(mut self) {
        let mut peer_rx = self.node.peer_change_rx.clone();
        let mut health_tick = tokio::time::interval(Duration::from_secs(30));
        health_tick.tick().await;
        tracing::info!(
            model_ref = self.model_ref,
            topology_id = self.active.topology_id,
            generation = self.active.generation,
            stages = ?split_stage_plan_labels(&self.active.stages),
            participants = ?split_participant_labels(&self.active.participants),
            "split topology coordinator active"
        );

        loop {
            tokio::select! {
                changed = peer_rx.changed() => {
                    if changed.is_err() {
                        tracing::debug!(model_ref = self.model_ref, "split topology coordinator peer watch closed");
                        break;
                    }
                    tokio::time::sleep(SPLIT_PARTICIPANT_STABLE_FOR).await;
                    while peer_rx.has_changed().unwrap_or(false) {
                        let _ = peer_rx.borrow_and_update();
                    }
                    if !self.evaluate_replan("membership_changed").await {
                        break;
                    }
                }
                _ = health_tick.tick() => {
                    if !self.evaluate_replan("periodic_check").await {
                        break;
                    }
                }
            }
        }
    }

    async fn evaluate_replan(&mut self, reason: &'static str) -> bool {
        let snapshot = collect_split_participants(
            &self.node,
            &self.model_name,
            &self.model_ref,
            &self.package,
            self.pinned_gpu.as_ref().map(|gpu| gpu.vram_bytes),
        )
        .await;
        self.node
            .refresh_stage_runtime_statuses(Duration::from_secs(2))
            .await;
        let runtime_statuses = self.node.stage_runtime_statuses().await;
        let missing_stage_nodes =
            split_missing_active_stage_nodes(&self.active, &snapshot.participants);
        let unavailable_stage_nodes = split_unavailable_active_stage_nodes(
            &self.active,
            &snapshot.participants,
            &runtime_statuses,
        );
        let planned_participants = snapshot.participants.clone();
        let candidate = if split_participants_meet_minimum(&planned_participants) {
            match self.plan_replan_candidate(&planned_participants) {
                Ok(candidate) => {
                    if candidate
                        .stages
                        .first()
                        .is_none_or(|stage0| stage0.node_id != self.node.id())
                    {
                        tracing::debug!(
                            model_ref = self.model_ref,
                            reason,
                            candidate_stages = ?split_stage_plan_labels(&candidate.stages),
                            "split topology replan skipped; stage 0 would move to another node"
                        );
                        None
                    } else {
                        Some(candidate)
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        model_ref = self.model_ref,
                        reason,
                        error = %err,
                        participants = ?split_participant_labels(&planned_participants),
                        excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
                        "split topology replan candidate failed"
                    );
                    None
                }
            }
        } else {
            tracing::debug!(
                model_ref = self.model_ref,
                reason,
                participants = ?split_participant_labels(&snapshot.participants),
                excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
                "split topology replan skipped; quorum not met"
            );
            None
        };

        match split_loss_recovery_decision(
            &self.active,
            &snapshot.participants,
            &unavailable_stage_nodes,
            candidate.as_ref(),
            self.local_model_fits(),
        ) {
            SplitLossRecoveryDecision::NoActiveStageLoss => {}
            SplitLossRecoveryDecision::ReplacementSplit => {
                let candidate = candidate.expect("replacement split decision requires a candidate");
                tracing::info!(
                    model_ref = self.model_ref,
                    reason,
                    active_topology_id = self.active.topology_id,
                    active_generation = self.active.generation,
                    candidate_topology_id = candidate.topology_id,
                    candidate_generation = candidate.generation,
                    missing_stage_nodes = ?split_node_labels(&missing_stage_nodes),
                    unavailable_stage_nodes = ?split_node_labels(&unavailable_stage_nodes),
                    active_stages = ?split_stage_plan_labels(&self.active.stages),
                    candidate_stages = ?split_stage_plan_labels(&candidate.stages),
                    participants = ?split_participant_labels(&candidate.participants),
                    "split topology lost an active stage peer; loading replacement split generation"
                );
                match self.load_and_publish_candidate(reason, candidate).await {
                    Ok(()) => return true,
                    Err(err) => {
                        tracing::warn!(
                            model_ref = self.model_ref,
                            reason,
                            error = %err,
                            "split topology replacement failed during load-and-cutover"
                        );
                    }
                }
                if self.local_model_fits() {
                    if let Err(err) = self
                        .request_local_fallback(reason, unavailable_stage_nodes.clone())
                        .await
                    {
                        tracing::warn!(
                            model_ref = self.model_ref,
                            reason,
                            error = %err,
                            "failed to publish split topology local fallback request"
                        );
                    } else {
                        return false;
                    }
                }
                if let Err(err) = self
                    .withdraw_active_generation(reason, unavailable_stage_nodes.clone())
                    .await
                {
                    tracing::warn!(
                        model_ref = self.model_ref,
                        reason,
                        error = %err,
                        "failed to publish split topology withdrawal"
                    );
                } else {
                    return false;
                }
                return true;
            }
            SplitLossRecoveryDecision::LocalFallback => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    topology_id = self.active.topology_id,
                    generation = self.active.generation,
                    missing_stage_nodes = ?split_node_labels(&missing_stage_nodes),
                    unavailable_stage_nodes = ?split_node_labels(&unavailable_stage_nodes),
                    "split topology lost an active stage peer; requesting local runtime fallback"
                );
                if let Err(err) = self
                    .request_local_fallback(reason, unavailable_stage_nodes.clone())
                    .await
                {
                    tracing::warn!(
                        model_ref = self.model_ref,
                        reason,
                        error = %err,
                        "failed to publish split topology local fallback request"
                    );
                } else {
                    return false;
                }
            }
            SplitLossRecoveryDecision::Withdraw => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    topology_id = self.active.topology_id,
                    generation = self.active.generation,
                    missing_stage_nodes = ?split_node_labels(&missing_stage_nodes),
                    unavailable_stage_nodes = ?split_node_labels(&unavailable_stage_nodes),
                    "split topology lost an active stage peer and no replacement path is available; withdrawing active generation"
                );
                if let Err(err) = self
                    .withdraw_active_generation(reason, unavailable_stage_nodes.clone())
                    .await
                {
                    tracing::warn!(
                        model_ref = self.model_ref,
                        reason,
                        error = %err,
                        "failed to publish split topology withdrawal"
                    );
                } else {
                    return false;
                }
            }
        }

        if snapshot.participants.len() < SPLIT_DEFAULT_MIN_PARTICIPANTS {
            return true;
        }

        let Some(candidate) = candidate else {
            return true;
        };

        let (replan_decision, replan_decision_reason) =
            split_replan_decision_with_reason(&self.active, &candidate);
        match replan_decision {
            SplitReplanDecision::Keep => {
                tracing::debug!(
                    model_ref = self.model_ref,
                    reason,
                    decision_reason = replan_decision_reason,
                    active_generation = self.active.generation,
                    active_stages = self.active.stages.len(),
                    candidate_stages = candidate.stages.len(),
                    active_participants = self.active.participants.len(),
                    candidate_participants = candidate.participants.len(),
                    "split topology replan skipped; candidate is not materially better"
                );
            }
            SplitReplanDecision::Candidate => {
                tracing::info!(
                    model_ref = self.model_ref,
                    reason,
                    decision_reason = replan_decision_reason,
                    active_topology_id = self.active.topology_id,
                    active_generation = self.active.generation,
                    candidate_topology_id = candidate.topology_id,
                    candidate_generation = candidate.generation,
                    active_stages = ?split_stage_plan_labels(&self.active.stages),
                    candidate_stages = ?split_stage_plan_labels(&candidate.stages),
                    participants = ?split_participant_labels(&candidate.participants),
                    "split topology replan candidate accepted; loading candidate generation"
                );
                if let Err(err) = self.load_and_publish_candidate(reason, candidate).await {
                    tracing::warn!(
                        model_ref = self.model_ref,
                        reason,
                        error = %err,
                        "split topology replan candidate failed during load-and-cutover"
                    );
                }
            }
        }
        true
    }

    fn plan_replan_candidate(
        &self,
        planned_participants: &[SplitParticipant],
    ) -> Result<SplitTopologyGeneration> {
        let generation = self.active.generation.saturating_add(1);
        let run_id = format!("mesh-split-{}-g{}", now_unix_nanos(), generation);
        let topology_id = format!("topology-{run_id}");
        let resources = SplitTopologyResourceInputs {
            ctx_size_override: Some(self.ctx_size),
            parallel_override: Some(self.slots),
            ..self.topology_resources
        };
        let planned = plan_runtime_slice_topology_with_resources(
            &topology_id,
            &self.model_ref,
            &self.package,
            planned_participants,
            &[],
            resources,
        )?;
        let stages = planned.stages;
        let participants = split_participants_for_stages(planned_participants, &stages);
        anyhow::ensure!(
            split_stages_meet_minimum(&stages),
            "split runtime needs at least two stage participants"
        );
        Ok(SplitTopologyGeneration::new(
            topology_id,
            run_id,
            generation,
            participants,
            stages,
        ))
    }

    fn local_model_fits(&self) -> bool {
        let local_capacity = self
            .pinned_gpu
            .as_ref()
            .map(|gpu| gpu.vram_bytes)
            .unwrap_or_else(|| self.node.vram_bytes());
        // Use the package's source model bytes when available — layer-package
        // refs use `hf://` pseudo-paths that `total_model_bytes` cannot stat.
        let model_bytes = if self.package.source_model_bytes > 0 {
            self.package.source_model_bytes
        } else {
            election::total_model_bytes(&self.model_path)
        };
        model_fits_runtime_capacity(model_bytes, local_capacity)
    }

    async fn load_and_publish_candidate(
        &mut self,
        reason: &'static str,
        candidate: SplitTopologyGeneration,
    ) -> Result<()> {
        let previous = self.active.clone();
        let loaded = load_split_runtime_generation(SplitGenerationLoadSpec {
            node: &self.node,
            model_ref: &self.model_ref,
            model_path: &self.model_path,
            package: &self.package,
            generation: &candidate,
            projector_path: self.projector_path.clone(),
            ctx_size: self.ctx_size,
            cache_type_k_override: self.cache_type_k_override.as_deref(),
            cache_type_v_override: self.cache_type_v_override.as_deref(),
            n_batch_override: self.n_batch_override,
            n_ubatch_override: self.n_ubatch_override,
            flash_attention_override: self.flash_attention_override,
            pinned_gpu: self.pinned_gpu.as_ref(),
            slots: self.slots,
            skippy_telemetry: self.skippy_telemetry.clone(),
        })
        .await?;
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::Replace(Box::new(SplitCoordinatorReplaceEvent {
            reason,
            generation: candidate.generation,
            loaded,
            ack: ack_tx,
        }));
        if let Err(err) = self.event_tx.send(event).await {
            let SplitCoordinatorEvent::Replace(event) = err.0 else {
                unreachable!("replace event send returned a non-replace event")
            };
            let event = *event;
            event.loaded.handle.shutdown().await;
            stop_split_generation(&self.node, &candidate, candidate.generation).await;
            anyhow::bail!("publish split topology candidate to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => {
                self.active = candidate;
                stop_split_generation(&self.node, &previous, self.active.generation).await;
                tracing::info!(
                    model_ref = self.model_ref,
                    topology_id = self.active.topology_id,
                    generation = self.active.generation,
                    stages = ?split_stage_plan_labels(&self.active.stages),
                    "split topology replan cutover complete"
                );
                Ok(())
            }
            Err(_) => {
                stop_split_generation(&self.node, &candidate, candidate.generation).await;
                anyhow::bail!("runtime loop dropped split topology candidate ack");
            }
        }
    }

    async fn request_local_fallback(
        &mut self,
        reason: &'static str,
        missing_stage_nodes: Vec<iroh::EndpointId>,
    ) -> Result<()> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::LocalFallback(SplitCoordinatorLocalFallbackEvent {
            reason,
            generation: self.active.generation,
            topology_id: self.active.topology_id.clone(),
            missing_stage_nodes,
            ack: ack_tx,
        });
        if self.event_tx.send(event).await.is_err() {
            anyhow::bail!("publish split topology local fallback to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => Ok(()),
            Err(_) => anyhow::bail!("runtime loop dropped split topology local fallback ack"),
        }
    }

    async fn withdraw_active_generation(
        &mut self,
        reason: &'static str,
        missing_stage_nodes: Vec<iroh::EndpointId>,
    ) -> Result<()> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::Withdraw(SplitCoordinatorWithdrawEvent {
            reason,
            generation: self.active.generation,
            topology_id: self.active.topology_id.clone(),
            missing_stage_nodes,
            ack: ack_tx,
        });
        if self.event_tx.send(event).await.is_err() {
            anyhow::bail!("publish split topology withdrawal to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => Ok(()),
            Err(_) => anyhow::bail!("runtime loop dropped split topology withdrawal ack"),
        }
    }
}

#[cfg(test)]
fn split_replan_decision(
    active: &SplitTopologyGeneration,
    candidate: &SplitTopologyGeneration,
) -> SplitReplanDecision {
    split_replan_decision_with_reason(active, candidate).0
}

fn split_replan_decision_with_reason(
    active: &SplitTopologyGeneration,
    candidate: &SplitTopologyGeneration,
) -> (SplitReplanDecision, &'static str) {
    if split_active_stage_participant_missing(active, &candidate.participants) {
        return (
            SplitReplanDecision::Candidate,
            "active_stage_participant_missing",
        );
    }
    if candidate.stages.len() > active.stages.len() {
        return (SplitReplanDecision::Candidate, "candidate_has_more_stages");
    }
    if candidate.participants.len() > active.participants.len()
        && candidate.stages.len() == active.stages.len()
    {
        return (
            SplitReplanDecision::Candidate,
            "candidate_has_more_participants",
        );
    }
    if split_stage_node_signature(&candidate.stages) != split_stage_node_signature(&active.stages)
        && split_stage_balance_score(&candidate.stages) < split_stage_balance_score(&active.stages)
    {
        return (SplitReplanDecision::Candidate, "candidate_improves_balance");
    }
    (SplitReplanDecision::Keep, "candidate_not_materially_better")
}

fn split_loss_recovery_decision(
    active: &SplitTopologyGeneration,
    current_participants: &[SplitParticipant],
    unavailable_stage_nodes: &[iroh::EndpointId],
    candidate: Option<&SplitTopologyGeneration>,
    local_model_fits: bool,
) -> SplitLossRecoveryDecision {
    if !split_active_stage_participant_missing(active, current_participants)
        && unavailable_stage_nodes.is_empty()
    {
        return SplitLossRecoveryDecision::NoActiveStageLoss;
    }
    if candidate.is_some_and(split_candidate_is_valid_replacement_split) {
        return SplitLossRecoveryDecision::ReplacementSplit;
    }
    if local_model_fits {
        return SplitLossRecoveryDecision::LocalFallback;
    }
    SplitLossRecoveryDecision::Withdraw
}

fn split_candidate_is_valid_replacement_split(candidate: &SplitTopologyGeneration) -> bool {
    split_participants_meet_minimum(&candidate.participants)
        && split_stages_meet_minimum(&candidate.stages)
}

fn split_participants_meet_minimum(participants: &[SplitParticipant]) -> bool {
    participants.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
}

fn split_stages_meet_minimum(stages: &[RuntimeSliceStagePlan]) -> bool {
    stages.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
}

fn split_active_stage_participant_missing(
    active: &SplitTopologyGeneration,
    current_participants: &[SplitParticipant],
) -> bool {
    !split_missing_active_stage_nodes(active, current_participants).is_empty()
}

fn split_missing_active_stage_nodes(
    active: &SplitTopologyGeneration,
    current_participants: &[SplitParticipant],
) -> Vec<iroh::EndpointId> {
    let mut missing = Vec::new();
    for stage in &active.stages {
        if current_participants
            .iter()
            .any(|participant| participant.node_id == stage.node_id)
            || missing.contains(&stage.node_id)
        {
            continue;
        }
        missing.push(stage.node_id);
    }
    missing
}

fn split_unavailable_active_stage_nodes(
    active: &SplitTopologyGeneration,
    current_participants: &[SplitParticipant],
    runtime_statuses: &[mesh::StageRuntimeStatus],
) -> Vec<iroh::EndpointId> {
    let mut unavailable = split_missing_active_stage_nodes(active, current_participants);
    for status in runtime_statuses {
        if !matches!(
            status.state,
            skippy::StageRuntimeState::Failed | skippy::StageRuntimeState::Stopped
        ) || status.topology_id != active.topology_id
            || status.run_id != active.run_id
            || active
                .stages
                .iter()
                .all(|stage| stage.stage_id != status.stage_id)
        {
            continue;
        }
        let Some(node_id) = status.node_id else {
            continue;
        };
        if !unavailable.contains(&node_id) {
            unavailable.push(node_id);
        }
    }
    unavailable
}

async fn stop_split_generation(
    node: &mesh::Node,
    generation: &SplitTopologyGeneration,
    shutdown_generation: u64,
) {
    for stage in generation.stages.iter().skip(1) {
        let stop = skippy::StageStopRequest {
            topology_id: generation.topology_id.clone(),
            run_id: generation.run_id.clone(),
            stage_id: stage.stage_id.clone(),
            shutdown_generation,
            coordinator_term: generation.coordinator_term,
        };
        let result = if stage.node_id == node.id() {
            node.send_local_stage_control(skippy::StageControlRequest::Stop(stop))
                .await
        } else {
            node.send_stage_control(stage.node_id, skippy::StageControlRequest::Stop(stop))
                .await
        };
        if let Err(err) = result {
            tracing::warn!(
                topology_id = %generation.topology_id,
                run_id = %generation.run_id,
                stage_id = %stage.stage_id,
                node = %stage.node_id.fmt_short(),
                error = %err,
                "failed to stop split stage generation"
            );
        }
        if stage.node_id != node.id() {
            node.stop_stage_transport_bridge(
                &generation.topology_id,
                &generation.run_id,
                &stage.stage_id,
            )
            .await;
        }
    }
}

fn split_stage_node_signature(stages: &[RuntimeSliceStagePlan]) -> Vec<iroh::EndpointId> {
    stages.iter().map(|stage| stage.node_id).collect()
}

fn split_stage_balance_score(stages: &[RuntimeSliceStagePlan]) -> u32 {
    let Some(min) = stages
        .iter()
        .map(|stage| stage.layer_end.saturating_sub(stage.layer_start))
        .min()
    else {
        return 0;
    };
    let max = stages
        .iter()
        .map(|stage| stage.layer_end.saturating_sub(stage.layer_start))
        .max()
        .unwrap_or(min);
    max.saturating_sub(min)
}

type SplitParticipantSignature = Vec<(String, u64, u64, u64, Option<u32>, bool, u32)>;

async fn wait_for_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    local_vram_override: Option<u64>,
    timeout: Duration,
) -> Result<SplitParticipantSnapshot> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut best: Vec<SplitParticipant> = Vec::new();
    let mut best_excluded: Vec<SplitParticipantExclusion> = Vec::new();
    let mut last_signature: SplitParticipantSignature = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    loop {
        let snapshot =
            collect_split_participants(node, model_name, model_ref, package, local_vram_override)
                .await;
        let signature = split_participant_signature(&snapshot.participants);
        let now = tokio::time::Instant::now();
        if signature != last_signature {
            stable_since = now;
            last_signature = signature;
            tracing::info!(
                model_ref,
                included = ?split_participant_labels(&snapshot.participants),
                excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
                "split topology participant set changed"
            );
        }
        if snapshot.participants.len() >= best.len() {
            best = snapshot.participants.clone();
            best_excluded = snapshot.excluded.clone();
        }

        let stable_for = now.saturating_duration_since(stable_since);
        if snapshot.participants.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
            && stable_for >= SPLIT_PARTICIPANT_STABLE_FOR
        {
            tracing::info!(
                model_ref,
                stable_for_ms = stable_for.as_millis(),
                participants = ?split_participant_labels(&snapshot.participants),
                "split topology participant set accepted"
            );
            return Ok(snapshot);
        }

        if now >= deadline {
            anyhow::ensure!(
                best.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS,
                "split runtime needs at least two participating nodes for {model_ref}; found {} eligible [{}]; excluded [{}]",
                best.len(),
                split_participant_labels(&best).join(", "),
                split_participant_exclusion_labels(&best_excluded).join(", ")
            );
            tracing::warn!(
                model_ref,
                participants = ?split_participant_labels(&best),
                excluded = ?split_participant_exclusion_labels(&best_excluded),
                "split topology participant wait timed out; using best observed set"
            );
            return Ok(SplitParticipantSnapshot {
                participants: best,
                excluded: best_excluded,
            });
        }

        tokio::time::sleep(SPLIT_PARTICIPANT_POLL_INTERVAL).await;
    }
}

async fn collect_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    local_vram_override: Option<u64>,
) -> SplitParticipantSnapshot {
    let mut participants = vec![SplitParticipant::local_package(
        node.id(),
        local_vram_override.unwrap_or_else(|| node.vram_bytes()),
        Some(node.first_joined_mesh_ts().await.unwrap_or(0)),
        package,
    )];
    let mut excluded = Vec::new();
    for peer in node.peers().await {
        if matches!(peer.role, NodeRole::Client) {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason: SplitParticipantExclusionReason::Client,
            });
            continue;
        }
        if peer.vram_bytes == 0 {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason: SplitParticipantExclusionReason::MissingVram,
            });
            continue;
        }
        let wants_model = peer
            .requested_models
            .iter()
            .any(|model| model == model_name)
            || peer.routes_model(model_ref)
            || peer.serving_models.iter().any(|model| model == model_name)
            || peer
                .available_models
                .iter()
                .any(|model| model == model_name)
            || peer
                .explicit_model_interests
                .iter()
                .any(|model| model == model_ref);
        if !wants_model {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason: SplitParticipantExclusionReason::MissingModelInterest,
            });
            continue;
        }

        if let Some(package_signal) =
            split_peer_package_signal(node, peer.id, model_ref, package).await
        {
            let artifact_transfer_allowed = node.artifact_transfer_allowed_for_peer(&peer).await;
            if package_signal.can_stage_with(package, artifact_transfer_allowed) {
                participants.push(
                    SplitParticipant::new(peer.id, peer.vram_bytes, peer.first_joined_mesh_ts)
                        .with_package_signals(
                            package_signal,
                            peer.rtt_ms,
                            artifact_transfer_allowed,
                        ),
                );
            } else {
                excluded.push(SplitParticipantExclusion {
                    node_id: peer.id,
                    reason: SplitParticipantExclusionReason::MissingModelSource,
                });
            }
        } else {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason: SplitParticipantExclusionReason::MissingModelSource,
            });
        }
    }
    participants.sort_by_key(|participant| participant.node_id.to_string());
    participants.dedup_by_key(|participant| participant.node_id);
    excluded.sort_by_key(|exclusion| exclusion.node_id.to_string());
    excluded.dedup_by_key(|exclusion| exclusion.node_id);
    SplitParticipantSnapshot {
        participants,
        excluded,
    }
}

async fn split_peer_package_signal(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
) -> Option<SplitParticipantPackageSignal> {
    let request = skippy::StageInventoryRequest {
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
    };
    let result = node
        .send_stage_control(peer_id, skippy::StageControlRequest::Inventory(request))
        .await;
    let Ok(skippy::StageControlResponse::Inventory(inventory)) = result else {
        return None;
    };
    Some(split_inventory_package_signal(&inventory, package))
}

fn split_inventory_package_signal(
    inventory: &skippy::StageLayerInventory,
    package: &skippy::SkippyPackageIdentity,
) -> SplitParticipantPackageSignal {
    let cached_slice_bytes = split_inventory_range_bytes(
        inventory
            .available_ranges
            .iter()
            .chain(inventory.ready_ranges.iter()),
        package,
    );
    let explicit_missing_bytes =
        split_inventory_range_bytes(inventory.missing_ranges.iter(), package);
    let missing_artifact_bytes = if explicit_missing_bytes > 0 {
        explicit_missing_bytes
    } else if cached_slice_bytes >= package.source_model_bytes {
        0
    } else if inventory.layer_count == 0 && cached_slice_bytes == 0 {
        package.source_model_bytes
    } else {
        package
            .source_model_bytes
            .saturating_sub(cached_slice_bytes)
    };
    SplitParticipantPackageSignal {
        cached_slice_bytes,
        missing_artifact_bytes,
        availability_score: split_inventory_covered_layers(
            inventory
                .available_ranges
                .iter()
                .chain(inventory.ready_ranges.iter()),
            package.layer_count,
        ),
    }
}

fn split_inventory_range_bytes<'a>(
    ranges: impl Iterator<Item = &'a skippy::LayerRange>,
    package: &skippy::SkippyPackageIdentity,
) -> u64 {
    if package.layer_count == 0 || package.source_model_bytes == 0 {
        return 0;
    }
    let covered_layers = u128::from(split_inventory_covered_layers(ranges, package.layer_count));
    let layer_count = u128::from(package.layer_count);
    let bytes = u128::from(package.source_model_bytes).saturating_mul(covered_layers) / layer_count;
    bytes.min(u128::from(package.source_model_bytes)) as u64
}

fn split_inventory_covered_layers<'a>(
    ranges: impl Iterator<Item = &'a skippy::LayerRange>,
    layer_count: u32,
) -> u32 {
    let mut ranges = ranges
        .filter_map(|range| {
            let start = range.layer_start.min(layer_count);
            let end = range.layer_end.min(layer_count);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    let mut covered = 0u32;
    let mut current: Option<(u32, u32)> = None;
    for (start, end) in ranges {
        match current {
            Some((current_start, current_end)) if start <= current_end => {
                current = Some((current_start, current_end.max(end)));
            }
            Some((current_start, current_end)) => {
                covered = covered.saturating_add(current_end.saturating_sub(current_start));
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        covered = covered.saturating_add(end.saturating_sub(start));
    }
    covered
}

fn split_participant_signature(participants: &[SplitParticipant]) -> SplitParticipantSignature {
    participants
        .iter()
        .map(|participant| {
            (
                participant.node_id.to_string(),
                participant.vram_bytes,
                participant.cached_slice_bytes,
                participant.missing_artifact_bytes,
                participant.rtt_ms,
                participant.artifact_transfer_supported,
                participant.availability_score,
            )
        })
        .collect()
}

fn split_participant_set_hash(participants: &[SplitParticipant]) -> String {
    let mut hasher = Sha256::new();
    for participant in split_participant_signature(participants) {
        hasher.update(participant.0.as_bytes());
        hasher.update(participant.1.to_le_bytes());
        hasher.update(participant.2.to_le_bytes());
        hasher.update(participant.3.to_le_bytes());
        hasher.update(participant.4.unwrap_or_default().to_le_bytes());
        hasher.update([u8::from(participant.5)]);
        hasher.update(participant.6.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn split_topology_hash(stages: &[RuntimeSliceStagePlan]) -> String {
    let mut hasher = Sha256::new();
    for stage in stages {
        hasher.update(stage.stage_id.as_bytes());
        hasher.update(stage.stage_index.to_le_bytes());
        hasher.update(stage.node_id.to_string().as_bytes());
        hasher.update(stage.layer_start.to_le_bytes());
        hasher.update(stage.layer_end.to_le_bytes());
        hasher.update(stage.parameter_bytes.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn split_node_labels(nodes: &[iroh::EndpointId]) -> Vec<String> {
    nodes
        .iter()
        .map(|node| node.fmt_short().to_string())
        .collect()
}

#[cfg(test)]
fn plan_runtime_slice_topology(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
) -> Result<Vec<RuntimeSliceStagePlan>> {
    plan_runtime_slice_topology_with_exclusions(topology_id, model_ref, package, participants, &[])
}

#[cfg(test)]
fn plan_runtime_slice_topology_with_exclusions(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
) -> Result<Vec<RuntimeSliceStagePlan>> {
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        "planning split runtime topology"
    );
    let topology_participants = participants
        .iter()
        .copied()
        .map(SplitParticipant::to_topology_participant)
        .collect::<Vec<_>>();
    let plan = skippy::plan_package_identity_topology(
        topology_id,
        model_ref,
        package,
        &topology_participants,
    )?;
    if !plan.diagnostics.is_empty() {
        tracing::debug!(
            topology_id,
            model_ref,
            diagnostics = ?plan.diagnostics,
            "package-aware split topology planner emitted diagnostics"
        );
    }
    let mut stages = plan
        .stages
        .into_iter()
        .map(|stage| RuntimeSliceStagePlan {
            stage_id: stage.stage_id,
            stage_index: stage.stage_index,
            node_id: stage.node_id,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            parameter_bytes: stage.parameter_bytes,
        })
        .collect::<Vec<_>>();
    stages.sort_by_key(|stage| stage.stage_index);
    validate_split_capacity(model_ref, package, participants, &stages, excluded)?;
    tracing::info!(
        topology_id,
        model_ref,
        stages = ?split_stage_plan_labels(&stages),
        "planned split runtime topology"
    );
    Ok(stages)
}

#[allow(clippy::too_many_arguments)]
fn split_stage0_config(
    topology_id: &str,
    run_id: &str,
    model_ref: &str,
    model_path: &Path,
    package: &skippy::SkippyPackageIdentity,
    stage0: &RuntimeSliceStagePlan,
    downstream_stage_id: String,
    downstream_stage_index: u32,
    downstream_endpoint: String,
    projector_path: Option<String>,
    ctx_size: u32,
    lane_count: u32,
    kv_cache: skippy::KvCachePolicy,
    cache_type_k_override: Option<&str>,
    cache_type_v_override: Option<&str>,
    n_batch_override: Option<u32>,
    n_ubatch_override: Option<u32>,
    flash_attention_override: FlashAttentionType,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    load_mode: LoadMode,
) -> StageConfig {
    let effective_cache_type_k = cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k())
        .to_string();
    let effective_cache_type_v = cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v())
        .to_string();
    let resolved_flash_attn_type =
        effective_flash_attention(flash_attention_override, &effective_cache_type_v);
    let family_policy = skippy::family_policy_for_model_path(model_path, Some(model_ref));
    let effective_model_path = if load_mode == LoadMode::LayerPackage {
        package.package_ref.clone()
    } else {
        model_path.to_string_lossy().to_string()
    };
    let mut config = StageConfig {
        run_id: run_id.to_string(),
        topology_id: topology_id.to_string(),
        model_id: model_ref.to_string(),
        package_ref: Some(package.package_ref.clone()),
        manifest_sha256: Some(package.manifest_sha256.clone()),
        source_model_path: Some(effective_model_path.clone()),
        source_model_sha256: Some(package.source_model_sha256.clone()),
        source_model_bytes: Some(package.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(effective_model_path),
        projector_path,
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        layer_start: stage0.layer_start,
        layer_end: stage0.layer_end,
        ctx_size,
        lane_count,
        n_batch: n_batch_override,
        n_ubatch: n_ubatch_override,
        n_gpu_layers: -1,
        cache_type_k: effective_cache_type_k,
        cache_type_v: effective_cache_type_v,
        flash_attn_type: resolved_flash_attn_type,
        filter_tensors_on_load: true,
        selected_device: pinned_gpu.map(|gpu| skippy_protocol::StageDevice {
            backend_device: gpu.backend_device.clone(),
            stable_id: Some(gpu.stable_id.clone()),
            index: Some(gpu.index),
            vram_bytes: Some(gpu.vram_bytes),
        }),
        kv_cache: None,
        load_mode,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: downstream_stage_id,
            stage_index: downstream_stage_index,
            endpoint: downstream_endpoint,
        }),
    };
    config.kv_cache = family_policy.stage_kv_cache_config_for_stage(&config);
    config
}

async fn prepare_split_stage(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: skippy::StageLoadRequest,
) -> Result<()> {
    let prepare = skippy::StagePrepareRequest {
        load,
        coordinator_id: Some(node.id()),
    };
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Prepare(prepare))
            .await
    } else {
        node.send_stage_control(stage_node_id, skippy::StageControlRequest::Prepare(prepare))
            .await
    }?;
    let skippy::StageControlResponse::PrepareAccepted(accepted) = response else {
        anyhow::bail!("unexpected response while preparing split stage");
    };
    anyhow::ensure!(
        accepted.accepted,
        "stage {} rejected prepare: {}",
        accepted.status.stage_id,
        accepted
            .error
            .unwrap_or_else(|| "unknown error".to_string())
    );
    Ok(())
}

async fn wait_for_split_stage_source(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
    timeout: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let inventory = query_stage_inventory(node, stage_node_id, load).await?;
        if inventory
            .available_ranges
            .iter()
            .chain(inventory.ready_ranges.iter())
            .any(|range| range.layer_start <= load.layer_start && range.layer_end >= load.layer_end)
        {
            tracing::info!(
                topology_id = %load.topology_id,
                run_id = %load.run_id,
                stage_id = %load.stage_id,
                node = %stage_node_id.fmt_short(),
                "split stage source is available; loading runtime"
            );
            return Ok(());
        }
        if let Some(failed) = inventory.preparing_ranges.iter().find(|status| {
            status.stage_id == load.stage_id
                && matches!(status.state, skippy::StagePreparationState::Failed)
        }) {
            anyhow::bail!(
                "stage {} source prepare failed: {}",
                load.stage_id,
                failed.error.as_deref().unwrap_or("unknown error")
            );
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for stage {} source availability after {:?}",
                load.stage_id,
                timeout
            );
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn query_stage_inventory(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
) -> Result<skippy::StageLayerInventory> {
    let request = skippy::StageInventoryRequest {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
    };
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Inventory(request))
            .await
    } else {
        node.send_stage_control(
            stage_node_id,
            skippy::StageControlRequest::Inventory(request),
        )
        .await
    }?;
    let skippy::StageControlResponse::Inventory(inventory) = response else {
        anyhow::bail!("unexpected response while querying stage inventory");
    };
    Ok(inventory)
}

fn split_stage_topology_instance(
    topology_id: &str,
    run_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    stages: &[RuntimeSliceStagePlan],
    ready_by_stage: &HashMap<String, skippy::StageStatusSnapshot>,
) -> mesh::StageTopologyInstance {
    mesh::StageTopologyInstance {
        topology_id: topology_id.to_string(),
        run_id: run_id.to_string(),
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        stages: stages
            .iter()
            .map(|stage| mesh::StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: mesh::StageEndpoint {
                    bind_addr: ready_by_stage
                        .get(&stage.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .unwrap_or_default(),
                },
            })
            .collect(),
    }
}

fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

async fn start_runtime_skippy_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
    plan: RuntimeResourcePlan,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let port = alloc_local_port().await?;
    let context_length = plan.context_length;
    let kv_cache = skippy::KvCachePolicy::for_model_size(spec.model_bytes);
    let effective_cache_type_k = spec
        .cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k());
    let effective_cache_type_v = spec
        .cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v());
    let resolved_flash_attn_type =
        effective_flash_attention(spec.flash_attention_override, effective_cache_type_v);
    tracing::info!(
        model = model_name,
        "KV cache: {} K + {} V, {}K context",
        effective_cache_type_k.to_ascii_uppercase(),
        effective_cache_type_v.to_ascii_uppercase(),
        context_length / 1024,
    );
    let projector_path = spec
        .mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&model_name))
        .filter(|path| path.exists());
    let mut options = skippy::SkippyModelLoadOptions::for_direct_gguf(&model_name, spec.model_path)
        .with_ctx_size(context_length)
        .with_generation_concurrency(plan.slots)
        .with_cache_types(effective_cache_type_k, effective_cache_type_v)
        .with_flash_attn_type(resolved_flash_attn_type)
        .with_telemetry(spec.skippy_telemetry.clone());
    if spec.n_batch_override.is_some() || spec.n_ubatch_override.is_some() {
        options = options.with_batch_sizes(spec.n_batch_override, spec.n_ubatch_override);
    }
    if let Some(projector_path) = projector_path {
        options = options.with_projector_path(projector_path);
    }
    if let Some(gpu) = spec.pinned_gpu {
        options = options.with_selected_device(skippy::SkippyDeviceDescriptor {
            backend_device: gpu.backend_device.clone(),
            stable_id: Some(gpu.stable_id.clone()),
            index: Some(gpu.index),
            vram_bytes: Some(gpu.vram_bytes),
        });
    }
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_name.clone(),
        source: None,
    });
    let node_for_hook = spec.node.clone();
    let skippy_model = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_with_hooks(
            options,
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
        )
    })
    .await
    .context("join load skippy direct GGUF task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_name.clone(),
        bytes: None,
    });
    let http = skippy_model.start_http(port);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            slots: plan.slots,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: skippy_model,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

async fn start_runtime_layer_package_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
    package: skippy::SkippyPackageIdentity,
    plan: RuntimeResourcePlan,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let context_length = plan.context_length;
    let kv_cache = skippy::KvCachePolicy::for_model_size(package.source_model_bytes);
    let effective_cache_type_k = spec
        .cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k())
        .to_string();
    let effective_cache_type_v = spec
        .cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v())
        .to_string();
    let resolved_flash_attn_type =
        effective_flash_attention(spec.flash_attention_override, &effective_cache_type_v);
    tracing::info!(
        model = model_name,
        "KV cache: {} K + {} V, {}K context",
        effective_cache_type_k.to_ascii_uppercase(),
        effective_cache_type_v.to_ascii_uppercase(),
        context_length / 1024,
    );
    let projector_path = spec
        .mmproj_override
        .map(|path| path.to_string_lossy().to_string())
        .or_else(|| {
            mmproj_path_for_model(&model_name).map(|path| path.to_string_lossy().to_string())
        })
        .filter(|path| Path::new(path).exists());
    let activation_width = skippy_stage_activation_width(package.activation_width, &model_name)?;
    let run_id = format!("mesh-skippy-{}", now_unix_nanos());
    let config = StageConfig {
        run_id: run_id.clone(),
        topology_id: format!("topology-{run_id}"),
        model_id: model_name.clone(),
        package_ref: Some(package.package_ref.clone()),
        manifest_sha256: Some(package.manifest_sha256.clone()),
        source_model_path: Some(package.package_ref.clone()),
        source_model_sha256: Some(package.source_model_sha256.clone()),
        source_model_bytes: Some(package.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(package.package_ref.clone()),
        projector_path,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: package.layer_count,
        ctx_size: context_length,
        lane_count: plan.slots as u32,
        n_batch: spec.n_batch_override,
        n_ubatch: spec.n_ubatch_override,
        n_gpu_layers: -1,
        cache_type_k: effective_cache_type_k,
        cache_type_v: effective_cache_type_v,
        flash_attn_type: resolved_flash_attn_type,
        filter_tensors_on_load: true,
        selected_device: spec.pinned_gpu.map(|gpu| skippy_protocol::StageDevice {
            backend_device: gpu.backend_device.clone(),
            stable_id: Some(gpu.stable_id.clone()),
            index: Some(gpu.index),
            vram_bytes: Some(gpu.vram_bytes),
        }),
        kv_cache: None,
        load_mode: LoadMode::LayerPackage,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    };
    let slots = plan.slots;
    let node_for_hook = spec.node.clone();
    let model_ref = model_name.clone();
    let skippy_telemetry = spec.skippy_telemetry.clone();
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_ref.clone(),
        source: None,
    });
    let handle = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_stage0_config(
            config,
            activation_width,
            slots,
            skippy_server::CONTEXT_BUDGET_MAX_TOKENS,
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
            skippy_telemetry,
        )
    })
    .await
    .context("join load skippy layer package task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http(alloc_local_port().await?);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            slots: plan.slots,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: handle,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

pub(super) fn local_process_payload(
    model_name: &str,
    instance_id: Option<&str>,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> api::RuntimeProcessPayload {
    local_process_snapshot(
        model_name,
        instance_id,
        backend,
        port,
        pid,
        slots,
        context_length,
    )
    .to_payload()
}

pub(super) fn local_process_snapshot(
    model_name: &str,
    instance_id: Option<&str>,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> crate::runtime_data::RuntimeProcessSnapshot {
    crate::runtime_data::RuntimeProcessSnapshot {
        model: model_name.to_string(),
        instance_id: instance_id.map(str::to_string),
        backend: backend.into(),
        pid,
        slots,
        port,
        context_length: Some(context_length),
        command: None,
        state: "ready".into(),
        start: None,
        health: Some("ready".into()),
    }
}

fn skippy_stage_activation_width(activation_width: u32, model_ref: &str) -> Result<i32> {
    i32::try_from(activation_width).with_context(|| {
        format!(
            "activation width {activation_width} for {model_ref} exceeds skippy stage ABI limit"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::sync::{Arc, Mutex as StdMutex};

    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn package(layer_count: u32) -> skippy::SkippyPackageIdentity {
        skippy::SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "manifest".to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "source".to_string(),
            source_model_bytes: u64::from(layer_count) * 1_000_000,
            source_files: Vec::new(),
            layer_count,
            activation_width: 2048,
            tensor_count: 100,
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn write_test_layer_package(dir: &Path, source_model_bytes: u64) {
        fs::create_dir_all(dir.join("layers")).unwrap();
        fs::write(dir.join("metadata.gguf"), b"metadata").unwrap();
        fs::write(dir.join("embeddings.gguf"), b"embeddings").unwrap();
        fs::write(dir.join("output.gguf"), b"output").unwrap();
        fs::write(dir.join("layers/00000.gguf"), b"layer0").unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "meshllm/test-layer-package",
            "source_model": {
                "path": "/models/test-layer-package.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "files": [{
                    "path": "/models/test-layer-package.gguf",
                    "size_bytes": source_model_bytes,
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }]
            },
            "format": "layer-package",
            "layer_count": 1,
            "activation_width": 4096,
            "shared": {
                "metadata": {
                    "path": "metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                },
                "embeddings": {
                    "path": "embeddings.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 10,
                    "sha256": sha256_hex(b"embeddings")
                },
                "output": {
                    "path": "output.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 6,
                    "sha256": sha256_hex(b"output")
                }
            },
            "layers": [{
                "layer_index": 0,
                "path": "layers/00000.gguf",
                "tensor_count": 1,
                "tensor_bytes": 1,
                "artifact_bytes": 6,
                "sha256": sha256_hex(b"layer0")
            }],
            "skippy_abi_version": "0.1.0",
        });
        fs::write(
            dir.join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn participant(seed: u8) -> SplitParticipant {
        SplitParticipant::new(make_id(seed), 24_000_000_000, None)
    }

    fn stage(
        seed: u8,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> RuntimeSliceStagePlan {
        RuntimeSliceStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index,
            node_id: make_id(seed),
            layer_start,
            layer_end,
            parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
        }
    }

    fn runtime_status_for_stage(
        generation: &SplitTopologyGeneration,
        stage: &RuntimeSliceStagePlan,
        state: skippy::StageRuntimeState,
    ) -> mesh::StageRuntimeStatus {
        mesh::StageRuntimeStatus {
            topology_id: generation.topology_id.clone(),
            run_id: generation.run_id.clone(),
            model_id: "model-a".to_string(),
            backend: "skippy".to_string(),
            package_ref: Some("gguf:///model.gguf".to_string()),
            manifest_sha256: Some("direct-gguf:1:model.gguf".to_string()),
            source_model_path: Some("/model.gguf".to_string()),
            source_model_sha256: None,
            source_model_bytes: Some(1),
            materialized_path: None,
            materialized_pinned: false,
            projector_path: None,
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index,
            node_id: Some(stage.node_id),
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            state,
            bind_addr: "127.0.0.1:31000".to_string(),
            activation_width: 896,
            wire_dtype: skippy::StageWireDType::F16,
            selected_device: None,
            ctx_size: 512,
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            flash_attn_type: FlashAttentionType::Auto,
            error: None,
            shutdown_generation: generation.generation,
        }
    }

    fn local_stage(
        node_id: iroh::EndpointId,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> RuntimeSliceStagePlan {
        RuntimeSliceStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index,
            node_id,
            layer_start,
            layer_end,
            parameter_bytes: u64::from(layer_end.saturating_sub(layer_start)) * 1_000_000,
        }
    }

    fn test_stage_status_from_load(
        load: &skippy::StageLoadRequest,
        state: skippy::StageRuntimeState,
    ) -> skippy::StageStatusSnapshot {
        skippy::StageStatusSnapshot {
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            model_id: load.model_id.clone(),
            backend: load.backend.clone(),
            package_ref: Some(load.package_ref.clone()),
            manifest_sha256: Some(load.manifest_sha256.clone()),
            source_model_path: load.model_path.clone(),
            source_model_sha256: None,
            source_model_bytes: load.source_model_bytes,
            materialized_path: None,
            materialized_pinned: false,
            projector_path: load.projector_path.clone(),
            stage_id: load.stage_id.clone(),
            stage_index: load.stage_index,
            layer_start: load.layer_start,
            layer_end: load.layer_end,
            state,
            bind_addr: "127.0.0.1:31000".to_string(),
            activation_width: load.activation_width as u32,
            wire_dtype: load.wire_dtype,
            selected_device: load.selected_device.clone(),
            ctx_size: load.ctx_size,
            lane_count: load.lane_count,
            n_batch: load.n_batch,
            n_ubatch: load.n_ubatch,
            flash_attn_type: load.flash_attn_type,
            error: None,
            shutdown_generation: load.shutdown_generation,
            coordinator_term: load.coordinator_term,
            coordinator_id: load.coordinator_id,
            lease_until_unix_ms: load.lease_until_unix_ms,
        }
    }

    fn test_stage_status_from_stop(stop: &skippy::StageStopRequest) -> skippy::StageStatusSnapshot {
        skippy::StageStatusSnapshot {
            topology_id: stop.topology_id.clone(),
            run_id: stop.run_id.clone(),
            model_id: String::new(),
            backend: "skippy".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            projector_path: None,
            stage_id: stop.stage_id.clone(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 0,
            state: skippy::StageRuntimeState::Stopped,
            bind_addr: String::new(),
            activation_width: 0,
            wire_dtype: skippy::StageWireDType::F16,
            selected_device: None,
            ctx_size: 0,
            lane_count: 0,
            n_batch: None,
            n_ubatch: None,
            flash_attn_type: FlashAttentionType::Auto,
            error: None,
            shutdown_generation: stop.shutdown_generation,
            coordinator_term: stop.coordinator_term,
            coordinator_id: None,
            lease_until_unix_ms: 0,
        }
    }

    fn test_preparation_status_from_load(
        load: &skippy::StageLoadRequest,
    ) -> skippy::StagePreparationStatus {
        skippy::StagePreparationStatus {
            topology_id: load.topology_id.clone(),
            run_id: load.run_id.clone(),
            model_id: load.model_id.clone(),
            backend: load.backend.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
            stage_id: load.stage_id.clone(),
            stage_index: load.stage_index,
            layer_start: load.layer_start,
            layer_end: load.layer_end,
            state: skippy::StagePreparationState::Available,
            bytes_done: load.source_model_bytes,
            bytes_total: load.source_model_bytes,
            bind_addr: None,
            error: None,
            shutdown_generation: load.shutdown_generation,
            coordinator_term: load.coordinator_term,
            coordinator_id: load.coordinator_id,
            lease_until_unix_ms: load.lease_until_unix_ms,
        }
    }

    fn test_inventory_from_request(
        request: &skippy::StageInventoryRequest,
    ) -> skippy::StageLayerInventory {
        skippy::StageLayerInventory {
            model_id: request.model_id.clone(),
            package_ref: request.package_ref.clone(),
            manifest_sha256: request.manifest_sha256.clone(),
            layer_count: 40,
            ready_ranges: Vec::new(),
            available_ranges: vec![skippy::LayerRange {
                layer_start: 0,
                layer_end: 40,
            }],
            missing_ranges: Vec::new(),
            preparing_ranges: Vec::new(),
            source_model_path: Some("/models/qwen.gguf".to_string()),
            source_model_bytes: Some(40_000_000),
            source_model_kind: skippy::SourceModelKind::LayerPackage,
        }
    }

    #[test]
    fn runtime_local_targets_keep_duplicate_same_model_ports() {
        let (target_tx, _target_rx) =
            tokio::sync::watch::channel(election::ModelTargets::default());
        let target_tx = std::sync::Arc::new(target_tx);

        add_runtime_local_target(&target_tx, "Qwen", 41001);
        add_runtime_local_target(&target_tx, "Qwen", 41002);
        add_runtime_local_target(&target_tx, "Qwen", 41002);

        let targets = target_tx.borrow().candidates("Qwen");
        assert_eq!(
            targets,
            vec![
                election::InferenceTarget::Local(41002),
                election::InferenceTarget::Local(41001),
            ]
        );
    }

    #[test]
    fn split_topology_planner_uses_all_eligible_participants() {
        let participants = vec![
            SplitParticipant::new(make_id(1), 16_000_000_000, None),
            SplitParticipant::new(make_id(2), 24_000_000_000, None),
            SplitParticipant::new(make_id(3), 32_000_000_000, None),
            SplitParticipant::new(make_id(4), 48_000_000_000, None),
        ];

        let stages = plan_runtime_slice_topology(
            "topology-test",
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
            &package(40),
            &participants,
        )
        .expect("topology plan");

        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].stage_index, 0);
        assert_eq!(stages[3].stage_index, 3);
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.stage_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(stages.first().unwrap().layer_start, 0);
        assert_eq!(stages.last().unwrap().layer_end, 40);
    }

    #[test]
    fn split_topology_planner_prefers_cached_participant_in_runtime_path() {
        let cold = SplitParticipant::new(make_id(1), 24_000_000_000, None).with_package_signals(
            SplitParticipantPackageSignal {
                cached_slice_bytes: 0,
                missing_artifact_bytes: 40_000_000,
                availability_score: 0,
            },
            Some(80),
            true,
        );
        let warm = SplitParticipant::new(make_id(2), 24_000_000_000, None).with_package_signals(
            SplitParticipantPackageSignal {
                cached_slice_bytes: 40_000_000,
                missing_artifact_bytes: 0,
                availability_score: 40,
            },
            Some(5),
            true,
        );

        let stages = plan_runtime_slice_topology(
            "topology-test",
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
            &package(40),
            &[cold, warm],
        )
        .expect("package-aware topology plan");

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].node_id, make_id(2));
        assert_eq!((stages[0].layer_start, stages[0].layer_end), (0, 20));
    }

    #[test]
    fn split_inventory_package_signal_counts_cached_and_missing_ranges() {
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 1_000,
            layer_count: 10,
            ..package(10)
        };
        let inventory = skippy::StageLayerInventory {
            model_id: "model-a".to_string(),
            package_ref: package.package_ref.clone(),
            manifest_sha256: package.manifest_sha256.clone(),
            layer_count: 10,
            ready_ranges: vec![skippy::LayerRange {
                layer_start: 4,
                layer_end: 6,
            }],
            available_ranges: vec![skippy::LayerRange {
                layer_start: 0,
                layer_end: 4,
            }],
            missing_ranges: vec![skippy::LayerRange {
                layer_start: 6,
                layer_end: 10,
            }],
            preparing_ranges: Vec::new(),
            source_model_path: None,
            source_model_bytes: None,
            source_model_kind: skippy::SourceModelKind::LayerPackage,
        };

        let signal = split_inventory_package_signal(&inventory, &package);

        assert_eq!(
            signal,
            SplitParticipantPackageSignal {
                cached_slice_bytes: 600,
                missing_artifact_bytes: 400,
                availability_score: 6,
            }
        );
        assert!(signal.can_stage_with(&package, true));
        assert!(!signal.can_stage_with(&package, false));
    }

    #[test]
    fn split_package_signal_allows_hf_fallback_when_peer_transfer_is_disabled() {
        let mut package = skippy::SkippyPackageIdentity {
            source_model_bytes: 1_000,
            layer_count: 10,
            ..package(10)
        };
        package.package_ref = "hf://meshllm/demo-layer-package@abc123".to_string();
        let signal = SplitParticipantPackageSignal {
            cached_slice_bytes: 200,
            missing_artifact_bytes: 800,
            availability_score: 2,
        };

        assert!(signal.can_stage_with(&package, false));
    }

    #[test]
    fn split_package_signal_still_requires_transfer_for_missing_local_package() {
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 1_000,
            layer_count: 10,
            ..package(10)
        };
        let signal = SplitParticipantPackageSignal {
            cached_slice_bytes: 200,
            missing_artifact_bytes: 800,
            availability_score: 2,
        };

        assert!(!signal.can_stage_with(&package, false));
        assert!(signal.can_stage_with(&package, true));
    }

    #[test]
    fn split_inventory_package_signal_treats_unknown_inventory_as_missing_package() {
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 1_000,
            layer_count: 10,
            ..package(10)
        };
        let inventory = skippy::StageLayerInventory {
            model_id: "model-a".to_string(),
            package_ref: package.package_ref.clone(),
            manifest_sha256: package.manifest_sha256.clone(),
            layer_count: 0,
            ready_ranges: Vec::new(),
            available_ranges: Vec::new(),
            missing_ranges: Vec::new(),
            preparing_ranges: Vec::new(),
            source_model_path: None,
            source_model_bytes: None,
            source_model_kind: skippy::SourceModelKind::Unknown,
        };

        let signal = split_inventory_package_signal(&inventory, &package);

        assert_eq!(
            signal,
            SplitParticipantPackageSignal {
                cached_slice_bytes: 0,
                missing_artifact_bytes: 1_000,
                availability_score: 0,
            }
        );
    }

    #[test]
    fn startup_runtime_plan_auto_splits_when_model_exceeds_local_capacity() {
        assert_eq!(
            startup_runtime_plan(false, 3_000_000_000, 4_800_000_000),
            StartupRuntimePlan::Split {
                reason: SplitRuntimeReason::LocalCapacity
            }
        );
    }

    #[test]
    fn runtime_model_planning_bytes_uses_layer_package_source_model_bytes() {
        let dir = tempfile::tempdir().unwrap();
        write_test_layer_package(dir.path(), 4_800_000_000);

        let model_bytes = runtime_model_planning_bytes(dir.path()).unwrap();

        assert_eq!(model_bytes, 4_800_000_000);
        assert_eq!(
            startup_runtime_plan(false, 3_000_000_000, model_bytes),
            StartupRuntimePlan::Split {
                reason: SplitRuntimeReason::LocalCapacity
            }
        );
    }

    #[test]
    fn startup_runtime_plan_keeps_local_when_model_fits_without_split_flag() {
        assert_eq!(
            startup_runtime_plan(false, 6_000_000_000, 4_800_000_000),
            StartupRuntimePlan::Local
        );
    }

    #[test]
    fn startup_runtime_plan_respects_explicit_split_for_fitting_model() {
        assert_eq!(
            startup_runtime_plan(true, 6_000_000_000, 4_800_000_000),
            StartupRuntimePlan::Split {
                reason: SplitRuntimeReason::Forced
            }
        );
    }

    #[test]
    fn split_topology_planner_accepts_constrained_nodes_with_enough_aggregate_capacity() {
        let participants = vec![
            SplitParticipant::new(make_id(1), 3_000_000_000, None),
            SplitParticipant::new(make_id(2), 3_000_000_000, None),
        ];
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 4_800_000_000,
            layer_count: 48,
            ..package(48)
        };

        let stages = plan_runtime_slice_topology(
            "topology-test",
            "Hermes-2-Pro-Mistral-7B-Q4_K_M",
            &package,
            &participants,
        )
        .expect("constrained nodes should form a split topology");

        assert_eq!(stages.len(), 2);
        assert_eq!(
            stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 24), (24, 48)]
        );
    }

    #[test]
    fn split_topology_planner_rejects_insufficient_aggregate_capacity() {
        let participants = vec![
            SplitParticipant::new(make_id(1), 2_000_000_000, None),
            SplitParticipant::new(make_id(2), 2_000_000_000, None),
        ];
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 4_800_000_000,
            layer_count: 48,
            ..package(48)
        };

        let error = plan_runtime_slice_topology(
            "topology-test",
            "Hermes-2-Pro-Mistral-7B-Q4_K_M",
            &package,
            &participants,
        )
        .expect_err("aggregate split capacity should be enforced")
        .to_string();

        assert!(error.contains("aggregate split capacity"));
        // Validation uses raw model weight (4.8GB) without the old 10%
        // headroom that was removed to avoid double-counting the topology
        // planner's own VRAM budget.
        assert!(error.contains("requires 4.8GB"));
        assert!(error.contains("has 4.0GB"));
        assert!(error.contains("short by 0.8GB"));
        assert!(error.contains("participants ["));
        assert!(error.contains(&format!("{}:2.0GB", make_id(1).fmt_short())));
        assert!(error.contains(&format!("{}:2.0GB", make_id(2).fmt_short())));
    }

    #[test]
    fn split_topology_planner_rejects_stage_that_exceeds_participant_capacity() {
        // Node 2 has 150 bytes but the planner assigns it at least 2 layers
        // (200 bytes), which exceeds its capacity.  The previous version of
        // this test used 200 bytes for node 2 which passes now that the old
        // 10% headroom is no longer applied on top of the planner budget.
        let participants = vec![
            SplitParticipant::new(make_id(1), 900, None),
            SplitParticipant::new(make_id(2), 150, None),
        ];
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 1_000,
            layer_count: 10,
            ..package(10)
        };

        let error = plan_runtime_slice_topology(
            "topology-test",
            "tiny-capacity-test",
            &package,
            &participants,
        )
        .expect_err("per-stage split capacity should be enforced")
        .to_string();

        assert!(error.contains("stage-1"));
        assert!(error.contains("exceeds node capacity"));
    }

    #[test]
    fn aggregate_split_capacity_error_reports_excluded_peers() {
        let participants = vec![SplitParticipant::new(make_id(1), 2_000_000_000, None)];
        let excluded = vec![
            SplitParticipantExclusion {
                node_id: make_id(2),
                reason: SplitParticipantExclusionReason::MissingModelInterest,
            },
            SplitParticipantExclusion {
                node_id: make_id(3),
                reason: SplitParticipantExclusionReason::MissingModelSource,
            },
        ];

        let error = format_aggregate_split_capacity_error(
            "Hermes-2-Pro-Mistral-7B-Q4_K_M",
            5_280_000_000,
            2_000_000_000,
            &participants,
            &excluded,
        );

        assert!(error.contains("short by 3.3GB"));
        assert!(error.contains("excluded ["));
        assert!(error.contains(&format!(
            "{}:missing_model_interest",
            make_id(2).fmt_short()
        )));
        assert!(error.contains(&format!("{}:missing_model_source", make_id(3).fmt_short())));
    }

    #[test]
    fn split_topology_planner_reports_exclusions_on_capacity_failure() {
        let participants = vec![
            SplitParticipant::new(make_id(1), 2_000_000_000, None),
            SplitParticipant::new(make_id(2), 2_000_000_000, None),
        ];
        let excluded = vec![SplitParticipantExclusion {
            node_id: make_id(3),
            reason: SplitParticipantExclusionReason::MissingModelInterest,
        }];
        let package = skippy::SkippyPackageIdentity {
            source_model_bytes: 4_800_000_000,
            layer_count: 48,
            ..package(48)
        };

        let error = plan_runtime_slice_topology_with_exclusions(
            "topology-test",
            "Hermes-2-Pro-Mistral-7B-Q4_K_M",
            &package,
            &participants,
            &excluded,
        )
        .expect_err("aggregate split capacity should be enforced")
        .to_string();

        // Raw model weight (4.8GB) minus aggregate VRAM (4.0GB) = 0.8GB
        // shortfall, without the old 10% headroom.
        assert!(error.contains("short by 0.8GB"));
        assert!(error.contains("excluded ["));
        assert!(error.contains(&format!(
            "{}:missing_model_interest",
            make_id(3).fmt_short()
        )));
    }

    #[test]
    fn stage_load_model_path_uses_local_path_outside_layer_packages() {
        let model_path = PathBuf::from("/models/runtime-slice.gguf");

        let layer_package = stage_load_model_path(
            LoadMode::LayerPackage,
            "hf://meshllm/demo-package",
            &model_path,
        );
        assert_eq!(layer_package, "hf://meshllm/demo-package");

        for mode in [LoadMode::RuntimeSlice, LoadMode::ArtifactSlice] {
            let path = stage_load_model_path(mode, "hf://meshllm/demo-package", &model_path);
            assert_eq!(path, "/models/runtime-slice.gguf");
        }
    }

    #[test]
    fn skippy_stage_activation_width_rejects_i32_overflow() {
        let error = skippy_stage_activation_width(i32::MAX as u32 + 1, "overflow-model")
            .unwrap_err()
            .to_string();

        assert!(error.contains("exceeds skippy stage ABI limit"));
        assert!(error.contains("overflow-model"));
    }

    #[test]
    fn split_participant_signature_includes_vram_for_stability() {
        let node_id = make_id(9);
        let first = vec![SplitParticipant::new(node_id, 16_000_000_000, None)];
        let second = vec![SplitParticipant::new(node_id, 24_000_000_000, None)];

        assert_ne!(
            split_participant_signature(&first),
            split_participant_signature(&second)
        );
    }

    #[test]
    fn split_participant_signature_includes_package_signals_for_stability() {
        let node_id = make_id(9);
        let first = vec![SplitParticipant::new(node_id, 24_000_000_000, None)];
        let second = vec![SplitParticipant::new(node_id, 24_000_000_000, None)
            .with_package_signals(
                SplitParticipantPackageSignal {
                    cached_slice_bytes: 12_000_000,
                    missing_artifact_bytes: 0,
                    availability_score: 12,
                },
                Some(20),
                true,
            )];

        assert_ne!(
            split_participant_signature(&first),
            split_participant_signature(&second)
        );
    }

    #[test]
    fn split_missing_active_stage_nodes_ignores_unused_lost_participants() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
        );
        let current_participants = vec![participant(1)];

        assert_eq!(
            split_missing_active_stage_nodes(&active, &current_participants),
            vec![make_id(2)]
        );
    }

    #[test]
    fn split_unavailable_active_stage_nodes_includes_failed_stage_without_missing_peer() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
        );
        let statuses = vec![runtime_status_for_stage(
            &active,
            &active.stages[1],
            skippy::StageRuntimeState::Failed,
        )];

        assert_eq!(
            split_unavailable_active_stage_nodes(
                &active,
                &[participant(1), participant(2), participant(3)],
                &statuses,
            ),
            vec![make_id(2)]
        );
    }

    #[tokio::test]
    async fn load_split_runtime_generation_stops_candidate_stages_after_partial_load_failure() {
        let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
            .await
            .unwrap();
        let (control_tx, mut control_rx) =
            tokio::sync::mpsc::unbounded_channel::<skippy::StageControlCommand>();
        node.set_stage_control_sender(control_tx).await;

        let requests = Arc::new(StdMutex::new(Vec::new()));
        let captured_requests = Arc::clone(&requests);
        tokio::spawn(async move {
            while let Some(command) = control_rx.recv().await {
                captured_requests
                    .lock()
                    .unwrap()
                    .push(command.request.clone());
                let response = match &command.request {
                    skippy::StageControlRequest::Prepare(prepare) => {
                        Ok(skippy::StageControlResponse::PrepareAccepted(
                            skippy::StagePrepareAcceptedResponse {
                                accepted: true,
                                status: test_preparation_status_from_load(&prepare.load),
                                error: None,
                            },
                        ))
                    }
                    skippy::StageControlRequest::Inventory(inventory) => {
                        Ok(skippy::StageControlResponse::Inventory(
                            test_inventory_from_request(inventory),
                        ))
                    }
                    skippy::StageControlRequest::Claim(claim) => {
                        Ok(skippy::StageControlResponse::ClaimAccepted(
                            skippy::StageCoordinatorClaimAck {
                                accepted: true,
                                claim: claim.clone(),
                                error: None,
                            },
                        ))
                    }
                    skippy::StageControlRequest::Load(load) if load.stage_id == "stage-1" => {
                        Err(anyhow::anyhow!("injected stage load failure"))
                    }
                    skippy::StageControlRequest::Load(load) => Ok(
                        skippy::StageControlResponse::Ready(skippy::StageReadyResponse {
                            accepted: true,
                            status: test_stage_status_from_load(
                                load,
                                skippy::StageRuntimeState::Ready,
                            ),
                            error: None,
                        }),
                    ),
                    skippy::StageControlRequest::Stop(stop) => Ok(
                        skippy::StageControlResponse::Ready(skippy::StageReadyResponse {
                            accepted: true,
                            status: test_stage_status_from_stop(stop),
                            error: None,
                        }),
                    ),
                    other => panic!("unexpected stage control request: {other:?}"),
                };
                let _ = command.resp.send(response);
            }
        });

        let mut package = package(40);
        package.package_ref = "hf://Mesh-LLM/test-split-package".to_string();
        let local_id = node.id();
        let generation = SplitTopologyGeneration::new(
            "candidate-topology".into(),
            "candidate-run".into(),
            2,
            vec![SplitParticipant::new(local_id, 24_000_000_000, None)],
            vec![
                local_stage(local_id, 0, 0, 12),
                local_stage(local_id, 1, 12, 24),
                local_stage(local_id, 2, 24, 40),
            ],
        );

        let error = match Box::pin(load_split_runtime_generation(SplitGenerationLoadSpec {
            node: &node,
            model_ref: "Qwen",
            model_path: Path::new("/models/qwen.gguf"),
            package: &package,
            generation: &generation,
            projector_path: None,
            ctx_size: 4096,
            pinned_gpu: None,
            slots: 1,
            cache_type_k_override: None,
            cache_type_v_override: None,
            n_batch_override: None,
            n_ubatch_override: None,
            flash_attention_override: FlashAttentionType::Auto,
            skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        }))
        .await
        {
            Ok(_) => panic!("candidate split generation load unexpectedly succeeded"),
            Err(error) => error,
        };

        let error_chain = format!("{error:#}");
        assert!(
            error_chain.contains("injected stage load failure"),
            "unexpected error: {error_chain}"
        );

        let requests = requests.lock().unwrap();
        let claim_count = requests
            .iter()
            .filter(|request| matches!(request, skippy::StageControlRequest::Claim(_)))
            .count();
        assert_eq!(claim_count, generation.stages.len());
        let load_stage_ids = requests
            .iter()
            .filter_map(|request| match request {
                skippy::StageControlRequest::Load(load) => Some(load.stage_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(load_stage_ids, vec!["stage-2", "stage-1"]);

        let stop_requests = requests
            .iter()
            .filter_map(|request| match request {
                skippy::StageControlRequest::Stop(stop) => Some(stop),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stop_requests.len(), 2);
        assert_eq!(stop_requests[0].stage_id, "stage-1");
        assert_eq!(stop_requests[1].stage_id, "stage-2");
        assert!(stop_requests.iter().all(|stop| {
            stop.topology_id == generation.topology_id
                && stop.run_id == generation.run_id
                && stop.shutdown_generation == generation.generation
        }));
    }

    #[test]
    fn split_replan_decision_accepts_more_stage_capacity() {
        let participants = vec![SplitParticipant::new(make_id(1), 16_000_000_000, None)];
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            participants.clone(),
            vec![RuntimeSliceStagePlan {
                stage_id: "stage-0".into(),
                stage_index: 0,
                node_id: make_id(1),
                layer_start: 0,
                layer_end: 40,
                parameter_bytes: 40_000_000,
            }],
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            participants,
            vec![
                RuntimeSliceStagePlan {
                    stage_id: "stage-0".into(),
                    stage_index: 0,
                    node_id: make_id(1),
                    layer_start: 0,
                    layer_end: 16,
                    parameter_bytes: 16_000_000,
                },
                RuntimeSliceStagePlan {
                    stage_id: "stage-1".into(),
                    stage_index: 1,
                    node_id: make_id(2),
                    layer_start: 16,
                    layer_end: 40,
                    parameter_bytes: 24_000_000,
                },
            ],
        );

        assert_eq!(
            split_replan_decision(&active, &candidate),
            SplitReplanDecision::Candidate
        );
        assert_eq!(
            split_replan_decision_with_reason(&active, &candidate),
            (SplitReplanDecision::Candidate, "candidate_has_more_stages")
        );
    }

    #[test]
    fn split_replan_decision_keeps_equivalent_topology() {
        let stages = vec![RuntimeSliceStagePlan {
            stage_id: "stage-0".into(),
            stage_index: 0,
            node_id: make_id(1),
            layer_start: 0,
            layer_end: 40,
            parameter_bytes: 40_000_000,
        }];
        let participants = vec![SplitParticipant::new(make_id(1), 16_000_000_000, None)];
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            participants.clone(),
            stages.clone(),
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            participants,
            stages,
        );

        assert_eq!(
            split_replan_decision(&active, &candidate),
            SplitReplanDecision::Keep
        );
        assert_eq!(
            split_replan_decision_with_reason(&active, &candidate),
            (SplitReplanDecision::Keep, "candidate_not_materially_better")
        );
    }

    #[test]
    fn split_replan_decision_accepts_degraded_topology_when_active_stage_peer_is_lost() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1), participant(3)],
            vec![stage(1, 0, 0, 15), stage(3, 1, 15, 30)],
        );

        assert_eq!(
            split_replan_decision(&active, &candidate),
            SplitReplanDecision::Candidate
        );
    }

    #[test]
    fn split_replan_decision_keeps_topology_when_only_unused_participant_is_lost() {
        let active_stages = vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)];
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            active_stages.clone(),
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1), participant(2)],
            active_stages,
        );

        assert_eq!(
            split_replan_decision(&active, &candidate),
            SplitReplanDecision::Keep
        );
    }

    #[test]
    fn split_loss_recovery_uses_replacement_split_when_active_stage_peer_is_lost() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1), participant(3)],
            vec![stage(1, 0, 0, 15), stage(3, 1, 15, 30)],
        );

        assert_eq!(
            split_loss_recovery_decision(
                &active,
                &[participant(1), participant(3)],
                &[],
                Some(&candidate),
                true,
            ),
            SplitLossRecoveryDecision::ReplacementSplit
        );
    }

    #[test]
    fn split_loss_recovery_uses_replacement_split_when_active_stage_has_failed() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1), participant(2), participant(3)],
            vec![stage(1, 0, 0, 15), stage(2, 1, 15, 30)],
        );
        assert_eq!(
            split_loss_recovery_decision(
                &active,
                &[participant(1), participant(2), participant(3)],
                &[make_id(2)],
                Some(&candidate),
                true,
            ),
            SplitLossRecoveryDecision::ReplacementSplit
        );
    }

    #[test]
    fn split_loss_recovery_falls_back_to_local_when_replacement_split_is_unavailable() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2)],
            vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
        );

        assert_eq!(
            split_loss_recovery_decision(&active, &[participant(1)], &[], None, true),
            SplitLossRecoveryDecision::LocalFallback
        );
    }

    #[test]
    fn split_loss_recovery_withdraws_when_split_and_local_paths_are_unavailable() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2)],
            vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
        );

        assert_eq!(
            split_loss_recovery_decision(&active, &[participant(1)], &[], None, false),
            SplitLossRecoveryDecision::Withdraw
        );
    }

    #[test]
    fn split_loss_recovery_rejects_single_participant_candidate_as_split_topology() {
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2)],
            vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1)],
            vec![stage(1, 0, 0, 40)],
        );

        assert_eq!(
            split_loss_recovery_decision(&active, &[participant(1)], &[], Some(&candidate), true),
            SplitLossRecoveryDecision::LocalFallback
        );
        assert!(!split_candidate_is_valid_replacement_split(&candidate));
    }

    #[test]
    fn split_loss_recovery_ignores_unused_participant_loss() {
        let active_stages = vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)];
        let active = SplitTopologyGeneration::new(
            "topology-a".into(),
            "run-a".into(),
            1,
            vec![participant(1), participant(2), participant(3)],
            active_stages.clone(),
        );
        let candidate = SplitTopologyGeneration::new(
            "topology-b".into(),
            "run-b".into(),
            2,
            vec![participant(1), participant(2)],
            active_stages,
        );

        assert_eq!(
            split_loss_recovery_decision(
                &active,
                &[participant(1), participant(2)],
                &[],
                Some(&candidate),
                false,
            ),
            SplitLossRecoveryDecision::NoActiveStageLoss
        );
    }

    #[test]
    fn split_topology_minimum_rejects_single_stage_split_candidate() {
        assert!(split_participants_meet_minimum(&[
            participant(1),
            participant(2)
        ]));
        assert!(!split_participants_meet_minimum(&[participant(1)]));
        assert!(split_stages_meet_minimum(&[
            stage(1, 0, 0, 20),
            stage(2, 1, 20, 40)
        ]));
        assert!(!split_stages_meet_minimum(&[stage(1, 0, 0, 40)]));
    }
}
