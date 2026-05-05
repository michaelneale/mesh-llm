use crate::api;
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::models;
use crate::network::router;
use anyhow::{Context, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig};
use skippy_topology::{
    infer_family_capability, plan_weighted_contiguous, BoundaryDecision, DiagnosticSeverity,
    LayerSpec, NodeSpec, PlannerPolicy, TopologyPlanRequest,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SPLIT_PARTICIPANT_POLL_INTERVAL: Duration = Duration::from_millis(500);
const SPLIT_PARTICIPANT_STABLE_FOR: Duration = Duration::from_secs(2);
const SPLIT_DEFAULT_MIN_PARTICIPANTS: usize = 2;
const SPLIT_INITIAL_SHUTDOWN_GENERATION: u64 = 1;

pub(super) enum RuntimeEvent {
    Exited { model: String, port: u16 },
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

    pub(super) async fn shutdown(self) {
        match self.inner {
            LocalRuntimeBackendHandle::Skippy { model, http, .. } => {
                let _ = http.shutdown().await;
                model.shutdown();
            }
        }
    }
}

pub(super) struct ManagedModelController {
    pub(super) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(super) task: tokio::task::JoinHandle<()>,
}

pub(super) struct LocalRuntimeModelStartSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) model_path: &'a Path,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) slots: usize,
}

#[allow(clippy::large_enum_variant)]
pub(super) enum SplitRuntimeStart {
    Started(SplitRuntimeGenerationHandle),
    Standby { coordinator: iroh::EndpointId },
}

pub(super) struct SplitRuntimeGenerationHandle {
    pub(super) loaded_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) cleanup: Option<SplitGenerationCleanup>,
    pub(super) coordinator_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) coordinator_task: Option<tokio::task::JoinHandle<()>>,
}

pub(super) struct SplitCoordinatorEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) loaded: SplitRuntimeGenerationHandle,
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
    entry.retain(|target| !matches!(target, election::InferenceTarget::Local(_)));
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
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let model_name = models::model_ref_for_path(spec.model_path);
    let model_bytes = election::total_model_bytes(spec.model_path);
    let my_vram = spec
        .pinned_gpu
        .map(|gpu| gpu.vram_bytes)
        .unwrap_or_else(|| spec.node.vram_bytes());
    anyhow::ensure!(
        my_vram >= (model_bytes as f64 * 1.1) as u64,
        "runtime load only supports models that fit locally on this node"
    );

    start_runtime_skippy_model(spec, model_name).await
}

pub(super) async fn start_runtime_split_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_ref: &str,
) -> Result<SplitRuntimeStart> {
    let package = skippy::synthetic_direct_gguf_package(model_ref, spec.model_path)?;
    let participants = wait_for_split_participants(
        spec.node,
        model_ref,
        model_ref,
        spec.pinned_gpu.map(|gpu| gpu.vram_bytes),
        Duration::from_secs(30),
    )
    .await?;
    let run_id = format!("mesh-split-{}", now_unix_nanos());
    let topology_id = format!("topology-{run_id}");
    let planned_participants = participants.clone();
    let stages =
        plan_runtime_slice_topology(&topology_id, model_ref, &package, &planned_participants)?;
    anyhow::ensure!(
        stages.len() > 1,
        "split runtime needs at least two stage participants"
    );
    let stage0 = stages
        .first()
        .context("split topology did not produce stage 0")?;
    if stage0.node_id != spec.node.id() {
        return Ok(SplitRuntimeStart::Standby {
            coordinator: stage0.node_id,
        });
    }

    let ctx_size = spec.ctx_size_override.unwrap_or(4096);
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
        slots: spec.slots,
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
        cache_type_k_override: spec.cache_type_k_override.map(str::to_string),
        cache_type_v_override: spec.cache_type_v_override.map(str::to_string),
        n_batch_override: spec.n_batch_override,
        n_ubatch_override: spec.n_ubatch_override,
        flash_attention_override: spec.flash_attention_override,
        pinned_gpu: spec.pinned_gpu.cloned(),
        slots: spec.slots,
        event_tx: coordinator_tx,
    }));

    Ok(SplitRuntimeStart::Started(loaded))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SplitParticipant {
    node_id: iroh::EndpointId,
    vram_bytes: u64,
    first_joined_mesh_ts: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitParticipantSnapshot {
    participants: Vec<SplitParticipant>,
    excluded: Vec<SplitParticipantExclusion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitParticipantExclusion {
    node_id: iroh::EndpointId,
    reason: SplitParticipantExclusionReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitParticipantExclusionReason {
    Client,
    MissingVram,
    MissingModelInterest,
}

impl SplitParticipantExclusionReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::MissingVram => "missing_vram",
            Self::MissingModelInterest => "missing_model_interest",
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
}

async fn load_split_runtime_generation(
    spec: SplitGenerationLoadSpec<'_>,
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

    let mut ready_by_stage: HashMap<String, skippy::StageStatusSnapshot> = HashMap::new();
    let mut downstream: Option<skippy::StagePeerDescriptor> = None;
    let kv_cache = skippy::KvCachePolicy::for_model_size(spec.package.source_model_bytes);
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
        model = spec.model_ref,
        "KV cache: {}",
        kv_cache.label(spec.package.source_model_bytes)
    );

    for stage in spec.generation.stages.iter().skip(1).rev() {
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
            model_path: Some(spec.model_ref.to_string()),
            projector_path: spec.projector_path.clone(),
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width: spec.package.activation_width as i32,
            wire_dtype: skippy::StageWireDType::F16,
            ctx_size: spec.ctx_size,
            lane_count: spec.slots as u32,
            n_batch: spec.n_batch_override,
            n_ubatch: spec.n_ubatch_override,
            n_gpu_layers: -1,
            cache_type_k: effective_cache_type_k.clone(),
            cache_type_v: effective_cache_type_v.clone(),
            flash_attn_type: resolved_flash_attn_type,
            shutdown_generation: spec.generation.generation,
            load_mode: LoadMode::RuntimeSlice,
            upstream: None,
            downstream: downstream.clone(),
        };
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
        spec.projector_path,
        spec.ctx_size,
        spec.slots as u32,
        kv_cache,
        spec.cache_type_k_override,
        spec.cache_type_v_override,
        spec.n_batch_override,
        spec.n_ubatch_override,
        spec.flash_attention_override,
        spec.pinned_gpu,
    );
    let handle = skippy::SkippyModelHandle::load_stage0_config(
        config,
        spec.package.activation_width as i32,
        spec.slots,
        skippy_server::openai::CONTEXT_BUDGET_MAX_TOKENS,
        Some(skippy::MeshAutoHookPolicy::new(spec.node.clone())),
    )?;
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeSliceStagePlan {
    stage_id: String,
    stage_index: u32,
    node_id: iroh::EndpointId,
    layer_start: u32,
    layer_end: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitTopologyGeneration {
    topology_id: String,
    run_id: String,
    generation: u64,
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
    cache_type_k_override: Option<String>,
    cache_type_v_override: Option<String>,
    n_batch_override: Option<u32>,
    n_ubatch_override: Option<u32>,
    flash_attention_override: FlashAttentionType,
    pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
    event_tx: tokio::sync::mpsc::Sender<SplitCoordinatorEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitReplanDecision {
    Keep,
    Candidate,
}

fn spawn_split_topology_coordinator(
    coordinator: SplitTopologyCoordinator,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        coordinator.run().await;
    })
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
                    self.evaluate_replan("membership_changed").await;
                }
                _ = health_tick.tick() => {
                    self.evaluate_replan("periodic_check").await;
                }
            }
        }
    }

    async fn evaluate_replan(&mut self, reason: &'static str) {
        let snapshot = collect_split_participants(
            &self.node,
            &self.model_name,
            &self.model_ref,
            self.pinned_gpu.as_ref().map(|gpu| gpu.vram_bytes),
        )
        .await;
        if snapshot.participants.len() < SPLIT_DEFAULT_MIN_PARTICIPANTS {
            tracing::debug!(
                model_ref = self.model_ref,
                reason,
                participants = ?split_participant_labels(&snapshot.participants),
                excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
                "split topology replan skipped; quorum not met"
            );
            return;
        }

        let planned_participants = snapshot.participants.clone();
        let generation = self.active.generation.saturating_add(1);
        let run_id = format!("mesh-split-{}-g{}", now_unix_nanos(), generation);
        let topology_id = format!("topology-{run_id}");
        let candidate = match plan_runtime_slice_topology(
            &topology_id,
            &self.model_ref,
            &self.package,
            &planned_participants,
        ) {
            Ok(stages) => SplitTopologyGeneration::new(
                topology_id,
                run_id,
                generation,
                planned_participants,
                stages,
            ),
            Err(err) => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    error = %err,
                    participants = ?split_participant_labels(&planned_participants),
                    excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
                    "split topology replan candidate failed"
                );
                return;
            }
        };
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
            return;
        }

        match split_replan_decision(&self.active, &candidate) {
            SplitReplanDecision::Keep => {
                tracing::debug!(
                    model_ref = self.model_ref,
                    reason,
                    active_generation = self.active.generation,
                    active_stages = self.active.stages.len(),
                    candidate_stages = candidate.stages.len(),
                    "split topology replan skipped; candidate is not materially better"
                );
            }
            SplitReplanDecision::Candidate => {
                tracing::info!(
                    model_ref = self.model_ref,
                    reason,
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
        })
        .await?;
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent {
            reason,
            generation: candidate.generation,
            loaded,
            ack: ack_tx,
        };
        if let Err(err) = self.event_tx.send(event).await {
            let event = err.0;
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
}

fn split_replan_decision(
    active: &SplitTopologyGeneration,
    candidate: &SplitTopologyGeneration,
) -> SplitReplanDecision {
    if candidate.stages.len() > active.stages.len() {
        return SplitReplanDecision::Candidate;
    }
    if candidate.participants.len() > active.participants.len()
        && candidate.stages.len() == active.stages.len()
    {
        return SplitReplanDecision::Candidate;
    }
    if split_stage_node_signature(&candidate.stages) != split_stage_node_signature(&active.stages)
        && split_stage_balance_score(&candidate.stages) < split_stage_balance_score(&active.stages)
    {
        return SplitReplanDecision::Candidate;
    }
    SplitReplanDecision::Keep
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

async fn wait_for_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    local_vram_override: Option<u64>,
    timeout: Duration,
) -> Result<Vec<SplitParticipant>> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut best: Vec<SplitParticipant> = Vec::new();
    let mut best_excluded: Vec<SplitParticipantExclusion> = Vec::new();
    let mut last_signature: Vec<(String, u64)> = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    loop {
        let snapshot =
            collect_split_participants(node, model_name, model_ref, local_vram_override).await;
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
            return Ok(snapshot.participants);
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
            return Ok(best);
        }

        tokio::time::sleep(SPLIT_PARTICIPANT_POLL_INTERVAL).await;
    }
}

async fn collect_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    local_vram_override: Option<u64>,
) -> SplitParticipantSnapshot {
    let mut participants = vec![SplitParticipant {
        node_id: node.id(),
        vram_bytes: local_vram_override.unwrap_or_else(|| node.vram_bytes()),
        first_joined_mesh_ts: Some(node.first_joined_mesh_ts().await.unwrap_or(0)),
    }];
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
        if wants_model {
            participants.push(SplitParticipant {
                node_id: peer.id,
                vram_bytes: peer.vram_bytes,
                first_joined_mesh_ts: peer.first_joined_mesh_ts,
            });
        } else {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason: SplitParticipantExclusionReason::MissingModelInterest,
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

fn split_participant_signature(participants: &[SplitParticipant]) -> Vec<(String, u64)> {
    participants
        .iter()
        .map(|participant| (participant.node_id.to_string(), participant.vram_bytes))
        .collect()
}

fn split_participant_labels(participants: &[SplitParticipant]) -> Vec<String> {
    participants
        .iter()
        .map(|participant| {
            format!(
                "{}:{}GB",
                participant.node_id.fmt_short(),
                participant.vram_bytes / 1_000_000_000
            )
        })
        .collect()
}

fn split_participant_exclusion_labels(excluded: &[SplitParticipantExclusion]) -> Vec<String> {
    excluded
        .iter()
        .map(|exclusion| {
            format!(
                "{}:{}",
                exclusion.node_id.fmt_short(),
                exclusion.reason.as_str()
            )
        })
        .collect()
}

fn plan_runtime_slice_topology(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
) -> Result<Vec<RuntimeSliceStagePlan>> {
    let node_by_id = participants
        .iter()
        .map(|participant| (participant.node_id.to_string(), participant.node_id))
        .collect::<HashMap<_, _>>();
    let fallback_layer_bytes = package.source_model_bytes / u64::from(package.layer_count.max(1));
    let layers = (0..package.layer_count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: false,
            parameter_bytes: fallback_layer_bytes,
        })
        .collect();
    let request = TopologyPlanRequest {
        topology_id: topology_id.to_string(),
        model_id: model_ref.to_string(),
        layers,
        nodes: participants
            .iter()
            .map(|participant| NodeSpec {
                node_id: participant.node_id.to_string(),
                cached_slice_bytes: 0,
                vram_bytes: participant.vram_bytes,
            })
            .collect(),
        family: infer_family_capability(model_ref, package.layer_count, package.activation_width),
        policy: PlannerPolicy::default(),
    };
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        "planning split runtime topology"
    );
    let plan = plan_weighted_contiguous(&request)?;
    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .map(|boundary| {
            format!(
                "rejected boundary at layer {}: {}",
                boundary.layer_boundary,
                boundary.messages.join("; ")
            )
        })
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        anyhow::bail!("{}", rejected.join("; "));
    }
    let errors = plan
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        anyhow::bail!("{}", errors.join("; "));
    }
    let mut stages = plan
        .stages
        .into_iter()
        .map(|stage| {
            let node_id = node_by_id.get(&stage.node_id).copied().with_context(|| {
                format!("topology planner returned unknown node {}", stage.node_id)
            })?;
            Ok(RuntimeSliceStagePlan {
                stage_id: stage.stage_id,
                stage_index: stage.stage_index,
                node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    stages.sort_by_key(|stage| stage.stage_index);
    tracing::info!(
        topology_id,
        model_ref,
        stages = ?split_stage_plan_labels(&stages),
        "planned split runtime topology"
    );
    Ok(stages)
}

fn split_stage_plan_labels(stages: &[RuntimeSliceStagePlan]) -> Vec<String> {
    stages
        .iter()
        .map(|stage| {
            format!(
                "{}:{}:{}..{}",
                stage.stage_id,
                stage.node_id.fmt_short(),
                stage.layer_start,
                stage.layer_end
            )
        })
        .collect()
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
) -> StageConfig {
    let effective_cache_type_k = cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k())
        .to_string();
    let effective_cache_type_v = cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v())
        .to_string();
    let resolved_flash_attn_type =
        effective_flash_attention(flash_attention_override, &effective_cache_type_v);
    StageConfig {
        run_id: run_id.to_string(),
        topology_id: topology_id.to_string(),
        model_id: model_ref.to_string(),
        package_ref: Some(package.package_ref.clone()),
        manifest_sha256: Some(package.manifest_sha256.clone()),
        source_model_path: Some(model_path.to_string_lossy().to_string()),
        source_model_sha256: Some(package.source_model_sha256.clone()),
        source_model_bytes: Some(package.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(model_path.to_string_lossy().to_string()),
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
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: downstream_stage_id,
            stage_index: downstream_stage_index,
            endpoint: downstream_endpoint,
        }),
    }
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
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let port = alloc_local_port().await?;
    let context_length = spec.ctx_size_override.unwrap_or(4096);
    let model_bytes = election::total_model_bytes(spec.model_path);
    let kv_cache = skippy::KvCachePolicy::for_model_size(model_bytes);
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
        "KV cache: {}",
        kv_cache.label(model_bytes)
    );
    let projector_path = spec
        .mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&model_name))
        .filter(|path| path.exists());
    let mut options = skippy::SkippyModelLoadOptions::for_direct_gguf(&model_name, spec.model_path)
        .with_ctx_size(context_length)
        .with_generation_concurrency(spec.slots)
        .with_cache_types(effective_cache_type_k, effective_cache_type_v)
        .with_flash_attn_type(resolved_flash_attn_type);
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
    let skippy_model = skippy::SkippyModelHandle::load_with_hooks(
        options,
        Some(skippy::MeshAutoHookPolicy::new(spec.node.clone())),
    )?;
    let http = skippy_model.start_http(port);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            slots: spec.slots,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: skippy_model,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

pub(super) fn local_process_payload(
    model_name: &str,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> api::RuntimeProcessPayload {
    local_process_snapshot(model_name, backend, port, pid, slots, context_length).to_payload()
}

pub(super) fn local_process_snapshot(
    model_name: &str,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> crate::runtime_data::RuntimeProcessSnapshot {
    crate::runtime_data::RuntimeProcessSnapshot {
        model: model_name.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

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

    #[test]
    fn split_topology_planner_uses_all_eligible_participants() {
        let participants = vec![
            SplitParticipant {
                node_id: make_id(1),
                vram_bytes: 16_000_000_000,
                first_joined_mesh_ts: None,
            },
            SplitParticipant {
                node_id: make_id(2),
                vram_bytes: 24_000_000_000,
                first_joined_mesh_ts: None,
            },
            SplitParticipant {
                node_id: make_id(3),
                vram_bytes: 32_000_000_000,
                first_joined_mesh_ts: None,
            },
            SplitParticipant {
                node_id: make_id(4),
                vram_bytes: 48_000_000_000,
                first_joined_mesh_ts: None,
            },
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
    fn split_participant_signature_includes_vram_for_stability() {
        let node_id = make_id(9);
        let first = vec![SplitParticipant {
            node_id,
            vram_bytes: 16_000_000_000,
            first_joined_mesh_ts: None,
        }];
        let second = vec![SplitParticipant {
            node_id,
            vram_bytes: 24_000_000_000,
            first_joined_mesh_ts: None,
        }];

        assert_ne!(
            split_participant_signature(&first),
            split_participant_signature(&second)
        );
    }

    #[test]
    fn split_replan_decision_accepts_more_stage_capacity() {
        let participants = vec![SplitParticipant {
            node_id: make_id(1),
            vram_bytes: 16_000_000_000,
            first_joined_mesh_ts: None,
        }];
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
                },
                RuntimeSliceStagePlan {
                    stage_id: "stage-1".into(),
                    stage_index: 1,
                    node_id: make_id(2),
                    layer_start: 16,
                    layer_end: 40,
                },
            ],
        );

        assert_eq!(
            split_replan_decision(&active, &candidate),
            SplitReplanDecision::Candidate
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
        }];
        let participants = vec![SplitParticipant {
            node_id: make_id(1),
            vram_bytes: 16_000_000_000,
            first_joined_mesh_ts: None,
        }];
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
    }
}
