use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{anyhow, Context, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig};
use skippy_server::{
    binary_transport::{BinaryStageOptions, WireCondition},
    telemetry::TelemetryLevel,
    EmbeddedServerHandle,
};
use tokio::sync::{mpsc, Mutex};

mod inventory;
#[cfg(test)]
mod tests;
mod types;

use inventory::{resolve_inventory_source, run_stage_prepare_task};
pub(crate) use types::*;

struct RunningStage {
    load: StageLoadRequest,
    server: EmbeddedServerHandle,
    materialized: Option<super::materialization::MaterializedStageArtifact>,
    _materialized_pin: Option<super::materialization::MaterializedStagePin>,
}

#[derive(Default)]
struct StageControlState {
    stages: HashMap<String, RunningStage>,
    preparations: Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
}

pub(crate) fn spawn_stage_control_loop() -> mpsc::UnboundedSender<StageControlCommand> {
    let (tx, mut rx) = mpsc::unbounded_channel::<StageControlCommand>();
    tokio::spawn(async move {
        let mut state = StageControlState::default();
        while let Some(command) = rx.recv().await {
            let result = state.handle(command.request).await;
            let _ = command.resp.send(result);
        }
    });
    tx
}

impl StageControlState {
    async fn handle(&mut self, request: StageControlRequest) -> Result<StageControlResponse> {
        match request {
            StageControlRequest::Load(load) => {
                self.load(load).await.map(StageControlResponse::Ready)
            }
            StageControlRequest::Stop(stop) => {
                self.stop(stop).await.map(StageControlResponse::Ready)
            }
            StageControlRequest::Status(filter) => {
                Ok(StageControlResponse::Status(self.statuses(&filter)))
            }
            StageControlRequest::Inventory(request) => Ok(StageControlResponse::Inventory(
                self.inventory(request).await,
            )),
            StageControlRequest::Prepare(request) => Ok(StageControlResponse::PrepareAccepted(
                self.prepare(request).await?,
            )),
            StageControlRequest::CancelPrepare(cancel) => Ok(
                StageControlResponse::PreparationStatus(preparation_status_from_cancel(cancel)),
            ),
            StageControlRequest::StatusUpdate(_status) => {
                Ok(StageControlResponse::StatusAck(StageStatusAck {
                    accepted: true,
                    error: None,
                }))
            }
        }
    }

    async fn inventory(&self, request: StageInventoryRequest) -> StageLayerInventory {
        let preparing_ranges = self
            .preparations
            .lock()
            .await
            .values()
            .filter(|status| {
                status.model_id == request.model_id
                    && status.package_ref == request.package_ref
                    && status.manifest_sha256 == request.manifest_sha256
            })
            .cloned()
            .collect::<Vec<_>>();
        let source = resolve_inventory_source(&request);
        let layer_count = source
            .as_ref()
            .map(|source| source.layer_count)
            .unwrap_or(0);
        let available_ranges = if source.is_some() && layer_count > 0 {
            vec![LayerRange {
                layer_start: 0,
                layer_end: layer_count,
            }]
        } else {
            Vec::new()
        };
        let ready_ranges = self
            .stages
            .values()
            .filter(|stage| {
                stage.load.model_id == request.model_id
                    && stage.load.package_ref == request.package_ref
                    && stage.load.manifest_sha256 == request.manifest_sha256
            })
            .map(|stage| LayerRange {
                layer_start: stage.load.layer_start,
                layer_end: stage.load.layer_end,
            })
            .collect::<Vec<_>>();
        let missing_ranges = if source.is_none() && layer_count > 0 {
            vec![LayerRange {
                layer_start: 0,
                layer_end: layer_count,
            }]
        } else {
            Vec::new()
        };
        StageLayerInventory {
            model_id: request.model_id,
            package_ref: request.package_ref,
            manifest_sha256: request.manifest_sha256,
            layer_count,
            ready_ranges,
            available_ranges,
            missing_ranges,
            preparing_ranges,
            source_model_path: source
                .as_ref()
                .map(|source| source.path.to_string_lossy().to_string()),
            source_model_bytes: source.as_ref().and_then(|source| source.bytes),
            source_model_kind: source
                .as_ref()
                .map(|source| source.kind)
                .unwrap_or(SourceModelKind::Unknown),
        }
    }

    async fn prepare(
        &mut self,
        request: StagePrepareRequest,
    ) -> Result<StagePrepareAcceptedResponse> {
        let key = stage_key(
            &request.load.topology_id,
            &request.load.run_id,
            &request.load.stage_id,
        );
        let status = preparation_status_from_load(&request.load, StagePreparationState::Assigned);
        self.preparations
            .lock()
            .await
            .insert(key.clone(), status.clone());
        let preparations = Arc::clone(&self.preparations);
        tokio::spawn(async move {
            run_stage_prepare_task(preparations, key, request.load).await;
        });
        Ok(StagePrepareAcceptedResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    async fn load(&mut self, load: StageLoadRequest) -> Result<StageReadyResponse> {
        anyhow::ensure!(
            load.backend == "skippy",
            "unsupported stage backend '{}'",
            load.backend
        );
        let key = stage_key(&load.topology_id, &load.run_id, &load.stage_id);
        if let Some(existing) = self.stages.remove(&key) {
            existing.server.shutdown().await?;
        }

        let bind_addr = materialize_stage_bind_addr(parse_bind_addr(&load.bind_addr)?)?;
        let mut effective_load = load;
        effective_load.bind_addr = bind_addr.to_string();
        super::configure_materialized_stage_cache();
        let materialized = super::materialize_stage_load(&effective_load)?;
        let config = stage_config(
            &effective_load,
            materialized.as_ref().map(|(artifact, _)| artifact),
        )?;
        let server = skippy_server::start_binary_stage(BinaryStageOptions {
            config,
            topology: None,
            bind_addr,
            activation_width: effective_load.activation_width,
            wire_dtype: effective_load.wire_dtype.into(),
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: TelemetryLevel::Off,
            max_inflight: effective_load.lane_count as usize,
            reply_credit_limit: None,
            async_prefill_forward: false,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            downstream_connect_timeout_secs: 30,
            openai: None,
        });
        if let Err(error) = wait_for_binary_stage_ready(bind_addr, Duration::from_secs(120)).await {
            let _ = server.shutdown().await;
            return Err(error);
        }

        self.stages.insert(
            key,
            RunningStage {
                load: effective_load.clone(),
                server,
                materialized: materialized.as_ref().map(|(artifact, _)| artifact.clone()),
                _materialized_pin: materialized.map(|(_, pin)| pin),
            },
        );
        let status = self
            .statuses(&StageStatusFilter {
                topology_id: Some(effective_load.topology_id.clone()),
                run_id: Some(effective_load.run_id.clone()),
                stage_id: Some(effective_load.stage_id.clone()),
            })
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("stage status missing after load"))?;
        Ok(StageReadyResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    async fn stop(&mut self, stop: StageStopRequest) -> Result<StageReadyResponse> {
        let key = stage_key(&stop.topology_id, &stop.run_id, &stop.stage_id);
        let Some(existing) = self.stages.remove(&key) else {
            let status = stopped_status(&stop);
            return Ok(StageReadyResponse {
                accepted: true,
                status,
                error: None,
            });
        };
        if stop.shutdown_generation < existing.load.shutdown_generation {
            let status = status_from_running(&existing);
            self.stages.insert(key, existing);
            return Ok(StageReadyResponse {
                accepted: false,
                status,
                error: Some("stale shutdown generation".to_string()),
            });
        }
        let mut status = status_from_running(&existing);
        status.state = StageRuntimeState::Stopping;
        existing.server.shutdown().await?;
        status.state = StageRuntimeState::Stopped;
        status.shutdown_generation = stop.shutdown_generation;
        Ok(StageReadyResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    fn statuses(&self, filter: &StageStatusFilter) -> Vec<StageStatusSnapshot> {
        self.stages
            .values()
            .filter(|stage| filter.matches(&stage.load))
            .map(status_from_running)
            .collect()
    }
}

impl StageStatusFilter {
    fn matches(&self, load: &StageLoadRequest) -> bool {
        self.topology_id
            .as_ref()
            .is_none_or(|value| value == &load.topology_id)
            && self
                .run_id
                .as_ref()
                .is_none_or(|value| value == &load.run_id)
            && self
                .stage_id
                .as_ref()
                .is_none_or(|value| value == &load.stage_id)
    }
}

fn stage_key(topology_id: &str, run_id: &str, stage_id: &str) -> String {
    format!("{topology_id}\n{run_id}\n{stage_id}")
}

fn parse_bind_addr(bind_addr: &str) -> Result<SocketAddr> {
    bind_addr
        .parse()
        .with_context(|| format!("parse stage bind_addr {bind_addr:?}"))
}

fn materialize_stage_bind_addr(bind_addr: SocketAddr) -> Result<SocketAddr> {
    if bind_addr.port() != 0 {
        return Ok(bind_addr);
    }
    let listener = std::net::TcpListener::bind(bind_addr)
        .with_context(|| format!("reserve ephemeral stage bind address for {bind_addr}"))?;
    listener
        .local_addr()
        .context("read reserved ephemeral stage bind address")
}

async fn wait_for_binary_stage_ready(bind_addr: SocketAddr, timeout: Duration) -> Result<()> {
    tokio::task::spawn_blocking(move || probe_binary_stage_ready(bind_addr, timeout))
        .await
        .context("join binary stage readiness probe")?
}

fn probe_binary_stage_ready(bind_addr: SocketAddr, timeout: Duration) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    let mut last_error = None;
    while std::time::Instant::now() < deadline {
        match std::net::TcpStream::connect(bind_addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
                match skippy_protocol::binary::recv_ready(&mut stream) {
                    Ok(()) => return Ok(()),
                    Err(error) => {
                        last_error =
                            Some(anyhow!(error).context("binary stage ready handshake failed"));
                    }
                }
            }
            Err(error) => {
                last_error = Some(anyhow!(error).context("connect binary stage listener"));
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out waiting for binary stage ready at {bind_addr}")))
}

fn stage_config(
    load: &StageLoadRequest,
    materialized: Option<&super::materialization::MaterializedStageArtifact>,
) -> Result<StageConfig> {
    anyhow::ensure!(!load.topology_id.is_empty(), "topology_id is required");
    anyhow::ensure!(!load.run_id.is_empty(), "run_id is required");
    anyhow::ensure!(!load.model_id.is_empty(), "model_id is required");
    anyhow::ensure!(!load.stage_id.is_empty(), "stage_id is required");
    anyhow::ensure!(
        load.layer_start < load.layer_end,
        "invalid stage layer range"
    );
    anyhow::ensure!(load.ctx_size > 0, "ctx_size must be greater than zero");
    anyhow::ensure!(load.lane_count > 0, "lane_count must be greater than zero");
    if let Some(device) = load.selected_device.as_ref() {
        anyhow::ensure!(
            !device.backend_device.is_empty(),
            "selected backend device must not be empty"
        );
    }
    let mut config = StageConfig {
        run_id: load.run_id.clone(),
        topology_id: load.topology_id.clone(),
        model_id: load.model_id.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: materialized
            .map(|artifact| artifact.source_model_path.clone())
            .or_else(|| load.model_path.clone()),
        source_model_sha256: materialized.map(|artifact| artifact.source_model_sha256.clone()),
        source_model_bytes: materialized.and_then(|artifact| artifact.source_model_bytes),
        materialized_path: materialized.map(|artifact| artifact.path.to_string_lossy().to_string()),
        materialized_pinned: materialized.is_some(),
        model_path: load.model_path.clone(),
        projector_path: load.projector_path.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        n_gpu_layers: load.n_gpu_layers,
        cache_type_k: empty_to_default(&load.cache_type_k, "f16"),
        cache_type_v: empty_to_default(&load.cache_type_v, "f16"),
        flash_attn_type: load.flash_attn_type,
        filter_tensors_on_load: matches!(
            load.load_mode,
            LoadMode::RuntimeSlice | LoadMode::LayerPackage
        ),
        selected_device: load.selected_device.clone(),
        kv_cache: None,
        load_mode: load.load_mode.clone(),
        bind_addr: load.bind_addr.clone(),
        upstream: load.upstream.as_ref().map(peer_config),
        downstream: load.downstream.as_ref().map(peer_config),
    };
    let family_policy = super::family_policy_for_stage_config(&config);
    config.kv_cache = family_policy.stage_kv_cache_config_for_stage(&config);
    Ok(config)
}

fn peer_config(peer: &StagePeerDescriptor) -> PeerConfig {
    PeerConfig {
        stage_id: peer.stage_id.clone(),
        stage_index: peer.stage_index,
        endpoint: peer.endpoint.clone(),
    }
}

fn empty_to_default(value: &str, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    }
}

fn status_from_running(stage: &RunningStage) -> StageStatusSnapshot {
    let server = stage.server.status();
    let state = match server.state {
        skippy_server::EmbeddedState::Starting => StageRuntimeState::Starting,
        skippy_server::EmbeddedState::Ready => StageRuntimeState::Ready,
        skippy_server::EmbeddedState::Stopping => StageRuntimeState::Stopping,
        skippy_server::EmbeddedState::Stopped => StageRuntimeState::Stopped,
        skippy_server::EmbeddedState::Failed => StageRuntimeState::Failed,
    };
    StageStatusSnapshot {
        topology_id: stage.load.topology_id.clone(),
        run_id: stage.load.run_id.clone(),
        model_id: stage.load.model_id.clone(),
        backend: stage.load.backend.clone(),
        package_ref: Some(stage.load.package_ref.clone()),
        manifest_sha256: Some(stage.load.manifest_sha256.clone()),
        source_model_path: stage
            .materialized
            .as_ref()
            .map(|artifact| artifact.source_model_path.clone())
            .or_else(|| stage.load.model_path.clone()),
        source_model_sha256: stage
            .materialized
            .as_ref()
            .map(|artifact| artifact.source_model_sha256.clone()),
        source_model_bytes: stage
            .materialized
            .as_ref()
            .and_then(|artifact| artifact.source_model_bytes),
        materialized_path: stage
            .materialized
            .as_ref()
            .map(|artifact| artifact.path.to_string_lossy().to_string()),
        materialized_pinned: stage.materialized.is_some(),
        projector_path: stage.load.projector_path.clone(),
        stage_id: stage.load.stage_id.clone(),
        stage_index: stage.load.stage_index,
        layer_start: stage.load.layer_start,
        layer_end: stage.load.layer_end,
        state,
        bind_addr: server.bind_addr.to_string(),
        activation_width: stage.load.activation_width.max(0) as u32,
        wire_dtype: stage.load.wire_dtype,
        selected_device: stage.load.selected_device.clone(),
        ctx_size: stage.load.ctx_size,
        lane_count: stage.load.lane_count,
        n_batch: stage.load.n_batch,
        n_ubatch: stage.load.n_ubatch,
        flash_attn_type: stage.load.flash_attn_type,
        error: server.last_error.clone(),
        shutdown_generation: stage.load.shutdown_generation,
    }
}

fn stopped_status(stop: &StageStopRequest) -> StageStatusSnapshot {
    StageStatusSnapshot {
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
        state: StageRuntimeState::Stopped,
        bind_addr: String::new(),
        activation_width: 0,
        wire_dtype: StageWireDType::F32,
        selected_device: None,
        ctx_size: 0,
        lane_count: 0,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: stop.shutdown_generation,
    }
}

fn preparation_status_from_load(
    load: &StageLoadRequest,
    state: StagePreparationState,
) -> StagePreparationStatus {
    StagePreparationStatus {
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
        state,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error: None,
        shutdown_generation: load.shutdown_generation,
    }
}

fn preparation_status_from_cancel(cancel: StageCancelPrepareRequest) -> StagePreparationStatus {
    StagePreparationStatus {
        topology_id: cancel.topology_id,
        run_id: cancel.run_id,
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: String::new(),
        manifest_sha256: String::new(),
        stage_id: cancel.stage_id,
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        state: StagePreparationState::Cancelled,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error: None,
        shutdown_generation: cancel.shutdown_generation,
    }
}

impl From<StageWireDType> for skippy_protocol::binary::WireActivationDType {
    fn from(value: StageWireDType) -> Self {
        match value {
            StageWireDType::F32 => Self::F32,
            StageWireDType::F16 => Self::F16,
            StageWireDType::Q8 => Self::Q8,
        }
    }
}
