use std::{collections::HashMap, net::SocketAddr, time::Duration};

use anyhow::{anyhow, Context, Result};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};
use skippy_server::{
    binary_transport::{BinaryStageOptions, WireCondition},
    telemetry::TelemetryLevel,
    EmbeddedServerHandle,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub(crate) struct StageControlCommand {
    pub(crate) request: StageControlRequest,
    pub(crate) resp: oneshot::Sender<Result<StageControlResponse>>,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum StageControlRequest {
    Load(StageLoadRequest),
    Stop(StageStopRequest),
    Status(StageStatusFilter),
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum StageControlResponse {
    Ready(StageReadyResponse),
    Status(Vec<StageStatusSnapshot>),
}

#[derive(Clone, Debug)]
pub(crate) struct StageLoadRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) model_path: Option<String>,
    pub(crate) projector_path: Option<String>,
    pub(crate) selected_device: Option<StageDevice>,
    pub(crate) bind_addr: String,
    pub(crate) activation_width: i32,
    pub(crate) wire_dtype: StageWireDType,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) shutdown_generation: u64,
    pub(crate) load_mode: LoadMode,
    pub(crate) upstream: Option<StagePeerDescriptor>,
    pub(crate) downstream: Option<StagePeerDescriptor>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStopRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) stage_id: String,
    pub(crate) shutdown_generation: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StageStatusFilter {
    pub(crate) topology_id: Option<String>,
    pub(crate) run_id: Option<String>,
    pub(crate) stage_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePeerDescriptor {
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) endpoint: String,
    pub(crate) node_id: Option<iroh::EndpointId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageWireDType {
    F32,
    F16,
    Q8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageRuntimeState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub(crate) struct StageReadyResponse {
    pub(crate) accepted: bool,
    pub(crate) status: StageStatusSnapshot,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStatusSnapshot {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) package_ref: Option<String>,
    pub(crate) manifest_sha256: Option<String>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) materialized_path: Option<String>,
    pub(crate) materialized_pinned: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) state: StageRuntimeState,
    pub(crate) bind_addr: String,
    pub(crate) activation_width: u32,
    pub(crate) wire_dtype: StageWireDType,
    pub(crate) selected_device: Option<StageDevice>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) error: Option<String>,
    pub(crate) shutdown_generation: u64,
}

struct RunningStage {
    load: StageLoadRequest,
    server: EmbeddedServerHandle,
    package_info: Option<super::materialization::StagePackageInfo>,
}

#[derive(Default)]
struct StageControlState {
    stages: HashMap<String, RunningStage>,
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
        }
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
        let package_info = if effective_load.load_mode == LoadMode::LayerPackage {
            Some(super::inspect_stage_package(&effective_load.package_ref)?)
        } else {
            None
        };
        let config = stage_config(&effective_load)?;
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
                package_info,
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

fn stage_config(load: &StageLoadRequest) -> Result<StageConfig> {
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
    Ok(StageConfig {
        run_id: load.run_id.clone(),
        topology_id: load.topology_id.clone(),
        model_id: load.model_id.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: load.model_path.clone(),
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
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
        load_mode: load.load_mode.clone(),
        bind_addr: load.bind_addr.clone(),
        upstream: load.upstream.as_ref().map(peer_config),
        downstream: load.downstream.as_ref().map(peer_config),
    })
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
            .package_info
            .as_ref()
            .map(|package| package.source_model_path.clone())
            .or_else(|| stage.load.model_path.clone()),
        source_model_sha256: stage
            .package_info
            .as_ref()
            .map(|package| package.source_model_sha256.clone()),
        source_model_bytes: stage
            .package_info
            .as_ref()
            .and_then(|package| package.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
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

impl From<StageWireDType> for skippy_protocol::binary::WireActivationDType {
    fn from(value: StageWireDType) -> Self {
        match value {
            StageWireDType::F32 => Self::F32,
            StageWireDType::F16 => Self::F16,
            StageWireDType::Q8 => Self::Q8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn load_request() -> StageLoadRequest {
        StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            backend: "skippy".to_string(),
            package_ref: "pkg-a".to_string(),
            manifest_sha256: "sha256".to_string(),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 12,
            model_path: Some("/models/model.gguf".to_string()),
            projector_path: Some("/models/mmproj.gguf".to_string()),
            selected_device: Some(StageDevice {
                backend_device: "CUDA0".to_string(),
                stable_id: Some("GPU-123".to_string()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            }),
            bind_addr: "127.0.0.1:0".to_string(),
            activation_width: 4096,
            wire_dtype: StageWireDType::F16,
            ctx_size: 8192,
            lane_count: 3,
            n_batch: Some(2048),
            n_ubatch: Some(512),
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "q8_0".to_string(),
            flash_attn_type: FlashAttentionType::Enabled,
            shutdown_generation: 7,
            load_mode: LoadMode::RuntimeSlice,
            upstream: None,
            downstream: Some(StagePeerDescriptor {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "127.0.0.1:9001".to_string(),
                node_id: None,
            }),
        }
    }

    #[test]
    fn stage_config_preserves_backend_neutral_load_fields() {
        let request = load_request();
        let config = stage_config(&request).unwrap();

        assert_eq!(config.topology_id, "topology-a");
        assert_eq!(config.run_id, "run-a");
        assert_eq!(config.model_id, "model-a");
        assert_eq!(config.package_ref.as_deref(), Some("pkg-a"));
        assert_eq!(config.manifest_sha256.as_deref(), Some("sha256"));
        assert_eq!(
            config.source_model_path.as_deref(),
            Some("/models/model.gguf")
        );
        assert!(config.materialized_path.is_none());
        assert!(!config.materialized_pinned);
        assert_eq!(config.stage_id, "stage-0");
        assert_eq!(config.stage_index, 0);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 12);
        assert_eq!(config.lane_count, 3);
        assert_eq!(config.n_batch, Some(2048));
        assert_eq!(config.n_ubatch, Some(512));
        assert_eq!(config.model_path.as_deref(), Some("/models/model.gguf"));
        assert_eq!(
            config.projector_path.as_deref(),
            Some("/models/mmproj.gguf")
        );
        assert_eq!(config.flash_attn_type, FlashAttentionType::Enabled);
        assert_eq!(
            config
                .selected_device
                .as_ref()
                .map(|d| d.backend_device.as_str()),
            Some("CUDA0")
        );
        assert_eq!(
            config.downstream.as_ref().map(|d| d.stage_id.as_str()),
            Some("stage-1")
        );
        assert!(config.filter_tensors_on_load);
    }

    #[test]
    fn stage_config_rejects_empty_selected_backend_device() {
        let mut request = load_request();
        request.selected_device = Some(StageDevice {
            backend_device: String::new(),
            stable_id: Some("uuid:GPU-123".into()),
            index: Some(0),
            vram_bytes: Some(24_000_000_000),
        });

        let err = stage_config(&request).unwrap_err().to_string();

        assert!(err.contains("selected backend device"));
    }

    #[test]
    fn stage_status_filter_matches_optional_identity_fields() {
        let load = load_request();
        assert!(StageStatusFilter {
            topology_id: Some("topology-a".to_string()),
            run_id: None,
            stage_id: Some("stage-0".to_string()),
        }
        .matches(&load));
        assert!(!StageStatusFilter {
            topology_id: Some("other".to_string()),
            run_id: None,
            stage_id: None,
        }
        .matches(&load));
    }

    #[test]
    fn materialize_stage_bind_addr_replaces_ephemeral_port() {
        let bind_addr = materialize_stage_bind_addr("127.0.0.1:0".parse().unwrap()).unwrap();
        assert_eq!(bind_addr.ip().to_string(), "127.0.0.1");
        assert_ne!(bind_addr.port(), 0);
    }

    #[tokio::test]
    async fn binary_stage_ready_probe_waits_for_wire_handshake() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let bind_addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(75));
            let (mut stream, _) = listener.accept().unwrap();
            skippy_protocol::binary::send_ready(&mut stream).unwrap();
        });

        let started = Instant::now();
        wait_for_binary_stage_ready(bind_addr, Duration::from_secs(2))
            .await
            .unwrap();
        assert!(started.elapsed() >= Duration::from_millis(50));
        server.join().unwrap();
    }
}
