#![allow(dead_code)]

mod deployment;
mod hooks;
mod kv_cache;
mod materialization;
mod package;
mod stage;
mod topology;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use openai_frontend::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, CompletionRequest,
    CompletionResponse, CompletionStream, ModelObject, OpenAiBackend, OpenAiHookPolicy,
    OpenAiRequestContext, OpenAiResult,
};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig, StageDevice};
use skippy_runtime::ModelInfo;
use skippy_server::{
    binary_transport::WireCondition, embedded_openai_backend, openai::CONTEXT_BUDGET_MAX_TOKENS,
    telemetry::Telemetry, telemetry::TelemetryLevel, EmbeddedOpenAiArgs, EmbeddedRuntimeOptions,
    EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedState, SkippyRuntimeHandle,
};

pub(crate) use hooks::MeshAutoHookPolicy;
pub(crate) use kv_cache::KvCachePolicy;
pub(crate) use materialization::{
    configure_materialized_stage_cache, materialize_stage_config, materialize_stage_load,
    materialized_stage_cache_dir, materialized_stages_for_sources,
    prune_unpinned_materialized_stages, remove_materialized_stages_for_sources,
    MaterializedStagePin,
};
pub(crate) use package::{synthetic_direct_gguf_package, SkippyPackageIdentity};
pub(crate) use stage::{
    spawn_stage_control_loop, StageControlCommand, StageControlRequest, StageControlResponse,
    StageLoadRequest, StagePeerDescriptor, StageReadyResponse, StageRuntimeState,
    StageStatusFilter, StageStatusSnapshot, StageStopRequest, StageWireDType,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SkippyModelState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyModelStatus {
    pub(crate) state: SkippyModelState,
    pub(crate) model_id: String,
    pub(crate) backend: &'static str,
    pub(crate) runtime_loaded: bool,
    pub(crate) package_ref: Option<String>,
    pub(crate) manifest_sha256: Option<String>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) materialized_path: Option<String>,
    pub(crate) materialized_pinned: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) lanes: Vec<SkippySessionLaneStatus>,
    pub(crate) max_session_tokens: u64,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) selected_device: Option<SkippyDeviceDescriptor>,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) stage_id: String,
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) started_at_unix_nanos: i64,
    pub(crate) stopped_at_unix_nanos: Option<i64>,
    pub(crate) last_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SkippySessionLaneStatus {
    pub(crate) index: usize,
    pub(crate) active: bool,
    pub(crate) session_id: Option<String>,
    pub(crate) token_count: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SkippyDeviceDescriptor {
    pub(crate) backend_device: String,
    pub(crate) stable_id: Option<String>,
    pub(crate) index: Option<usize>,
    pub(crate) vram_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyModelLoadOptions {
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) ctx_size: u32,
    pub(crate) n_gpu_layers: i32,
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) generation_concurrency: usize,
    pub(crate) default_max_tokens: u32,
    pub(crate) layer_end: Option<u32>,
    pub(crate) selected_device: Option<SkippyDeviceDescriptor>,
    pub(crate) package_identity: Option<SkippyPackageIdentity>,
    pub(crate) projector_path: Option<PathBuf>,
}

impl SkippyModelLoadOptions {
    pub(crate) fn for_direct_gguf(
        model_id: impl Into<String>,
        model_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_path: model_path.into(),
            ctx_size: 4096,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            n_batch: None,
            n_ubatch: None,
            flash_attn_type: FlashAttentionType::Auto,
            generation_concurrency: 1,
            default_max_tokens: CONTEXT_BUDGET_MAX_TOKENS,
            layer_end: None,
            selected_device: None,
            package_identity: None,
            projector_path: None,
        }
    }

    pub(crate) fn with_ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub(crate) fn with_generation_concurrency(mut self, generation_concurrency: usize) -> Self {
        self.generation_concurrency = generation_concurrency;
        self
    }

    pub(crate) fn with_cache_types(mut self, cache_type_k: &str, cache_type_v: &str) -> Self {
        self.cache_type_k = cache_type_k.to_string();
        self.cache_type_v = cache_type_v.to_string();
        self
    }

    pub(crate) fn with_batch_sizes(mut self, n_batch: Option<u32>, n_ubatch: Option<u32>) -> Self {
        self.n_batch = n_batch;
        self.n_ubatch = n_ubatch;
        self
    }

    pub(crate) fn with_flash_attn_type(mut self, flash_attn_type: FlashAttentionType) -> Self {
        self.flash_attn_type = flash_attn_type;
        self
    }

    pub(crate) fn with_layer_end(mut self, layer_end: u32) -> Self {
        self.layer_end = Some(layer_end);
        self
    }

    pub(crate) fn with_selected_device(mut self, selected_device: SkippyDeviceDescriptor) -> Self {
        self.selected_device = Some(selected_device);
        self
    }

    pub(crate) fn with_projector_path(mut self, projector_path: impl Into<PathBuf>) -> Self {
        self.projector_path = Some(projector_path.into());
        self
    }

    #[cfg(test)]
    pub(crate) fn with_package_identity(mut self, package_identity: SkippyPackageIdentity) -> Self {
        self.package_identity = Some(package_identity);
        self
    }
}

#[derive(Debug)]
struct HandleState {
    state: SkippyModelState,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

pub(crate) struct SkippyModelHandle {
    runtime: SkippyRuntimeHandle,
    backend: Arc<dyn OpenAiBackend>,
    config: StageConfig,
    started_at_unix_nanos: i64,
    status: Arc<Mutex<HandleState>>,
    _materialized_pin: Option<MaterializedStagePin>,
}

pub(crate) struct SkippyHttpHandle {
    port: u16,
    server: EmbeddedServerHandle,
}

impl SkippyHttpHandle {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) async fn shutdown(self) -> Result<()> {
        self.server.shutdown().await
    }
}

impl SkippyModelHandle {
    pub(crate) fn load(options: SkippyModelLoadOptions) -> Result<Self> {
        Self::load_with_hooks(options, None)
    }

    pub(crate) fn load_with_hooks(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    ) -> Result<Self> {
        let stage_config = single_stage_config(&options)?;
        let runtime = SkippyRuntimeHandle::load(EmbeddedRuntimeOptions {
            config: stage_config.clone(),
            topology: None,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: TelemetryLevel::Off,
        })
        .with_context(|| {
            format!(
                "load skippy runtime for model {} from {}",
                options.model_id,
                options.model_path.display()
            )
        })?;
        let telemetry = Telemetry::new(None, 0, stage_config.clone(), TelemetryLevel::Off);
        let binding = embedded_openai_backend(EmbeddedOpenAiArgs {
            bind_addr: "127.0.0.1:0"
                .parse()
                .expect("static bind address should parse"),
            config: stage_config.clone(),
            runtime: runtime.runtime(),
            model_id: Some(options.model_id.clone()),
            default_max_tokens: options.default_max_tokens,
            generation_concurrency: options.generation_concurrency,
            prefill_chunk_size: 64,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: 64,
            prefill_adaptive_step: 64,
            prefill_adaptive_max: 512,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            activation_width: 0,
            wire_dtype: skippy_protocol::binary::WireActivationDType::F32,
            downstream_connect_timeout_secs: 30,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            telemetry,
            hook_policy,
        })
        .context("construct skippy OpenAI backend")?;
        Ok(Self {
            runtime,
            backend: binding.backend,
            config: stage_config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _materialized_pin: None,
        })
    }

    pub(crate) fn load_stage0_config(
        mut config: StageConfig,
        activation_width: i32,
        generation_concurrency: usize,
        default_max_tokens: u32,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    ) -> Result<Self> {
        configure_materialized_stage_cache();
        let materialized = materialize_stage_config(&config)?;
        let materialized_pin = materialized.map(|(artifact, pin)| {
            config.manifest_sha256 = Some(artifact.manifest_sha256);
            config.source_model_path = Some(artifact.source_model_path);
            config.source_model_sha256 = Some(artifact.source_model_sha256);
            config.source_model_bytes = artifact.source_model_bytes;
            config.materialized_path = Some(artifact.path.to_string_lossy().to_string());
            config.materialized_pinned = true;
            pin
        });
        let runtime = SkippyRuntimeHandle::load(EmbeddedRuntimeOptions {
            config: config.clone(),
            topology: None,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: TelemetryLevel::Off,
        })
        .with_context(|| {
            format!(
                "load skippy stage 0 runtime for model {} from {:?}",
                config.model_id, config.model_path
            )
        })?;
        let telemetry = Telemetry::new(None, 0, config.clone(), TelemetryLevel::Off);
        let binding = embedded_openai_backend(EmbeddedOpenAiArgs {
            bind_addr: "127.0.0.1:0"
                .parse()
                .expect("static bind address should parse"),
            config: config.clone(),
            runtime: runtime.runtime(),
            model_id: Some(config.model_id.clone()),
            default_max_tokens,
            generation_concurrency,
            prefill_chunk_size: 64,
            prefill_chunk_policy: "fixed".to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: 64,
            prefill_adaptive_step: 64,
            prefill_adaptive_max: 512,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            activation_width,
            wire_dtype: skippy_protocol::binary::WireActivationDType::F16,
            downstream_connect_timeout_secs: 30,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            telemetry,
            hook_policy,
        })
        .context("construct skippy stage 0 OpenAI backend")?;
        Ok(Self {
            runtime,
            backend: binding.backend,
            config,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _materialized_pin: materialized_pin,
        })
    }

    pub(crate) fn backend(&self) -> Arc<dyn OpenAiBackend> {
        self.backend.clone()
    }

    pub(crate) fn start_http(&self, port: u16) -> SkippyHttpHandle {
        let bind_addr = ([127, 0, 0, 1], port).into();
        let server = skippy_server::start_openai_backend(bind_addr, self.backend());
        SkippyHttpHandle { port, server }
    }

    pub(crate) fn status(&self) -> SkippyModelStatus {
        let embedded = self.runtime.status();
        let local = self.status.lock().expect("skippy status lock poisoned");
        status_from_parts(&self.config, &embedded, &local, self.started_at_unix_nanos)
    }

    pub(crate) fn shutdown(&self) {
        {
            let mut state = self.status.lock().expect("skippy status lock poisoned");
            if matches!(state.state, SkippyModelState::Stopped) {
                return;
            }
            state.state = SkippyModelState::Stopping;
        }
        self.runtime.shutdown();
        let mut state = self.status.lock().expect("skippy status lock poisoned");
        state.state = SkippyModelState::Stopped;
        state.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for SkippyModelHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[async_trait]
impl OpenAiBackend for SkippyModelHandle {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.backend.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }
}

pub(crate) fn single_stage_config(options: &SkippyModelLoadOptions) -> Result<StageConfig> {
    anyhow::ensure!(
        options.ctx_size > 0,
        "skippy ctx_size must be greater than zero"
    );
    anyhow::ensure!(
        options.generation_concurrency > 0,
        "skippy generation_concurrency must be greater than zero"
    );
    if let Some(device) = options.selected_device.as_ref() {
        anyhow::ensure!(
            !device.backend_device.is_empty(),
            "skippy selected backend device must not be empty"
        );
    }
    let package_identity = match options.package_identity.as_ref() {
        Some(identity) => identity.clone(),
        None => synthetic_direct_gguf_package(&options.model_id, &options.model_path)?,
    };
    let layer_end = options.layer_end.unwrap_or(package_identity.layer_count);
    anyhow::ensure!(
        layer_end > 0,
        "skippy stage layer_end must be greater than zero"
    );
    let run_id = format!("mesh-skippy-{}", now_unix_nanos());
    Ok(StageConfig {
        run_id: run_id.clone(),
        topology_id: format!("topology-{run_id}"),
        model_id: options.model_id.clone(),
        package_ref: Some(package_identity.package_ref),
        manifest_sha256: Some(package_identity.manifest_sha256),
        source_model_path: Some(
            package_identity
                .source_model_path
                .to_string_lossy()
                .to_string(),
        ),
        source_model_sha256: Some(package_identity.source_model_sha256),
        source_model_bytes: Some(package_identity.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(options.model_path.to_string_lossy().to_string()),
        projector_path: options
            .projector_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size: options.ctx_size,
        lane_count: options.generation_concurrency as u32,
        n_batch: options.n_batch,
        n_ubatch: options.n_ubatch,
        n_gpu_layers: options.n_gpu_layers,
        cache_type_k: options.cache_type_k.clone(),
        cache_type_v: options.cache_type_v.clone(),
        flash_attn_type: options.flash_attn_type,
        filter_tensors_on_load: false,
        selected_device: options.selected_device.clone().map(Into::into),
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    })
}

impl From<SkippyDeviceDescriptor> for StageDevice {
    fn from(device: SkippyDeviceDescriptor) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

impl From<StageDevice> for SkippyDeviceDescriptor {
    fn from(device: StageDevice) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

pub(crate) fn infer_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model metadata {}", path.display()))?;
    let layer_count = info
        .tensors()
        .with_context(|| format!("read model tensors {}", path.display()))?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .with_context(|| format!("infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

fn status_from_parts(
    config: &StageConfig,
    embedded: &EmbeddedRuntimeStatus,
    local: &HandleState,
    started_at_unix_nanos: i64,
) -> SkippyModelStatus {
    SkippyModelStatus {
        state: match local.state {
            SkippyModelState::Starting => SkippyModelState::Starting,
            SkippyModelState::Ready => map_embedded_state(embedded.state),
            SkippyModelState::Stopping => SkippyModelState::Stopping,
            SkippyModelState::Stopped => SkippyModelState::Stopped,
            SkippyModelState::Failed => SkippyModelState::Failed,
        },
        model_id: config.model_id.clone(),
        backend: "skippy",
        runtime_loaded: embedded.runtime_loaded,
        package_ref: config.package_ref.clone(),
        manifest_sha256: config.manifest_sha256.clone(),
        source_model_path: config.source_model_path.clone(),
        source_model_sha256: config.source_model_sha256.clone(),
        source_model_bytes: config.source_model_bytes,
        materialized_path: config.materialized_path.clone(),
        materialized_pinned: config.materialized_pinned,
        projector_path: config.projector_path.clone(),
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        lanes: embedded
            .sessions
            .lanes
            .iter()
            .map(|lane| SkippySessionLaneStatus {
                index: lane.index,
                active: lane.active,
                session_id: lane.session_id.clone(),
                token_count: lane.token_count,
            })
            .collect(),
        max_session_tokens: embedded.sessions.max_session_tokens,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_gpu_layers: config.n_gpu_layers,
        flash_attn_type: config.flash_attn_type,
        selected_device: config.selected_device.clone().map(Into::into),
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        stage_id: config.stage_id.clone(),
        topology_id: config.topology_id.clone(),
        run_id: config.run_id.clone(),
        started_at_unix_nanos,
        stopped_at_unix_nanos: local
            .stopped_at_unix_nanos
            .or(embedded.stopped_at_unix_nanos),
        last_error: local
            .last_error
            .clone()
            .or_else(|| embedded.last_error.clone()),
    }
}

fn map_embedded_state(state: EmbeddedState) -> SkippyModelState {
    match state {
        EmbeddedState::Starting => SkippyModelState::Starting,
        EmbeddedState::Ready => SkippyModelState::Ready,
        EmbeddedState::Stopping => SkippyModelState::Stopping,
        EmbeddedState::Stopped => SkippyModelState::Stopped,
        EmbeddedState::Failed => SkippyModelState::Failed,
    }
}

fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_package_identity(layer_count: u32) -> SkippyPackageIdentity {
        SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_string(),
            source_model_bytes: 1234,
            source_files: Vec::new(),
            layer_count,
            activation_width: 4096,
            tensor_count: 100,
        }
    }

    #[test]
    fn single_stage_config_materializes_direct_gguf_runtime_slice() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36));

        let config = single_stage_config(&options).unwrap();

        assert_eq!(config.model_id, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.model_path.as_deref(), Some("/models/qwen.gguf"));
        assert_eq!(
            config.package_ref.as_deref(),
            Some("gguf:///models/qwen.gguf")
        );
        assert_eq!(
            config.manifest_sha256.as_deref(),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        );
        assert_eq!(
            config.source_model_path.as_deref(),
            Some("/models/qwen.gguf")
        );
        assert_eq!(config.source_model_bytes, Some(1234));
        assert!(config.materialized_path.is_none());
        assert!(!config.materialized_pinned);
        assert_eq!(config.stage_id, "stage-0");
        assert_eq!(config.stage_index, 0);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 36);
        assert_eq!(config.ctx_size, 8192);
        assert_eq!(config.n_gpu_layers, -1);
        assert!(config.selected_device.is_none());
        assert_eq!(config.load_mode, LoadMode::RuntimeSlice);
        assert!(config.upstream.is_none());
        assert!(config.downstream.is_none());
    }

    #[test]
    fn single_stage_config_preserves_projector_path() {
        let options = SkippyModelLoadOptions::for_direct_gguf("Qwen2.5-VL", "/models/qwen-vl.gguf")
            .with_layer_end(36)
            .with_package_identity(fake_package_identity(36))
            .with_projector_path("/models/mmproj-qwen-vl.gguf");

        let config = single_stage_config(&options).unwrap();

        assert_eq!(
            config.projector_path.as_deref(),
            Some("/models/mmproj-qwen-vl.gguf")
        );
    }

    #[test]
    fn single_stage_config_preserves_selected_device_descriptor() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36))
                .with_selected_device(SkippyDeviceDescriptor {
                    backend_device: "CUDA3".into(),
                    stable_id: Some("uuid:GPU-123".into()),
                    index: Some(3),
                    vram_bytes: Some(24_000_000_000),
                });

        let config = single_stage_config(&options).unwrap();
        let device = config.selected_device.expect("device descriptor");

        assert_eq!(device.backend_device, "CUDA3");
        assert_eq!(device.stable_id.as_deref(), Some("uuid:GPU-123"));
        assert_eq!(device.index, Some(3));
        assert_eq!(device.vram_bytes, Some(24_000_000_000));
    }

    #[test]
    fn single_stage_config_rejects_empty_selected_backend_device() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(1)
            .with_selected_device(SkippyDeviceDescriptor {
                backend_device: String::new(),
                stable_id: Some("uuid:GPU-123".into()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            });

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("selected backend device"));
    }

    #[test]
    fn single_stage_config_rejects_empty_layer_range() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(0)
            .with_package_identity(fake_package_identity(1));

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("layer_end"));
    }

    #[test]
    fn embedded_state_maps_to_mesh_skippy_state() {
        assert_eq!(
            map_embedded_state(EmbeddedState::Starting),
            SkippyModelState::Starting
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Ready),
            SkippyModelState::Ready
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Failed),
            SkippyModelState::Failed
        );
    }
}
