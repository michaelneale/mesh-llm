use anyhow::Result;
use std::collections::HashMap;
use std::future::Future;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;
use tokio::net::TcpListener;

fn mmproj_path_for_model(model_name: &str) -> Option<PathBuf> {
    crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|model| {
            model.name == model_name
                || model.file.strip_suffix(".gguf").unwrap_or(&model.file) == model_name
        })
        .and_then(|model| model.mmproj.as_ref())
        .map(|asset| crate::models::catalog::models_dir().join(&asset.file))
        .filter(|path| path.exists())
}

/// Backend-neutral runtime handle for a serving instance.
///
/// This sits above any concrete backend implementation:
/// - llama.cpp subprocesses
/// - MLX in-process serving
/// - future plugin-hosted inference runtimes
#[derive(Clone, Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
    shutdown_tx: Option<tokio::sync::watch::Sender<bool>>,
}

impl InferenceServerHandle {
    pub(crate) fn process(pid: u32, expected_exit: Arc<AtomicBool>) -> Self {
        Self {
            pid,
            expected_exit,
            shutdown_tx: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn in_process(shutdown_tx: tokio::sync::watch::Sender<bool>) -> Self {
        Self {
            pid: std::process::id(),
            expected_exit: Arc::new(AtomicBool::new(true)),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(true);
            return;
        }
        self.expected_exit.store(true, Ordering::Relaxed);
        crate::inference::launch::terminate_process(self.pid).await;
    }
}

/// Backend-neutral runtime result for a started serving instance.
#[derive(Debug)]
pub struct InferenceServerProcess {
    pub handle: InferenceServerHandle,
    pub death_rx: tokio::sync::oneshot::Receiver<()>,
    pub context_length: u32,
    pub listen_port: u16,
}

/// Backend-neutral request to start a serving endpoint for a model.
///
/// Backends can treat `worker_tunnel_ports` / `tensor_split` as the
/// distributed-host shape, or ignore them for single-node serving.
#[derive(Clone, Debug)]
pub struct InferenceEndpointRequest {
    pub preferred_provider_id: Option<String>,
    pub model_path: PathBuf,
    pub listen_port: u16,
    pub worker_tunnel_ports: Vec<u16>,
    pub tensor_split: Option<String>,
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub local_vram_bytes: u64,
    pub mmproj_path: Option<PathBuf>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram_bytes: Option<u64>,
}

impl InferenceEndpointRequest {
    pub fn local(
        model_path: impl Into<PathBuf>,
        listen_port: u16,
        model_bytes: u64,
        local_vram_bytes: u64,
    ) -> Self {
        Self {
            preferred_provider_id: None,
            model_path: model_path.into(),
            listen_port,
            worker_tunnel_ports: Vec::new(),
            tensor_split: None,
            draft_model_path: None,
            draft_max: 0,
            model_bytes,
            local_vram_bytes,
            mmproj_path: None,
            ctx_size_override: None,
            total_group_vram_bytes: None,
        }
    }

    pub fn with_ctx_size_override(mut self, ctx_size_override: Option<u32>) -> Self {
        self.ctx_size_override = ctx_size_override;
        self
    }

    #[allow(dead_code)]
    pub fn with_preferred_provider_id(
        mut self,
        preferred_provider_id: Option<impl Into<String>>,
    ) -> Self {
        self.preferred_provider_id = preferred_provider_id.map(Into::into);
        self
    }

    pub fn with_mmproj_path(mut self, mmproj_path: Option<impl AsRef<Path>>) -> Self {
        self.mmproj_path = mmproj_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn distributed_host(
        model_path: impl Into<PathBuf>,
        listen_port: u16,
        worker_tunnel_ports: Vec<u16>,
        model_bytes: u64,
        local_vram_bytes: u64,
    ) -> Self {
        Self {
            preferred_provider_id: None,
            model_path: model_path.into(),
            listen_port,
            worker_tunnel_ports,
            tensor_split: None,
            draft_model_path: None,
            draft_max: 0,
            model_bytes,
            local_vram_bytes,
            mmproj_path: None,
            ctx_size_override: None,
            total_group_vram_bytes: None,
        }
    }

    pub fn with_tensor_split(mut self, tensor_split: Option<impl Into<String>>) -> Self {
        self.tensor_split = tensor_split.map(Into::into);
        self
    }

    pub fn with_draft_model_path(mut self, draft_model_path: Option<impl AsRef<Path>>) -> Self {
        self.draft_model_path = draft_model_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn with_draft_max(mut self, draft_max: u16) -> Self {
        self.draft_max = draft_max;
        self
    }

    pub fn with_total_group_vram_bytes(mut self, total_group_vram_bytes: Option<u64>) -> Self {
        self.total_group_vram_bytes = total_group_vram_bytes;
        self
    }
}

/// Backend-neutral request to start a worker-side runtime helper.
#[derive(Clone, Debug, Default)]
pub struct InferenceWorkerRequest {
    pub preferred_provider_id: Option<String>,
    pub model_path: Option<PathBuf>,
    pub device_hint: Option<String>,
}

impl InferenceWorkerRequest {
    pub fn with_model_path(mut self, model_path: Option<impl AsRef<Path>>) -> Self {
        self.model_path = model_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    #[allow(dead_code)]
    pub fn with_preferred_provider_id(
        mut self,
        preferred_provider_id: Option<impl Into<String>>,
    ) -> Self {
        self.preferred_provider_id = preferred_provider_id.map(Into::into);
        self
    }

    pub fn with_device_hint(mut self, device_hint: Option<impl Into<String>>) -> Self {
        self.device_hint = device_hint.map(Into::into);
        self
    }
}

type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Clone, Debug)]
pub struct FullAnalyzeRankingRequest {
    pub bin_dir: PathBuf,
    pub model_name: String,
    pub model_path: PathBuf,
    pub cached_path: PathBuf,
}

impl FullAnalyzeRankingRequest {
    pub fn new(
        bin_dir: impl Into<PathBuf>,
        model_name: impl Into<String>,
        model_path: impl Into<PathBuf>,
        cached_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            bin_dir: bin_dir.into(),
            model_name: model_name.into(),
            model_path: model_path.into(),
            cached_path: cached_path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MicroAnalyzeRankingRequest {
    pub bin_dir: PathBuf,
    pub model_name: String,
    pub model_path: PathBuf,
    pub options: crate::inference::moe::MoeRuntimeOptions,
}

impl MicroAnalyzeRankingRequest {
    pub fn new(
        bin_dir: impl Into<PathBuf>,
        model_name: impl Into<String>,
        model_path: impl Into<PathBuf>,
        options: crate::inference::moe::MoeRuntimeOptions,
    ) -> Self {
        Self {
            bin_dir: bin_dir.into(),
            model_name: model_name.into(),
            model_path: model_path.into(),
            options,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HeuristicRankingRequest {
    pub model_path: PathBuf,
    pub expert_count: u32,
    pub method: crate::inference::moe::HeuristicScoreMethod,
}

impl HeuristicRankingRequest {
    pub fn new(
        model_path: impl Into<PathBuf>,
        expert_count: u32,
        method: crate::inference::moe::HeuristicScoreMethod,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            expert_count,
            method,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoeDetectionRequest {
    pub model_path: PathBuf,
}

impl MoeDetectionRequest {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CachedRankingRequest {
    pub model_path: PathBuf,
}

impl CachedRankingRequest {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SharedRankingArtifactLookupRequest {
    pub model_path: PathBuf,
}

impl SharedRankingArtifactLookupRequest {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SharedRankingArtifactImportRequest {
    pub model_path: PathBuf,
    pub artifact: crate::inference::moe::SharedRankingArtifact,
}

impl SharedRankingArtifactImportRequest {
    pub fn new(
        model_path: impl Into<PathBuf>,
        artifact: crate::inference::moe::SharedRankingArtifact,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            artifact,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoeShardPreparationRequest {
    pub bin_dir: PathBuf,
    pub model_path: PathBuf,
    pub assignment: crate::inference::moe::NodeAssignment,
    pub output_path: PathBuf,
}

impl MoeShardPreparationRequest {
    pub fn new(
        bin_dir: impl Into<PathBuf>,
        model_path: impl Into<PathBuf>,
        assignment: crate::inference::moe::NodeAssignment,
        output_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            bin_dir: bin_dir.into(),
            model_path: model_path.into(),
            assignment,
            output_path: output_path.into(),
        }
    }
}

pub trait MoeRankingProvider: Send + Sync {
    fn detect_moe(&self, request: &MoeDetectionRequest)
        -> Option<crate::models::gguf::GgufMoeInfo>;

    fn load_cached_ranking(&self, request: &CachedRankingRequest) -> Option<Vec<u32>>;

    fn best_shared_ranking_artifact(
        &self,
        request: &SharedRankingArtifactLookupRequest,
    ) -> Option<crate::inference::moe::SharedRankingArtifact>;

    fn import_shared_ranking_artifact(
        &self,
        request: &SharedRankingArtifactImportRequest,
    ) -> Result<bool>;

    fn ensure_full_analyze_ranking(
        &self,
        request: &FullAnalyzeRankingRequest,
    ) -> Result<crate::inference::moe::SharedRankingArtifact>;

    fn ensure_micro_analyze_ranking(
        &self,
        request: &MicroAnalyzeRankingRequest,
    ) -> Result<crate::inference::moe::SharedRankingArtifact>;

    fn resolve_heuristic_ranking(
        &self,
        request: &HeuristicRankingRequest,
    ) -> Result<(Vec<u32>, String, String)>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceProviderCapabilities {
    pub supports_local_runtime: bool,
    pub supports_distributed_host_runtime: bool,
    pub requires_worker_runtime: bool,
    pub supports_moe_shard_runtime: bool,
}

#[derive(Clone, Debug)]
pub struct InferenceProviderDescriptor {
    selection: InferenceProviderSelection,
    matches_local_endpoint: fn(&InferenceEndpointRequest) -> bool,
    matches_distributed_endpoint: fn(&InferenceEndpointRequest) -> bool,
    matches_worker_runtime: fn(&InferenceWorkerRequest) -> bool,
}

impl InferenceProviderDescriptor {
    pub const fn new(
        selection: InferenceProviderSelection,
        matches_local_endpoint: fn(&InferenceEndpointRequest) -> bool,
        matches_distributed_endpoint: fn(&InferenceEndpointRequest) -> bool,
        matches_worker_runtime: fn(&InferenceWorkerRequest) -> bool,
    ) -> Self {
        Self {
            selection,
            matches_local_endpoint,
            matches_distributed_endpoint,
            matches_worker_runtime,
        }
    }
}

#[derive(Clone)]
pub struct InferenceProviderSelection {
    provider_id: Arc<str>,
    backend_label: Arc<str>,
    capabilities: InferenceProviderCapabilities,
    provider: Arc<dyn InferenceProvider>,
    moe_ranking_provider: Option<Arc<dyn MoeRankingProvider>>,
}

impl InferenceProviderSelection {
    pub fn new(
        provider_id: impl Into<Arc<str>>,
        backend_label: impl Into<Arc<str>>,
        capabilities: InferenceProviderCapabilities,
        provider: Arc<dyn InferenceProvider>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            backend_label: backend_label.into(),
            capabilities,
            provider,
            moe_ranking_provider: None,
        }
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn backend_label(&self) -> &str {
        &self.backend_label
    }

    pub fn capabilities(&self) -> InferenceProviderCapabilities {
        self.capabilities
    }

    pub fn provider(&self) -> &dyn InferenceProvider {
        self.provider.as_ref()
    }

    pub fn with_moe_ranking_provider(
        mut self,
        moe_ranking_provider: Arc<dyn MoeRankingProvider>,
    ) -> Self {
        self.moe_ranking_provider = Some(moe_ranking_provider);
        self
    }

    fn cloned_moe_ranking_provider(&self) -> Option<Arc<dyn MoeRankingProvider>> {
        self.moe_ranking_provider.clone()
    }
}

impl std::fmt::Debug for InferenceProviderSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceProviderSelection")
            .field("provider_id", &self.provider_id())
            .field("backend_label", &self.backend_label())
            .finish()
    }
}

impl PartialEq for InferenceProviderSelection {
    fn eq(&self, other: &Self) -> bool {
        self.provider_id() == other.provider_id()
    }
}

impl Eq for InferenceProviderSelection {}

#[derive(Clone)]
pub struct MoeRankingProviderSelection {
    provider_id: Arc<str>,
    backend_label: Arc<str>,
    ranking_provider: Arc<dyn MoeRankingProvider>,
}

impl MoeRankingProviderSelection {
    pub fn new(
        provider_id: impl Into<Arc<str>>,
        backend_label: impl Into<Arc<str>>,
        ranking_provider: Arc<dyn MoeRankingProvider>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            backend_label: backend_label.into(),
            ranking_provider,
        }
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn backend_label(&self) -> &str {
        &self.backend_label
    }

    pub fn ranking_provider(&self) -> &dyn MoeRankingProvider {
        self.ranking_provider.as_ref()
    }
}

impl std::fmt::Debug for MoeRankingProviderSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeRankingProviderSelection")
            .field("provider_id", &self.provider_id())
            .field("backend_label", &self.backend_label())
            .finish()
    }
}

impl PartialEq for MoeRankingProviderSelection {
    fn eq(&self, other: &Self) -> bool {
        self.provider_id() == other.provider_id()
    }
}

impl Eq for MoeRankingProviderSelection {}

#[derive(Clone, Debug)]
pub struct MoeRankingProviderDescriptor {
    selection: MoeRankingProviderSelection,
    matches_model: fn(&Path) -> bool,
}

impl MoeRankingProviderDescriptor {
    pub const fn new(
        selection: MoeRankingProviderSelection,
        matches_model: fn(&Path) -> bool,
    ) -> Self {
        Self {
            selection,
            matches_model,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferenceProviderRegistry {
    builtin_providers: Vec<InferenceProviderDescriptor>,
}

impl InferenceProviderRegistry {
    pub fn new(builtin_providers: Vec<InferenceProviderDescriptor>) -> Self {
        Self { builtin_providers }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn register_provider(&self, descriptor: InferenceProviderDescriptor) {
        let mut providers = registered_provider_descriptors()
            .write()
            .expect("registered inference provider lock poisoned");
        providers.retain(|existing| existing.selection != descriptor.selection);
        providers.push(descriptor);
    }

    fn select_provider(
        &self,
        preferred_provider_id: Option<&str>,
        capability_filter: impl Fn(InferenceProviderCapabilities) -> bool,
        dynamic_matches: impl Fn(&InferenceProviderDescriptor) -> bool,
        builtin_matches: impl Fn(&InferenceProviderDescriptor) -> bool,
        empty_message: &'static str,
    ) -> InferenceProviderSelection {
        if let Some(selection) =
            self.select_preferred_provider(preferred_provider_id, &capability_filter)
        {
            return selection;
        }

        if let Some(selection) = registered_provider_descriptors()
            .read()
            .expect("registered inference provider lock poisoned")
            .iter()
            .find(|descriptor| {
                capability_filter(descriptor.selection.capabilities())
                    && dynamic_matches(descriptor)
            })
            .map(|descriptor| descriptor.selection.clone())
        {
            return selection;
        }

        self.builtin_providers
            .iter()
            .find(|descriptor| {
                capability_filter(descriptor.selection.capabilities())
                    && builtin_matches(descriptor)
            })
            .map(|descriptor| descriptor.selection.clone())
            .expect(empty_message)
    }

    fn select_preferred_provider(
        &self,
        preferred_provider_id: Option<&str>,
        capability_filter: &impl Fn(InferenceProviderCapabilities) -> bool,
    ) -> Option<InferenceProviderSelection> {
        let preferred_provider_id = preferred_provider_id?;

        if let Some(selection) = registered_provider_descriptors()
            .read()
            .expect("registered inference provider lock poisoned")
            .iter()
            .find(|descriptor| {
                descriptor.selection.provider_id() == preferred_provider_id
                    && capability_filter(descriptor.selection.capabilities())
            })
            .map(|descriptor| descriptor.selection.clone())
        {
            return Some(selection);
        }

        self.builtin_providers
            .iter()
            .find(|descriptor| {
                descriptor.selection.provider_id() == preferred_provider_id
                    && capability_filter(descriptor.selection.capabilities())
            })
            .map(|descriptor| descriptor.selection.clone())
    }

    pub fn select_local_endpoint_provider(
        &self,
        request: &InferenceEndpointRequest,
    ) -> InferenceProviderSelection {
        self.select_provider(
            request.preferred_provider_id.as_deref(),
            |capabilities| capabilities.supports_local_runtime,
            |descriptor| (descriptor.matches_local_endpoint)(request),
            |descriptor| (descriptor.matches_local_endpoint)(request),
            "at least one local inference provider must be registered",
        )
    }

    pub fn select_distributed_endpoint_provider(
        &self,
        request: &InferenceEndpointRequest,
    ) -> InferenceProviderSelection {
        self.select_provider(
            request.preferred_provider_id.as_deref(),
            |capabilities| capabilities.supports_distributed_host_runtime,
            |descriptor| (descriptor.matches_distributed_endpoint)(request),
            |descriptor| (descriptor.matches_distributed_endpoint)(request),
            "at least one distributed inference provider must be registered",
        )
    }

    pub fn select_worker_provider(
        &self,
        request: &InferenceWorkerRequest,
    ) -> InferenceProviderSelection {
        self.select_provider(
            request.preferred_provider_id.as_deref(),
            |capabilities| capabilities.requires_worker_runtime,
            |descriptor| (descriptor.matches_worker_runtime)(request),
            |descriptor| (descriptor.matches_worker_runtime)(request),
            "at least one worker inference provider must be registered",
        )
    }
}

pub trait InferenceProvider: Send + Sync {
    fn start_endpoint<'a>(
        &'a self,
        bin_dir: &'a Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceEndpointRequest,
    ) -> ProviderFuture<'a, InferenceServerProcess>;

    fn start_worker<'a>(
        &'a self,
        bin_dir: &'a Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceWorkerRequest,
    ) -> ProviderFuture<'a, u16>;

    fn prepare_moe_shard(&self, request: &MoeShardPreparationRequest) -> Result<()>;
}

/// Built-in provider adapter for the current llama.cpp runtime path.
///
/// This is the first step toward a pluggable backend provider interface:
/// core call sites depend on the provider contract, while the built-in
/// provider still delegates to the existing llama-specific launch code.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinLlamaProvider;

#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinLlamaMoeRankingProvider;

impl MoeRankingProvider for BuiltinLlamaMoeRankingProvider {
    fn detect_moe(
        &self,
        request: &MoeDetectionRequest,
    ) -> Option<crate::models::gguf::GgufMoeInfo> {
        builtin_llama_detect_moe(request)
    }

    fn load_cached_ranking(&self, request: &CachedRankingRequest) -> Option<Vec<u32>> {
        builtin_llama_load_cached_ranking(request)
    }

    fn best_shared_ranking_artifact(
        &self,
        request: &SharedRankingArtifactLookupRequest,
    ) -> Option<crate::inference::moe::SharedRankingArtifact> {
        builtin_llama_best_shared_ranking_artifact(request)
    }

    fn import_shared_ranking_artifact(
        &self,
        request: &SharedRankingArtifactImportRequest,
    ) -> Result<bool> {
        builtin_llama_import_shared_ranking_artifact(request)
    }

    fn ensure_full_analyze_ranking(
        &self,
        request: &FullAnalyzeRankingRequest,
    ) -> Result<crate::inference::moe::SharedRankingArtifact> {
        builtin_llama_ensure_full_analyze_ranking(request)
    }

    fn ensure_micro_analyze_ranking(
        &self,
        request: &MicroAnalyzeRankingRequest,
    ) -> Result<crate::inference::moe::SharedRankingArtifact> {
        builtin_llama_ensure_micro_analyze_ranking(request)
    }

    fn resolve_heuristic_ranking(
        &self,
        request: &HeuristicRankingRequest,
    ) -> Result<(Vec<u32>, String, String)> {
        builtin_llama_resolve_heuristic_ranking(request)
    }
}

fn builtin_llama_detect_moe(
    request: &MoeDetectionRequest,
) -> Option<crate::models::gguf::GgufMoeInfo> {
    crate::models::gguf::detect_moe(&request.model_path)
}

fn builtin_llama_load_cached_ranking(request: &CachedRankingRequest) -> Option<Vec<u32>> {
    let ranking_path = crate::inference::moe::ranking_cache_path(&request.model_path);
    crate::inference::moe::load_cached_ranking(&ranking_path)
}

fn builtin_llama_best_shared_ranking_artifact(
    request: &SharedRankingArtifactLookupRequest,
) -> Option<crate::inference::moe::SharedRankingArtifact> {
    crate::inference::moe::best_shared_ranking_artifact(&request.model_path)
}

fn builtin_llama_import_shared_ranking_artifact(
    request: &SharedRankingArtifactImportRequest,
) -> Result<bool> {
    crate::inference::moe::cache_shared_ranking_if_stronger(&request.model_path, &request.artifact)
}

fn builtin_llama_ensure_full_analyze_ranking(
    request: &FullAnalyzeRankingRequest,
) -> Result<crate::inference::moe::SharedRankingArtifact> {
    let bin_dir = request.bin_dir.as_path();
    let model_name = request.model_name.as_str();
    let model_path = request.model_path.as_path();
    let cached_path = request.cached_path.as_path();
    if let Some(artifact) = crate::inference::moe::load_shared_ranking_artifact(
        cached_path,
        crate::inference::moe::SharedRankingKind::Analyze,
        crate::inference::moe::SharedRankingOrigin::LegacyCache,
        None,
        None,
        None,
    ) {
        eprintln!(
            "🧩 [{model_name}] Using cached MoE ranking mode=full-analyze origin={} cache={}",
            artifact.origin.label(),
            cached_path.display()
        );
        return Ok(artifact);
    }
    if let Some(parent) = cached_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let started = std::time::Instant::now();
    eprintln!(
        "🧩 [{model_name}] MoE analysis mode=full-analyze cache={}",
        cached_path.display()
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState::default()));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "full-analyze",
        Arc::clone(&progress),
        started,
    );
    let mut child = Command::new(&analyze_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--all-layers",
            "--export-ranking",
            &cached_path.to_string_lossy(),
            "-n",
            "32",
            "-c",
            "4096",
            "-ngl",
            "99",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout_relay = child.stdout.take().map(|stdout| {
        spawn_moe_analyze_log_relay(stdout, model_name.to_string(), Arc::clone(&progress))
    });
    let stderr_relay = child.stderr.take().map(|stderr| {
        spawn_moe_analyze_log_relay(stderr, model_name.to_string(), Arc::clone(&progress))
    });
    let status = child.wait()?;
    if let Some(handle) = stdout_relay {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_relay {
        let _ = handle.join();
    }
    if let Ok(mut state) = progress.lock() {
        if let Some(total) = state.total_prompts {
            state.current_prompt = total;
        }
        state.done = true;
    }
    let _ = spinner.join();
    anyhow::ensure!(status.success(), "llama-moe-analyze exited with {status}");
    let ranking = crate::inference::moe::load_cached_ranking(cached_path).ok_or_else(|| {
        anyhow::anyhow!(
            "No ranking produced by full analyze at {}",
            cached_path.display()
        )
    })?;
    let artifact = crate::inference::moe::SharedRankingArtifact {
        kind: crate::inference::moe::SharedRankingKind::Analyze,
        origin: crate::inference::moe::SharedRankingOrigin::LocalFullAnalyze,
        ranking,
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    crate::inference::moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    eprintln!(
        "  Full moe-analyze cached at {} in {:.1}s (origin={})",
        cached_path.display(),
        started.elapsed().as_secs_f64(),
        artifact.origin.label()
    );
    Ok(artifact)
}

fn builtin_llama_ensure_micro_analyze_ranking(
    request: &MicroAnalyzeRankingRequest,
) -> Result<crate::inference::moe::SharedRankingArtifact> {
    let bin_dir = request.bin_dir.as_path();
    let model_name = request.model_name.as_str();
    let model_path = request.model_path.as_path();
    let options = &request.options;
    let cached_path = crate::inference::moe::micro_ranking_cache_path(
        model_path,
        options.micro_prompt_count,
        options.micro_tokens,
        options.micro_layer_scope,
    );
    if let Some(artifact) = crate::inference::moe::load_shared_ranking_artifact(
        &cached_path,
        crate::inference::moe::SharedRankingKind::MicroAnalyze,
        crate::inference::moe::SharedRankingOrigin::LegacyCache,
        Some(options.micro_prompt_count),
        Some(options.micro_tokens),
        Some(options.micro_layer_scope),
    ) {
        eprintln!(
            "🧩 [{model_name}] Using cached MoE ranking mode=micro-analyze origin={} cache={}",
            artifact.origin.label(),
            cached_path.display()
        );
        return Ok(artifact);
    }
    let ranking = run_micro_analyze_ranking(bin_dir, model_name, model_path, options)?;
    let artifact = crate::inference::moe::SharedRankingArtifact {
        kind: crate::inference::moe::SharedRankingKind::MicroAnalyze,
        origin: crate::inference::moe::SharedRankingOrigin::LocalMicroAnalyze,
        ranking,
        micro_prompt_count: Some(options.micro_prompt_count),
        micro_tokens: Some(options.micro_tokens),
        micro_layer_scope: Some(options.micro_layer_scope),
    };
    crate::inference::moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    eprintln!(
        "  Micro moe-analyze cached at {} (origin={})",
        cached_path.display(),
        artifact.origin.label()
    );
    Ok(artifact)
}

fn builtin_llama_resolve_heuristic_ranking(
    request: &HeuristicRankingRequest,
) -> Result<(Vec<u32>, String, String)> {
    let model_path = request.model_path.as_path();
    let expert_count = request.expert_count;
    let method = request.method;
    let cached = crate::inference::moe::heuristic_ranking_cache_path_for_method(model_path, method);
    if let Some(ranking) = crate::inference::moe::load_cached_ranking(&cached) {
        return Ok((
            ranking,
            format!("heuristic-{}", method.cache_suffix()),
            "local-heuristic-cache".to_string(),
        ));
    }
    let ranking = crate::inference::moe::compute_heuristic_ranking_with_method(
        model_path,
        expert_count,
        method,
    )?;
    crate::inference::moe::write_cached_ranking(&cached, &ranking)?;
    Ok((
        ranking,
        format!("heuristic-{}", method.cache_suffix()),
        "local-heuristic".to_string(),
    ))
}

impl InferenceProvider for BuiltinLlamaProvider {
    fn start_endpoint<'a>(
        &'a self,
        bin_dir: &'a Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceEndpointRequest,
    ) -> ProviderFuture<'a, InferenceServerProcess> {
        Box::pin(async move {
            crate::inference::launch::start_llama_server(bin_dir, binary_flavor, request).await
        })
    }

    fn start_worker<'a>(
        &'a self,
        bin_dir: &'a Path,
        binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceWorkerRequest,
    ) -> ProviderFuture<'a, u16> {
        Box::pin(async move {
            crate::inference::launch::start_rpc_server(bin_dir, binary_flavor, request).await
        })
    }

    fn prepare_moe_shard(&self, request: &MoeShardPreparationRequest) -> Result<()> {
        crate::inference::moe::run_split(
            &request.bin_dir,
            &request.model_path,
            &request.assignment,
            &request.output_path,
        )
    }
}

fn resolve_analyze_binary(bin_dir: &Path) -> Result<std::path::PathBuf> {
    let candidates = [
        bin_dir.join("llama-moe-analyze"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-analyze"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    anyhow::bail!(
        "llama-moe-analyze not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn should_suppress_moe_analyze_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with("print_info:")
}

fn should_relay_moe_analyze_warning(line: &str) -> bool {
    let trimmed = line.trim();
    if should_suppress_moe_analyze_line(trimmed) {
        return false;
    }

    trimmed.starts_with("W ")
        || trimmed.starts_with("E ")
        || trimmed.to_ascii_lowercase().contains("failed")
        || trimmed.to_ascii_lowercase().contains("error")
}

#[derive(Default)]
struct MoeAnalyzeProgressState {
    current_prompt: usize,
    total_prompts: Option<usize>,
    done: bool,
}

fn format_moe_analysis_progress_line(
    model_name: &str,
    mode: &str,
    spinner: &str,
    current: usize,
    total: Option<usize>,
    elapsed: std::time::Duration,
) -> String {
    let progress = match total {
        Some(total) if total > 0 => format!(
            "{:>5.1}%  {}/{}",
            (current as f64 / total as f64) * 100.0,
            current,
            total
        ),
        Some(total) => format!("       0/{}", total),
        None => "starting".to_string(),
    };
    format!(
        "🧩 [{}] {:<17} {}  {:>3}s",
        model_name,
        format!("MoE {mode}"),
        format!("{spinner} {progress}"),
        elapsed.as_secs()
    )
}

fn spawn_moe_analysis_spinner(
    model_name: String,
    mode: &'static str,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
    started: std::time::Instant,
) -> thread::JoinHandle<()> {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    thread::spawn(move || {
        let mut frame_idx = 0usize;
        loop {
            let (current, total, done) = progress
                .lock()
                .map(|state| (state.current_prompt, state.total_prompts, state.done))
                .unwrap_or((0, None, true));
            let spinner = if done {
                "✓"
            } else {
                FRAMES[frame_idx % FRAMES.len()]
            };
            let line = format_moe_analysis_progress_line(
                &model_name,
                mode,
                spinner,
                current,
                total,
                started.elapsed(),
            );
            eprint!("\r\x1b[2K{line}");
            let _ = std::io::stderr().flush();
            if done {
                eprintln!();
                break;
            }
            frame_idx += 1;
            thread::sleep(std::time::Duration::from_millis(125));
        }
    })
}

fn parse_moe_analyze_prompt_total(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Running ")?;
    let prompt_count = rest.split_whitespace().next()?;
    prompt_count.parse::<usize>().ok()
}

fn parse_moe_analyze_prompt_progress(line: &str) -> Option<(usize, usize)> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Prompt ")?;
    let progress = rest.split(':').next()?.trim();
    let (current, total) = progress.split_once('/')?;
    Some((current.parse::<usize>().ok()?, total.parse::<usize>().ok()?))
}

fn spawn_moe_analyze_log_relay<R: std::io::Read + Send + 'static>(
    reader: R,
    model_name: String,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(total) = parse_moe_analyze_prompt_total(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                }
                continue;
            }
            if let Some((current, total)) = parse_moe_analyze_prompt_progress(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                    state.current_prompt = current.saturating_sub(1);
                }
                continue;
            }
            if should_relay_moe_analyze_warning(&line) {
                eprint!("\r\x1b[2K");
                eprintln!("  [{model_name}] {line}");
            }
        }
    })
}

#[derive(Clone, Copy)]
struct AnalyzeMassRow {
    expert_id: u32,
    gate_mass: f64,
}

fn run_micro_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &crate::inference::moe::MoeRuntimeOptions,
) -> Result<Vec<u32>> {
    let prompts = default_micro_prompts();
    let prompt_count = options.micro_prompt_count.max(1).min(prompts.len());
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let timestamp_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!(
        "mesh-llm-micro-live-{}-{}",
        std::process::id(),
        timestamp_nanos
    ));
    std::fs::create_dir_all(&tmp_dir)?;
    let started = std::time::Instant::now();
    let mut mass_by_expert: HashMap<u32, f64> = HashMap::new();
    eprintln!(
        "🧩 [{model_name}] MoE analysis mode=micro-analyze prompts={} tokens={} layers={} cache=pending",
        prompt_count,
        options.micro_tokens,
        match options.micro_layer_scope {
            crate::inference::moe::MoeMicroLayerScope::All => "all",
            crate::inference::moe::MoeMicroLayerScope::First => "first",
        }
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState {
        current_prompt: 0,
        total_prompts: Some(prompt_count),
        done: false,
    }));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "micro-analyze",
        Arc::clone(&progress),
        started,
    );

    for (idx, prompt) in prompts.iter().take(prompt_count).enumerate() {
        let output_path = tmp_dir.join(format!("prompt-{idx}.csv"));
        let mut command = Command::new(&analyze_bin);
        command.args([
            "-m",
            &model_path.to_string_lossy(),
            "--export-ranking",
            &output_path.to_string_lossy(),
            "-n",
            &options.micro_tokens.to_string(),
            "-c",
            "4096",
            "-ngl",
            "99",
            "-p",
            prompt,
        ]);
        if matches!(
            options.micro_layer_scope,
            crate::inference::moe::MoeMicroLayerScope::All
        ) {
            command.arg("--all-layers");
        }
        let output = command.output()?;
        if !output.status.success() {
            if let Ok(mut state) = progress.lock() {
                state.done = true;
            }
            let _ = spinner.join();
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut details = stderr
                .lines()
                .chain(stdout.lines())
                .filter(|line| !should_suppress_moe_analyze_line(line))
                .collect::<Vec<_>>();
            if details.len() > 20 {
                details.truncate(20);
            }
            let detail_text = if details.is_empty() {
                String::new()
            } else {
                format!(": {}", details.join(" | "))
            };
            anyhow::bail!(
                "llama-moe-analyze exited with {}{}",
                output.status,
                detail_text
            );
        }
        for row in load_analyze_mass_rows(&output_path)? {
            *mass_by_expert.entry(row.expert_id).or_insert(0.0) += row.gate_mass;
        }
        if let Ok(mut state) = progress.lock() {
            state.current_prompt = idx + 1;
        }
    }
    if let Ok(mut state) = progress.lock() {
        state.current_prompt = prompt_count;
        state.done = true;
    }
    let _ = spinner.join();

    let mut rows = mass_by_expert.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let ranking = rows.into_iter().map(|(expert_id, _)| expert_id).collect();
    let _ = std::fs::remove_dir_all(&tmp_dir);
    eprintln!(
        "  Micro moe-analyze used {} prompt(s), {} token(s), {} in {:.1}s",
        prompt_count,
        options.micro_tokens,
        match options.micro_layer_scope {
            crate::inference::moe::MoeMicroLayerScope::All => "all layers",
            crate::inference::moe::MoeMicroLayerScope::First => "first layer",
        },
        started.elapsed().as_secs_f64()
    );
    Ok(ranking)
}

fn load_analyze_mass_rows(path: &Path) -> Result<Vec<AnalyzeMassRow>> {
    let content = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        rows.push(AnalyzeMassRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
        });
    }
    Ok(rows)
}

fn default_micro_prompts() -> &'static [&'static str] {
    &[
        "User: Explain how mixture-of-experts routing works in a language model.\nAssistant:",
        "User: Write a short professional email asking for feedback on a technical design.\nAssistant:",
        "User: Outline a debugging plan for a flaky distributed systems test.\nAssistant:",
        "User: Summarize the tradeoffs between latency and quality in MoE inference.\nAssistant:",
    ]
}

const fn always_match_local_endpoint(_request: &InferenceEndpointRequest) -> bool {
    true
}

const fn always_match_distributed_endpoint(_request: &InferenceEndpointRequest) -> bool {
    true
}

const fn always_match_worker_runtime(_request: &InferenceWorkerRequest) -> bool {
    true
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct PluginInferenceProviderRegistration {
    provider_id: String,
    backend_label: String,
    capabilities: InferenceProviderCapabilities,
    moe_ranking_provider: Option<Arc<dyn MoeRankingProvider>>,
}

impl PluginInferenceProviderRegistration {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(
        provider_id: impl Into<String>,
        backend_label: impl Into<String>,
        capabilities: InferenceProviderCapabilities,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            backend_label: backend_label.into(),
            capabilities,
            moe_ranking_provider: None,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_moe_ranking_provider(
        mut self,
        moe_ranking_provider: Arc<dyn MoeRankingProvider>,
    ) -> Self {
        self.moe_ranking_provider = Some(moe_ranking_provider);
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn into_descriptor(
        self,
        provider: Arc<dyn InferenceProvider>,
    ) -> InferenceProviderDescriptor {
        let selection = if let Some(moe_ranking_provider) = self.moe_ranking_provider {
            InferenceProviderSelection::new(
                self.provider_id,
                self.backend_label,
                self.capabilities,
                provider,
            )
            .with_moe_ranking_provider(moe_ranking_provider)
        } else {
            InferenceProviderSelection::new(
                self.provider_id,
                self.backend_label,
                self.capabilities,
                provider,
            )
        };
        InferenceProviderDescriptor::new(
            selection,
            never_match_local_endpoint,
            never_match_distributed_endpoint,
            never_match_worker_runtime,
        )
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct PluginMoeRankingProviderRegistration {
    provider_id: String,
    backend_label: String,
    matches_model: fn(&Path) -> bool,
}

impl PluginMoeRankingProviderRegistration {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(provider_id: impl Into<String>, backend_label: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            backend_label: backend_label.into(),
            matches_model: never_match_model_for_moe_ranking,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_model_matcher(mut self, matches_model: fn(&Path) -> bool) -> Self {
        self.matches_model = matches_model;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn into_descriptor(
        self,
        ranking_provider: Arc<dyn MoeRankingProvider>,
    ) -> MoeRankingProviderDescriptor {
        MoeRankingProviderDescriptor::new(
            MoeRankingProviderSelection::new(
                self.provider_id,
                self.backend_label,
                ranking_provider,
            ),
            self.matches_model,
        )
    }
}

fn builtin_llama_selection() -> InferenceProviderSelection {
    InferenceProviderSelection::new(
        "builtin.llama",
        "llama",
        InferenceProviderCapabilities {
            supports_local_runtime: true,
            supports_distributed_host_runtime: true,
            requires_worker_runtime: true,
            supports_moe_shard_runtime: true,
        },
        Arc::new(BuiltinLlamaProvider),
    )
    .with_moe_ranking_provider(Arc::new(BuiltinLlamaMoeRankingProvider))
}

fn builtin_provider_registry() -> &'static InferenceProviderRegistry {
    static BUILTIN_PROVIDER_REGISTRY: OnceLock<InferenceProviderRegistry> = OnceLock::new();
    BUILTIN_PROVIDER_REGISTRY.get_or_init(|| {
        InferenceProviderRegistry::new(vec![InferenceProviderDescriptor::new(
            builtin_llama_selection(),
            always_match_local_endpoint,
            always_match_distributed_endpoint,
            always_match_worker_runtime,
        )])
    })
}

pub fn provider_registry() -> &'static InferenceProviderRegistry {
    builtin_provider_registry()
}

fn registered_provider_descriptors() -> &'static RwLock<Vec<InferenceProviderDescriptor>> {
    static REGISTERED_PROVIDER_DESCRIPTORS: OnceLock<RwLock<Vec<InferenceProviderDescriptor>>> =
        OnceLock::new();
    REGISTERED_PROVIDER_DESCRIPTORS.get_or_init(|| RwLock::new(Vec::new()))
}

fn registered_moe_ranking_provider_descriptors(
) -> &'static RwLock<Vec<MoeRankingProviderDescriptor>> {
    static REGISTERED_MOE_RANKING_PROVIDER_DESCRIPTORS: OnceLock<
        RwLock<Vec<MoeRankingProviderDescriptor>>,
    > = OnceLock::new();
    REGISTERED_MOE_RANKING_PROVIDER_DESCRIPTORS.get_or_init(|| RwLock::new(Vec::new()))
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn register_provider(descriptor: InferenceProviderDescriptor) {
    provider_registry().register_provider(descriptor);
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn register_plugin_provider(
    registration: PluginInferenceProviderRegistration,
    provider: Arc<dyn InferenceProvider>,
) {
    register_provider(registration.into_descriptor(provider));
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn register_plugin_moe_ranking_provider(
    registration: PluginMoeRankingProviderRegistration,
    ranking_provider: Arc<dyn MoeRankingProvider>,
) {
    registered_moe_ranking_provider_descriptors()
        .write()
        .expect("registered moe ranking provider lock poisoned")
        .push(registration.into_descriptor(ranking_provider));
}

#[cfg_attr(not(test), allow(dead_code))]
const fn never_match_local_endpoint(_request: &InferenceEndpointRequest) -> bool {
    false
}

#[cfg_attr(not(test), allow(dead_code))]
const fn never_match_distributed_endpoint(_request: &InferenceEndpointRequest) -> bool {
    false
}

#[cfg_attr(not(test), allow(dead_code))]
const fn never_match_worker_runtime(_request: &InferenceWorkerRequest) -> bool {
    false
}

#[cfg_attr(not(test), allow(dead_code))]
const fn never_match_model_for_moe_ranking(_model_path: &Path) -> bool {
    false
}

#[cfg(test)]
fn clear_registered_providers_for_tests() {
    registered_provider_descriptors()
        .write()
        .expect("registered inference provider lock poisoned")
        .clear();
    registered_moe_ranking_provider_descriptors()
        .write()
        .expect("registered moe ranking provider lock poisoned")
        .clear();
}

pub fn select_local_endpoint_provider(
    request: &InferenceEndpointRequest,
) -> InferenceProviderSelection {
    provider_registry().select_local_endpoint_provider(request)
}

pub fn select_distributed_endpoint_provider(
    request: &InferenceEndpointRequest,
) -> InferenceProviderSelection {
    provider_registry().select_distributed_endpoint_provider(request)
}

pub fn select_worker_provider(request: &InferenceWorkerRequest) -> InferenceProviderSelection {
    provider_registry().select_worker_provider(request)
}

pub fn provider_requires_worker_runtime(model_path: &Path) -> bool {
    let request = InferenceEndpointRequest::local(model_path, 0, 0, 0);
    select_local_endpoint_provider(&request)
        .capabilities()
        .requires_worker_runtime
}

pub fn primary_backend_label_for_model(
    model_path: &Path,
    model_bytes: u64,
    local_vram_bytes: u64,
) -> String {
    let request = InferenceEndpointRequest::local(model_path, 0, model_bytes, local_vram_bytes);
    select_local_endpoint_provider(&request)
        .backend_label()
        .to_string()
}

pub fn select_moe_ranking_provider(
    model_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Option<MoeRankingProviderSelection> {
    if let Some(preferred_provider_id) = preferred_provider_id {
        if let Some(selection) = registered_moe_ranking_provider_descriptors()
            .read()
            .expect("registered moe ranking provider lock poisoned")
            .iter()
            .find(|descriptor| descriptor.selection.provider_id() == preferred_provider_id)
            .map(|descriptor| descriptor.selection.clone())
        {
            return Some(selection);
        }
    } else if let Some(selection) = registered_moe_ranking_provider_descriptors()
        .read()
        .expect("registered moe ranking provider lock poisoned")
        .iter()
        .find(|descriptor| (descriptor.matches_model)(model_path))
        .map(|descriptor| descriptor.selection.clone())
    {
        return Some(selection);
    }

    let request = InferenceEndpointRequest::local(model_path, 0, 0, 0)
        .with_preferred_provider_id(preferred_provider_id);
    let selection = select_local_endpoint_provider(&request);
    selection
        .cloned_moe_ranking_provider()
        .map(|ranking_provider| {
            MoeRankingProviderSelection::new(
                selection.provider_id().to_string(),
                selection.backend_label().to_string(),
                ranking_provider,
            )
        })
}

pub fn detect_moe_for_model(
    model_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Option<crate::models::gguf::GgufMoeInfo> {
    let selection = select_moe_ranking_provider(model_path, preferred_provider_id)?;
    let request = MoeDetectionRequest::new(model_path);
    selection.ranking_provider().detect_moe(&request)
}

pub fn load_cached_moe_ranking_for_model(
    model_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Option<Vec<u32>> {
    let selection = select_moe_ranking_provider(model_path, preferred_provider_id)?;
    let request = CachedRankingRequest::new(model_path);
    selection.ranking_provider().load_cached_ranking(&request)
}

pub fn best_shared_moe_ranking_artifact_for_model(
    model_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Option<crate::inference::moe::SharedRankingArtifact> {
    let selection = select_moe_ranking_provider(model_path, preferred_provider_id)?;
    selection
        .ranking_provider()
        .best_shared_ranking_artifact(&SharedRankingArtifactLookupRequest::new(model_path))
}

pub fn import_shared_moe_ranking_artifact_for_model(
    model_path: &Path,
    artifact: &crate::inference::moe::SharedRankingArtifact,
    preferred_provider_id: Option<&str>,
) -> Result<bool> {
    let selection =
        select_moe_ranking_provider(model_path, preferred_provider_id).ok_or_else(|| {
            anyhow::anyhow!(
                "no MoE ranking provider available for {}",
                model_path.display()
            )
        })?;
    selection.ranking_provider().import_shared_ranking_artifact(
        &SharedRankingArtifactImportRequest::new(model_path, artifact.clone()),
    )
}

pub fn ensure_full_analyze_ranking_for_model(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    cached_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Result<crate::inference::moe::SharedRankingArtifact> {
    let request =
        FullAnalyzeRankingRequest::new(bin_dir, model_name.to_string(), model_path, cached_path);
    let selection =
        select_moe_ranking_provider(model_path, preferred_provider_id).ok_or_else(|| {
            anyhow::anyhow!(
                "no MoE ranking provider available for {}",
                model_path.display()
            )
        })?;
    selection
        .ranking_provider()
        .ensure_full_analyze_ranking(&request)
}

pub fn ensure_micro_analyze_ranking_for_model(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &crate::inference::moe::MoeRuntimeOptions,
    preferred_provider_id: Option<&str>,
) -> Result<crate::inference::moe::SharedRankingArtifact> {
    let request = MicroAnalyzeRankingRequest::new(
        bin_dir,
        model_name.to_string(),
        model_path,
        options.clone(),
    );
    let selection =
        select_moe_ranking_provider(model_path, preferred_provider_id).ok_or_else(|| {
            anyhow::anyhow!(
                "no MoE ranking provider available for {}",
                model_path.display()
            )
        })?;
    selection
        .ranking_provider()
        .ensure_micro_analyze_ranking(&request)
}

pub fn resolve_heuristic_ranking_for_model(
    model_path: &Path,
    expert_count: u32,
    method: crate::inference::moe::HeuristicScoreMethod,
    preferred_provider_id: Option<&str>,
) -> Result<(Vec<u32>, String, String)> {
    let request = HeuristicRankingRequest::new(model_path, expert_count, method);
    let selection =
        select_moe_ranking_provider(model_path, preferred_provider_id).ok_or_else(|| {
            anyhow::anyhow!(
                "no MoE ranking provider available for {}",
                model_path.display()
            )
        })?;
    selection
        .ranking_provider()
        .resolve_heuristic_ranking(&request)
}

/// Start a distributed-host endpoint through the selected inference provider.
///
/// This keeps mesh election in control of placement and worker selection,
/// while moving the backend launch entry point out of `election.rs`.
pub async fn start_distributed_host(
    node: &crate::mesh::Node,
    tunnel_mgr: &crate::network::tunnel::Manager,
    bin_dir: &Path,
    model: &Path,
    model_name: &str,
    model_peers: &[crate::mesh::PeerInfo],
    draft: Option<&Path>,
    draft_max: u16,
    force_split: bool,
    binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
) -> Option<(u16, InferenceServerProcess)> {
    let my_vram = node.vram_bytes();
    let model_bytes = crate::inference::election::total_model_bytes(model);
    let min_vram = (model_bytes as f64 * 1.1) as u64;

    let need_split = force_split || my_vram < min_vram;

    let worker_ids: Vec<_> = if need_split {
        let mut candidates: Vec<_> = model_peers
            .iter()
            .filter(|p| {
                matches!(p.role, crate::mesh::NodeRole::Worker) || p.is_assigned_model(model_name)
            })
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .filter(|p| match p.rtt_ms {
                Some(rtt) if rtt > crate::mesh::MAX_SPLIT_RTT_MS => {
                    eprintln!(
                        "  ⚠ Skipping {} — RTT {}ms exceeds {}ms limit",
                        p.id.fmt_short(),
                        rtt,
                        crate::mesh::MAX_SPLIT_RTT_MS
                    );
                    false
                }
                _ => true,
            })
            .collect();

        candidates.sort_by_key(|p| p.rtt_ms.unwrap_or(u32::MAX));

        let mut accumulated_vram = my_vram;
        let mut selected = Vec::new();
        for p in &candidates {
            if accumulated_vram >= min_vram && !(force_split && selected.is_empty()) {
                break;
            }
            accumulated_vram += p.vram_bytes;
            let rtt_str = p
                .rtt_ms
                .map(|r| format!("{}ms", r))
                .unwrap_or("?ms".to_string());
            eprintln!(
                "  ✓ Adding {} — {:.1}GB VRAM, RTT {rtt_str}",
                p.id.fmt_short(),
                p.vram_bytes as f64 / 1e9
            );
            selected.push(p.id);
        }
        if accumulated_vram < min_vram {
            eprintln!(
                "  ⚠ Total VRAM {:.1}GB still short of {:.1}GB — using all {} candidates",
                accumulated_vram as f64 / 1e9,
                min_vram as f64 / 1e9,
                candidates.len()
            );
            selected = candidates.iter().map(|p| p.id).collect();
        }
        selected
    } else {
        let worker_count = model_peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count();
        if worker_count > 0 {
            eprintln!(
                "  Model fits on host ({:.1}GB VRAM for {:.1}GB model) — serving entirely",
                my_vram as f64 / 1e9,
                model_bytes as f64 / 1e9
            );
            eprintln!("  Use --split to force distributed mode");
        }
        vec![]
    };

    if !worker_ids.is_empty() {
        eprintln!("  Waiting for tunnels to {} worker(s)...", worker_ids.len());
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tunnel_mgr.wait_for_peers(worker_ids.len()),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let my_map = tunnel_mgr.peer_ports_map().await;
        let _ = node.broadcast_tunnel_map(my_map).await;
        let _ = node
            .wait_for_tunnel_maps(worker_ids.len(), std::time::Duration::from_secs(10))
            .await;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;
    }

    let all_ports = tunnel_mgr.peer_ports_map().await;
    let mut rpc_ports: Vec<u16> = Vec::new();
    for id in &worker_ids {
        if let Some(&port) = all_ports.get(id) {
            rpc_ports.push(port);
        }
    }

    let my_vram_f = my_vram as f64;
    let mut all_vrams: Vec<f64> = Vec::new();
    for id in &worker_ids {
        if let Some(peer) = model_peers.iter().find(|p| p.id == *id) {
            all_vrams.push(if peer.vram_bytes > 0 {
                peer.vram_bytes as f64
            } else {
                my_vram_f
            });
        }
    }
    all_vrams.push(my_vram_f);
    let total: f64 = all_vrams.iter().sum();
    let split = if total > 0.0 && !rpc_ports.is_empty() {
        let s: Vec<String> = all_vrams
            .iter()
            .map(|v| format!("{:.2}", v / total))
            .collect();
        let split_str = s.join(",");
        eprintln!(
            "  Tensor split: {split_str} ({} node(s), {:.0}GB total)",
            rpc_ports.len() + 1,
            total / 1e9
        );
        Some(split_str)
    } else {
        eprintln!("  Serving entirely ({:.0}GB VRAM)", my_vram_f / 1e9);
        None
    };

    let listen_port = match find_free_port().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Failed to find free port: {e}");
            return None;
        }
    };

    let mmproj_path = mmproj_path_for_model(model_name);
    let group_vram = if !rpc_ports.is_empty() {
        Some(total as u64)
    } else {
        None
    };

    let request = InferenceEndpointRequest::distributed_host(
        model,
        listen_port,
        rpc_ports,
        model_bytes,
        my_vram,
    )
    .with_tensor_split(split.as_deref())
    .with_draft_model_path(draft)
    .with_draft_max(draft_max)
    .with_mmproj_path(mmproj_path.as_deref())
    .with_ctx_size_override(ctx_size_override)
    .with_total_group_vram_bytes(group_vram);
    let selected_provider = select_distributed_endpoint_provider(&request);
    if !selected_provider
        .capabilities()
        .supports_distributed_host_runtime
    {
        eprintln!(
            "  {} ({}) does not support distributed host runtime for {}",
            selected_provider.backend_label(),
            selected_provider.provider_id(),
            model_name
        );
        return None;
    }
    match selected_provider
        .provider()
        .start_endpoint(bin_dir, binary_flavor, &request)
        .await
    {
        Ok(process) => Some((listen_port, process)),
        Err(e) => {
            eprintln!(
                "  Failed to start {} ({}) runtime: {e}",
                selected_provider.backend_label(),
                selected_provider.provider_id()
            );
            None
        }
    }
}

async fn find_free_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};
    use std::sync::{Mutex, OnceLock};

    #[derive(Clone, Copy, Debug, Default)]
    struct TestLocalProvider;

    #[derive(Clone, Copy, Debug, Default)]
    struct TestRankingProvider;

    impl InferenceProvider for TestLocalProvider {
        fn start_endpoint<'a>(
            &'a self,
            _bin_dir: &'a Path,
            _binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
            _request: &'a InferenceEndpointRequest,
        ) -> ProviderFuture<'a, InferenceServerProcess> {
            Box::pin(async { unreachable!("test provider start_endpoint should not run") })
        }

        fn start_worker<'a>(
            &'a self,
            _bin_dir: &'a Path,
            _binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
            _request: &'a InferenceWorkerRequest,
        ) -> ProviderFuture<'a, u16> {
            Box::pin(async { unreachable!("test provider start_worker should not run") })
        }

        fn prepare_moe_shard(&self, _request: &MoeShardPreparationRequest) -> Result<()> {
            unreachable!("test provider prepare_moe_shard should not run")
        }
    }

    impl MoeRankingProvider for TestRankingProvider {
        fn detect_moe(
            &self,
            _request: &MoeDetectionRequest,
        ) -> Option<crate::models::gguf::GgufMoeInfo> {
            None
        }

        fn load_cached_ranking(&self, _request: &CachedRankingRequest) -> Option<Vec<u32>> {
            Some(vec![9, 4, 1])
        }

        fn best_shared_ranking_artifact(
            &self,
            _request: &SharedRankingArtifactLookupRequest,
        ) -> Option<crate::inference::moe::SharedRankingArtifact> {
            Some(crate::inference::moe::SharedRankingArtifact {
                kind: crate::inference::moe::SharedRankingKind::Analyze,
                origin: crate::inference::moe::SharedRankingOrigin::PeerImport,
                ranking: vec![9, 4, 1],
                micro_prompt_count: None,
                micro_tokens: None,
                micro_layer_scope: None,
            })
        }

        fn import_shared_ranking_artifact(
            &self,
            _request: &SharedRankingArtifactImportRequest,
        ) -> Result<bool> {
            Ok(true)
        }

        fn ensure_full_analyze_ranking(
            &self,
            _request: &FullAnalyzeRankingRequest,
        ) -> Result<crate::inference::moe::SharedRankingArtifact> {
            unreachable!("test ranking provider full analyze should not run")
        }

        fn ensure_micro_analyze_ranking(
            &self,
            _request: &MicroAnalyzeRankingRequest,
        ) -> Result<crate::inference::moe::SharedRankingArtifact> {
            unreachable!("test ranking provider micro analyze should not run")
        }

        fn resolve_heuristic_ranking(
            &self,
            _request: &HeuristicRankingRequest,
        ) -> Result<(Vec<u32>, String, String)> {
            unreachable!("test ranking provider heuristic resolve should not run")
        }
    }

    const fn never_match_distributed_endpoint_for_tests(
        _request: &InferenceEndpointRequest,
    ) -> bool {
        false
    }

    const fn never_match_worker_runtime_for_tests(_request: &InferenceWorkerRequest) -> bool {
        false
    }

    fn provider_registry_test_lock() -> &'static Mutex<()> {
        static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn write_gguf_string(file: &mut std::fs::File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn test_local_descriptor() -> InferenceProviderDescriptor {
        InferenceProviderDescriptor::new(
            InferenceProviderSelection::new(
                "test.local",
                "test-local",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: false,
                    requires_worker_runtime: false,
                    supports_moe_shard_runtime: false,
                },
                Arc::new(TestLocalProvider),
            ),
            always_match_local_endpoint,
            never_match_distributed_endpoint_for_tests,
            never_match_worker_runtime_for_tests,
        )
    }

    #[test]
    fn registered_provider_takes_precedence_over_builtin_for_matching_local_runtime() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_provider(test_local_descriptor());

        let request = InferenceEndpointRequest::local("/tmp/model.gguf", 8080, 1, 1);
        let selection = select_local_endpoint_provider(&request);

        assert_eq!(selection.provider_id(), "test.local");
        assert_eq!(selection.backend_label(), "test-local");

        clear_registered_providers_for_tests();
    }

    #[test]
    fn preferred_provider_id_selects_registered_provider_without_matcher_path() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_provider(test_local_descriptor());

        let request = InferenceEndpointRequest::local("/tmp/model.gguf", 8080, 1, 1)
            .with_preferred_provider_id(Some("test.local"));
        let selection = select_local_endpoint_provider(&request);

        assert_eq!(selection.provider_id(), "test.local");
        assert_eq!(selection.backend_label(), "test-local");

        clear_registered_providers_for_tests();
    }

    #[test]
    fn plugin_registration_creates_preferred_only_provider_descriptor() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_plugin_provider(
            PluginInferenceProviderRegistration::new(
                "plugin.notes",
                "plugin-notes",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: false,
                    requires_worker_runtime: false,
                    supports_moe_shard_runtime: false,
                },
            ),
            Arc::new(TestLocalProvider),
        );

        let default_selection = select_local_endpoint_provider(&InferenceEndpointRequest::local(
            "/tmp/model.gguf",
            8080,
            1,
            1,
        ));
        assert_eq!(default_selection.provider_id(), "builtin.llama");

        let explicit_selection = select_local_endpoint_provider(
            &InferenceEndpointRequest::local("/tmp/model.gguf", 8080, 1, 1)
                .with_preferred_provider_id(Some("plugin.notes")),
        );
        assert_eq!(explicit_selection.provider_id(), "plugin.notes");
        assert_eq!(explicit_selection.backend_label(), "plugin-notes");

        clear_registered_providers_for_tests();
    }

    #[test]
    fn builtin_llama_selection_exposes_moe_ranking_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();

        let root =
            std::env::temp_dir().join(format!("mesh-llm-provider-moe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create test root");
        let model_path = root.join("model.gguf");
        std::fs::write(&model_path, b"gguf").expect("write model placeholder");
        let ranking_path = crate::inference::moe::ranking_cache_path(&model_path);
        if let Some(parent) = ranking_path.parent() {
            std::fs::create_dir_all(parent).expect("create ranking cache dir");
        }
        std::fs::write(&ranking_path, "3\n1\n4\n").expect("write ranking cache");

        let selection =
            select_moe_ranking_provider(&model_path, None).expect("builtin llama ranking provider");
        assert_eq!(selection.provider_id(), "builtin.llama");
        assert_eq!(
            load_cached_moe_ranking_for_model(&model_path, None),
            Some(vec![3, 1, 4])
        );
        let artifact =
            best_shared_moe_ranking_artifact_for_model(&model_path, None).expect("artifact");
        assert_eq!(
            artifact.kind,
            crate::inference::moe::SharedRankingKind::Analyze
        );
        assert_eq!(
            artifact.origin,
            crate::inference::moe::SharedRankingOrigin::LegacyCache
        );
        assert_eq!(artifact.ranking, vec![3, 1, 4]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preferred_provider_without_ranking_provider_disables_moe_ranking_selection() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_plugin_provider(
            PluginInferenceProviderRegistration::new(
                "plugin.notes",
                "plugin-notes",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: false,
                    requires_worker_runtime: false,
                    supports_moe_shard_runtime: false,
                },
            ),
            Arc::new(TestLocalProvider),
        );

        let selection =
            select_moe_ranking_provider(Path::new("/tmp/model.gguf"), Some("plugin.notes"));
        assert!(selection.is_none());

        clear_registered_providers_for_tests();
    }

    #[test]
    fn plugin_registration_can_attach_moe_ranking_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_plugin_provider(
            PluginInferenceProviderRegistration::new(
                "plugin.ranker",
                "plugin-ranker",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: false,
                    requires_worker_runtime: false,
                    supports_moe_shard_runtime: false,
                },
            )
            .with_moe_ranking_provider(Arc::new(TestRankingProvider)),
            Arc::new(TestLocalProvider),
        );

        let selection =
            select_moe_ranking_provider(Path::new("/tmp/model.gguf"), Some("plugin.ranker"))
                .expect("plugin ranking provider");
        assert_eq!(selection.provider_id(), "plugin.ranker");
        assert_eq!(
            load_cached_moe_ranking_for_model(Path::new("/tmp/model.gguf"), Some("plugin.ranker")),
            Some(vec![9, 4, 1])
        );

        let artifact = best_shared_moe_ranking_artifact_for_model(
            Path::new("/tmp/model.gguf"),
            Some("plugin.ranker"),
        )
        .expect("shared artifact");
        assert_eq!(artifact.ranking, vec![9, 4, 1]);
        assert_eq!(
            artifact.origin,
            crate::inference::moe::SharedRankingOrigin::PeerImport
        );

        clear_registered_providers_for_tests();
    }

    #[test]
    fn standalone_moe_ranking_provider_can_be_registered_without_execution_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_plugin_moe_ranking_provider(
            PluginMoeRankingProviderRegistration::new("plugin.rank-only", "plugin-rank-only"),
            Arc::new(TestRankingProvider),
        );

        let selection =
            select_moe_ranking_provider(Path::new("/tmp/model.gguf"), Some("plugin.rank-only"))
                .expect("standalone ranking provider");
        assert_eq!(selection.provider_id(), "plugin.rank-only");
        assert_eq!(selection.backend_label(), "plugin-rank-only");
        assert_eq!(
            load_cached_moe_ranking_for_model(
                Path::new("/tmp/model.gguf"),
                Some("plugin.rank-only")
            ),
            Some(vec![9, 4, 1])
        );

        clear_registered_providers_for_tests();
    }

    #[test]
    fn standalone_moe_ranking_provider_matcher_can_select_without_preferred_id() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();
        register_plugin_moe_ranking_provider(
            PluginMoeRankingProviderRegistration::new("plugin.rank-match", "plugin-rank-match")
                .with_model_matcher(|model_path| {
                    model_path.extension().is_some_and(|ext| ext == "gguf")
                }),
            Arc::new(TestRankingProvider),
        );

        let selection = select_moe_ranking_provider(Path::new("/tmp/model.gguf"), None)
            .expect("matcher-based ranking provider");
        assert_eq!(selection.provider_id(), "plugin.rank-match");
        assert_eq!(
            load_cached_moe_ranking_for_model(Path::new("/tmp/model.gguf"), None),
            Some(vec![9, 4, 1])
        );

        clear_registered_providers_for_tests();
    }

    #[test]
    fn builtin_llama_selection_resolves_heuristic_ranking_via_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();

        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-provider-heuristic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("heuristic.gguf");

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1i64.to_le_bytes()).unwrap();
        file.write_all(&3i64.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "general.alignment");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&32u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "qwen.expert_count");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "qwen.expert_used_count");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "blk.0.ffn_gate_inp.weight");
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2i64.to_le_bytes()).unwrap();
        file.write_all(&4i64.to_le_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        let meta_end = file.stream_position().unwrap();
        let aligned = ((meta_end + 31) / 32) * 32;
        if aligned > meta_end {
            file.write_all(&vec![0u8; (aligned - meta_end) as usize])
                .unwrap();
        }

        let values = [3.0f32, 4.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);

        let (ranking, source, origin) = resolve_heuristic_ranking_for_model(
            &path,
            4,
            crate::inference::moe::HeuristicScoreMethod::MeanL2,
            None,
        )
        .unwrap();
        assert_eq!(ranking, vec![0, 3, 2, 1]);
        assert_eq!(source, "heuristic-heuristic");
        assert_eq!(origin, "local-heuristic");

        let (_, _, cached_origin) = resolve_heuristic_ranking_for_model(
            &path,
            4,
            crate::inference::moe::HeuristicScoreMethod::MeanL2,
            None,
        )
        .unwrap();
        assert_eq!(cached_origin, "local-heuristic-cache");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn builtin_llama_selection_imports_shared_ranking_via_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();

        let dir =
            std::env::temp_dir().join(format!("mesh-llm-provider-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("import.gguf");

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1i64.to_le_bytes()).unwrap();
        file.write_all(&3i64.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "general.alignment");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&32u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "qwen.expert_count");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "qwen.expert_used_count");
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();

        write_gguf_string(&mut file, "blk.0.ffn_gate_inp.weight");
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2i64.to_le_bytes()).unwrap();
        file.write_all(&4i64.to_le_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        let meta_end = file.stream_position().unwrap();
        for dim in [2u64, 4u64] {
            file.write_all(&dim.to_le_bytes()).unwrap();
        }
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&meta_end.to_le_bytes()).unwrap();

        for value in [
            10.0f32, 1.0, 1.0, 1.0, //
            1.0, 9.0, 1.0, 1.0,
        ] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);

        let artifact = crate::inference::moe::SharedRankingArtifact {
            kind: crate::inference::moe::SharedRankingKind::Analyze,
            origin: crate::inference::moe::SharedRankingOrigin::PeerImport,
            ranking: vec![0, 3, 2, 1],
            micro_prompt_count: None,
            micro_tokens: None,
            micro_layer_scope: None,
        };
        let imported =
            import_shared_moe_ranking_artifact_for_model(&path, &artifact, None).unwrap();
        assert!(imported);

        let cached = best_shared_moe_ranking_artifact_for_model(&path, None).unwrap();
        assert_eq!(
            cached.origin,
            crate::inference::moe::SharedRankingOrigin::PeerImport
        );
        assert_eq!(cached.ranking, vec![0, 3, 2, 1]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
