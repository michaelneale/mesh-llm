use anyhow::{Context, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use tokio::net::TcpListener;

#[cfg(target_os = "macos")]
pub(crate) fn matches_mlx_model_dir(request: &InferenceEndpointRequest) -> bool {
    crate::mlx::is_mlx_model_dir(request.model_path.as_path())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn matches_mlx_model_dir(_request: &InferenceEndpointRequest) -> bool {
    false
}

#[cfg(target_os = "macos")]
pub(crate) fn matches_mlx_worker_runtime(request: &InferenceWorkerRequest) -> bool {
    request
        .model_path
        .as_deref()
        .map(crate::mlx::is_mlx_model_dir)
        .unwrap_or(false)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn matches_mlx_worker_runtime(_request: &InferenceWorkerRequest) -> bool {
    false
}

fn is_gguf_model_file(model_path: &Path) -> bool {
    model_path.is_file()
        && model_path
            .extension()
            .map(|ext| ext == "gguf")
            .unwrap_or(false)
}

pub(crate) fn matches_gguf_model_file(request: &InferenceEndpointRequest) -> bool {
    is_gguf_model_file(request.model_path.as_path())
}

pub(crate) fn matches_gguf_worker_runtime(request: &InferenceWorkerRequest) -> bool {
    request
        .model_path
        .as_deref()
        .map(is_gguf_model_file)
        .unwrap_or(false)
}

/// Backend-neutral runtime handle for a serving instance.
///
/// This sits above any concrete backend implementation:
/// - llama.cpp subprocesses
/// - MLX plugin-managed serving
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

pub trait MoeRankingProvider: Send + Sync {
    fn detect_moe(&self, model_path: &Path) -> Option<crate::models::gguf::GgufMoeInfo>;

    fn load_cached_ranking(&self, model_path: &Path) -> Option<Vec<u32>>;
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

    pub fn replace_plugin_providers(&self, descriptors: Vec<InferenceProviderDescriptor>) {
        let mut providers = registered_provider_descriptors()
            .write()
            .expect("registered inference provider lock poisoned");
        *providers = descriptors;
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

    fn prepare_moe_shard<'a>(
        &'a self,
        bin_dir: &'a Path,
        model_path: &'a Path,
        assignment: &'a crate::inference::moe::NodeAssignment,
        output_path: &'a Path,
    ) -> ProviderFuture<'a, ()>;
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
    fn detect_moe(&self, model_path: &Path) -> Option<crate::models::gguf::GgufMoeInfo> {
        crate::models::gguf::detect_moe(model_path)
    }

    fn load_cached_ranking(&self, model_path: &Path) -> Option<Vec<u32>> {
        let ranking_path = crate::inference::moe::ranking_cache_path(model_path);
        crate::inference::moe::load_cached_ranking(&ranking_path)
    }
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

    fn prepare_moe_shard<'a>(
        &'a self,
        bin_dir: &'a Path,
        model_path: &'a Path,
        assignment: &'a crate::inference::moe::NodeAssignment,
        output_path: &'a Path,
    ) -> ProviderFuture<'a, ()> {
        Box::pin(async move {
            crate::inference::moe::run_split(bin_dir, model_path, assignment, output_path)
        })
    }
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
        self.into_descriptor_with_local_match(provider, never_match_local_endpoint)
    }

    pub fn into_descriptor_with_local_match(
        self,
        provider: Arc<dyn InferenceProvider>,
        matches_local_endpoint: fn(&InferenceEndpointRequest) -> bool,
    ) -> InferenceProviderDescriptor {
        self.into_descriptor_with_runtime_matchers(
            provider,
            matches_local_endpoint,
            never_match_distributed_endpoint,
            never_match_worker_runtime,
        )
    }

    pub fn into_descriptor_with_runtime_matchers(
        self,
        provider: Arc<dyn InferenceProvider>,
        matches_local_endpoint: fn(&InferenceEndpointRequest) -> bool,
        matches_distributed_endpoint: fn(&InferenceEndpointRequest) -> bool,
        matches_worker_runtime: fn(&InferenceWorkerRequest) -> bool,
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
            matches_local_endpoint,
            matches_distributed_endpoint,
            matches_worker_runtime,
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

#[derive(Clone)]
pub struct PluginManagedEndpointProvider {
    provider_id: String,
    plugin_name: String,
    endpoint_id: String,
    plugin_manager: crate::plugin::PluginManager,
}

impl PluginManagedEndpointProvider {
    pub fn new(
        provider_id: impl Into<String>,
        plugin_name: impl Into<String>,
        endpoint_id: impl Into<String>,
        plugin_manager: crate::plugin::PluginManager,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            plugin_name: plugin_name.into(),
            endpoint_id: endpoint_id.into(),
            plugin_manager,
        }
    }
}

impl InferenceProvider for PluginManagedEndpointProvider {
    fn start_endpoint<'a>(
        &'a self,
        _bin_dir: &'a Path,
        _binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceEndpointRequest,
    ) -> ProviderFuture<'a, InferenceServerProcess> {
        Box::pin(async move {
            let ensured = self
                .plugin_manager
                .ensure_managed_inference_endpoint(&self.plugin_name, &self.endpoint_id, request)
                .await?;
            let listen_port = parse_endpoint_port(&ensured.address).with_context(|| {
                format!(
                    "Plugin-managed inference provider '{}' returned invalid address '{}'",
                    self.provider_id, ensured.address
                )
            })?;
            Ok(in_process_endpoint_process(
                listen_port,
                ensured.context_length,
            ))
        })
    }

    fn start_worker<'a>(
        &'a self,
        _bin_dir: &'a Path,
        _binary_flavor: Option<crate::inference::launch::BinaryFlavor>,
        request: &'a InferenceWorkerRequest,
    ) -> ProviderFuture<'a, u16> {
        Box::pin(async move {
            let response = self
                .plugin_manager
                .ensure_managed_inference_worker(
                    &self.plugin_name,
                    request.model_path.as_deref(),
                    request.device_hint.as_deref(),
                )
                .await?;
            Ok(response.port)
        })
    }

    fn prepare_moe_shard<'a>(
        &'a self,
        _bin_dir: &'a Path,
        model_path: &'a Path,
        assignment: &'a crate::inference::moe::NodeAssignment,
        output_path: &'a Path,
    ) -> ProviderFuture<'a, ()> {
        Box::pin(async move {
            self.plugin_manager
                .prepare_managed_moe_shard(&self.plugin_name, model_path, output_path, assignment)
                .await
        })
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
        let mut providers = Vec::new();

        providers.push(InferenceProviderDescriptor::new(
            builtin_llama_selection(),
            always_match_local_endpoint,
            always_match_distributed_endpoint,
            always_match_worker_runtime,
        ));

        InferenceProviderRegistry::new(providers)
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

pub fn sync_plugin_provider_descriptors(descriptors: Vec<InferenceProviderDescriptor>) {
    provider_registry().replace_plugin_providers(descriptors);
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
pub fn plugin_provider_id(plugin_name: &str, endpoint_id: &str) -> String {
    format!("plugin.{plugin_name}.{endpoint_id}")
}

fn parse_endpoint_port(address: &str) -> Result<u16> {
    let without_scheme = address
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(address);
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (_, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("address '{address}' does not include a usable port"))?;
    port.parse::<u16>()
        .with_context(|| format!("address '{address}' returned an invalid port"))
}

fn in_process_endpoint_process(listen_port: u16, context_length: u32) -> InferenceServerProcess {
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let handle = InferenceServerHandle::in_process(shutdown_tx);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        loop {
            if shutdown_rx.changed().await.is_err() {
                break;
            }
            if *shutdown_rx.borrow() {
                break;
            }
        }
        let _ = death_tx.send(());
    });
    InferenceServerProcess {
        handle,
        death_rx,
        context_length,
        listen_port,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
const fn never_match_model_for_moe_ranking(_model_path: &Path) -> bool {
    false
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

#[cfg(test)]
pub(crate) fn clear_registered_providers_for_tests() {
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
    selection.ranking_provider().detect_moe(model_path)
}

pub fn load_cached_moe_ranking_for_model(
    model_path: &Path,
    preferred_provider_id: Option<&str>,
) -> Option<Vec<u32>> {
    let selection = select_moe_ranking_provider(model_path, preferred_provider_id)?;
    selection.ranking_provider().load_cached_ranking(model_path)
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

    let mmproj_path = crate::models::find_mmproj_path(model_name, model);
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
        Ok(process) => Some((process.listen_port, process)),
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

        fn prepare_moe_shard<'a>(
            &'a self,
            _bin_dir: &'a Path,
            _model_path: &'a Path,
            _assignment: &'a crate::inference::moe::NodeAssignment,
            _output_path: &'a Path,
        ) -> ProviderFuture<'a, ()> {
            Box::pin(async { unreachable!("test provider prepare_moe_shard should not run") })
        }
    }

    impl MoeRankingProvider for TestRankingProvider {
        fn detect_moe(&self, _model_path: &Path) -> Option<crate::models::gguf::GgufMoeInfo> {
            None
        }

        fn load_cached_ranking(&self, _model_path: &Path) -> Option<Vec<u32>> {
            Some(vec![9, 4, 1])
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
        std::fs::create_dir_all(root.join("moe-rankings")).expect("create moe ranking dir");
        let model_path = root.join("model.gguf");
        std::fs::write(&model_path, b"gguf").expect("write model placeholder");
        std::fs::write(root.join("moe-rankings/model.csv"), "3\n1\n4\n")
            .expect("write ranking cache");

        let selection =
            select_moe_ranking_provider(&model_path, None).expect("builtin llama ranking provider");
        assert_eq!(selection.provider_id(), "builtin.llama");
        assert_eq!(
            load_cached_moe_ranking_for_model(&model_path, None),
            Some(vec![3, 1, 4])
        );

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

    #[cfg(target_os = "macos")]
    #[test]
    fn plugin_registration_with_local_match_selects_mlx_provider() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();

        let root =
            std::env::temp_dir().join(format!("mesh-llm-provider-mlx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{"model_type":"deepseek_v3","architectures":["DeepseekV3ForCausalLM"]}"#,
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"placeholder").unwrap();

        register_provider(
            PluginInferenceProviderRegistration::new(
                "plugin.mlx.local-mlx",
                "mlx",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: true,
                    requires_worker_runtime: true,
                    supports_moe_shard_runtime: false,
                },
            )
            .into_descriptor_with_runtime_matchers(
                Arc::new(TestLocalProvider),
                matches_mlx_model_dir,
                matches_mlx_model_dir,
                matches_mlx_worker_runtime,
            ),
        );

        let request = InferenceEndpointRequest::local(&root, 8080, 1, 1);
        let selection = select_local_endpoint_provider(&request);

        assert_eq!(selection.provider_id(), "plugin.mlx.local-mlx");
        assert_eq!(selection.backend_label(), "mlx");

        let distributed = select_distributed_endpoint_provider(
            &InferenceEndpointRequest::distributed_host(&root, 8123, vec![7001], 1, 1),
        );
        assert_eq!(distributed.provider_id(), "plugin.mlx.local-mlx");

        let worker =
            select_worker_provider(&InferenceWorkerRequest::default().with_model_path(Some(&root)));
        assert_eq!(worker.provider_id(), "plugin.mlx.local-mlx");

        clear_registered_providers_for_tests();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn plugin_registration_with_gguf_match_selects_runtime_types() {
        let _guard = provider_registry_test_lock()
            .lock()
            .expect("provider registry test lock poisoned");
        clear_registered_providers_for_tests();

        let root = std::env::temp_dir().join(format!(
            "mesh-llm-provider-gguf-test-{}.gguf",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&root);
        std::fs::write(&root, b"gguf").unwrap();

        register_provider(
            PluginInferenceProviderRegistration::new(
                "plugin.llama.local",
                "llama",
                InferenceProviderCapabilities {
                    supports_local_runtime: true,
                    supports_distributed_host_runtime: true,
                    requires_worker_runtime: true,
                    supports_moe_shard_runtime: true,
                },
            )
            .into_descriptor_with_runtime_matchers(
                Arc::new(TestLocalProvider),
                matches_gguf_model_file,
                matches_gguf_model_file,
                matches_gguf_worker_runtime,
            ),
        );

        let local =
            select_local_endpoint_provider(&InferenceEndpointRequest::local(&root, 8080, 1, 1));
        assert_eq!(local.provider_id(), "plugin.llama.local");

        let distributed = select_distributed_endpoint_provider(
            &InferenceEndpointRequest::distributed_host(&root, 8123, vec![7001], 1, 1),
        );
        assert_eq!(distributed.provider_id(), "plugin.llama.local");

        let worker =
            select_worker_provider(&InferenceWorkerRequest::default().with_model_path(Some(&root)));
        assert_eq!(worker.provider_id(), "plugin.llama.local");

        clear_registered_providers_for_tests();
        let _ = std::fs::remove_file(&root);
    }
}
