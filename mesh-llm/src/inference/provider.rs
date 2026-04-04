use anyhow::Result;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use tokio::net::TcpListener;

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
}

/// Backend-neutral request to start a serving endpoint for a model.
///
/// Backends can treat `worker_tunnel_ports` / `tensor_split` as the
/// distributed-host shape, or ignore them for single-node serving.
#[derive(Clone, Debug)]
pub struct InferenceEndpointRequest {
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
    pub model_path: Option<PathBuf>,
    pub device_hint: Option<String>,
}

impl InferenceWorkerRequest {
    pub fn with_model_path(mut self, model_path: Option<impl AsRef<Path>>) -> Self {
        self.model_path = model_path.map(|path| path.as_ref().to_path_buf());
        self
    }

    pub fn with_device_hint(mut self, device_hint: Option<impl Into<String>>) -> Self {
        self.device_hint = device_hint.map(Into::into);
        self
    }
}

type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceProviderCapabilities {
    pub supports_local_runtime: bool,
    pub supports_distributed_host_runtime: bool,
    pub requires_worker_runtime: bool,
    pub supports_moe_shard_runtime: bool,
}

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy)]
pub struct InferenceProviderSelection {
    provider_id: &'static str,
    provider: &'static dyn InferenceProvider,
}

impl InferenceProviderSelection {
    pub const fn new(provider_id: &'static str, provider: &'static dyn InferenceProvider) -> Self {
        Self {
            provider_id,
            provider,
        }
    }

    pub fn provider_id(&self) -> &'static str {
        self.provider_id
    }

    pub fn backend_label(&self) -> &'static str {
        self.provider.backend_label()
    }

    pub fn capabilities(&self) -> InferenceProviderCapabilities {
        self.provider.capabilities()
    }

    pub fn provider(&self) -> &'static dyn InferenceProvider {
        self.provider
    }
}

impl std::fmt::Debug for InferenceProviderSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceProviderSelection")
            .field("provider_id", &self.provider_id)
            .field("backend_label", &self.backend_label())
            .finish()
    }
}

impl PartialEq for InferenceProviderSelection {
    fn eq(&self, other: &Self) -> bool {
        self.provider_id == other.provider_id
    }
}

impl Eq for InferenceProviderSelection {}

#[derive(Clone, Copy, Debug)]
pub struct InferenceProviderRegistry {
    builtin_providers: &'static [InferenceProviderDescriptor],
}

impl InferenceProviderRegistry {
    pub const fn new(builtin_providers: &'static [InferenceProviderDescriptor]) -> Self {
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
        capability_filter: impl Fn(InferenceProviderCapabilities) -> bool,
        dynamic_matches: impl Fn(&InferenceProviderDescriptor) -> bool,
        builtin_matches: impl Fn(&InferenceProviderDescriptor) -> bool,
        empty_message: &'static str,
    ) -> InferenceProviderSelection {
        if let Some(selection) = registered_provider_descriptors()
            .read()
            .expect("registered inference provider lock poisoned")
            .iter()
            .find(|descriptor| {
                capability_filter(descriptor.selection.capabilities())
                    && dynamic_matches(descriptor)
            })
            .map(|descriptor| descriptor.selection)
        {
            return selection;
        }

        self.builtin_providers
            .iter()
            .find(|descriptor| {
                capability_filter(descriptor.selection.capabilities())
                    && builtin_matches(descriptor)
            })
            .map(|descriptor| descriptor.selection)
            .expect(empty_message)
    }

    pub fn select_local_endpoint_provider(
        &self,
        request: &InferenceEndpointRequest,
    ) -> InferenceProviderSelection {
        self.select_provider(
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
            |capabilities| capabilities.requires_worker_runtime,
            |descriptor| (descriptor.matches_worker_runtime)(request),
            |descriptor| (descriptor.matches_worker_runtime)(request),
            "at least one worker inference provider must be registered",
        )
    }
}

pub trait InferenceProvider: Send + Sync {
    fn backend_label(&self) -> &'static str;

    fn capabilities(&self) -> InferenceProviderCapabilities;

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
}

/// Built-in provider adapter for the current llama.cpp runtime path.
///
/// This is the first step toward a pluggable backend provider interface:
/// core call sites depend on the provider contract, while the built-in
/// provider still delegates to the existing llama-specific launch code.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinLlamaProvider;

impl InferenceProvider for BuiltinLlamaProvider {
    fn backend_label(&self) -> &'static str {
        "llama"
    }

    fn capabilities(&self) -> InferenceProviderCapabilities {
        InferenceProviderCapabilities {
            supports_local_runtime: true,
            supports_distributed_host_runtime: true,
            requires_worker_runtime: true,
            supports_moe_shard_runtime: true,
        }
    }

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
}

static BUILTIN_LLAMA_PROVIDER: BuiltinLlamaProvider = BuiltinLlamaProvider;
static BUILTIN_LLAMA_SELECTION: InferenceProviderSelection =
    InferenceProviderSelection::new("builtin.llama", &BUILTIN_LLAMA_PROVIDER);
static BUILTIN_LLAMA_DESCRIPTOR: InferenceProviderDescriptor = InferenceProviderDescriptor::new(
    BUILTIN_LLAMA_SELECTION,
    always_match_local_endpoint,
    always_match_distributed_endpoint,
    always_match_worker_runtime,
);
static BUILTIN_PROVIDER_REGISTRY: InferenceProviderRegistry =
    InferenceProviderRegistry::new(&[BUILTIN_LLAMA_DESCRIPTOR]);

const fn always_match_local_endpoint(_request: &InferenceEndpointRequest) -> bool {
    true
}

const fn always_match_distributed_endpoint(_request: &InferenceEndpointRequest) -> bool {
    true
}

const fn always_match_worker_runtime(_request: &InferenceWorkerRequest) -> bool {
    true
}

pub fn provider_registry() -> &'static InferenceProviderRegistry {
    &BUILTIN_PROVIDER_REGISTRY
}

fn registered_provider_descriptors() -> &'static RwLock<Vec<InferenceProviderDescriptor>> {
    static REGISTERED_PROVIDER_DESCRIPTORS: OnceLock<RwLock<Vec<InferenceProviderDescriptor>>> =
        OnceLock::new();
    REGISTERED_PROVIDER_DESCRIPTORS.get_or_init(|| RwLock::new(Vec::new()))
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn register_provider(descriptor: InferenceProviderDescriptor) {
    provider_registry().register_provider(descriptor);
}

#[cfg(test)]
fn clear_registered_providers_for_tests() {
    registered_provider_descriptors()
        .write()
        .expect("registered inference provider lock poisoned")
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
) -> &'static str {
    let request = InferenceEndpointRequest::local(model_path, 0, model_bytes, local_vram_bytes);
    select_local_endpoint_provider(&request).backend_label()
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

    #[derive(Clone, Copy, Debug, Default)]
    struct TestLocalProvider;

    impl InferenceProvider for TestLocalProvider {
        fn backend_label(&self) -> &'static str {
            "test-local"
        }

        fn capabilities(&self) -> InferenceProviderCapabilities {
            InferenceProviderCapabilities {
                supports_local_runtime: true,
                supports_distributed_host_runtime: false,
                requires_worker_runtime: false,
                supports_moe_shard_runtime: false,
            }
        }

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
    }

    static TEST_LOCAL_PROVIDER: TestLocalProvider = TestLocalProvider;
    static TEST_LOCAL_SELECTION: InferenceProviderSelection =
        InferenceProviderSelection::new("test.local", &TEST_LOCAL_PROVIDER);
    static TEST_LOCAL_DESCRIPTOR: InferenceProviderDescriptor = InferenceProviderDescriptor::new(
        TEST_LOCAL_SELECTION,
        always_match_local_endpoint,
        never_match_distributed_endpoint_for_tests,
        never_match_worker_runtime_for_tests,
    );

    const fn never_match_distributed_endpoint_for_tests(
        _request: &InferenceEndpointRequest,
    ) -> bool {
        false
    }

    const fn never_match_worker_runtime_for_tests(_request: &InferenceWorkerRequest) -> bool {
        false
    }

    #[test]
    fn registered_provider_takes_precedence_over_builtin_for_matching_local_runtime() {
        clear_registered_providers_for_tests();
        register_provider(TEST_LOCAL_DESCRIPTOR);

        let request = InferenceEndpointRequest::local("/tmp/model.gguf", 8080, 1, 1);
        let selection = select_local_endpoint_provider(&request);

        assert_eq!(selection.provider_id(), "test.local");
        assert_eq!(selection.backend_label(), "test-local");

        clear_registered_providers_for_tests();
    }
}
