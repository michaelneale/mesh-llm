use crate::api;
use crate::cli::ServingBackend;
use crate::inference::{election, launch, skippy};
use crate::mesh;
use crate::models;
use crate::network::openai::backend;
use crate::network::router;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub(super) enum RuntimeEvent {
    Exited { model: String, port: u16 },
}

pub(super) enum LocalRuntimeBackendHandle {
    Llama {
        process: launch::InferenceServerHandle,
        backend_proxy: backend::BackendProxyHandle,
    },
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
            LocalRuntimeBackendHandle::Llama { process, .. } => process.pid(),
            LocalRuntimeBackendHandle::Skippy { .. } => std::process::id(),
        }
    }

    pub(super) async fn shutdown(self) {
        match self.inner {
            LocalRuntimeBackendHandle::Llama {
                process,
                backend_proxy,
            } => {
                backend_proxy.shutdown().await;
                process.shutdown().await;
            }
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
    pub(super) runtime: &'a crate::runtime::instance::InstanceRuntime,
    pub(super) bin_dir: &'a Path,
    pub(super) binary_flavor: Option<launch::BinaryFlavor>,
    pub(super) node: &'a mesh::Node,
    pub(super) model_path: &'a Path,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) slots: usize,
    pub(super) serving_backend: ServingBackend,
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
    let model_name = resolved_model_name(spec.model_path);
    let model_bytes = election::total_model_bytes(spec.model_path);
    let my_vram = spec
        .pinned_gpu
        .map(|gpu| gpu.vram_bytes)
        .unwrap_or_else(|| spec.node.vram_bytes());
    anyhow::ensure!(
        my_vram >= (model_bytes as f64 * 1.1) as u64,
        "runtime load only supports models that fit locally on this node"
    );

    match spec.serving_backend {
        ServingBackend::Llama => {
            return start_runtime_llama_model(spec, model_name, model_bytes, my_vram).await;
        }
        ServingBackend::Skippy => {
            return start_runtime_skippy_model(spec, model_name).await;
        }
    }
}

async fn start_runtime_llama_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
    model_bytes: u64,
    my_vram: u64,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let llama_port = alloc_local_port().await?;
    let mmproj_path = spec
        .mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&model_name));
    let process = launch::start_llama_server(
        spec.runtime,
        spec.bin_dir,
        spec.binary_flavor,
        launch::ModelLaunchSpec {
            model: spec.model_path,
            http_port: llama_port,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: election::local_multi_gpu_split_mode(spec.binary_flavor),
            draft: None,
            draft_max: 0,
            model_bytes,
            my_vram,
            mmproj: mmproj_path.as_deref(),
            ctx_size_override: spec.ctx_size_override,
            total_group_vram: None,
            selected_gpu: spec.pinned_gpu,
            slots: spec.slots,
            runtime_data_producer: Some(spec.node.runtime_data_collector().producer(
                crate::runtime_data::RuntimeDataSource {
                    scope: "runtime",
                    plugin_data_key: None,
                    plugin_endpoint_key: None,
                },
            )),
        },
    )
    .await?;
    let backend_proxy = backend::start_backend_proxy(llama_port).await?;
    let port = backend_proxy.port();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port,
            backend: "llama".into(),
            context_length: process.context_length,
            slots: spec.slots,
            inner: LocalRuntimeBackendHandle::Llama {
                process: process.handle,
                backend_proxy,
            },
        },
        process.death_rx,
    ))
}

async fn start_runtime_skippy_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    anyhow::ensure!(
        spec.mmproj_override.is_none(),
        "skippy runtime load does not support multimodal projector overrides yet"
    );
    let port = alloc_local_port().await?;
    let context_length = spec.ctx_size_override.unwrap_or(4096);
    let mut options = skippy::SkippyModelLoadOptions::for_direct_gguf(&model_name, spec.model_path)
        .with_ctx_size(context_length)
        .with_generation_concurrency(spec.slots);
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
