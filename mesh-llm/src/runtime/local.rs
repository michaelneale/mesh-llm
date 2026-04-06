use crate::api;
use crate::inference::{election, launch};
use crate::mesh;
use crate::models;
use crate::network::router;
use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(super) enum RuntimeEvent {
    Exited { model: String, port: u16 },
}

#[derive(Debug)]
pub(super) struct LocalRuntimeModelHandle {
    pub(super) port: u16,
    pub(super) backend: String,
    pub(super) process: launch::InferenceServerHandle,
    pub(super) context_length: u32,
}

pub(super) struct ManagedModelController {
    pub(super) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(super) task: tokio::task::JoinHandle<()>,
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
    bin_dir: &Path,
    binary_flavor: Option<launch::BinaryFlavor>,
    node: &mesh::Node,
    model_path: &Path,
    mmproj_override: Option<&Path>,
    ctx_size_override: Option<u32>,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let model_name = resolved_model_name(model_path);
    let model_bytes = election::total_model_bytes(model_path);
    let my_vram = node.vram_bytes();
    anyhow::ensure!(
        my_vram >= (model_bytes as f64 * 1.1) as u64,
        "runtime load only supports models that fit locally on this node"
    );

    let port = alloc_local_port().await?;
    let mmproj_path = mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&model_name));
    let process = launch::start_llama_server(
        bin_dir,
        binary_flavor,
        launch::ModelLaunchSpec {
            model: model_path,
            http_port: port,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: election::local_multi_gpu_split_mode(binary_flavor),
            draft: None,
            draft_max: 0,
            model_bytes,
            my_vram,
            mmproj: mmproj_path.as_deref(),
            ctx_size_override,
            total_group_vram: None,
        },
    )
    .await?;

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port,
            backend: "llama".into(),
            process: process.handle,
            context_length: process.context_length,
        },
        process.death_rx,
    ))
}

pub(super) fn local_process_payload(
    model_name: &str,
    backend: &str,
    port: u16,
    pid: u32,
) -> api::RuntimeProcessPayload {
    api::RuntimeProcessPayload {
        name: model_name.to_string(),
        backend: backend.into(),
        status: "ready".into(),
        port,
        pid,
    }
}
