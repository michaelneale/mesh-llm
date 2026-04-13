use crate::mesh;
use crate::network::affinity;
use crate::plugin;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

pub enum RuntimeControlRequest {
    Load {
        spec: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<String>>,
    },
    Unload {
        model: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
}

#[derive(Clone, Serialize)]
pub struct RuntimeModelPayload {
    pub name: String,
    pub backend: String,
    pub status: String,
    pub port: Option<u16>,
}

#[derive(Clone, Serialize)]
pub struct RuntimeProcessPayload {
    pub name: String,
    pub backend: String,
    pub status: String,
    pub port: u16,
    pub pid: u32,
}

#[derive(Clone)]
pub struct MeshApi {
    pub(super) inner: Arc<Mutex<ApiInner>>,
}

pub(super) struct ApiInner {
    pub(super) node: mesh::Node,
    pub(super) plugin_manager: plugin::PluginManager,
    pub(super) affinity_router: affinity::AffinityRouter,
    pub(super) is_host: bool,
    pub(super) is_client: bool,
    pub(super) llama_ready: bool,
    pub(super) llama_port: Option<u16>,
    pub(super) model_name: String,
    pub(super) primary_backend: Option<String>,
    pub(super) draft_name: Option<String>,
    pub(super) api_port: u16,
    pub(super) model_size_bytes: u64,
    pub(super) mesh_name: Option<String>,
    pub(super) latest_version: Option<String>,
    pub(super) nostr_relays: Vec<String>,
    pub(super) nostr_discovery: bool,
    pub(super) runtime_control: Option<tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>>,
    pub(super) local_processes: Vec<RuntimeProcessPayload>,
    pub(super) sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
    pub(super) inventory_scan_running: bool,
    pub(super) inventory_scan_waiters:
        Vec<tokio::sync::oneshot::Sender<crate::models::LocalModelInventorySnapshot>>,
    pub(super) local_instances: Arc<Mutex<Vec<crate::runtime::instance::LocalInstanceSnapshot>>>,
}
