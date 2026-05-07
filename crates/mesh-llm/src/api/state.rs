use crate::mesh;
use crate::network::affinity;
use crate::plugin;
use crate::runtime_data;
use serde::{Serialize, Serializer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Best-effort publication state for mesh nodes (Issue #240).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicationState {
    /// No --publish requested; mesh is private.
    Private,
    /// The latest publish attempt succeeded.
    Public,
    /// The latest publish attempt failed after `--publish` was requested.
    PublishFailed,
}

impl PublicationState {
    pub fn as_str(&self) -> &'static str {
        match self {
            PublicationState::Private => "private",
            PublicationState::Public => "public",
            PublicationState::PublishFailed => "publish_failed",
        }
    }
}

impl Serialize for PublicationState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

pub enum RuntimeControlRequest {
    Load {
        spec: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<RuntimeLoadResponse>>,
    },
    Unload {
        target: String,
        resp: tokio::sync::oneshot::Sender<anyhow::Result<RuntimeUnloadResponse>>,
    },
    Shutdown,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLoadResponse {
    pub model: String,
    pub instance_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeUnloadResponse {
    pub model: String,
    pub instance_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeModelPayload {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub backend: String,
    pub status: String,
    pub port: Option<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeProcessPayload {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub backend: String,
    pub status: String,
    pub port: u16,
    pub pid: u32,
    pub slots: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LocalModelInterest {
    pub model_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submission_source: Option<String>,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
}

#[derive(Clone)]
pub struct MeshApi {
    pub(super) inner: Arc<Mutex<ApiInner>>,
}

pub(super) struct ApiInner {
    pub(super) node: mesh::Node,
    pub(super) plugin_manager: plugin::PluginManager,
    pub(super) affinity_router: affinity::AffinityRouter,
    pub(super) runtime_data_collector: runtime_data::RuntimeDataCollector,
    pub(super) runtime_data_producer: runtime_data::RuntimeDataProducer,
    pub(super) headless: bool,
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
    pub(super) publication_state: PublicationState,
    pub(super) runtime_control: Option<tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>>,
    pub(super) local_processes: Vec<RuntimeProcessPayload>,
    pub(super) sse_clients: Vec<tokio::sync::mpsc::UnboundedSender<String>>,
    pub(super) model_interests: HashMap<String, LocalModelInterest>,
    pub(super) wakeable_inventory: crate::runtime::wakeable::WakeableInventory,
}
