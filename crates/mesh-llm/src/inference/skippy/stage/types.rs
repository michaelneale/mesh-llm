use anyhow::Result;
use skippy_protocol::{FlashAttentionType, LoadMode, StageDevice};
use tokio::sync::oneshot;

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
    Inventory(StageInventoryRequest),
    Prepare(StagePrepareRequest),
    CancelPrepare(StageCancelPrepareRequest),
    StatusUpdate(StagePreparationStatus),
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum StageControlResponse {
    Ready(StageReadyResponse),
    Status(Vec<StageStatusSnapshot>),
    Inventory(StageLayerInventory),
    PrepareAccepted(StagePrepareAcceptedResponse),
    PreparationStatus(StagePreparationStatus),
    StatusAck(StageStatusAck),
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
pub(crate) struct StageInventoryRequest {
    pub(crate) model_id: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePrepareRequest {
    pub(crate) load: StageLoadRequest,
    pub(crate) coordinator_id: Option<iroh::EndpointId>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageCancelPrepareRequest {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) stage_id: String,
    pub(crate) shutdown_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerRange {
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct StageLayerInventory {
    pub(crate) model_id: String,
    pub(crate) package_ref: String,
    pub(crate) manifest_sha256: String,
    pub(crate) layer_count: u32,
    pub(crate) ready_ranges: Vec<LayerRange>,
    pub(crate) available_ranges: Vec<LayerRange>,
    pub(crate) missing_ranges: Vec<LayerRange>,
    pub(crate) preparing_ranges: Vec<StagePreparationStatus>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) source_model_kind: SourceModelKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceModelKind {
    Unknown,
    LayerPackage,
    PlainGguf,
    SplitGguf,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StagePreparationState {
    Assigned,
    Downloading,
    Available,
    Resolving,
    Loading,
    Ready,
    Failed,
    Cancelled,
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

#[derive(Clone, Debug)]
pub(crate) struct StagePreparationStatus {
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
    pub(crate) state: StagePreparationState,
    pub(crate) bytes_done: Option<u64>,
    pub(crate) bytes_total: Option<u64>,
    pub(crate) bind_addr: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) shutdown_generation: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct StagePrepareAcceptedResponse {
    pub(crate) accepted: bool,
    pub(crate) status: StagePreparationStatus,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StageStatusAck {
    pub(crate) accepted: bool,
    pub(crate) error: Option<String>,
}
