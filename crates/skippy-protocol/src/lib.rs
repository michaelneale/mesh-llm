use serde::{Deserialize, Serialize};

pub mod binary;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    Ready,
    PrefillChunk,
    FinalPrefillChunk,
    DecodeToken,
    StateImport,
    StateExport,
    Ack,
    TokenReply,
    Stop,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageIdentity {
    pub run_id: String,
    pub request_id: String,
    pub session_id: String,
    pub topology_id: String,
    pub stage_id: String,
    pub stage_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoadMode {
    RuntimeSlice,
    LayerPackage,
    ArtifactSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageConfig {
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    #[serde(default)]
    pub package_ref: Option<String>,
    #[serde(default)]
    pub manifest_sha256: Option<String>,
    #[serde(default)]
    pub source_model_path: Option<String>,
    #[serde(default)]
    pub source_model_sha256: Option<String>,
    #[serde(default)]
    pub source_model_bytes: Option<u64>,
    #[serde(default)]
    pub materialized_path: Option<String>,
    #[serde(default)]
    pub materialized_pinned: bool,
    #[serde(default)]
    pub model_path: Option<String>,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default)]
    pub n_gpu_layers: i32,
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    #[serde(default)]
    pub filter_tensors_on_load: bool,
    #[serde(default)]
    pub selected_device: Option<StageDevice>,
    pub load_mode: LoadMode,
    pub bind_addr: String,
    #[serde(default)]
    pub upstream: Option<PeerConfig>,
    #[serde(default)]
    pub downstream: Option<PeerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageDevice {
    pub backend_device: String,
    #[serde(default)]
    pub stable_id: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub vram_bytes: Option<u64>,
}

fn default_ctx_size() -> u32 {
    512
}

fn default_cache_type() -> String {
    "f16".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PeerConfig {
    pub stage_id: String,
    pub stage_index: u32,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageTopology {
    pub topology_id: String,
    pub model_id: String,
    pub stages: Vec<StageTopologyEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageTopologyEntry {
    pub stage_id: String,
    pub stage_index: u32,
    pub host: Option<String>,
    pub endpoint: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub load_mode: LoadMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationDType {
    Unknown,
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationLayout {
    Opaque,
    TokenMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ActivationDescriptor {
    pub version: u32,
    pub dtype: ActivationDType,
    pub layout: ActivationLayout,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub payload_bytes: u64,
    #[serde(default)]
    pub flags: u64,
    #[serde(default)]
    pub payload_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "message_type", rename_all = "snake_case")]
pub enum StageMessage {
    Ready(ReadyMessage),
    PrefillChunk(PrefillChunkMessage),
    FinalPrefillChunk(FinalPrefillChunkMessage),
    DecodeToken(DecodeTokenMessage),
    StateImport(StateImportMessage),
    StateExport(StateExportMessage),
    Ack(AckMessage),
    TokenReply(TokenReplyMessage),
    Stop(StopMessage),
    Error(ErrorMessage),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MessageBase {
    pub schema_version: u32,
    pub run_id: String,
    pub request_id: String,
    pub session_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub topology_id: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub tokenizer_id: Option<String>,
    #[serde(default)]
    pub chat_template_id: Option<String>,
    #[serde(default)]
    pub seq: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ReadyMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PrefillChunkMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub token_ids: Vec<i32>,
    pub prompt_token_start: u32,
    #[serde(default)]
    pub activation_dtype: Option<String>,
    #[serde(default)]
    pub activation_bytes: Option<u64>,
    #[serde(default)]
    pub activation: Option<ActivationDescriptor>,
    #[serde(default)]
    pub activation_ref: Option<String>,
    #[serde(default)]
    pub is_final: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct FinalPrefillChunkMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub token_ids: Vec<i32>,
    pub prompt_token_start: u32,
    pub is_final: bool,
    #[serde(default)]
    pub activation_dtype: Option<String>,
    #[serde(default)]
    pub activation_bytes: Option<u64>,
    #[serde(default)]
    pub activation: Option<ActivationDescriptor>,
    #[serde(default)]
    pub activation_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DecodeTokenMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub token_id: i32,
    pub decode_index: u32,
    #[serde(default)]
    pub activation_dtype: Option<String>,
    #[serde(default)]
    pub activation_bytes: Option<u64>,
    #[serde(default)]
    pub activation: Option<ActivationDescriptor>,
    #[serde(default)]
    pub activation_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StateImportMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub layer_start: u32,
    pub layer_end: u32,
    pub state_bytes: u64,
    #[serde(default)]
    pub state_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StateExportMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct AckMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub acked_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TokenReplyMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub token_id: i32,
    #[serde(default)]
    pub decode_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StopMessage {
    #[serde(flatten)]
    pub base: MessageBase,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ErrorMessage {
    #[serde(flatten)]
    pub base: MessageBase,
    pub error_code: String,
    pub error_message: String,
}

impl StageConfig {
    pub fn ready_message(&self) -> StageMessage {
        StageMessage::Ready(ReadyMessage {
            base: MessageBase {
                schema_version: SCHEMA_VERSION,
                run_id: self.run_id.clone(),
                request_id: "stage-ready".to_string(),
                session_id: "stage-lifecycle".to_string(),
                stage_id: self.stage_id.clone(),
                stage_index: self.stage_index,
                topology_id: self.topology_id.clone(),
                model_id: Some(self.model_id.clone()),
                tokenizer_id: None,
                chat_template_id: None,
                seq: Some(0),
            },
            layer_start: self.layer_start,
            layer_end: self.layer_end,
        })
    }
}

impl StageMessage {
    pub fn base(&self) -> &MessageBase {
        match self {
            Self::Ready(message) => &message.base,
            Self::PrefillChunk(message) => &message.base,
            Self::FinalPrefillChunk(message) => &message.base,
            Self::DecodeToken(message) => &message.base,
            Self::StateImport(message) => &message.base,
            Self::StateExport(message) => &message.base,
            Self::Ack(message) => &message.base,
            Self::TokenReply(message) => &message.base,
            Self::Stop(message) => &message.base,
            Self::Error(message) => &message.base,
        }
    }

    pub fn kind(&self) -> MessageKind {
        match self {
            Self::Ready(_) => MessageKind::Ready,
            Self::PrefillChunk(_) => MessageKind::PrefillChunk,
            Self::FinalPrefillChunk(_) => MessageKind::FinalPrefillChunk,
            Self::DecodeToken(_) => MessageKind::DecodeToken,
            Self::StateImport(_) => MessageKind::StateImport,
            Self::StateExport(_) => MessageKind::StateExport,
            Self::Ack(_) => MessageKind::Ack,
            Self::TokenReply(_) => MessageKind::TokenReply,
            Self::Stop(_) => MessageKind::Stop,
            Self::Error(_) => MessageKind::Error,
        }
    }

    pub fn ack_for(&self, stage: &StageConfig) -> StageMessage {
        let base = self.base();
        StageMessage::Ack(AckMessage {
            base: MessageBase {
                schema_version: SCHEMA_VERSION,
                run_id: base.run_id.clone(),
                request_id: base.request_id.clone(),
                session_id: base.session_id.clone(),
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                topology_id: stage.topology_id.clone(),
                model_id: Some(stage.model_id.clone()),
                tokenizer_id: base.tokenizer_id.clone(),
                chat_template_id: base.chat_template_id.clone(),
                seq: base.seq,
            },
            acked_seq: base.seq.unwrap_or(0),
        })
    }
}
