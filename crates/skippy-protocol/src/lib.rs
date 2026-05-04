use serde::{Deserialize, Serialize};

pub mod binary;
pub mod proto {
    pub mod stage {
        include!(concat!(env!("OUT_DIR"), "/skippy.stage.v1.rs"));
    }
}

pub const SCHEMA_VERSION: u32 = 1;
pub const STAGE_ALPN_V1: &[u8] = b"skippy-stage/1";
pub const STAGE_PROTOCOL_GENERATION: u32 = 1;
pub const STAGE_STREAM_CONTROL: u8 = 0x01;
pub const STAGE_STREAM_TRANSPORT: u8 = 0x02;
pub const MAX_STAGE_FRAME_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageFrameError {
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    MissingStageControlCommand,
    MissingStageControlResponse,
    MissingStageTransportTarget,
}

impl std::fmt::Display for StageFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageFrameError::BadGeneration { got } => write!(
                f,
                "bad skippy stage generation: expected {}, got {}",
                STAGE_PROTOCOL_GENERATION, got
            ),
            StageFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {got}")
            }
            StageFrameError::MissingStageControlCommand => {
                write!(f, "stage control command is required but missing")
            }
            StageFrameError::MissingStageControlResponse => {
                write!(f, "stage control response is required but missing")
            }
            StageFrameError::MissingStageTransportTarget => {
                write!(f, "stage transport target is required but missing")
            }
        }
    }
}

impl std::error::Error for StageFrameError {}

pub fn validate_stage_control_request(
    frame: &proto::stage::StageControlRequest,
) -> Result<(), StageFrameError> {
    validate_generation(frame.gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.command.is_none() {
        return Err(StageFrameError::MissingStageControlCommand);
    }
    Ok(())
}

pub fn validate_stage_control_response(
    frame: &proto::stage::StageControlResponse,
) -> Result<(), StageFrameError> {
    validate_generation(frame.gen)?;
    if frame.response.is_none() {
        return Err(StageFrameError::MissingStageControlResponse);
    }
    Ok(())
}

pub fn validate_stage_transport_open(
    frame: &proto::stage::StageTransportOpen,
) -> Result<(), StageFrameError> {
    validate_generation(frame.gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.topology_id.is_empty() || frame.run_id.is_empty() || frame.stage_id.is_empty() {
        return Err(StageFrameError::MissingStageTransportTarget);
    }
    Ok(())
}

fn validate_generation(gen: u32) -> Result<(), StageFrameError> {
    if gen != STAGE_PROTOCOL_GENERATION {
        return Err(StageFrameError::BadGeneration { got: gen });
    }
    Ok(())
}

fn validate_endpoint_id(len: usize) -> Result<(), StageFrameError> {
    if len != 32 {
        return Err(StageFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}

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
    #[serde(default)]
    pub projector_path: Option<String>,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_lane_count")]
    pub lane_count: u32,
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

fn default_lane_count() -> u32 {
    4
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

#[cfg(test)]
mod tests {
    use prost::Message as _;

    use super::proto::stage::{
        stage_control_request, stage_control_response, GetStageStatus, LoadStage,
        StageControlRequest, StageControlResponse, StageReady, StageRuntimeState, StageStatus,
        StageTransportOpen, StageWireDType, StopStage,
    };
    use super::{
        validate_stage_control_request, validate_stage_control_response,
        validate_stage_transport_open, StageFrameError, STAGE_PROTOCOL_GENERATION,
    };

    #[test]
    fn stage_control_request_validates_generation_sender_and_command() {
        let frame = StageControlRequest {
            gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![9u8; 32],
            command: Some(stage_control_request::Command::GetStageStatus(
                GetStageStatus {
                    topology_id: Some("topology-a".to_string()),
                    run_id: Some("run-a".to_string()),
                    stage_id: Some("stage-0".to_string()),
                },
            )),
        };
        validate_stage_control_request(&frame).unwrap();

        let load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadStage(LoadStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                model_id: "qwen".to_string(),
                backend: "skippy".to_string(),
                package_ref: "hf://repo/model".to_string(),
                manifest_sha256: "a5".repeat(32),
                stage_id: "stage-0".to_string(),
                layer_end: 16,
                activation_width: 4096,
                projector_path: Some("/models/mmproj.gguf".to_string()),
                ..Default::default()
            })),
            ..frame.clone()
        };
        let decoded = StageControlRequest::decode(load.encode_to_vec().as_slice()).unwrap();
        match decoded.command {
            Some(stage_control_request::Command::LoadStage(load)) => {
                assert_eq!(load.projector_path.as_deref(), Some("/models/mmproj.gguf"));
            }
            other => panic!("expected LoadStage, got {other:?}"),
        }

        let stop = StageControlRequest {
            command: Some(stage_control_request::Command::StopStage(StopStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                stage_id: "stage-0".to_string(),
                shutdown_generation: 7,
            })),
            ..frame.clone()
        };
        validate_stage_control_request(&stop).unwrap();

        let missing_command = StageControlRequest {
            command: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&missing_command),
            Err(StageFrameError::MissingStageControlCommand)
        ));

        let wrong_gen = StageControlRequest { gen: 2, ..frame };
        assert!(matches!(
            validate_stage_control_request(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 2 })
        ));
    }

    #[test]
    fn stage_control_response_validates_generation_and_response() {
        let frame = StageControlResponse {
            gen: STAGE_PROTOCOL_GENERATION,
            response: Some(stage_control_response::Response::StageReady(StageReady {
                accepted: true,
                status: Some(StageStatus {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    model_id: "qwen".to_string(),
                    backend: "skippy".to_string(),
                    stage_id: "stage-0".to_string(),
                    stage_index: 0,
                    layer_start: 0,
                    layer_end: 16,
                    state: StageRuntimeState::Ready as i32,
                    bind_addr: "127.0.0.1:0".to_string(),
                    activation_width: 4096,
                    wire_dtype: StageWireDType::StageWireDtypeF16 as i32,
                    shutdown_generation: 7,
                    ctx_size: 8192,
                    lane_count: 2,
                    projector_path: Some("/models/mmproj.gguf".to_string()),
                    ..Default::default()
                }),
                error: None,
            })),
        };
        let decoded = StageControlResponse::decode(frame.encode_to_vec().as_slice()).unwrap();
        validate_stage_control_response(&decoded).unwrap();
        match decoded.response {
            Some(stage_control_response::Response::StageReady(ready)) => {
                let status = ready.status.expect("stage-ready status");
                assert_eq!(
                    status.projector_path.as_deref(),
                    Some("/models/mmproj.gguf")
                );
                assert_eq!(status.lane_count, 2);
            }
            other => panic!("expected StageReady, got {other:?}"),
        }

        let missing_response = StageControlResponse {
            response: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_response(&missing_response),
            Err(StageFrameError::MissingStageControlResponse)
        ));

        let wrong_gen = StageControlResponse { gen: 2, ..frame };
        assert!(matches!(
            validate_stage_control_response(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 2 })
        ));
    }

    #[test]
    fn stage_transport_open_validates_generation_sender_and_target() {
        let frame = StageTransportOpen {
            gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![7u8; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-1".to_string(),
        };
        validate_stage_transport_open(&frame).unwrap();

        let missing_target = StageTransportOpen {
            stage_id: String::new(),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_transport_open(&missing_target),
            Err(StageFrameError::MissingStageTransportTarget)
        ));

        let wrong_gen = StageTransportOpen { gen: 2, ..frame };
        assert!(matches!(
            validate_stage_transport_open(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 2 })
        ));
    }
}
