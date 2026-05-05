use std::io;

use super::{
    activation::{decode_f16_to_f32_bytes, decode_q8_to_f32_bytes},
    invalid_data,
};

pub const STAGE_STATE_VERSION: i32 = 4;
pub const MAX_STAGE_LOGIT_BIAS: usize = 256;
pub const READY_MAGIC: i32 = 0x5352_4459; // "SRDY"
pub const LLAMA_TOKEN_NULL: i32 = -1;
pub const STAGE_STATE_HEADER_BYTES: usize = 10 * 4;
pub const STAGE_SAMPLING_CONFIG_BASE_BYTES: usize = 10 * 4;
pub const STAGE_LOGIT_BIAS_WIRE_BYTES: usize = 4 + 4;
pub const STAGE_WIRE_FIXED_HEADER_BYTES: usize = 4 * 4 + STAGE_STATE_HEADER_BYTES + 2 * 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireActivationDType {
    F32 = 0,
    F16 = 1,
    Q8 = 2,
}

impl TryFrom<i32> for WireActivationDType {
    type Error = io::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q8),
            _ => Err(invalid_data("unknown activation wire dtype")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireMessageKind {
    PrefillEmbd = 1,
    DecodeEmbd = 2,
    Stop = 3,
    PrefillFinalEmbd = 4,
    DecodeReplayEmbd = 5,
    DecodeReplayFinalEmbd = 6,
    StateImport = 7,
    DecodeReadout = 8,
    DecodeLightCtx = 9,
    VerifySpan = 10,
    CheckpointSession = 11,
    RestoreSession = 12,
    StateExport = 13,
    ConfigureGeneration = 14,
    ProbePrefill = 15,
    RestorePrefill = 16,
    TryRestorePrefill = 17,
    TryRestorePrefillDecode = 18,
}

impl WireMessageKind {
    pub fn is_prefill(self) -> bool {
        matches!(self, Self::PrefillEmbd | Self::PrefillFinalEmbd)
    }

    pub fn is_decode_replay(self) -> bool {
        matches!(self, Self::DecodeReplayEmbd | Self::DecodeReplayFinalEmbd)
    }

    pub fn is_decode_light_context(self) -> bool {
        matches!(self, Self::DecodeLightCtx)
    }

    pub fn requires_predicted_reply(self) -> bool {
        matches!(
            self,
            Self::DecodeEmbd
                | Self::DecodeReadout
                | Self::DecodeLightCtx
                | Self::VerifySpan
                | Self::PrefillFinalEmbd
                | Self::DecodeReplayFinalEmbd
        )
    }

    pub fn is_session_control(self) -> bool {
        matches!(self, Self::CheckpointSession | Self::RestoreSession)
    }

    pub fn is_generation_control(self) -> bool {
        matches!(self, Self::ConfigureGeneration)
    }

    pub fn is_prefix_cache_control(self) -> bool {
        matches!(
            self,
            Self::ProbePrefill
                | Self::RestorePrefill
                | Self::TryRestorePrefill
                | Self::TryRestorePrefillDecode
        )
    }

    pub fn is_activationless_prefix_cache_control(self) -> bool {
        matches!(
            self,
            Self::ProbePrefill | Self::RestorePrefill | Self::TryRestorePrefill
        )
    }
}

impl TryFrom<i32> for WireMessageKind {
    type Error = io::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::PrefillEmbd),
            2 => Ok(Self::DecodeEmbd),
            3 => Ok(Self::Stop),
            4 => Ok(Self::PrefillFinalEmbd),
            5 => Ok(Self::DecodeReplayEmbd),
            6 => Ok(Self::DecodeReplayFinalEmbd),
            7 => Ok(Self::StateImport),
            8 => Ok(Self::DecodeReadout),
            9 => Ok(Self::DecodeLightCtx),
            10 => Ok(Self::VerifySpan),
            11 => Ok(Self::CheckpointSession),
            12 => Ok(Self::RestoreSession),
            13 => Ok(Self::StateExport),
            14 => Ok(Self::ConfigureGeneration),
            15 => Ok(Self::ProbePrefill),
            16 => Ok(Self::RestorePrefill),
            17 => Ok(Self::TryRestorePrefill),
            18 => Ok(Self::TryRestorePrefillDecode),
            _ => Err(invalid_data("unknown stage message kind")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireReplyKind {
    Ack = 1,
    PredictedToken = 2,
    PredictedTokens = 3,
}

impl TryFrom<i32> for WireReplyKind {
    type Error = io::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ack),
            2 => Ok(Self::PredictedToken),
            3 => Ok(Self::PredictedTokens),
            _ => Err(invalid_data("unknown stage reply kind")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireStagePhase {
    Prefill = 1,
    Decode = 2,
    DecodeReplay = 3,
    DecodeLight = 4,
}

pub mod state_flags {
    pub const FINAL_CHUNK: i32 = 1 << 0;
    pub const LIGHT_CONTEXT: i32 = 1 << 1;
    pub const SKIP_VERIFY_CHECKPOINT: i32 = 1 << 2;
    pub const SAMPLING: i32 = 1 << 3;
    pub const FULL_STATE: i32 = 1 << 4;
    pub const CHAT_SAMPLING_METADATA: i32 = 1 << 5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageStateHeader {
    pub version: i32,
    pub seq_id: i32,
    pub phase: i32,
    pub flags: i32,
    pub checkpoint_generation: i32,
    pub prompt_token_count: i32,
    pub decode_step: i32,
    pub current_token: i32,
    pub source_stage_index: i32,
    pub reserved: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageLogitBias {
    pub token_id: i32,
    pub bias: f32,
}

impl Default for StageLogitBias {
    fn default() -> Self {
        Self {
            token_id: 0,
            bias: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageSamplingConfig {
    pub flags: u32,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: i32,
    pub logit_bias: Vec<StageLogitBias>,
}

impl Default for StageSamplingConfig {
    fn default() -> Self {
        Self {
            flags: 0,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            penalty_last_n: -1,
            logit_bias: Vec::new(),
        }
    }
}

impl StageSamplingConfig {
    pub fn enabled(&self) -> bool {
        self.flags != 0
    }
}

impl StageStateHeader {
    pub fn new(kind: WireMessageKind, dtype: WireActivationDType) -> Self {
        let mut header = Self {
            version: STAGE_STATE_VERSION,
            seq_id: 0,
            phase: expected_phase(kind) as i32,
            flags: 0,
            checkpoint_generation: 0,
            prompt_token_count: 0,
            decode_step: -1,
            current_token: LLAMA_TOKEN_NULL,
            source_stage_index: -1,
            reserved: dtype as i32,
        };
        if matches!(
            kind,
            WireMessageKind::PrefillFinalEmbd | WireMessageKind::DecodeReplayFinalEmbd
        ) {
            header.flags |= state_flags::FINAL_CHUNK;
        }
        if kind.is_decode_light_context() {
            header.flags |= state_flags::LIGHT_CONTEXT;
        }
        header
    }

    pub fn dtype(self) -> io::Result<WireActivationDType> {
        WireActivationDType::try_from(self.reserved)
    }

    pub fn matches_kind(self, kind: WireMessageKind) -> bool {
        if matches!(
            kind,
            WireMessageKind::StateImport | WireMessageKind::StateExport
        ) || kind.is_session_control()
            || kind.is_generation_control()
        {
            return true;
        }
        if self.phase != expected_phase(kind) as i32 {
            return false;
        }
        let expected_final = matches!(
            kind,
            WireMessageKind::PrefillFinalEmbd | WireMessageKind::DecodeReplayFinalEmbd
        );
        let actual_final = (self.flags & state_flags::FINAL_CHUNK) != 0;
        let expected_light = kind.is_decode_light_context();
        let actual_light = (self.flags & state_flags::LIGHT_CONTEXT) != 0;
        expected_final == actual_final && expected_light == actual_light
    }
}

impl Default for StageStateHeader {
    fn default() -> Self {
        Self::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageWireMessage {
    pub kind: WireMessageKind,
    pub pos_start: i32,
    pub token_count: i32,
    pub state: StageStateHeader,
    pub request_id: u64,
    pub session_id: u64,
    pub sampling: Option<StageSamplingConfig>,
    pub chat_sampling_metadata: Option<String>,
    pub tokens: Vec<i32>,
    pub activation: Vec<u8>,
    pub raw_bytes: Vec<u8>,
}

impl StageWireMessage {
    pub fn stop(dtype: WireActivationDType) -> Self {
        Self::stop_with_identity(dtype, 0, 0)
    }

    pub fn stop_with_identity(
        dtype: WireActivationDType,
        request_id: u64,
        session_id: u64,
    ) -> Self {
        Self {
            kind: WireMessageKind::Stop,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::Stop, dtype),
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    pub fn configure_generation(
        dtype: WireActivationDType,
        request_id: u64,
        session_id: u64,
        prompt_token_count: i32,
        sampling: Option<StageSamplingConfig>,
        chat_sampling_metadata: Option<String>,
    ) -> Self {
        let mut state = StageStateHeader::new(WireMessageKind::ConfigureGeneration, dtype);
        state.prompt_token_count = prompt_token_count;
        Self {
            kind: WireMessageKind::ConfigureGeneration,
            pos_start: 0,
            token_count: 0,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    pub fn activation_f32_payload(&self, n_embd: i32) -> io::Result<Vec<u8>> {
        if self.activation.is_empty() {
            return Ok(Vec::new());
        }
        match self.state.dtype()? {
            WireActivationDType::F32 => Ok(self.activation.clone()),
            WireActivationDType::F16 => decode_f16_to_f32_bytes(&self.activation),
            WireActivationDType::Q8 => {
                decode_q8_to_f32_bytes(&self.activation, self.token_count, n_embd)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReply {
    pub kind: WireReplyKind,
    pub predicted: i32,
    pub predicted_tokens: Vec<i32>,
    pub stats: StageReplyStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageReplyStats {
    pub kv_lookup_hits: i64,
    pub kv_lookup_misses: i64,
    pub kv_lookup_errors: i64,
    pub kv_imported_pages: i64,
    pub kv_imported_tokens: i64,
    pub kv_recorded_pages: i64,
    pub kv_recorded_bytes: i64,
    pub kv_hit_stage_mask: i64,
    pub kv_record_stage_mask: i64,
    pub checkpoint_flush_us: i64,
    pub checkpoint_prefill_drain_us: i64,
    pub checkpoint_local_us: i64,
    pub checkpoint_downstream_write_us: i64,
    pub checkpoint_downstream_wait_us: i64,
    pub checkpoint_total_us: i64,
    pub checkpoint_prefill_drained_replies: i64,
    pub restore_flush_us: i64,
    pub restore_prefill_drain_us: i64,
    pub restore_local_us: i64,
    pub restore_downstream_write_us: i64,
    pub restore_downstream_wait_us: i64,
    pub restore_total_us: i64,
    pub restore_prefill_drained_replies: i64,
    pub verify_span_compute_us: i64,
    pub verify_span_forward_write_us: i64,
    pub verify_span_downstream_wait_us: i64,
    pub verify_span_total_us: i64,
    pub verify_span_stage_count: i64,
    pub verify_span_request_count: i64,
    pub verify_span_token_count: i64,
    pub verify_span_max_tokens: i64,
    pub verify_span_checkpointed_requests: i64,
    pub verify_span_skip_checkpoint_requests: i64,
}

impl StageReplyStats {
    pub fn merge(&mut self, other: Self) {
        self.kv_lookup_hits += other.kv_lookup_hits;
        self.kv_lookup_misses += other.kv_lookup_misses;
        self.kv_lookup_errors += other.kv_lookup_errors;
        self.kv_imported_pages += other.kv_imported_pages;
        self.kv_imported_tokens += other.kv_imported_tokens;
        self.kv_recorded_pages += other.kv_recorded_pages;
        self.kv_recorded_bytes += other.kv_recorded_bytes;
        self.kv_hit_stage_mask |= other.kv_hit_stage_mask;
        self.kv_record_stage_mask |= other.kv_record_stage_mask;
        self.checkpoint_flush_us += other.checkpoint_flush_us;
        self.checkpoint_prefill_drain_us += other.checkpoint_prefill_drain_us;
        self.checkpoint_local_us += other.checkpoint_local_us;
        self.checkpoint_downstream_write_us += other.checkpoint_downstream_write_us;
        self.checkpoint_downstream_wait_us += other.checkpoint_downstream_wait_us;
        self.checkpoint_total_us += other.checkpoint_total_us;
        self.checkpoint_prefill_drained_replies += other.checkpoint_prefill_drained_replies;
        self.restore_flush_us += other.restore_flush_us;
        self.restore_prefill_drain_us += other.restore_prefill_drain_us;
        self.restore_local_us += other.restore_local_us;
        self.restore_downstream_write_us += other.restore_downstream_write_us;
        self.restore_downstream_wait_us += other.restore_downstream_wait_us;
        self.restore_total_us += other.restore_total_us;
        self.restore_prefill_drained_replies += other.restore_prefill_drained_replies;
        self.verify_span_compute_us += other.verify_span_compute_us;
        self.verify_span_forward_write_us += other.verify_span_forward_write_us;
        self.verify_span_downstream_wait_us += other.verify_span_downstream_wait_us;
        self.verify_span_total_us += other.verify_span_total_us;
        self.verify_span_stage_count += other.verify_span_stage_count;
        self.verify_span_request_count += other.verify_span_request_count;
        self.verify_span_token_count += other.verify_span_token_count;
        self.verify_span_max_tokens = self
            .verify_span_max_tokens
            .max(other.verify_span_max_tokens);
        self.verify_span_checkpointed_requests += other.verify_span_checkpointed_requests;
        self.verify_span_skip_checkpoint_requests += other.verify_span_skip_checkpoint_requests;
    }

    pub fn is_empty(self) -> bool {
        self.kv_lookup_hits == 0
            && self.kv_lookup_misses == 0
            && self.kv_lookup_errors == 0
            && self.kv_imported_pages == 0
            && self.kv_imported_tokens == 0
            && self.kv_recorded_pages == 0
            && self.kv_recorded_bytes == 0
            && self.kv_hit_stage_mask == 0
            && self.kv_record_stage_mask == 0
            && self.checkpoint_flush_us == 0
            && self.checkpoint_prefill_drain_us == 0
            && self.checkpoint_local_us == 0
            && self.checkpoint_downstream_write_us == 0
            && self.checkpoint_downstream_wait_us == 0
            && self.checkpoint_total_us == 0
            && self.checkpoint_prefill_drained_replies == 0
            && self.restore_flush_us == 0
            && self.restore_prefill_drain_us == 0
            && self.restore_local_us == 0
            && self.restore_downstream_write_us == 0
            && self.restore_downstream_wait_us == 0
            && self.restore_total_us == 0
            && self.restore_prefill_drained_replies == 0
            && self.verify_span_compute_us == 0
            && self.verify_span_forward_write_us == 0
            && self.verify_span_downstream_wait_us == 0
            && self.verify_span_total_us == 0
            && self.verify_span_stage_count == 0
            && self.verify_span_request_count == 0
            && self.verify_span_token_count == 0
            && self.verify_span_max_tokens == 0
            && self.verify_span_checkpointed_requests == 0
            && self.verify_span_skip_checkpoint_requests == 0
    }
}

fn expected_phase(kind: WireMessageKind) -> WireStagePhase {
    if kind.is_prefill()
        || matches!(
            kind,
            WireMessageKind::StateImport | WireMessageKind::StateExport
        )
        || kind.is_session_control()
        || kind.is_generation_control()
        || kind.is_prefix_cache_control()
    {
        WireStagePhase::Prefill
    } else if kind.is_decode_replay() {
        WireStagePhase::DecodeReplay
    } else if kind.is_decode_light_context() {
        WireStagePhase::DecodeLight
    } else {
        WireStagePhase::Decode
    }
}
