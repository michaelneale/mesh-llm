use std::path::{Path, PathBuf};

use skippy_protocol::{FlashAttentionType, StageKvCacheMode, StageKvCachePayload};
use skippy_server::EmbeddedOpenAiRequestDefaults;

use super::super::StageWireDType;
use crate::plugin::{MeshConfig, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig};

pub(super) const BUILTIN_CTX_SIZE: u32 = 4096;
pub(super) const BUILTIN_BATCH: u32 = 512;
pub(super) const BUILTIN_UBATCH: u32 = 128;
pub(super) const BUILTIN_PARALLEL: usize = 1;
pub(super) const BUILTIN_PREFILL_CHUNK_SIZE: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_START: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_STEP: usize = 64;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_MAX: usize = 512;
pub(super) const BUILTIN_SAFETY_MARGIN_GB: f64 = 2.0;

#[derive(Clone, Debug)]
pub(crate) struct SkippyConfigResolveRequest<'a> {
    pub(crate) mesh_config: &'a MeshConfig,
    pub(crate) model_id: &'a str,
    pub(crate) model_path: &'a Path,
    pub(crate) model_bytes: u64,
    pub(crate) allocatable_memory_bytes: Option<u64>,
    pub(crate) request_defaults: Option<&'a RequestDefaultsConfig>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSkippyConfig {
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) model_fit: ResolvedModelFitConfig,
    pub(crate) hardware: ResolvedHardwareConfig,
    pub(crate) throughput: ResolvedThroughputConfig,
    pub(crate) skippy: ResolvedSkippyExecutionConfig,
    pub(crate) speculative: ResolvedSpeculativeConfig,
    pub(crate) request_defaults: ResolvedRequestDefaultsConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedModelFitConfig {
    pub(crate) ctx_size: u32,
    pub(crate) batch: u32,
    pub(crate) ubatch: u32,
    pub(crate) cache_type_k: String,
    pub(crate) cache_type_v: String,
    pub(crate) kv_cache_policy: String,
    pub(crate) prefix_cache: ResolvedStageKvCache,
    pub(crate) kv_offload: String,
    pub(crate) flash_attention: FlashAttentionType,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedHardwareConfig {
    pub(crate) device: Option<String>,
    pub(crate) gpu_layers: i32,
    pub(crate) fit_target_mib: Option<u64>,
    pub(crate) resolved_model_path: PathBuf,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) stage_layer_start: Option<u32>,
    pub(crate) stage_layer_end: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedThroughputConfig {
    pub(crate) parallel: usize,
    pub(crate) continuous_batching: String,
    pub(crate) threads: Option<usize>,
    pub(crate) threads_batch: Option<usize>,
    pub(crate) tuning_profile: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSkippyExecutionConfig {
    pub(crate) activation_wire_dtype: StageWireDType,
    pub(crate) activation_wire_dtype_explicit: bool,
    pub(crate) binary_stage_transport: String,
    pub(crate) prefill_chunking: String,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) prefill_chunk_schedule: Option<String>,
    pub(crate) prefill_controls_explicit: bool,
    pub(crate) lifecycle_startup_timeout_ms: Option<u64>,
    pub(crate) lifecycle_readiness_interval_ms: Option<u64>,
    pub(crate) lifecycle_health_interval_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSpeculativeConfig {
    pub(crate) mode: String,
    pub(crate) draft_model_path: Option<PathBuf>,
    pub(crate) pairing_fault: String,
    pub(crate) draft_max_tokens: u32,
    pub(crate) explicit: bool,
    pub(crate) draft_n_gpu_layers: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedStageKvCache {
    FamilyDefault,
    Disabled,
    Explicit(ResolvedStageKvCacheTemplate),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedStageKvCacheTemplate {
    pub(crate) mode: StageKvCacheMode,
    pub(crate) payload: StageKvCachePayload,
    pub(crate) max_entries: Option<usize>,
    pub(crate) max_bytes: Option<u64>,
    pub(crate) min_tokens: Option<u64>,
    pub(crate) shared_prefix_stride_tokens: Option<u64>,
    pub(crate) shared_prefix_record_limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedRequestDefaultsConfig {
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) presence_penalty: Option<f64>,
    pub(crate) frequency_penalty: Option<f64>,
    pub(crate) seed: Option<i64>,
    pub(crate) logit_bias: Option<toml::Value>,
    pub(crate) top_k: Option<i64>,
    pub(crate) min_p: Option<f64>,
    pub(crate) repeat_penalty: Option<f64>,
    pub(crate) repeat_last_n: Option<i64>,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) reasoning_format: Option<String>,
    pub(crate) reasoning_enabled: Option<ReasoningEnabled>,
    pub(crate) reasoning_budget: Option<ReasoningBudget>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedEmbeddedOpenAiArgs {
    pub(crate) model_id: Option<String>,
    pub(crate) default_max_tokens: u32,
    pub(crate) request_defaults: EmbeddedOpenAiRequestDefaults,
    pub(crate) generation_concurrency: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) prefill_chunk_policy: String,
    pub(crate) prefill_chunk_schedule: Option<String>,
    pub(crate) prefill_adaptive_start: usize,
    pub(crate) prefill_adaptive_step: usize,
    pub(crate) prefill_adaptive_max: usize,
    pub(crate) draft_model_path: Option<PathBuf>,
    pub(crate) speculative_window: usize,
    pub(crate) adaptive_speculative_window: bool,
    pub(crate) draft_n_gpu_layers: Option<i32>,
    pub(crate) activation_width: i32,
    pub(crate) wire_dtype: skippy_protocol::binary::WireActivationDType,
    pub(crate) reply_credit_limit: Option<usize>,
    pub(crate) downstream_connect_timeout_secs: u64,
}
