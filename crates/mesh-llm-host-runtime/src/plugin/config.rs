use super::{
    PluginSummary, BLACKBOARD_PLUGIN_ID, BLOBSTORE_PLUGIN_ID, FLASH_MOE_PLUGIN_ID,
    OPENAI_ENDPOINT_PLUGIN_ID, TELEMETRY_PLUGIN_ID,
};
use anyhow::{bail, Context, Result};
use mesh_llm_plugin::MeshVisibility;
use serde::{Deserialize, Serialize};
use skippy_protocol::FlashAttentionType;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const FLASH_MOE_INSTALL_HINT: &str = "Install Flash-MoE separately and set \
                                     `command` to its infer binary, or set \
                                     `url` to an already-running Flash-MoE /v1 endpoint.";

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub owner_control: OwnerControlConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OwnerControlConfig {
    #[serde(default)]
    pub bind: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub advertise_addr: Option<std::net::SocketAddr>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct GpuConfig {
    #[serde(default)]
    pub assignment: GpuAssignment,
    #[serde(default)]
    pub parallel: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuAssignment {
    #[default]
    Auto,
    Pinned,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelConfigDefaults {
    #[serde(default)]
    pub model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    pub hardware: Option<HardwareConfig>,
    #[serde(default)]
    pub throughput: Option<ThroughputConfig>,
    #[serde(default)]
    pub skippy: Option<SkippyConfig>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    pub multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    pub advanced: Option<AdvancedConfig>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelConfigEntry {
    pub model: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub gpu_id: Option<String>,
    #[serde(default)]
    pub parallel: Option<usize>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub flash_attention: Option<FlashAttentionType>,
    #[serde(default)]
    pub model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    pub hardware: Option<HardwareConfig>,
    #[serde(default)]
    pub throughput: Option<ThroughputConfig>,
    #[serde(default)]
    pub skippy: Option<SkippyConfig>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    pub multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    pub advanced: Option<AdvancedConfig>,
    #[serde(skip)]
    pub gpu_id_from_legacy_shim: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelFitConfig {
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub kv_cache_policy: Option<String>,
    #[serde(default)]
    pub kv_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub kv_unified: Option<BoolOrAuto>,
    #[serde(default)]
    pub cache_ram_mib: Option<u64>,
    #[serde(default)]
    pub cache_idle_slots: Option<u32>,
    #[serde(default)]
    pub prompt_cache: Option<BoolOrAuto>,
    #[serde(default)]
    pub prefix_cache: Option<PrefixCacheConfig>,
    #[serde(default)]
    pub keep_tokens: Option<u32>,
    #[serde(default)]
    pub context_shift: Option<BoolOrAuto>,
    #[serde(default)]
    pub swa_full: Option<bool>,
    #[serde(default)]
    pub checkpoint_interval: Option<u32>,
    #[serde(default)]
    pub checkpoint_count: Option<u32>,
    #[serde(default)]
    pub lookup_cache_static: Option<String>,
    #[serde(default)]
    pub lookup_cache_dynamic: Option<String>,
    #[serde(default)]
    pub flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_entries: Option<u32>,
    #[serde(default)]
    pub max_bytes: Option<u64>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub shared_stride_tokens: Option<u32>,
    #[serde(default)]
    pub shared_record_limit: Option<u32>,
    #[serde(default)]
    pub payload_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HardwareConfig {
    #[serde(default)]
    pub model_runtime: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub gpu_layers: Option<IntegerOrString>,
    #[serde(default)]
    pub stage_layer_start: Option<u32>,
    #[serde(default)]
    pub stage_layer_end: Option<u32>,
    #[serde(default)]
    pub placement: Option<String>,
    #[serde(default)]
    pub tensor_split: Option<TensorSplitConfig>,
    #[serde(default)]
    pub split_mode: Option<String>,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub cpu_moe: Option<BoolOrAuto>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub rpc_backend: Option<toml::Value>,
    #[serde(default)]
    pub fit_target_mib: Option<u64>,
    #[serde(default)]
    pub safety_margin_gb: Option<f64>,
    #[serde(default)]
    pub fit_context: Option<BoolOrAuto>,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub hf_file: Option<String>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub lora_adapters: Vec<String>,
    #[serde(default)]
    pub control_vectors: Vec<String>,
    #[serde(default)]
    pub check_tensors: Option<bool>,
    #[serde(default)]
    pub mmap: Option<BoolOrAuto>,
    #[serde(default)]
    pub mlock: Option<bool>,
    #[serde(default)]
    pub direct_io: Option<bool>,
    #[serde(default)]
    pub repack: Option<bool>,
    #[serde(default)]
    pub op_offload: Option<bool>,
    #[serde(default)]
    pub no_host_buffer: Option<bool>,
    #[serde(default)]
    pub warmup: Option<BoolOrAuto>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThroughputConfig {
    #[serde(default)]
    pub parallel: Option<usize>,
    #[serde(default)]
    pub continuous_batching: Option<BoolOrAuto>,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub threads_batch: Option<usize>,
    #[serde(default)]
    pub threads_http: Option<usize>,
    #[serde(default)]
    pub priority: Option<IntegerOrString>,
    #[serde(default)]
    pub poll: Option<BoolOrString>,
    #[serde(default)]
    pub cpu_affinity: Option<StringOrStringList>,
    #[serde(default)]
    pub numa: Option<String>,
    #[serde(default)]
    pub slot_prompt_similarity: Option<f64>,
    #[serde(default)]
    pub sleep_idle_seconds: Option<u64>,
    #[serde(default)]
    pub tuning_profile: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SkippyConfig {
    #[serde(default)]
    pub stage_model_path: Option<String>,
    #[serde(default)]
    pub stage_role: Option<String>,
    #[serde(default)]
    pub stage_topology: Option<String>,
    #[serde(default)]
    pub activation_wire_dtype: Option<String>,
    #[serde(default)]
    pub binary_stage_transport: Option<String>,
    #[serde(default)]
    pub openai_frontend_mode: Option<toml::Value>,
    #[serde(default)]
    pub lifecycle_startup_timeout_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_readiness_interval_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_health_interval_ms: Option<u64>,
    #[serde(default)]
    pub prefill_chunking: Option<String>,
    #[serde(default)]
    pub prefill_chunk_size: Option<u32>,
    #[serde(default)]
    pub prefill_chunk_schedule: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeConfig {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub draft_model_path: Option<String>,
    #[serde(default)]
    pub draft_hf_repo: Option<String>,
    #[serde(default)]
    pub draft_hf_file: Option<String>,
    #[serde(default)]
    pub draft_selection_policy: Option<String>,
    #[serde(default)]
    pub pairing_fault: Option<String>,
    #[serde(default)]
    pub draft_max_tokens: Option<u32>,
    #[serde(default)]
    pub draft_min_tokens: Option<u32>,
    #[serde(default)]
    pub draft_acceptance_threshold: Option<f64>,
    #[serde(default)]
    pub draft_split_probability: Option<f64>,
    #[serde(default)]
    pub draft_gpu_layers: Option<i32>,
    #[serde(default)]
    pub draft_device: Option<String>,
    #[serde(default)]
    pub draft_threads: Option<usize>,
    #[serde(default)]
    pub draft_cache_type_k: Option<String>,
    #[serde(default)]
    pub draft_cache_type_v: Option<String>,
    #[serde(default)]
    pub ngram_min: Option<u32>,
    #[serde(default)]
    pub ngram_max: Option<u32>,
    #[serde(default)]
    pub spec_default: Option<BoolOrAuto>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequestDefaultsConfig {
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StringOrStringList>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub typical_p: Option<f64>,
    #[serde(default)]
    pub top_nsigma: Option<f64>,
    #[serde(default)]
    pub dynatemp_range: Option<f64>,
    #[serde(default)]
    pub dynatemp_exponent: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_last_n: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub dry: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub xtc: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub adaptive: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub mirostat_mode: Option<IntegerOrString>,
    #[serde(default)]
    pub mirostat_entropy: Option<f64>,
    #[serde(default)]
    pub mirostat_learning_rate: Option<f64>,
    #[serde(default)]
    pub samplers: Option<Vec<String>>,
    #[serde(default)]
    pub sampler_sequence: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub logit_bias: Option<toml::Value>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub backend_sampling: Option<toml::Value>,
    #[serde(default)]
    pub reasoning_format: Option<String>,
    #[serde(default)]
    pub reasoning_enabled: Option<ReasoningEnabled>,
    #[serde(default)]
    pub reasoning_budget: Option<ReasoningBudget>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_file: Option<String>,
    #[serde(default)]
    pub jinja: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<toml::Value>,
    #[serde(default)]
    pub skip_chat_parsing: Option<bool>,
    #[serde(default)]
    pub prefill_assistant: Option<toml::Value>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub grammar: Option<toml::Value>,
    #[serde(default)]
    pub json_schema: Option<toml::Value>,
    #[serde(default)]
    pub logprobs: Option<toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultimodalConfig {
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_url: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    #[serde(default)]
    pub embeddings: Option<toml::Value>,
    #[serde(default)]
    pub reranking: Option<toml::Value>,
    #[serde(default)]
    pub pooling: Option<toml::Value>,
    #[serde(default)]
    pub vocoder: Option<toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedConfig {
    #[serde(default)]
    pub server: Option<AdvancedServerConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedServerConfig {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub reuse_port: Option<bool>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub metrics: Option<bool>,
    #[serde(default)]
    pub slots: Option<bool>,
    #[serde(default)]
    pub props: Option<bool>,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub api_prefix: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrAuto {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrString {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum IntegerOrString {
    Integer(i64),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrStringList {
    String(String),
    List(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TensorSplitConfig {
    Ratios(Vec<f64>),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningEnabled {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningBudget {
    Integer(u32),
    String(String),
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReservedObjectConfig {}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawMeshConfig {
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    gpu: GpuConfig,
    #[serde(default)]
    owner_control: OwnerControlConfig,
    #[serde(default)]
    telemetry: TelemetryConfig,
    #[serde(default)]
    defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigDefaults {
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigEntry {
    model: String,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
}

impl<'de> Deserialize<'de> for MeshConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawMeshConfig::deserialize(deserializer)?;
        Ok(Self {
            version: raw.version,
            gpu: raw.gpu,
            owner_control: raw.owner_control,
            telemetry: raw.telemetry,
            defaults: raw.defaults,
            models: raw.models,
            plugins: raw.plugins,
            extra: raw.extra,
        })
    }
}

impl<'de> Deserialize<'de> for ModelConfigDefaults {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigDefaults::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl<'de> Deserialize<'de> for ModelConfigEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigEntry::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl ModelConfigDefaults {
    fn from_raw(raw: RawModelConfigDefaults) -> Self {
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k,
            raw.cache_type_v,
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let hardware = merge_hardware(raw.hardware, raw.gpu_id, None, None);
        let throughput = merge_throughput(raw.throughput, raw.parallel);
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj);
        Self {
            model_fit,
            hardware,
            throughput,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
        }
    }
}

impl ModelConfigEntry {
    fn from_raw(raw: RawModelConfigEntry) -> Self {
        let gpu_id_from_legacy_shim = raw.gpu_id.is_some();
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k.clone(),
            raw.cache_type_v.clone(),
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj.clone());
        let hardware = merge_hardware(
            raw.hardware,
            raw.gpu_id.clone(),
            multimodal.as_ref().and_then(|m| m.mmproj.clone()),
            multimodal.as_ref().and_then(|m| m.mmproj_offload.clone()),
        );
        let throughput = merge_throughput(raw.throughput, raw.parallel);

        Self {
            model: raw.model,
            mmproj: multimodal
                .as_ref()
                .and_then(|config| config.mmproj.clone())
                .or_else(|| hardware.as_ref().and_then(|config| config.mmproj.clone()))
                .or(raw.mmproj),
            ctx_size: model_fit.as_ref().and_then(|config| config.ctx_size),
            gpu_id: hardware
                .as_ref()
                .and_then(|config| config.device.clone())
                .or(raw.gpu_id),
            parallel: throughput.as_ref().and_then(|config| config.parallel),
            cache_type_k: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_k.clone())
                .or(raw.cache_type_k),
            cache_type_v: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_v.clone())
                .or(raw.cache_type_v),
            batch: model_fit.as_ref().and_then(|config| config.batch),
            ubatch: model_fit.as_ref().and_then(|config| config.ubatch),
            flash_attention: model_fit
                .as_ref()
                .and_then(|config| config.flash_attention)
                .or(raw.flash_attention),
            model_fit,
            hardware,
            throughput,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
            gpu_id_from_legacy_shim,
        }
    }
}

fn merge_model_fit(
    current: Option<ModelFitConfig>,
    ctx_size: Option<u32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    batch: Option<u32>,
    ubatch: Option<u32>,
    flash_attention: Option<FlashAttentionType>,
) -> Option<ModelFitConfig> {
    let mut config = current.unwrap_or_default();
    config.ctx_size = config.ctx_size.or(ctx_size);
    config.cache_type_k = config.cache_type_k.or(cache_type_k);
    config.cache_type_v = config.cache_type_v.or(cache_type_v);
    config.batch = config.batch.or(batch);
    config.ubatch = config.ubatch.or(ubatch);
    config.flash_attention = config.flash_attention.or(flash_attention);
    if is_model_fit_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn merge_hardware(
    current: Option<HardwareConfig>,
    gpu_id: Option<String>,
    mmproj: Option<String>,
    mmproj_offload: Option<BoolOrAuto>,
) -> Option<HardwareConfig> {
    let mut config = current.unwrap_or_default();
    config.device = config.device.or(gpu_id);
    config.mmproj = config.mmproj.or(mmproj);
    config.mmproj_offload = config.mmproj_offload.or(mmproj_offload);
    if is_hardware_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn merge_throughput(
    current: Option<ThroughputConfig>,
    parallel: Option<usize>,
) -> Option<ThroughputConfig> {
    let mut config = current.unwrap_or_default();
    config.parallel = config.parallel.or(parallel);
    if is_throughput_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn merge_multimodal(
    current: Option<MultimodalConfig>,
    mmproj: Option<String>,
) -> Option<MultimodalConfig> {
    let mut config = current.unwrap_or_default();
    config.mmproj = config.mmproj.or(mmproj);
    if is_multimodal_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn is_model_fit_empty(config: &ModelFitConfig) -> bool {
    config == &ModelFitConfig::default()
}

fn is_hardware_empty(config: &HardwareConfig) -> bool {
    config == &HardwareConfig::default()
}

fn is_throughput_empty(config: &ThroughputConfig) -> bool {
    config == &ThroughputConfig::default()
}

fn is_multimodal_empty(config: &MultimodalConfig) -> bool {
    config == &MultimodalConfig::default()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub service_name: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub export_interval_secs: Option<u64>,
    #[serde(default)]
    pub queue_size: Option<usize>,
    #[serde(default)]
    pub prompt_shape_metrics: bool,
    #[serde(default)]
    pub metrics: TelemetryMetricsConfig,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryMetricsConfig {
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Base URL for inference endpoint plugins (e.g. http://localhost:8000/v1).
    #[serde(default)]
    pub url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
    pub inactive: Vec<PluginSummary>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// Backend URL for inference endpoint plugins.
    pub url: Option<String>,
    /// Extra environment passed only to the plugin process.
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PluginHostMode {
    pub mesh_visibility: MeshVisibility,
}

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MESH_LLM_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    if !path.exists() {
        return Ok(MeshConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    let config: MeshConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse config {}", path.display()))?;
    validate_config(&config).with_context(|| format!("Invalid config {}", path.display()))?;
    Ok(config)
}

pub(crate) fn validate_config(config: &MeshConfig) -> Result<()> {
    if let Some(version) = config.version {
        if version != 1 {
            bail!("unsupported config version {version}; expected version = 1");
        }
    }
    if let Some(bind) = config.owner_control.bind {
        if bind.port() == 0 && !bind.ip().is_loopback() {
            bail!(
                "owner_control.bind must use a concrete port when binding a non-loopback address"
            );
        }
    }
    if let Some(advertise_addr) = config.owner_control.advertise_addr {
        if advertise_addr.port() == 0 {
            bail!("owner_control.advertise_addr must use a concrete port");
        }
        if advertise_addr.ip().is_unspecified() {
            bail!("owner_control.advertise_addr must not use an unspecified IP address");
        }
    }
    if let Some(parallel) = config.gpu.parallel {
        if parallel < 1 {
            bail!("gpu.parallel must be at least 1, got {parallel}");
        }
    }
    validate_telemetry_config(&config.telemetry)?;
    if let Some(defaults) = &config.defaults {
        validate_model_defaults(defaults, "defaults", config.gpu.assignment)?;
    }
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            bail!("models[{index}].model must not be empty");
        }
        validate_model_entry(model, &format!("models[{index}]"), config.gpu.assignment)?;
    }
    Ok(())
}

fn validate_model_defaults(
    defaults: &ModelConfigDefaults,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    if let Some(model_fit) = &defaults.model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    match defaults.hardware.as_ref() {
        Some(hardware) => {
            validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?
        }
        None if matches!(gpu_assignment, GpuAssignment::Pinned) => validate_hardware(
            &HardwareConfig::default(),
            &format!("{base_path}.hardware"),
            gpu_assignment,
        )?,
        None => {}
    }
    if let Some(throughput) = &defaults.throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &defaults.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &defaults.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &defaults.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        defaults.hardware.as_ref(),
        defaults.multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &defaults.multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &defaults.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    validate_gpu_assignment_constraints(
        defaults.hardware.as_ref(),
        None,
        &format!("{base_path}.hardware.device"),
        gpu_assignment,
    )?;
    Ok(())
}

fn validate_model_entry(
    model: &ModelConfigEntry,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    let model_fit = merge_model_fit(
        model.model_fit.clone(),
        model.ctx_size,
        model.cache_type_k.clone(),
        model.cache_type_v.clone(),
        model.batch,
        model.ubatch,
        model.flash_attention,
    );
    let multimodal = merge_multimodal(model.multimodal.clone(), model.mmproj.clone());
    let hardware = merge_hardware(
        model.hardware.clone(),
        model.gpu_id.clone(),
        multimodal.as_ref().and_then(|config| config.mmproj.clone()),
        multimodal
            .as_ref()
            .and_then(|config| config.mmproj_offload.clone()),
    );
    let throughput = merge_throughput(model.throughput.clone(), model.parallel);

    if let Some(mmproj) = &model.mmproj {
        validate_non_empty(mmproj, &format!("{base_path}.multimodal.mmproj"))?;
    }
    if let Some(model_fit) = &model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    match hardware.as_ref() {
        Some(hardware) => {
            validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?
        }
        None if matches!(gpu_assignment, GpuAssignment::Pinned) => validate_hardware(
            &HardwareConfig::default(),
            &format!("{base_path}.hardware"),
            gpu_assignment,
        )?,
        None => {}
    }
    if let Some(throughput) = &throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &model.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &model.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &model.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        hardware.as_ref(),
        multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &model.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    validate_gpu_assignment_constraints(
        hardware.as_ref(),
        model
            .gpu_id_from_legacy_shim
            .then_some(model.gpu_id.as_deref())
            .flatten(),
        &format!("{base_path}.hardware.device"),
        gpu_assignment,
    )?;
    Ok(())
}

fn validate_gpu_assignment_constraints(
    hardware: Option<&HardwareConfig>,
    legacy_gpu_id: Option<&str>,
    device_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    if matches!(gpu_assignment, GpuAssignment::Auto) && legacy_gpu_id.is_some() {
        bail!("{device_path} must not be set when gpu.assignment = \"auto\"");
    }
    if matches!(gpu_assignment, GpuAssignment::Pinned) {
        match hardware.and_then(|config| config.device.as_deref()) {
            Some(device) if !device.trim().is_empty() && !device.eq_ignore_ascii_case("auto") => {}
            _ => {
                bail!(
                    "{device_path} must be set to a non-empty value when gpu.assignment = \"pinned\""
                );
            }
        }
    }
    Ok(())
}

fn validate_model_fit(config: &ModelFitConfig, base_path: &str) -> Result<()> {
    validate_optional_positive_u32(config.ctx_size, &format!("{base_path}.ctx_size"))?;
    validate_optional_positive_u32(config.batch, &format!("{base_path}.batch"))?;
    validate_optional_positive_u32(config.ubatch, &format!("{base_path}.ubatch"))?;
    if let (Some(batch), Some(ubatch)) = (config.batch, config.ubatch) {
        if ubatch > batch {
            bail!("{base_path}.ubatch must be less than or equal to {base_path}.batch");
        }
    }
    validate_optional_non_empty(
        config.cache_type_k.as_deref(),
        &format!("{base_path}.cache_type_k"),
    )?;
    validate_optional_non_empty(
        config.cache_type_v.as_deref(),
        &format!("{base_path}.cache_type_v"),
    )?;
    validate_optional_enum(
        config.kv_cache_policy.as_deref(),
        &["auto", "quality", "balanced", "saver"],
        &format!("{base_path}.kv_cache_policy"),
    )?;
    validate_bool_or_auto(
        config.kv_offload.as_ref(),
        &format!("{base_path}.kv_offload"),
    )?;
    validate_bool_or_auto(
        config.kv_unified.as_ref(),
        &format!("{base_path}.kv_unified"),
    )?;
    validate_bool_or_auto(
        config.prompt_cache.as_ref(),
        &format!("{base_path}.prompt_cache"),
    )?;
    validate_bool_or_auto(
        config.context_shift.as_ref(),
        &format!("{base_path}.context_shift"),
    )?;
    if let Some(cache_idle_slots) = config.cache_idle_slots {
        if cache_idle_slots > 0 && matches!(config.prompt_cache, Some(BoolOrAuto::Bool(false))) {
            bail!("{base_path}.cache_idle_slots requires {base_path}.prompt_cache = true");
        }
    }
    if let Some(prefix_cache) = &config.prefix_cache {
        validate_prefix_cache(prefix_cache, &format!("{base_path}.prefix_cache"))?;
    }
    if let (Some(keep_tokens), Some(ctx_size)) = (config.keep_tokens, config.ctx_size) {
        if keep_tokens > ctx_size {
            bail!("{base_path}.keep_tokens must be less than or equal to {base_path}.ctx_size");
        }
    }
    validate_optional_positive_u32(
        config.checkpoint_interval,
        &format!("{base_path}.checkpoint_interval"),
    )?;
    validate_optional_positive_u32(
        config.checkpoint_count,
        &format!("{base_path}.checkpoint_count"),
    )?;
    validate_optional_non_empty(
        config.lookup_cache_static.as_deref(),
        &format!("{base_path}.lookup_cache_static"),
    )?;
    validate_optional_non_empty(
        config.lookup_cache_dynamic.as_deref(),
        &format!("{base_path}.lookup_cache_dynamic"),
    )?;
    Ok(())
}

fn validate_prefix_cache(config: &PrefixCacheConfig, base_path: &str) -> Result<()> {
    if config.enabled == Some(false) {
        return Ok(());
    }
    if config.enabled == Some(true) {
        validate_optional_positive_u32(config.max_entries, &format!("{base_path}.max_entries"))?;
        validate_optional_positive_u32(config.min_tokens, &format!("{base_path}.min_tokens"))?;
        validate_optional_positive_u32(
            config.shared_stride_tokens,
            &format!("{base_path}.shared_stride_tokens"),
        )?;
        validate_optional_positive_u32(
            config.shared_record_limit,
            &format!("{base_path}.shared_record_limit"),
        )?;
    }
    validate_optional_enum(
        config.payload_mode.as_deref(),
        &["resident-kv", "kv-recurrent", "full-state", "auto"],
        &format!("{base_path}.payload_mode"),
    )?;
    Ok(())
}

fn validate_hardware(
    config: &HardwareConfig,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    validate_optional_enum(
        config.model_runtime.as_deref(),
        &["auto", "cpu", "cuda", "rocm", "metal", "vulkan"],
        &format!("{base_path}.model_runtime"),
    )?;
    if let Some(device) = &config.device {
        validate_non_empty(device, &format!("{base_path}.device"))?;
        if matches!(gpu_assignment, GpuAssignment::Pinned) && device.eq_ignore_ascii_case("auto") {
            bail!("{base_path}.device must not be \"auto\" when gpu.assignment = \"pinned\"");
        }
    }
    if let Some(gpu_layers) = &config.gpu_layers {
        match gpu_layers {
            IntegerOrString::Integer(value) if *value >= -1 => {}
            IntegerOrString::Integer(_) => bail!("{base_path}.gpu_layers must be at least -1"),
            IntegerOrString::String(value) => {
                validate_allowed(value, &["auto"], &format!("{base_path}.gpu_layers"))?
            }
        }
    }
    match (config.stage_layer_start, config.stage_layer_end) {
        (Some(start), Some(end)) if end <= start => {
            bail!("{base_path}.stage_layer_end must be greater than {base_path}.stage_layer_start");
        }
        (Some(_), None) => bail!(
            "{base_path}.stage_layer_end must be set when {base_path}.stage_layer_start is set"
        ),
        (None, Some(_)) => bail!(
            "{base_path}.stage_layer_start must be set when {base_path}.stage_layer_end is set"
        ),
        _ => {}
    }
    validate_optional_enum(
        config.placement.as_deref(),
        &["auto", "pooled", "separated"],
        &format!("{base_path}.placement"),
    )?;
    if let Some(tensor_split) = &config.tensor_split {
        match tensor_split {
            TensorSplitConfig::Ratios(ratios) => {
                for ratio in ratios {
                    if *ratio < 0.0 {
                        bail!("{base_path}.tensor_split must contain only non-negative ratios");
                    }
                }
            }
            TensorSplitConfig::String(value) => {
                validate_non_empty(value, &format!("{base_path}.tensor_split"))?
            }
        }
    }
    validate_optional_enum(
        config.split_mode.as_deref(),
        &["auto", "none", "layer", "row"],
        &format!("{base_path}.split_mode"),
    )?;
    if let Some(value) = &config.cpu_moe {
        validate_bool_or_auto(Some(value), &format!("{base_path}.cpu_moe"))?;
    }
    if config.rpc_backend.is_some() {
        bail!("{base_path}.rpc_backend is documented-rejected and must not be set");
    }
    if let Some(fit_context) = &config.fit_context {
        validate_bool_or_auto(Some(fit_context), &format!("{base_path}.fit_context"))?;
    }
    validate_non_negative_f64(
        config.safety_margin_gb,
        &format!("{base_path}.safety_margin_gb"),
    )?;
    validate_hf_pair(
        config.hf_repo.as_deref(),
        config.hf_file.as_deref(),
        &format!("{base_path}.hf_repo"),
        &format!("{base_path}.hf_file"),
    )?;
    validate_optional_non_empty(
        config.model_path.as_deref(),
        &format!("{base_path}.model_path"),
    )?;
    validate_optional_non_empty(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    validate_bool_or_auto(config.mmap.as_ref(), &format!("{base_path}.mmap"))?;
    validate_bool_or_auto(config.warmup.as_ref(), &format!("{base_path}.warmup"))?;
    validate_string_list(&config.lora_adapters, &format!("{base_path}.lora_adapters"))?;
    validate_string_list(
        &config.control_vectors,
        &format!("{base_path}.control_vectors"),
    )?;
    Ok(())
}

fn validate_throughput(config: &ThroughputConfig, base_path: &str) -> Result<()> {
    if let Some(parallel) = config.parallel {
        if parallel < 1 {
            bail!("{base_path}.parallel must be at least 1, got {parallel}");
        }
    }
    validate_bool_or_auto(
        config.continuous_batching.as_ref(),
        &format!("{base_path}.continuous_batching"),
    )?;
    // `0` is a canonical auto/default sentinel for threads and threads_batch.
    if config.threads_http.is_some() {
        bail!("{base_path}.threads_http is documented-rejected and must not be set");
    }
    if let Some(BoolOrString::String(value)) = &config.poll {
        validate_allowed(
            value,
            &["auto", "busy", "sleep"],
            &format!("{base_path}.poll"),
        )?;
    }
    if let Some(cpu_affinity) = &config.cpu_affinity {
        match cpu_affinity {
            StringOrStringList::String(value) => {
                validate_non_empty(value, &format!("{base_path}.cpu_affinity"))?
            }
            StringOrStringList::List(values) => {
                validate_string_list(values, &format!("{base_path}.cpu_affinity"))?
            }
        }
    }
    validate_optional_non_empty(config.numa.as_deref(), &format!("{base_path}.numa"))?;
    if let Some(slot_prompt_similarity) = config.slot_prompt_similarity {
        if slot_prompt_similarity < 0.0 {
            bail!("{base_path}.slot_prompt_similarity must be non-negative");
        }
    }
    if config.sleep_idle_seconds.is_some() {
        bail!("{base_path}.sleep_idle_seconds is documented-rejected and must not be set");
    }
    validate_optional_enum(
        config.tuning_profile.as_deref(),
        &["throughput", "balanced", "saver"],
        &format!("{base_path}.tuning_profile"),
    )?;
    Ok(())
}

fn validate_skippy(config: &SkippyConfig, base_path: &str) -> Result<()> {
    validate_optional_non_empty(
        config.stage_model_path.as_deref(),
        &format!("{base_path}.stage_model_path"),
    )?;
    validate_optional_non_empty(
        config.stage_role.as_deref(),
        &format!("{base_path}.stage_role"),
    )?;
    validate_optional_non_empty(
        config.stage_topology.as_deref(),
        &format!("{base_path}.stage_topology"),
    )?;
    validate_optional_enum(
        config.activation_wire_dtype.as_deref(),
        &["auto", "f16", "f32", "q8"],
        &format!("{base_path}.activation_wire_dtype"),
    )?;
    validate_optional_non_empty(
        config.binary_stage_transport.as_deref(),
        &format!("{base_path}.binary_stage_transport"),
    )?;
    if config.openai_frontend_mode.is_some() {
        bail!("{base_path}.openai_frontend_mode is documented-rejected and must not be set");
    }
    validate_optional_positive_u64(
        config.lifecycle_startup_timeout_ms,
        &format!("{base_path}.lifecycle_startup_timeout_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_readiness_interval_ms,
        &format!("{base_path}.lifecycle_readiness_interval_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_health_interval_ms,
        &format!("{base_path}.lifecycle_health_interval_ms"),
    )?;
    validate_optional_enum(
        config.prefill_chunking.as_deref(),
        &["auto", "fixed", "schedule", "adaptive-ramp"],
        &format!("{base_path}.prefill_chunking"),
    )?;
    if let Some(schedule) = &config.prefill_chunk_schedule {
        validate_non_empty(schedule, &format!("{base_path}.prefill_chunk_schedule"))?;
        for item in schedule.split(',') {
            let trimmed = item.trim();
            if trimmed.is_empty()
                || trimmed
                    .parse::<u32>()
                    .ok()
                    .filter(|value| *value > 0)
                    .is_none()
            {
                bail!("{base_path}.prefill_chunk_schedule must contain only comma-separated positive integers");
            }
        }
    }
    Ok(())
}

fn validate_speculative(config: &SpeculativeConfig, base_path: &str) -> Result<()> {
    validate_optional_enum(
        config.mode.as_deref(),
        &["auto", "disabled", "draft", "ngram"],
        &format!("{base_path}.mode"),
    )?;
    validate_hf_pair(
        config.draft_hf_repo.as_deref(),
        config.draft_hf_file.as_deref(),
        &format!("{base_path}.draft_hf_repo"),
        &format!("{base_path}.draft_hf_file"),
    )?;
    validate_optional_enum(
        config.draft_selection_policy.as_deref(),
        &["manual", "auto"],
        &format!("{base_path}.draft_selection_policy"),
    )?;
    validate_optional_enum(
        config.pairing_fault.as_deref(),
        &[
            "warn_disable",
            "fail-open",
            "fail-closed",
            "fail_open",
            "fail_closed",
        ],
        &format!("{base_path}.pairing_fault"),
    )?;
    validate_optional_positive_u32(
        config.draft_max_tokens,
        &format!("{base_path}.draft_max_tokens"),
    )?;
    if let (Some(min), Some(max)) = (config.draft_min_tokens, config.draft_max_tokens) {
        if min > max {
            bail!("{base_path}.draft_min_tokens must be less than or equal to {base_path}.draft_max_tokens");
        }
    }
    validate_probability(
        config.draft_acceptance_threshold,
        &format!("{base_path}.draft_acceptance_threshold"),
    )?;
    validate_probability(
        config.draft_split_probability,
        &format!("{base_path}.draft_split_probability"),
    )?;
    if let Some(gpu_layers) = config.draft_gpu_layers {
        if gpu_layers < -1 {
            bail!("{base_path}.draft_gpu_layers must be at least -1");
        }
    }
    validate_optional_non_empty(
        config.draft_device.as_deref(),
        &format!("{base_path}.draft_device"),
    )?;
    validate_optional_positive_usize(config.draft_threads, &format!("{base_path}.draft_threads"))?;
    validate_optional_non_empty(
        config.draft_cache_type_k.as_deref(),
        &format!("{base_path}.draft_cache_type_k"),
    )?;
    validate_optional_non_empty(
        config.draft_cache_type_v.as_deref(),
        &format!("{base_path}.draft_cache_type_v"),
    )?;
    validate_optional_positive_u32(config.ngram_min, &format!("{base_path}.ngram_min"))?;
    validate_optional_positive_u32(config.ngram_max, &format!("{base_path}.ngram_max"))?;
    if let (Some(min), Some(max)) = (config.ngram_min, config.ngram_max) {
        if max < min {
            bail!("{base_path}.ngram_max must be greater than or equal to {base_path}.ngram_min");
        }
    }
    validate_bool_or_auto(
        config.spec_default.as_ref(),
        &format!("{base_path}.spec_default"),
    )?;
    if config.mode.as_deref() == Some("draft")
        && config.draft_model_path.is_none()
        && config.draft_hf_repo.is_none()
        && config.draft_selection_policy.is_none()
    {
        bail!("{base_path}.draft_selection_policy must be set when {base_path}.mode = \"draft\" and no explicit draft model source is configured");
    }
    Ok(())
}

fn validate_request_defaults(config: &RequestDefaultsConfig, base_path: &str) -> Result<()> {
    validate_optional_positive_u32(config.max_tokens, &format!("{base_path}.max_tokens"))?;
    if let Some(stop) = &config.stop {
        match stop {
            StringOrStringList::String(value) => {
                validate_non_empty(value, &format!("{base_path}.stop"))?
            }
            StringOrStringList::List(values) => {
                validate_string_list(values, &format!("{base_path}.stop"))?
            }
        }
    }
    validate_non_negative_f64(config.temperature, &format!("{base_path}.temperature"))?;
    validate_probability(config.top_p, &format!("{base_path}.top_p"))?;
    if let Some(top_k) = config.top_k {
        if top_k < 0 {
            bail!("{base_path}.top_k must be greater than or equal to 0");
        }
    }
    validate_probability(config.min_p, &format!("{base_path}.min_p"))?;
    validate_probability(config.typical_p, &format!("{base_path}.typical_p"))?;
    validate_non_negative_f64(config.top_nsigma, &format!("{base_path}.top_nsigma"))?;
    validate_non_negative_f64(
        config.dynatemp_range,
        &format!("{base_path}.dynatemp_range"),
    )?;
    validate_non_negative_f64(
        config.dynatemp_exponent,
        &format!("{base_path}.dynatemp_exponent"),
    )?;
    validate_non_negative_f64(
        config.repeat_penalty,
        &format!("{base_path}.repeat_penalty"),
    )?;
    if let Some(repeat_last_n) = config.repeat_last_n {
        if repeat_last_n < -1 {
            bail!("{base_path}.repeat_last_n must be greater than or equal to -1");
        }
    }
    validate_non_negative_f64(
        config.presence_penalty,
        &format!("{base_path}.presence_penalty"),
    )?;
    validate_non_negative_f64(
        config.frequency_penalty,
        &format!("{base_path}.frequency_penalty"),
    )?;
    if let Some(mode) = &config.mirostat_mode {
        match mode {
            IntegerOrString::Integer(value) if *value == 1 || *value == 2 => {}
            IntegerOrString::String(value) => validate_allowed(
                value,
                &["disabled", "1", "2"],
                &format!("{base_path}.mirostat_mode"),
            )?,
            _ => bail!("{base_path}.mirostat_mode must be one of: disabled, 1, 2"),
        }
    }
    validate_positive_f64(
        config.mirostat_entropy,
        &format!("{base_path}.mirostat_entropy"),
    )?;
    validate_positive_f64(
        config.mirostat_learning_rate,
        &format!("{base_path}.mirostat_learning_rate"),
    )?;
    if let Some(samplers) = &config.samplers {
        validate_string_list(samplers, &format!("{base_path}.samplers"))?;
    }
    validate_optional_non_empty(
        config.sampler_sequence.as_deref(),
        &format!("{base_path}.sampler_sequence"),
    )?;
    if config.backend_sampling.is_some() {
        bail!("{base_path}.backend_sampling is documented-rejected and must not be set");
    }
    validate_optional_enum(
        config.reasoning_format.as_deref(),
        &["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
        &format!("{base_path}.reasoning_format"),
    )?;
    if let Some(reasoning_enabled) = &config.reasoning_enabled {
        match reasoning_enabled {
            ReasoningEnabled::Bool(_) => {}
            ReasoningEnabled::String(value) => validate_allowed(
                value,
                &["auto", "off", "on"],
                &format!("{base_path}.reasoning_enabled"),
            )?,
        }
    }
    if let Some(reasoning_budget) = &config.reasoning_budget {
        match reasoning_budget {
            ReasoningBudget::Integer(_) => {}
            ReasoningBudget::String(value) => validate_allowed(
                value,
                &["auto", "low", "medium", "high"],
                &format!("{base_path}.reasoning_budget"),
            )?,
        }
    }
    validate_optional_non_empty(
        config.chat_template.as_deref(),
        &format!("{base_path}.chat_template"),
    )?;
    validate_optional_non_empty(
        config.chat_template_file.as_deref(),
        &format!("{base_path}.chat_template_file"),
    )?;
    validate_optional_non_empty(
        config.system_prompt.as_deref(),
        &format!("{base_path}.system_prompt"),
    )?;
    if config.grammar.is_some() {
        bail!("{base_path}.grammar is documented-rejected and must not be set");
    }
    if config.json_schema.is_some() {
        bail!("{base_path}.json_schema is documented-rejected and must not be set");
    }
    if config.logprobs.is_some() {
        bail!("{base_path}.logprobs is documented-rejected and must not be set");
    }
    Ok(())
}

fn validate_multimodal_pair(
    hardware: Option<&HardwareConfig>,
    multimodal: Option<&MultimodalConfig>,
    hardware_path: &str,
    multimodal_path: &str,
) -> Result<()> {
    if let (Some(hardware), Some(multimodal)) = (hardware, multimodal) {
        if let (Some(hardware_mmproj), Some(multimodal_mmproj)) =
            (hardware.mmproj.as_deref(), multimodal.mmproj.as_deref())
        {
            if hardware_mmproj != multimodal_mmproj {
                bail!(
                    "{multimodal_path}.mmproj must match {hardware_path}.mmproj when both are set"
                );
            }
        }
        if let (Some(hardware_offload), Some(multimodal_offload)) = (
            hardware.mmproj_offload.as_ref(),
            multimodal.mmproj_offload.as_ref(),
        ) {
            if hardware_offload != multimodal_offload {
                bail!("{multimodal_path}.mmproj_offload must match {hardware_path}.mmproj_offload when both are set");
            }
        }
    }
    Ok(())
}

fn validate_multimodal(config: &MultimodalConfig, base_path: &str) -> Result<()> {
    validate_optional_non_empty(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_optional_non_empty(
        config.mmproj_url.as_deref(),
        &format!("{base_path}.mmproj_url"),
    )?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    if let (Some(min), Some(max)) = (config.image_min_tokens, config.image_max_tokens) {
        if min > max {
            bail!("{base_path}.image_min_tokens must be less than or equal to {base_path}.image_max_tokens");
        }
    }
    if config.embeddings.is_some() {
        bail!("{base_path}.embeddings is documented-rejected and must not be set");
    }
    if config.reranking.is_some() {
        bail!("{base_path}.reranking is documented-rejected and must not be set");
    }
    if config.pooling.is_some() {
        bail!("{base_path}.pooling is documented-rejected and must not be set");
    }
    if config.vocoder.is_some() {
        bail!("{base_path}.vocoder is documented-rejected and must not be set");
    }
    Ok(())
}

fn validate_advanced(config: &AdvancedConfig, base_path: &str) -> Result<()> {
    if let Some(server) = &config.server {
        if server.host.is_some() {
            bail!("{base_path}.server.host is documented-rejected and must not be set");
        }
        if server.port.is_some() {
            bail!("{base_path}.server.port is documented-rejected and must not be set");
        }
        if server.reuse_port.is_some() {
            bail!("{base_path}.server.reuse_port is documented-rejected and must not be set");
        }
        if server.timeout.is_some() {
            bail!("{base_path}.server.timeout is documented-rejected and must not be set");
        }
        if server.metrics.is_some() {
            bail!("{base_path}.server.metrics is documented-rejected and must not be set");
        }
        if server.slots.is_some() {
            bail!("{base_path}.server.slots is documented-rejected and must not be set");
        }
        if server.props.is_some() {
            bail!("{base_path}.server.props is documented-rejected and must not be set");
        }
        if server.api_prefix.is_some() {
            bail!("{base_path}.server.api_prefix is documented-rejected and must not be set");
        }
        validate_optional_non_empty(
            server.alias.as_deref(),
            &format!("{base_path}.server.alias"),
        )?;
    }
    Ok(())
}

fn validate_optional_positive_u32(value: Option<u32>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_positive_u64(value: Option<u64>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_positive_usize(value: Option<usize>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_non_empty(value: Option<&str>, path: &str) -> Result<()> {
    if let Some(value) = value {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

fn validate_non_empty(value: &str, path: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{path} must not be empty when set");
    }
    Ok(())
}

fn validate_optional_enum(value: Option<&str>, allowed: &[&str], path: &str) -> Result<()> {
    if let Some(value) = value {
        validate_allowed(value, allowed, path)?;
    }
    Ok(())
}

fn validate_allowed(value: &str, allowed: &[&str], path: &str) -> Result<()> {
    validate_non_empty(value, path)?;
    if !allowed
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        bail!("{path} must be one of: {}", allowed.join(", "));
    }
    Ok(())
}

fn validate_bool_or_auto(value: Option<&BoolOrAuto>, path: &str) -> Result<()> {
    if let Some(BoolOrAuto::String(value)) = value {
        validate_allowed(value, &["auto"], path)?;
    }
    Ok(())
}

fn validate_probability(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value {
        if !(0.0..=1.0).contains(&value) {
            bail!("{path} must be between 0.0 and 1.0");
        }
    }
    Ok(())
}

fn validate_non_negative_f64(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value {
        if value < 0.0 {
            bail!("{path} must be greater than or equal to 0.0");
        }
    }
    Ok(())
}

fn validate_positive_f64(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value {
        if value <= 0.0 {
            bail!("{path} must be greater than 0.0");
        }
    }
    Ok(())
}

fn validate_hf_pair(
    repo: Option<&str>,
    file: Option<&str>,
    repo_path: &str,
    file_path: &str,
) -> Result<()> {
    validate_optional_non_empty(repo, repo_path)?;
    validate_optional_non_empty(file, file_path)?;
    match (repo, file) {
        (Some(_), None) => bail!("{file_path} must be set when {repo_path} is set"),
        (None, Some(_)) => bail!("{repo_path} must be set when {file_path} is set"),
        _ => Ok(()),
    }
}

fn validate_string_list(values: &[String], path: &str) -> Result<()> {
    for value in values {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

fn validate_telemetry_config(config: &TelemetryConfig) -> Result<()> {
    if let Some(service_name) = &config.service_name {
        if service_name.trim().is_empty() {
            bail!("telemetry.service_name must not be empty when set");
        }
    }
    if let Some(endpoint) = &config.endpoint {
        if endpoint.trim().is_empty() {
            bail!("telemetry.endpoint must not be empty when set");
        }
    }
    if let Some(endpoint) = &config.metrics.endpoint {
        if endpoint.trim().is_empty() {
            bail!("telemetry.metrics.endpoint must not be empty when set");
        }
    }
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            bail!("telemetry.headers keys must not be empty");
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs {
        if export_interval_secs < 1 {
            bail!("telemetry.export_interval_secs must be at least 1");
        }
    }
    if let Some(queue_size) = config.queue_size {
        if queue_size < 1 {
            bail!("telemetry.queue_size must be at least 1");
        }
    }
    if config.prompt_shape_metrics {
        bail!("telemetry.prompt_shape_metrics is not supported yet and must remain false");
    }
    Ok(())
}

pub(crate) fn telemetry_plugin_enabled(config: &MeshConfig) -> bool {
    config
        .plugins
        .iter()
        .find(|entry| entry.name == TELEMETRY_PLUGIN_ID)
        .map(|entry| entry.enabled.unwrap_or(true))
        .unwrap_or(true)
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blackboard_enabled = true;
    let mut blobstore_enabled = true;
    let mut openai_endpoint_enabled = false;
    let mut openai_endpoint_url: Option<String> = None;
    let mut flash_moe_entry: Option<&PluginConfigEntry> = None;
    let mut telemetry_enabled = true;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLACKBOARD_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLACKBOARD_PLUGIN_ID
                );
            }
            blackboard_enabled = enabled;
            continue;
        }
        if entry.name == BLOBSTORE_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLOBSTORE_PLUGIN_ID
                );
            }
            blobstore_enabled = enabled;
            continue;
        }
        if entry.name == OPENAI_ENDPOINT_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` and `url` may be set",
                    OPENAI_ENDPOINT_PLUGIN_ID
                );
            }
            openai_endpoint_enabled = enabled;
            if let Some(ref url) = entry.url {
                openai_endpoint_url = Some(url.clone());
            }
            continue;
        }
        if entry.name == FLASH_MOE_PLUGIN_ID {
            if !enabled {
                continue;
            }
            flash_moe_entry = Some(entry);
            continue;
        }
        if entry.name == TELEMETRY_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() || entry.url.is_some() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    TELEMETRY_PLUGIN_ID
                );
            }
            telemetry_enabled = enabled;
            continue;
        }
        if !enabled {
            continue;
        }
        let command = entry
            .command
            .clone()
            .with_context(|| format!("Plugin '{}' is enabled but missing command", entry.name))?;
        externals.push(ExternalPluginSpec {
            name: entry.name.clone(),
            command,
            args: entry.args.clone(),
            url: None,
            env: BTreeMap::new(),
        });
    }

    if blackboard_enabled {
        externals.insert(0, blackboard_plugin_spec()?);
    }
    if telemetry_enabled {
        let insert_at = usize::from(blackboard_enabled).min(externals.len());
        externals.insert(insert_at, telemetry_plugin_spec()?);
    }
    if openai_endpoint_enabled {
        let mut spec = openai_endpoint_plugin_spec()?;
        spec.url = openai_endpoint_url;
        externals.push(spec);
    }
    if let Some(entry) = flash_moe_entry {
        externals.push(flash_moe_plugin_spec(entry)?);
    }
    if blobstore_enabled {
        externals.push(blobstore_plugin_spec()?);
    }

    Ok(ResolvedPlugins {
        externals,
        inactive,
    })
}

pub fn blackboard_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLACKBOARD_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLACKBOARD_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn blobstore_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLOBSTORE_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            BLOBSTORE_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn openai_endpoint_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: OPENAI_ENDPOINT_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            OPENAI_ENDPOINT_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

pub fn flash_moe_plugin_spec(entry: &PluginConfigEntry) -> Result<ExternalPluginSpec> {
    let backend_command = entry
        .command
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let endpoint_url = entry
        .url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if backend_command.is_some() && endpoint_url.is_some() {
        bail!(
            "Plugin '{}' accepts either `command` for a managed flash-moe process or `url` for an already-running endpoint, not both",
            FLASH_MOE_PLUGIN_ID
        );
    }
    if backend_command.is_none() && endpoint_url.is_none() {
        bail!(
            "Plugin '{}' requires `command` or `url`. {}",
            FLASH_MOE_PLUGIN_ID,
            FLASH_MOE_INSTALL_HINT
        );
    }
    if backend_command.is_none() && !entry.args.is_empty() {
        bail!("Plugin '{}' args require `command`", FLASH_MOE_PLUGIN_ID);
    }
    if entry
        .args
        .iter()
        .any(|arg| arg == "--serve" || arg.starts_with("--serve="))
    {
        bail!(
            "Plugin '{}' owns the flash-moe `--serve` port; remove `--serve` from args",
            FLASH_MOE_PLUGIN_ID
        );
    }

    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    let mut env = BTreeMap::new();
    if let Some(backend_command) = backend_command {
        env.insert(
            "MESH_LLM_FLASH_MOE_COMMAND".to_string(),
            backend_command.to_string(),
        );
        env.insert(
            "MESH_LLM_FLASH_MOE_ARGS_JSON".to_string(),
            serde_json::to_string(&entry.args)?,
        );
    }
    if let Some(url) = endpoint_url {
        env.insert("MESH_LLM_FLASH_MOE_URL".to_string(), url.to_string());
    }

    Ok(ExternalPluginSpec {
        name: FLASH_MOE_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            FLASH_MOE_PLUGIN_ID.into(),
        ],
        url: None,
        env,
    })
}

pub fn telemetry_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: TELEMETRY_PLUGIN_ID.to_string(),
        command,
        args: vec![
            "--log-format".into(),
            "json".into(),
            "--plugin".into(),
            TELEMETRY_PLUGIN_ID.into(),
        ],
        url: None,
        env: BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const FULL_SURFACE_VALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");
    const FULL_SURFACE_INVALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_invalid.toml");

    fn documented_matrix_key_paths() -> BTreeSet<String> {
        let matrix = include_str!("../../../../docs/skippy/CONFIGURATION.md");
        matrix
            .lines()
            .filter(|line| line.starts_with('|'))
            .filter_map(|line| {
                let columns: Vec<_> = line.split('|').map(str::trim).collect();
                columns.get(3).copied()
            })
            .filter(|cell| cell.contains('`'))
            .flat_map(|cell| {
                cell.split("<br>")
                    .filter_map(|part| {
                        let trimmed = part.trim();
                        trimmed
                            .strip_prefix('`')
                            .and_then(|value| value.strip_suffix('`'))
                    })
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn test_model(name: &str) -> ModelConfigEntry {
        ModelConfigEntry {
            model: name.into(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            model_fit: None,
            hardware: None,
            throughput: None,
            skippy: None,
            speculative: None,
            request_defaults: None,
            multimodal: None,
            advanced: None,
            gpu_id_from_legacy_shim: false,
        }
    }

    #[test]
    fn parse_unified_config_keeps_plugins_and_models() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[owner_control]
bind = "127.0.0.1:7447"
advertise_addr = "203.0.113.10:18443"

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[[models]]
model = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/model.gguf"
mmproj = "bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf"

[[plugin]]
name = "demo"
command = "/tmp/demo"
"#,
        )
        .unwrap();

        assert_eq!(config.version, Some(1));
        assert_eq!(
            config.owner_control.bind,
            Some("127.0.0.1:7447".parse().unwrap())
        );
        assert_eq!(
            config.owner_control.advertise_addr,
            Some("203.0.113.10:18443".parse().unwrap())
        );
        assert_eq!(config.gpu.assignment, GpuAssignment::Auto);
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].model, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.models[0].gpu_id, None);
        assert_eq!(config.models[0].cache_type_k, None);
        assert_eq!(config.models[0].cache_type_v, None);
        assert_eq!(config.models[0].batch, None);
        assert_eq!(config.models[0].ubatch, None);
        assert_eq!(config.models[0].flash_attention, None);
        assert_eq!(
            config.models[1].mmproj.as_deref(),
            Some("bartowski/Qwen2.5-VL-7B-Instruct-GGUF/mmproj.gguf")
        );
        assert_eq!(config.models[1].gpu_id, None);
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].name, "demo");
    }

    #[test]
    fn telemetry_config_deserializes_standard_metrics_settings() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048
prompt_shape_metrics = false

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"

[[plugin]]
name = "telemetry"
enabled = true
"#,
        )
        .unwrap();

        assert_eq!(config.telemetry.enabled, Some(true));
        assert_eq!(config.telemetry.service_name.as_deref(), Some("mesh-llm"));
        assert_eq!(
            config.telemetry.endpoint.as_deref(),
            Some("https://otel.example.com")
        );
        assert_eq!(
            config.telemetry.metrics.endpoint.as_deref(),
            Some("https://otel.example.com/v1/metrics")
        );
        assert_eq!(
            config
                .telemetry
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer TOKEN")
        );
        assert_eq!(config.telemetry.export_interval_secs, Some(15));
        assert_eq!(config.telemetry.queue_size, Some(2048));
        assert!(!config.telemetry.prompt_shape_metrics);
    }

    #[test]
    fn telemetry_config_rejects_zero_queue_size() {
        let config: MeshConfig = toml::from_str(
            r#"
[telemetry]
queue_size = 0
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("telemetry.queue_size must be at least 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn owner_control_config_rejects_ephemeral_non_loopback_bind() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "0.0.0.0:0"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "owner_control.bind must use a concrete port when binding a non-loopback address"
        ));
    }

    #[test]
    fn owner_control_config_rejects_unspecified_advertise_addr() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "0.0.0.0:18443"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("owner_control.advertise_addr must not use an unspecified IP address"));
    }

    #[test]
    fn owner_control_config_rejects_ephemeral_advertise_addr() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "127.0.0.1:0"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("owner_control.advertise_addr must use a concrete port"));
    }

    #[test]
    fn telemetry_config_rejects_prompt_shape_metrics_until_reviewed() {
        let config: MeshConfig = toml::from_str(
            r#"
[telemetry]
prompt_shape_metrics = true
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("telemetry.prompt_shape_metrics is not supported yet"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn flash_moe_config_requires_external_command_or_endpoint_with_install_hint() {
        let entry = PluginConfigEntry {
            name: FLASH_MOE_PLUGIN_ID.to_string(),
            enabled: Some(true),
            command: None,
            args: Vec::new(),
            url: None,
        };

        let err = flash_moe_plugin_spec(&entry)
            .expect_err("flash-moe requires a managed command or attached endpoint");
        let message = err.to_string();

        assert!(message.contains("Install Flash-MoE separately"));
        assert!(message.contains("command"));
        assert!(message.contains("url"));
    }

    #[test]
    fn pinned_gpu_config_accepted_pinned_config() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = "pci:0000:65:00.0"
ctx_size = 8192
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    }

    #[test]
    fn pinned_gpu_config_missing_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains(
            "models[0].hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
        ));
    }

    #[test]
    fn pinned_gpu_config_empty_gpu_id_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Pinned,
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                gpu_id: Some("  \t  ".into()),
                gpu_id_from_legacy_shim: true,
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].hardware.device must not be empty when set"));
    }

    #[test]
    fn pinned_gpu_config_auto_assignment_rejects_gpu_id() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            models: vec![ModelConfigEntry {
                gpu_id: Some("pci:0000:65:00.0".into()),
                gpu_id_from_legacy_shim: true,
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].hardware.device must not be set when gpu.assignment = \"auto\""));
    }

    #[test]
    fn pinned_gpu_config_preserves_accepted_gpu_id_string_exactly() {
        let raw = r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
gpu_id = " pci:0000:65:00.0 "
"#;

        let config: MeshConfig = toml::from_str(raw).unwrap();
        validate_config(&config).unwrap();

        assert_eq!(
            config.models[0].gpu_id.as_deref(),
            Some(" pci:0000:65:00.0 ")
        );
    }

    // ── gpu.parallel validation ──

    #[test]
    fn gpu_parallel_field_deserializes_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"
parallel = 8

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, Some(8));
    }

    #[test]
    fn gpu_parallel_defaults_to_none_when_omitted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        assert_eq!(config.gpu.parallel, None);
    }

    #[test]
    fn gpu_parallel_zero_rejected() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(0),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu.parallel must be at least 1, got 0"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn gpu_parallel_one_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(1),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_none_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_large_value_accepted() {
        let config = MeshConfig {
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: Some(64),
            },
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };

        validate_config(&config).unwrap();
    }

    #[test]
    fn gpu_parallel_unwrap_or_default_is_4() {
        fn parsed_parallel(value: Option<usize>) -> usize {
            value.unwrap_or(4)
        }

        assert_eq!(parsed_parallel(None), 4);
        assert_eq!(parsed_parallel(Some(1)), 1);
        assert_eq!(parsed_parallel(Some(8)), 8);
        assert_eq!(parsed_parallel(Some(64)), 64);
    }

    #[test]
    fn per_model_parallel_valid_value_accepted() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                parallel: Some(8),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn per_model_parallel_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                parallel: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string()
                .contains("models[0].throughput.parallel must be at least 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn per_model_parallel_none_accepted() {
        let config = MeshConfig {
            models: vec![test_model("Qwen3-8B-Q4_K_M")],
            ..MeshConfig::default()
        };
        validate_config(&config).unwrap();
    }

    #[test]
    fn model_runtime_overrides_deserialize_from_toml() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 2048
ubatch = 512
flash_attention = "enabled"
"#,
        )
        .unwrap();

        assert_eq!(config.models[0].cache_type_k.as_deref(), Some("q8_0"));
        assert_eq!(config.models[0].cache_type_v.as_deref(), Some("q4_0"));
        assert_eq!(config.models[0].batch, Some(2048));
        assert_eq!(config.models[0].ubatch, Some(512));
        assert_eq!(
            config.models[0].flash_attention,
            Some(FlashAttentionType::Enabled)
        );
    }

    #[test]
    fn model_cache_type_k_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                cache_type_k: Some("   ".into()),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].model_fit.cache_type_k must not be empty when set"));
    }

    #[test]
    fn model_cache_type_v_empty_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                cache_type_v: Some("   ".into()),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].model_fit.cache_type_v must not be empty when set"));
    }

    #[test]
    fn model_batch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                batch: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].model_fit.batch must be at least 1 when set"));
    }

    #[test]
    fn model_ubatch_zero_rejected() {
        let config = MeshConfig {
            models: vec![ModelConfigEntry {
                ubatch: Some(0),
                ..test_model("Qwen3-8B-Q4_K_M")
            }],
            ..MeshConfig::default()
        };

        let err = validate_config(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("models[0].model_fit.ubatch must be at least 1 when set"));
    }

    #[test]
    fn defaults_nested_sections_preserve_existing_behavior_when_omitted() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
parallel = 4
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        assert!(config.defaults.is_none());
        assert_eq!(config.models[0].ctx_size, Some(8192));
        assert_eq!(config.models[0].parallel, Some(4));
        assert_eq!(
            config.models[0].model_fit.as_ref().and_then(|v| v.ctx_size),
            Some(8192)
        );
        assert_eq!(
            config.models[0]
                .throughput
                .as_ref()
                .and_then(|v| v.parallel),
            Some(4)
        );
    }

    #[test]
    fn nested_defaults_parse_representative_sections() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.model_fit]
ctx_size = 4096
kv_cache_policy = "balanced"

[defaults.hardware]
model_runtime = "cuda"

[defaults.throughput]
parallel = 2

[defaults.skippy]
activation_wire_dtype = "f16"

[defaults.speculative]
mode = "ngram"

[defaults.request_defaults]
temperature = 0.2

[defaults.multimodal]
image_max_tokens = 4096

[defaults.advanced.server]
alias = "qwen-local"

[[models]]
model = "Qwen3-8B-Q4_K_M"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let defaults = config.defaults.expect("defaults should parse");
        assert_eq!(defaults.model_fit.and_then(|v| v.ctx_size), Some(4096));
        assert_eq!(
            defaults.hardware.and_then(|v| v.model_runtime),
            Some("cuda".into())
        );
        assert_eq!(defaults.throughput.and_then(|v| v.parallel), Some(2));
        assert_eq!(
            defaults.skippy.and_then(|v| v.activation_wire_dtype),
            Some("f16".into())
        );
        assert_eq!(
            defaults.speculative.and_then(|v| v.mode),
            Some("ngram".into())
        );
    }

    #[test]
    fn canonical_plan_example_auto_sentinels_parse_and_validate() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[defaults.model_fit]
ctx_size = 8192
batch = 512
ubatch = 128
kv_cache_policy = "auto"
cache_type_k = "auto"
cache_type_v = "auto"
kv_offload = "auto"
kv_unified = "auto"
cache_ram_mib = 0
cache_idle_slots = 0
prompt_cache = "auto"
context_shift = "auto"

[defaults.hardware]
model_runtime = "auto"
device = "auto"
gpu_layers = "auto"
tensor_split = []
split_mode = "auto"
main_gpu = 0
placement = "auto"
safety_margin_gb = 2.0
mmap = "auto"
mlock = false
direct_io = false
warmup = "auto"

[defaults.throughput]
parallel = 1
continuous_batching = "auto"
threads = 0
threads_batch = 0
tuning_profile = "balanced"
numa = "auto"
cpu_affinity = []

[defaults.skippy]
activation_wire_dtype = "auto"
prefill_chunking = "auto"
prefill_chunk_size = 0
binary_stage_transport = "auto"

[defaults.speculative]
mode = "auto"
draft_selection_policy = "auto"
pairing_fault = "warn_disable"
draft_max_tokens = 16
draft_min_tokens = 0
draft_acceptance_threshold = 0.0

[defaults.request_defaults]
temperature = 0.8
top_p = 0.95
top_k = 40
min_p = 0.0
repeat_penalty = 1.0
repeat_last_n = 64
reasoning_format = "auto"
reasoning_budget = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192

[models.model_fit]
ctx_size = 16384
cache_type_k = "q8_0"
cache_type_v = "q8_0"

[models.hardware]
gpu_layers = 99
device = "cuda:0"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let defaults = config.defaults.as_ref().expect("defaults should parse");
        assert!(matches!(
            defaults.model_fit.as_ref().and_then(|v| v.kv_unified.as_ref()),
            Some(BoolOrAuto::String(value)) if value == "auto"
        ));
        assert!(matches!(
            defaults.hardware.as_ref().and_then(|v| v.gpu_layers.as_ref()),
            Some(IntegerOrString::String(value)) if value == "auto"
        ));
        assert!(matches!(
            defaults.hardware.as_ref().and_then(|v| v.tensor_split.as_ref()),
            Some(TensorSplitConfig::Ratios(values)) if values.is_empty()
        ));
        assert!(matches!(
            defaults.request_defaults.as_ref().and_then(|v| v.reasoning_budget.as_ref()),
            Some(ReasoningBudget::String(value)) if value == "auto"
        ));
        assert_eq!(config.models[0].ctx_size, Some(16384));
        assert_eq!(config.models[0].gpu_id.as_deref(), Some("cuda:0"));
    }

    #[test]
    fn legacy_flat_fields_normalize_into_nested_sections() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 8192
gpu_id = "pci:0000:65:00.0"
parallel = 6
cache_type_k = "q8_0"
cache_type_v = "q4_0"
batch = 1024
ubatch = 256
flash_attention = "enabled"
mmproj = "projector.gguf"
"#,
        )
        .unwrap();

        let model = &config.models[0];
        assert_eq!(
            model.model_fit.as_ref().and_then(|v| v.ctx_size),
            Some(8192)
        );
        assert_eq!(
            model.hardware.as_ref().and_then(|v| v.device.as_deref()),
            Some("pci:0000:65:00.0")
        );
        assert_eq!(model.throughput.as_ref().and_then(|v| v.parallel), Some(6));
        assert_eq!(
            model
                .model_fit
                .as_ref()
                .and_then(|v| v.cache_type_k.as_deref()),
            Some("q8_0")
        );
        assert_eq!(model.model_fit.as_ref().and_then(|v| v.batch), Some(1024));
        assert_eq!(
            model.multimodal.as_ref().and_then(|v| v.mmproj.as_deref()),
            Some("projector.gguf")
        );
    }

    #[test]
    fn nested_values_override_legacy_shims() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "pinned"

[[models]]
model = "Qwen3-8B-Q4_K_M"
ctx_size = 4096
gpu_id = "legacy-gpu"
parallel = 2
batch = 256
mmproj = "legacy.gguf"

[models.model_fit]
ctx_size = 8192
batch = 1024

[models.hardware]
device = "nested-gpu"

[models.throughput]
parallel = 8

[models.multimodal]
mmproj = "nested.gguf"
"#,
        )
        .unwrap();

        validate_config(&config).unwrap();
        let model = &config.models[0];
        assert_eq!(model.ctx_size, Some(8192));
        assert_eq!(model.batch, Some(1024));
        assert_eq!(model.gpu_id.as_deref(), Some("nested-gpu"));
        assert_eq!(model.parallel, Some(8));
        assert_eq!(model.mmproj.as_deref(), Some("nested.gguf"));
    }

    #[test]
    fn invalid_model_fit_batch_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
batch = 0
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].model_fit.batch must be at least 1 when set"
        );
    }

    #[test]
    fn invalid_split_mode_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
split_mode = "diagonal"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].hardware.split_mode must be one of: auto, none, layer, row"
        );
    }

    #[test]
    fn invalid_reasoning_format_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
        );
    }

    #[test]
    fn deepseek_legacy_reasoning_format_is_accepted() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "deepseek-legacy"
"#,
        )
        .unwrap();

        validate_config(&config).expect("deepseek-legacy should remain accepted");
    }

    #[test]
    fn invalid_speculative_draft_requires_policy_path_is_stable() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.speculative]
mode = "draft"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].speculative.draft_selection_policy must be set when models[0].speculative.mode = \"draft\" and no explicit draft model source is configured"
        );
    }

    #[test]
    fn invalid_mmproj_conflict_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
mmproj = "hardware.gguf"

[models.multimodal]
mmproj = "multimodal.gguf"
"#,
        )
        .unwrap();

        let err = validate_config(&config).unwrap_err();
        assert_eq!(
            err.to_string(),
            "models[0].multimodal.mmproj must match models[0].hardware.mmproj when both are set"
        );
    }

    #[test]
    fn integrated_full_surface_fixture_parses_validates_and_tracks_docs() {
        let config: MeshConfig = toml::from_str(FULL_SURFACE_VALID_FIXTURE).unwrap();

        validate_config(&config).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(
            config.owner_control.bind,
            Some("127.0.0.1:7447".parse().unwrap())
        );
        assert_eq!(
            config.owner_control.advertise_addr,
            Some("203.0.113.10:18443".parse().unwrap())
        );

        let defaults = config.defaults.as_ref().expect("defaults should parse");
        assert_eq!(
            defaults.model_fit.as_ref().and_then(|fit| fit.ctx_size),
            Some(8192)
        );
        assert_eq!(
            defaults
                .request_defaults
                .as_ref()
                .and_then(|request_defaults| request_defaults.temperature),
            Some(0.2)
        );

        let explicit = &config.models[0];
        assert_eq!(explicit.model, "Qwen/Qwen3-0.6B:Q4_K_M");
        assert_eq!(
            explicit.model_fit.as_ref().and_then(|fit| fit.ctx_size),
            Some(16384)
        );
        assert_eq!(
            explicit
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.stage_layer_start),
            Some(12)
        );
        assert_eq!(
            explicit
                .skippy
                .as_ref()
                .and_then(|skippy| skippy.prefill_chunk_schedule.as_deref()),
            Some("128,256,384")
        );

        let omitted = &config.models[1];
        assert_eq!(omitted.model, "ggml-org/gemma-3-270m-it-GGUF:Q8_0");
        assert!(
            omitted.model_fit.is_none(),
            "omitted per-model model_fit should stay absent"
        );
        assert!(
            omitted.request_defaults.is_none(),
            "omitted per-model request defaults should stay absent"
        );

        let matrix = include_str!("../../../../docs/skippy/CONFIGURATION.md");
        let matrix_keys = documented_matrix_key_paths();
        assert!(
            matrix_keys.len() >= 100,
            "expected a substantial canonical key-path set, found {}",
            matrix_keys.len()
        );
        for key in [
            "model_fit.ctx_size",
            "model_fit.prefix_cache.max_entries",
            "hardware.stage_layer_start",
            "hardware.stage_layer_end",
            "skippy.prefill_chunk_schedule",
            "speculative.draft_gpu_layers",
            "request_defaults.reasoning_budget",
            "multimodal.mmproj",
            "advanced.server.alias",
        ] {
            assert!(matrix.contains(key), "missing matrix doc entry {key}");
        }

        let docs_readme = include_str!("../../../../docs/README.md");
        let usage = include_str!("../../../../docs/USAGE.md");
        let cli = include_str!("../../../../docs/CLI.md");
        assert!(docs_readme.contains("[skippy/CONFIGURATION.md](skippy/CONFIGURATION.md)"));
        assert!(usage.contains("request payload values still win"));
        assert!(cli.contains("Request defaults only fill absent or null request fields"));
        assert!(cli.contains("Staged-only controls stay staged-only."));
    }

    #[test]
    fn integrated_invalid_fixture_reports_batch_then_pinned_device_paths() {
        let invalid: MeshConfig = toml::from_str(FULL_SURFACE_INVALID_FIXTURE).unwrap();
        let batch_error = validate_config(&invalid).unwrap_err().to_string();
        assert_eq!(
            batch_error,
            "models[0].model_fit.batch must be at least 1 when set"
        );

        let repaired_batch = FULL_SURFACE_INVALID_FIXTURE.replace("batch = 0", "batch = 64");
        let repaired_batch =
            repaired_batch.replace("[defaults.hardware]\ndevice = \"CUDA0\"\n\n", "");
        let repaired: MeshConfig = toml::from_str(&repaired_batch).unwrap();
        let pinned_error = validate_config(&repaired).unwrap_err().to_string();
        assert_eq!(
            pinned_error,
            "defaults.hardware.device must be set to a non-empty value when gpu.assignment = \"pinned\""
        );
    }
}
