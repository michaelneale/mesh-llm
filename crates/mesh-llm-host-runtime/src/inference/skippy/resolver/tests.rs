use super::*;
use crate::inference::skippy::{SkippyPackageIdentity, SkippyTelemetryOptions, StageWireDType};
use crate::plugin::{MeshConfig, ReasoningBudget, RequestDefaultsConfig};
use serde_json::Value;
use skippy_protocol::{LoadMode, StageKvCachePayload};
use skippy_server::{EmbeddedReasoningEnabled, EmbeddedReasoningFormat};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../../../tests/fixtures/skippy_full_surface_valid.toml");
const FULL_SURFACE_INVALID_FIXTURE: &str =
    include_str!("../../../../tests/fixtures/skippy_full_surface_invalid.toml");

type FullSurfaceFixture = (MeshConfig, NamedTempFile, NamedTempFile, NamedTempFile);

fn fake_package_identity(layer_count: u32) -> SkippyPackageIdentity {
    SkippyPackageIdentity {
        package_ref: "gguf:///models/qwen.gguf".to_string(),
        manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .to_string(),
        source_model_path: PathBuf::from("/models/qwen.gguf"),
        source_model_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            .to_string(),
        source_model_bytes: 1234,
        source_files: Vec::new(),
        layer_count,
        activation_width: 4096,
        tensor_count: 100,
    }
}

fn parse_config(toml: &str) -> MeshConfig {
    toml::from_str(toml).expect("config should parse")
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(bytes, value);
}

fn temp_model_file() -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("temp model file");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    file.write_all(&bytes).expect("write fake gguf");
    file.flush().expect("flush fake gguf");
    file
}

#[test]
fn resolver_applies_precedence_and_keeps_request_defaults_out_of_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
ctx_size = 8192
batch = 512
ubatch = 128
cache_type_v = "q8_0"

[defaults.hardware]
device = "CUDA0"

[defaults.throughput]
parallel = 2

[defaults.skippy]
activation_wire_dtype = "q8"

[defaults.request_defaults]
temperature = 0.2
max_tokens = 128

[[models]]
model = "ggml-org/gemma-3-270m-it-GGUF:Q8_0"

[models.model_fit]
ctx_size = 16384
batch = 1024
cache_type_k = "f16"

[models.hardware]
device = "CUDA1"

[models.throughput]
parallel = 3

[models.skippy]
activation_wire_dtype = "f32"

[models.request_defaults]
temperature = 0.4
"#,
    );
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.7),
        max_tokens: Some(256),
        reasoning_budget: Some(ReasoningBudget::Integer(512)),
        ..Default::default()
    };
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "ggml-org/gemma-3-270m-it-GGUF:Q8_0",
        model_path: model_file.path(),
        model_bytes: 8 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(16 * 1024 * 1024 * 1024),
        request_defaults: Some(&request_defaults),
    })
    .unwrap();

    assert_eq!(resolved.model_fit.ctx_size, 16384);
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 128);
    assert_eq!(resolved.hardware.device.as_deref(), Some("CUDA1"));
    assert_eq!(resolved.throughput.parallel, 3);
    assert_eq!(resolved.skippy.activation_wire_dtype, StageWireDType::F32);
    assert_eq!(resolved.request_defaults.max_tokens, 256);
    assert_eq!(resolved.request_defaults.temperature, Some(0.7));
    assert_eq!(
        resolved.request_defaults.reasoning_budget,
        Some(ReasoningBudget::Integer(512))
    );

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let serialized: Value = serde_json::to_value(&stage_config).expect("stage config json");
    let object = serialized.as_object().expect("stage config object");
    assert!(!object.contains_key("request_defaults"));
    assert!(!object.contains_key("temperature"));
    assert_eq!(object.get("ctx_size").and_then(Value::as_u64), Some(16384));
}

#[test]
fn resolver_macro_expands_kv_cache_tuning_profile_and_safety_margin() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
kv_cache_policy = "saver"

[defaults.hardware]
safety_margin_gb = 1.5

[defaults.throughput]
tuning_profile = "throughput"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(12 * 1024 * 1024 * 1024),
        request_defaults: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q8_0");
    assert_eq!(resolved.model_fit.kv_offload, "true");
    assert_eq!(resolved.throughput.tuning_profile, "throughput");
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 256);
    assert_eq!(resolved.throughput.parallel, 2);
    assert_eq!(resolved.throughput.continuous_batching, "true");
    assert_eq!(resolved.hardware.fit_target_mib, Some(10_752));
}

#[test]
fn per_model_kv_macro_beats_global_explicit_cache_fields_unless_model_explicit_exists() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
cache_type_k = "f16"
cache_type_v = "f16"
kv_offload = false

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
kv_cache_policy = "saver"
cache_type_v = "q4_0"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap();

    assert_eq!(resolved.model_fit.kv_cache_policy, "saver");
    assert_eq!(resolved.model_fit.cache_type_k, "q8_0");
    assert_eq!(resolved.model_fit.cache_type_v, "q4_0");
    assert_eq!(resolved.model_fit.kv_offload, "true");
}

#[test]
fn per_model_throughput_macro_beats_global_explicit_fields_unless_model_explicit_exists() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
batch = 64
ubatch = 32

[defaults.throughput]
parallel = 7
continuous_batching = false

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.model_fit]
ubatch = 999

[models.throughput]
tuning_profile = "throughput"
parallel = 11
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: Path::new("/models/qwen.gguf"),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap();

    assert_eq!(resolved.throughput.tuning_profile, "throughput");
    assert_eq!(resolved.model_fit.batch, 1024);
    assert_eq!(resolved.model_fit.ubatch, 999);
    assert_eq!(resolved.throughput.parallel, 11);
    assert_eq!(resolved.throughput.continuous_batching, "true");
}

fn resolve_qwen_test_config(
    mesh_config: &MeshConfig,
    model_path: &Path,
    request_defaults: Option<&RequestDefaultsConfig>,
) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path,
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults,
    })
    .unwrap()
}

fn assert_load_time_config_unchanged(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    assert_eq!(without_request.model_fit, with_request.model_fit);
    assert_eq!(without_request.hardware, with_request.hardware);
    assert_eq!(without_request.skippy, with_request.skippy);
}

fn assert_request_defaults_follow_request_overrides(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    assert_eq!(without_request.request_defaults.temperature, Some(0.2));
    assert_eq!(with_request.request_defaults.temperature, Some(0.9));
    assert_eq!(without_request.request_defaults.max_tokens, 128);
    assert_eq!(with_request.request_defaults.max_tokens, 32);
}

fn assert_stage_config_ignores_request_defaults(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    let baseline_stage = without_request
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("baseline stage config should build");
    let override_stage = with_request
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("override stage config should build");

    assert_eq!(baseline_stage.model_id, override_stage.model_id);
    assert_eq!(baseline_stage.model_path, override_stage.model_path);
    assert_eq!(baseline_stage.ctx_size, override_stage.ctx_size);
    assert_eq!(baseline_stage.lane_count, override_stage.lane_count);
    assert_eq!(baseline_stage.n_batch, override_stage.n_batch);
    assert_eq!(baseline_stage.n_ubatch, override_stage.n_ubatch);
    assert_eq!(baseline_stage.n_gpu_layers, override_stage.n_gpu_layers);
    assert_eq!(baseline_stage.cache_type_k, override_stage.cache_type_k);
    assert_eq!(baseline_stage.cache_type_v, override_stage.cache_type_v);
    assert_eq!(
        baseline_stage.flash_attn_type,
        override_stage.flash_attn_type
    );
    assert_eq!(
        baseline_stage.selected_device,
        override_stage.selected_device
    );
    assert_eq!(baseline_stage.load_mode, override_stage.load_mode);
}

fn assert_embedded_openai_max_tokens_follow_request_overrides(
    without_request: &ResolvedSkippyConfig,
    with_request: &ResolvedSkippyConfig,
) {
    let baseline_openai = without_request
        .to_embedded_openai_args(4096, true)
        .expect("baseline openai args should build");
    let override_openai = with_request
        .to_embedded_openai_args(4096, true)
        .expect("override openai args should build");

    assert_eq!(baseline_openai.default_max_tokens, 128);
    assert_eq!(override_openai.default_max_tokens, 32);
}

#[test]
fn request_overrides_change_request_time_defaults_without_mutating_load_time_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
ctx_size = 4096

[defaults.request_defaults]
temperature = 0.2
max_tokens = 128
"#,
    );
    let model_file = temp_model_file();
    let without_request = resolve_qwen_test_config(&mesh_config, model_file.path(), None);
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.9),
        max_tokens: Some(32),
        ..Default::default()
    };
    let with_request =
        resolve_qwen_test_config(&mesh_config, model_file.path(), Some(&request_defaults));

    assert_load_time_config_unchanged(&without_request, &with_request);
    assert_request_defaults_follow_request_overrides(&without_request, &with_request);
    assert_stage_config_ignores_request_defaults(&without_request, &with_request);
    assert_embedded_openai_max_tokens_follow_request_overrides(&without_request, &with_request);
}

#[test]
fn supported_request_defaults_translate_into_embedded_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.request_defaults]
presence_penalty = 1.0
frequency_penalty = 0.5
seed = 7
logit_bias = { "12" = -4.0 }
repeat_last_n = 32
reasoning_format = "deepseek-legacy"
reasoning_enabled = "on"
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap();

    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("embedded OpenAI args should build");
    assert_eq!(openai.request_defaults.presence_penalty, Some(1.0));
    assert_eq!(openai.request_defaults.frequency_penalty, Some(0.5));
    assert_eq!(openai.request_defaults.seed, Some(7));
    assert_eq!(openai.request_defaults.repeat_last_n, Some(32));
    assert_eq!(
        openai.request_defaults.reasoning_format,
        Some(EmbeddedReasoningFormat::DeepseekLegacy)
    );
    assert_eq!(
        openai.request_defaults.reasoning_enabled,
        Some(EmbeddedReasoningEnabled::Enabled)
    );
    assert_eq!(
        openai
            .request_defaults
            .logit_bias
            .as_ref()
            .and_then(|value| value.get("12"))
            .and_then(serde_json::Value::as_f64),
        Some(-4.0)
    );

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let serialized = serde_json::to_value(&stage_config).expect("stage config json");
    let object = serialized.as_object().expect("stage config object");
    assert!(!object.contains_key("presence_penalty"));
    assert!(!object.contains_key("repeat_last_n"));
    assert!(!object.contains_key("logit_bias"));
}

#[test]
fn unsupported_request_defaults_fail_closed_during_resolution() {
    let mesh_config = parse_config(
        r#"
[defaults.request_defaults]
chat_template = "unsafe-template"
"#,
    );
    let model_file = temp_model_file();

    let err = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("defaults.request_defaults.chat_template"));
}

#[test]
fn family_policy_beats_builtin_wire_dtype_when_config_is_unset() {
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &MeshConfig::default(),
        model_id: "ggml-org/gemma-3-270m-it-GGUF:Q8_0",
        model_path: Path::new("/models/gemma.gguf"),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap();

    assert_eq!(resolved.skippy.activation_wire_dtype, StageWireDType::F32);
}

#[test]
fn staged_controls_propagate_into_stage_config_and_embedded_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.model_fit]
prompt_cache = true

[defaults.model_fit.prefix_cache]
enabled = true
max_entries = 9
min_tokens = 96
shared_stride_tokens = 48
shared_record_limit = 3
payload_mode = "resident-kv"

[defaults.skippy]
activation_wire_dtype = "q8"
prefill_chunking = "schedule"
prefill_chunk_size = 128
prefill_chunk_schedule = "128,256,384"

[defaults.speculative]
mode = "draft"
draft_model_path = "/models/qwen3-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail-open"
draft_max_tokens = 8
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .expect("config should resolve");

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    let kv_cache = stage_config
        .kv_cache
        .expect("kv cache should be configured");
    assert_eq!(kv_cache.max_entries, 9);
    assert_eq!(kv_cache.min_tokens, 96);
    assert_eq!(kv_cache.shared_prefix_stride_tokens, 48);
    assert_eq!(kv_cache.shared_prefix_record_limit, 3);
    assert_eq!(kv_cache.payload, StageKvCachePayload::ResidentKv);

    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("embedded args should build");
    assert_eq!(openai.prefill_chunk_policy, "schedule");
    assert_eq!(openai.prefill_chunk_size, 128);
    assert_eq!(
        openai.prefill_chunk_schedule.as_deref(),
        Some("128,256,384")
    );
    assert_eq!(openai.speculative_window, 8);
    assert_eq!(
        openai.draft_model_path.as_deref(),
        Some(Path::new("/models/qwen3-draft.gguf"))
    );
    assert_eq!(
        openai.wire_dtype,
        skippy_protocol::binary::WireActivationDType::Q8
    );
}

#[test]
fn staged_only_controls_fail_closed_for_single_stage_loads() {
    let mesh_config = parse_config(
        r#"
[defaults.skippy]
prefill_chunk_size = 128
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .expect("config should resolve");

    let err = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(err.contains("prefill chunk controls require staged serving"));
}

#[test]
fn incompatible_draft_pairing_warn_disable_turns_speculation_off() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = "/models/llama-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "warn_disable"
draft_max_tokens = 8
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .expect("warn_disable should resolve");

    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.draft_model_path.is_none());
}

#[test]
fn incompatible_draft_pairing_fail_closed_rejects_before_launch() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = "/models/llama-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 8
"#,
    );
    let model_file = temp_model_file();

    let err = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("incompatible speculative draft pairing"));
}

#[test]
fn manual_stage_layer_range_is_staged_only_and_reaches_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.hardware]
stage_layer_start = 12
stage_layer_end = 24
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .expect("config should resolve");

    let err = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(err.contains("staged-only controls"));

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(32)), LoadMode::RuntimeSlice)
        .expect("stage config should preserve explicit layer range");
    assert_eq!((stage_config.layer_start, stage_config.layer_end), (12, 24));
}

#[test]
fn unsupported_speculative_thresholds_fail_closed() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
draft_acceptance_threshold = 0.5
"#,
    );
    let model_file = temp_model_file();

    let err = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("draft_acceptance_threshold"));
}

fn full_surface_fixture_with_model_paths() -> FullSurfaceFixture {
    let mut mesh_config = parse_config(FULL_SURFACE_VALID_FIXTURE);
    let explicit_model = temp_model_file();
    let defaults_model = temp_model_file();
    let projector_file = NamedTempFile::new().expect("temp projector");

    let explicit_projector_path = projector_file.path().display().to_string();
    let explicit_hardware = mesh_config.models[0]
        .hardware
        .as_mut()
        .expect("explicit hardware");
    explicit_hardware.model_path = Some(explicit_model.path().display().to_string());
    explicit_hardware.mmproj = Some(explicit_projector_path.clone());
    mesh_config.models[0]
        .multimodal
        .as_mut()
        .expect("explicit multimodal")
        .mmproj = Some(explicit_projector_path);
    mesh_config.models[1]
        .hardware
        .as_mut()
        .expect("defaults hardware")
        .model_path = Some(defaults_model.path().display().to_string());

    (mesh_config, explicit_model, defaults_model, projector_file)
}

fn resolve_full_surface_model(
    mesh_config: &MeshConfig,
    model_id: &str,
    model_path: &Path,
    model_bytes: u64,
    request_defaults: Option<&RequestDefaultsConfig>,
) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config,
        model_id,
        model_path,
        model_bytes,
        allocatable_memory_bytes: Some(12 * 1024 * 1024 * 1024),
        request_defaults,
    })
    .expect("full-surface fixture model should resolve")
}

fn assert_explicit_full_surface_resolution(explicit: &ResolvedSkippyConfig) {
    assert_eq!(explicit.model_fit.ctx_size, 16384);
    assert_eq!(explicit.model_fit.batch, 1024);
    assert_eq!(explicit.model_fit.ubatch, 128);
    assert_eq!(explicit.hardware.device.as_deref(), Some("CUDA1"));
    assert_eq!(explicit.hardware.stage_layer_start, Some(12));
    assert_eq!(explicit.hardware.stage_layer_end, Some(24));
    assert_eq!(explicit.throughput.parallel, 3);
    assert_eq!(explicit.throughput.threads, Some(10));
    assert_eq!(explicit.throughput.threads_batch, Some(6));
    assert_eq!(explicit.request_defaults.temperature, Some(0.7));
    assert_eq!(explicit.request_defaults.max_tokens, 256);
}

fn assert_explicit_full_surface_stage_config(explicit: &ResolvedSkippyConfig) {
    let stage = explicit
        .to_stage_config(Some(fake_package_identity(32)), LoadMode::RuntimeSlice)
        .expect("stage config should build");

    assert_eq!((stage.layer_start, stage.layer_end), (12, 24));
    assert_eq!(stage.n_batch, Some(1024));
    assert_eq!(stage.n_ubatch, Some(128));
    assert_eq!(stage.n_gpu_layers, 99);
}

fn assert_explicit_full_surface_runtime_options(explicit: &ResolvedSkippyConfig) {
    let runtime = explicit
        .to_embedded_runtime_options(
            &SkippyTelemetryOptions::off(),
            Some(fake_package_identity(32)),
            LoadMode::RuntimeSlice,
        )
        .expect("embedded runtime options should build");

    assert_eq!(runtime.n_threads, Some(10));
    assert_eq!(runtime.n_threads_batch, Some(6));
    assert_eq!(runtime.config.layer_start, 12);
    assert_eq!(runtime.config.layer_end, 24);
}

fn assert_explicit_full_surface_openai_args(explicit: &ResolvedSkippyConfig) {
    let openai = explicit
        .to_embedded_openai_args(4096, true)
        .expect("embedded openai args should build");

    assert_eq!(openai.prefill_chunk_policy, "schedule");
    assert_eq!(openai.prefill_chunk_size, 128);
    assert_eq!(
        openai.prefill_chunk_schedule.as_deref(),
        Some("128,256,384")
    );
    assert_eq!(openai.speculative_window, 8);
    assert_eq!(openai.draft_n_gpu_layers, Some(12));
    assert_eq!(openai.default_max_tokens, 256);
}

fn assert_defaults_only_full_surface_resolution(omitted: &ResolvedSkippyConfig) {
    assert_eq!(omitted.model_fit.ctx_size, 8192);
    assert_eq!(omitted.model_fit.batch, 512);
    assert_eq!(omitted.model_fit.ubatch, 128);
    assert_eq!(omitted.hardware.device.as_deref(), Some("CUDA2"));
    assert_eq!(omitted.throughput.parallel, 2);
    assert_eq!(omitted.request_defaults.temperature, Some(0.2));
    assert_eq!(omitted.request_defaults.max_tokens, 128);
}

fn assert_defaults_only_full_surface_model_load_options(omitted: &ResolvedSkippyConfig) {
    let single_stage = omitted
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("defaults-only model should remain single-stage safe");

    assert_eq!(single_stage.ctx_size, 8192);
    assert_eq!(single_stage.n_batch, Some(512));
    assert_eq!(single_stage.n_ubatch, Some(128));
}

#[test]
fn integrated_full_surface_fixture_resolves_defaults_overrides_staged_and_runtime_paths() {
    let (mesh_config, explicit_model, defaults_model, _projector_file) =
        full_surface_fixture_with_model_paths();
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.7),
        max_tokens: Some(256),
        ..Default::default()
    };

    let explicit = resolve_full_surface_model(
        &mesh_config,
        "Qwen/Qwen3-0.6B:Q4_K_M",
        explicit_model.path(),
        4 * 1024 * 1024 * 1024,
        Some(&request_defaults),
    );
    assert_explicit_full_surface_resolution(&explicit);
    assert_explicit_full_surface_stage_config(&explicit);
    assert_explicit_full_surface_runtime_options(&explicit);
    assert_explicit_full_surface_openai_args(&explicit);

    let omitted = resolve_full_surface_model(
        &mesh_config,
        "ggml-org/gemma-3-270m-it-GGUF:Q8_0",
        defaults_model.path(),
        2 * 1024 * 1024 * 1024,
        None,
    );
    assert_defaults_only_full_surface_resolution(&omitted);
    assert_defaults_only_full_surface_model_load_options(&omitted);
}

#[test]
fn integrated_invalid_fixture_fails_closed_for_request_defaults_and_single_stage_staged_knobs() {
    let repaired_batch = FULL_SURFACE_INVALID_FIXTURE.replace("batch = 0", "batch = 64");
    let repaired_device = format!(
        "{repaired_batch}\ndevice = \"CUDA0\"\n",
        repaired_batch = repaired_batch.trim_end()
    );

    let unsupported_request = parse_config(&repaired_device);
    let model_file = temp_model_file();
    let unsupported_error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &unsupported_request,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .unwrap_err()
    .to_string();
    assert!(unsupported_error.contains("defaults.request_defaults.chat_template"));

    let staged_only_config = parse_config(&repaired_device.replace(
        "\n[defaults.request_defaults]\nchat_template = \"unsafe-template\"\n",
        "\n",
    ));
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &staged_only_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
    })
    .expect("staged-only config should resolve before translation gating");
    let staged_only_error = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(staged_only_error.contains("prefill chunk controls require staged serving"));
}
