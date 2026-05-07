use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use skippy_cache::{
    ExactStateCache, PrefixCandidatePolicy, ResidentActivationCache, ResidentCacheConfig,
    ResidentPrefixCache,
};
use skippy_protocol::{
    LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::ModelInfo;

use super::{ExactStateExtra, KvStageIntegration, StageKvMode, StagePrefixCachePayload};

impl KvStageIntegration {
    pub fn from_config(config: &StageConfig) -> Result<Option<Self>> {
        let Some(cache_config) = effective_cache_config(config) else {
            return Ok(None);
        };
        let mode = match cache_config.mode {
            StageKvCacheMode::Disabled | StageKvCacheMode::Auto => StageKvMode::Disabled,
            StageKvCacheMode::Record => StageKvMode::Record,
            StageKvCacheMode::LookupRecord => StageKvMode::LookupRecord,
        };
        if mode == StageKvMode::Disabled {
            return Ok(None);
        }
        let payload = effective_cache_payload(config, cache_config.payload);
        if payload == StagePrefixCachePayload::Disabled {
            return Ok(None);
        }
        if model_requires_recurrent_state(config)
            && matches!(payload, StagePrefixCachePayload::ResidentKv)
        {
            return Ok(None);
        }
        let candidate_policy = PrefixCandidatePolicy::from_cache(&cache_config);
        let resident_config = ResidentCacheConfig::from_stage(config, &cache_config);
        Ok(Some(Self {
            mode,
            payload,
            correctness_mode: false,
            trust_local_writes: true,
            candidate_policy,
            inflight_records: Arc::new(Mutex::new(BTreeSet::new())),
            resident: Arc::new(Mutex::new(ResidentPrefixCache::new(resident_config))),
            activations: Arc::new(Mutex::new(ResidentActivationCache::new(resident_config))),
            exact_states: Arc::new(Mutex::new(ExactStateCache::<ExactStateExtra>::new(
                cache_config.max_entries.clamp(1, 512),
                cache_config.max_bytes,
            ))),
        }))
    }
}

fn effective_cache_payload(
    config: &StageConfig,
    requested: StageKvCachePayload,
) -> StagePrefixCachePayload {
    if matches!(requested, StageKvCachePayload::Auto) && model_requires_recurrent_state(config) {
        return StagePrefixCachePayload::KvRecurrent;
    }
    match requested {
        StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
        StageKvCachePayload::KvRecurrent => StagePrefixCachePayload::KvRecurrent,
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => infer_cache_payload(config),
    }
}

fn model_requires_recurrent_state(config: &StageConfig) -> bool {
    let Some(path) = kv_cache_inspection_path(config) else {
        return false;
    };
    let Ok(info) = ModelInfo::open(path) else {
        return false;
    };
    let Ok(tensors) = info.tensors() else {
        return false;
    };
    tensors
        .iter()
        .any(|tensor| tensor_name_requires_recurrent_state(&tensor.name))
}

fn kv_cache_inspection_path(config: &StageConfig) -> Option<PathBuf> {
    let path = config.model_path.as_deref()?;
    match config.load_mode {
        LoadMode::LayerPackage => {
            let package_dir = std::path::Path::new(path);
            layer_package_inspection_path(package_dir, config.layer_start, config.layer_end)
                .or_else(|| layer_package_metadata_path(package_dir))
                .or_else(|| Some(PathBuf::from(path)))
        }
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => Some(PathBuf::from(path)),
    }
}

fn layer_package_inspection_path(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
) -> Option<PathBuf> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(manifest_path).ok()?).ok()?;
    let layers = manifest.get("layers")?.as_array()?;
    let selected = layers
        .iter()
        .enumerate()
        .find(|(index, layer)| {
            let layer_index = layer
                .get("layer_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(*index as u32);
            layer_index >= layer_start && layer_index < layer_end
        })
        .or_else(|| layers.first().map(|layer| (0, layer)))?
        .1;
    let path = selected.get("path")?.as_str()?;
    let path = PathBuf::from(path);
    if path.is_absolute() {
        return None;
    }
    let absolute = package_dir.join(path);
    absolute.is_file().then_some(absolute)
}

fn layer_package_metadata_path(package_dir: &Path) -> Option<PathBuf> {
    let metadata = package_dir.join("shared/metadata.gguf");
    metadata.is_file().then_some(metadata)
}

fn tensor_name_requires_recurrent_state(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".ssm")
        || lower.contains("ssm_")
        || lower.contains("time_mix")
        || lower.contains("recurrent")
        || lower.contains("rwkv")
}

fn infer_cache_payload(config: &StageConfig) -> StagePrefixCachePayload {
    let identity = format!(
        "{} {}",
        config.model_id,
        config.model_path.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();

    if identity.contains("falcon-h1")
        || identity.contains("qwen3next")
        || identity.contains("qwen3-next")
        || identity.contains("qwen3.6")
        || identity.contains("qwen3_6")
        || identity.contains("kimi-linear")
        || identity.contains("kimi_linear")
    {
        return StagePrefixCachePayload::KvRecurrent;
    }
    if identity.contains("llama")
        || identity.contains("qwen3")
        || identity.contains("deepseek")
        || identity.contains("glm4")
        || identity.contains("glm-4.7")
        || identity.contains("glm47")
        || identity.contains("glm4.7")
        || identity.contains("olmo")
        || identity.contains("gemma")
        || identity.contains("minimax")
    {
        return StagePrefixCachePayload::ResidentKv;
    }
    StagePrefixCachePayload::Disabled
}

fn effective_cache_config(config: &StageConfig) -> Option<StageKvCacheConfig> {
    if let Some(cache) = config.kv_cache.clone() {
        return Some(cache);
    }
    let mode = std::env::var("SKIPPY_KV_CACHE")
        .or_else(|_| std::env::var("SKIPPY_PREFIX_CACHE"))
        .ok()
        .and_then(|value| parse_cache_mode(&value));
    let mode = mode?;
    let max_entries = std::env::var("SKIPPY_KV_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let max_bytes = std::env::var("SKIPPY_KV_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let min_tokens = std::env::var("SKIPPY_KV_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let shared_prefix_stride_tokens = std::env::var("SKIPPY_KV_CACHE_SHARED_STRIDE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let shared_prefix_record_limit = std::env::var("SKIPPY_KV_CACHE_SHARED_RECORD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let payload = std::env::var("SKIPPY_KV_CACHE_PAYLOAD")
        .ok()
        .and_then(|value| parse_cache_payload(&value))
        .unwrap_or(StageKvCachePayload::Auto);
    Some(StageKvCacheConfig {
        mode,
        payload,
        max_entries,
        max_bytes,
        min_tokens,
        shared_prefix_stride_tokens,
        shared_prefix_record_limit,
    })
}

fn parse_cache_payload(value: &str) -> Option<StageKvCachePayload> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCachePayload::Auto),
        "resident" | "resident-kv" | "kv" => Some(StageKvCachePayload::ResidentKv),
        "kv-recurrent" | "kvrecurrent" => Some(StageKvCachePayload::KvRecurrent),
        "full" | "full-state" | "fullstate" => Some(StageKvCachePayload::FullState),
        _ => None,
    }
}

fn parse_cache_mode(value: &str) -> Option<StageKvCacheMode> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCacheMode::Auto),
        "0" | "off" | "false" | "disabled" | "disable" => Some(StageKvCacheMode::Disabled),
        "record" => Some(StageKvCacheMode::Record),
        "1" | "on" | "true" | "lookup-record" | "lookuprecord" | "exact" => {
            Some(StageKvCacheMode::LookupRecord)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::FlashAttentionType;

    #[test]
    fn recurrent_tensor_names_require_exact_state_cache() {
        assert!(tensor_name_requires_recurrent_state("blk.0.ssm_a"));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.ssm_conv1d.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.time_mix_k.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.rwkv_gate.weight"
        ));
        assert!(!tensor_name_requires_recurrent_state("blk.0.attn_q.weight"));
        assert!(!tensor_name_requires_recurrent_state(
            "blk.0.ffn_down.weight"
        ));
    }

    #[test]
    fn layer_package_inspection_uses_representative_layer_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::create_dir_all(dir.path().join("layers")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(dir.path().join("layers/00000.gguf"), b"layer0").unwrap();
        fs::write(dir.path().join("layers/00001.gguf"), b"layer1").unwrap();
        let manifest = serde_json::json!({
            "layers": [
                { "layer_index": 0, "path": "layers/00000.gguf" },
                { "layer_index": 1, "path": "layers/00001.gguf" }
            ]
        });
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = test_config("example/recurrent-package");
        config.load_mode = LoadMode::LayerPackage;
        config.model_path = Some(dir.path().to_string_lossy().to_string());
        config.layer_start = 1;
        config.layer_end = 2;

        assert_eq!(
            kv_cache_inspection_path(&config),
            Some(dir.path().join("layers/00001.gguf"))
        );
    }

    #[test]
    fn layer_package_inspection_falls_back_to_shared_metadata() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        let manifest = serde_json::json!({
            "layers": [
                { "layer_index": 0, "path": "layers/missing.gguf" }
            ]
        });
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = test_config("example/recurrent-package");
        config.load_mode = LoadMode::LayerPackage;
        config.model_path = Some(dir.path().to_string_lossy().to_string());

        assert_eq!(
            kv_cache_inspection_path(&config),
            Some(dir.path().join("shared/metadata.gguf"))
        );
    }

    #[test]
    fn cache_payload_inference_selects_recurrent_and_dense_families() {
        assert_eq!(
            infer_cache_payload(&test_config("tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M")),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            infer_cache_payload(&test_config(
                "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF:IQ2_XXS"
            )),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            infer_cache_payload(&test_config(
                "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M"
            )),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M")),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("unsloth/GLM-4.7-Flash-GGUF:Q4_K_M")),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("example/unknown-model:Q4_K_M")),
            StagePrefixCachePayload::Disabled
        );
    }

    #[test]
    fn explicit_cache_payload_overrides_identity_inference() {
        let config = test_config("tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M");

        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::ResidentKv),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::KvRecurrent),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::FullState),
            StagePrefixCachePayload::FullState
        );
    }

    #[test]
    fn parses_cache_mode_and_payload_aliases() {
        assert_eq!(
            parse_cache_mode("lookup_record"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(
            parse_cache_mode("exact"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(parse_cache_mode("off"), Some(StageKvCacheMode::Disabled));
        assert_eq!(
            parse_cache_payload("kv_recurrent"),
            Some(StageKvCachePayload::KvRecurrent)
        );
        assert_eq!(
            parse_cache_payload("resident"),
            Some(StageKvCachePayload::ResidentKv)
        );
        assert_eq!(parse_cache_payload("nope"), None);
    }

    fn test_config(model_id: &str) -> StageConfig {
        StageConfig {
            run_id: "test-run".to_string(),
            topology_id: "test-topology".to_string(),
            model_id: model_id.to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 256,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }
}
